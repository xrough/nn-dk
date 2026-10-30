"""Scientific and software invariants for learned dynamics and statistical heads."""
import time

import numpy as np
import pytest
import torch

from nn_dk.models import ConservativeTransition, DirectMomentRegressor, RidgeMomentRegressor
from nn_dk.observations import tangent_basis
from nn_dk.training import (
    fit_baselines,
    fit_models,
    independent_pair_labels,
    load_a,
    load_b,
    load_baselines,
)
from nn_dk.types import Context, QueryBank, TargetSpec, TrajectoryBatch


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)
    torch.manual_seed(12)


def ensemble(context, target, seed=0):
    """Positive test paths, with two requested times and independent path innovations."""
    rng = np.random.default_rng(seed)
    times = np.asarray([0, 1/64, 2/64])
    x = rng.dirichlet(np.full(target.n_cells, 100.), size=(6, len(times)))
    return TrajectoryBatch(times, x, target)


def groups():
    target = TargetSpec(n_cells=8)
    a, b = Context(n_particles=512), Context(cos_coeffs=(0.1, 0., 0.), n_particles=512)
    return target, QueryBank(times=(1/64, 2/64)), [(a, ensemble(a,target))], [(b,ensemble(b,target,1))]


def test_flux_covariance_conserve_and_tangent_likelihood_matches_distribution():
    target = TargetSpec(n_cells=8)
    model = ConservativeTransition(target)
    states = torch.full((3, 8), 1/8, dtype=torch.float64)
    contexts = torch.tensor(np.tile(Context().features(), (3,1)), dtype=torch.float32)
    drift, covariance = model(states, contexts, 0.)
    assert torch.max(drift.sum(-1).abs()) < 1e-14
    assert torch.max(covariance.sum(-1).abs()) < 1e-12
    q = torch.tensor(tangent_basis(8), dtype=torch.float64)
    reduced = q.T@covariance@q * model.lag/contexts[:,9].double().exp()[:,None,None]
    assert torch.linalg.eigvalsh(reduced).min() > 0
    increment = torch.randn(3,7,dtype=torch.float64)*1e-4
    next_states = states + model.lag*drift + increment@q.T
    actual = model.nll(states, next_states, contexts, 0.)
    expected = -torch.distributions.MultivariateNormal(torch.zeros_like(increment), covariance_matrix=reduced).log_prob(increment).mean()
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
    actual.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_model_inputs_are_causal_and_query_covariance_is_joint_psd():
    target = TargetSpec(n_cells=8)
    context = Context()
    a = ConservativeTransition(target)
    states = torch.full((2,8), 1/8)
    features = torch.tensor(np.tile(context.features(),(2,1)),dtype=torch.float32)
    inputs = a.inputs(states, features, torch.tensor([0.,0.2]))
    torch.testing.assert_close(inputs[0,:-1], inputs[1,:-1])
    assert not torch.equal(inputs[0,-1],inputs[1,-1])
    b = DirectMomentRegressor(target, QueryBank())
    prediction = b.predict(context, QueryBank())
    assert prediction.mean.shape == (12,)
    assert np.linalg.eigvalsh(prediction.scaled_covariance).min() > -1e-7
    other = Context(kappa=0.3, cos_coeffs=(0.1,0.2,0.))
    batched = b.predict_batch([context, other], QueryBank())
    for c, combined in zip((context, other), batched):
        single = b.predict(c, QueryBank())
        np.testing.assert_allclose(combined.mean, single.mean, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(combined.scaled_covariance, single.scaled_covariance, rtol=1e-5, atol=1e-7)
        assert combined.context_id == c.context_id
        assert combined.diagnostics['batch_size'] == 2
    with pytest.raises(ValueError, match="fixed query bank"):
        b.predict(context, QueryBank(times=(0.5,1.)))


def test_replicate_difference_labels_use_disjoint_pairs():
    target = TargetSpec(n_cells=8)
    c = Context()
    bank = QueryBank(times=(1/64,))
    batch = ensemble(c,target)
    x, outcomes, covariance = independent_pair_labels([(c,batch)],bank)
    expected = bank.extract(batch).reshape(3,2,4)
    np.testing.assert_allclose(outcomes.numpy(), expected, rtol=1e-6, atol=1e-7)
    difference = expected[:,0]-expected[:,1]
    np.testing.assert_allclose(covariance.numpy(), c.n_particles/2*np.einsum('ni,nj->nij',difference,difference), rtol=1e-6, atol=1e-7)
    assert len(x) == len(batch.states)//2


def test_sampler_stops_invalid_paths_and_rejects_lag_transfer():
    target = TargetSpec(n_cells=8)
    model = ConservativeTransition(target)
    # A deliberately catastrophic face current must fail visibly, without repair.
    def components(states, contexts, elapsed):
        n = len(states)
        drift = torch.zeros((n,8),dtype=torch.float64)
        drift[:,0], drift[:,1] = -1e5, 1e5
        return drift, torch.zeros(n,8,8,dtype=torch.float64), torch.zeros(n,8,8,dtype=torch.float64)
    model.components = components
    result = model.sample(Context(),[0.,1/64,2/64],4,13)
    assert not result.valid.any()
    np.testing.assert_allclose(result.failure_times,1/64)
    assert np.isnan(result.states[:,1:]).all()
    np.testing.assert_allclose(result.states[:,0].sum(-1),1.,atol=1e-14)
    with pytest.raises(ValueError,match="multiples"):
        model.sample(Context(),[0.01],1,1)
    model.supported_particles=(512,)
    with pytest.raises(ValueError,match="retrain"):
        model.sample(Context(n_particles=2048),[0.],1,1)


def test_group_leakage_is_rejected(tmp_path):
    target, bank, train, valid = groups()
    with pytest.raises(ValueError,match="overlap"):
        fit_models(train, train, target, bank, 1/64, {"epochs":1},tmp_path,1)


@pytest.mark.integration
def test_checkpoint_roundtrip_exact_resume_and_baselines(tmp_path):
    target, bank, train, valid = groups()
    config = dict(epochs=1,batch_size=4,patience=10,threads=1)
    interrupted = tmp_path/'interrupted'
    metadata = fit_models(train,valid,target,bank,1/64,config,interrupted,7,deadline=time.monotonic()-1)
    assert not metadata['complete']
    assert load_a(interrupted/'a_seed7.pt').supported_particles == (512,)
    assert load_b(interrupted/'b_seed7.pt').query_bank == bank
    first_epoch = fit_models(train,valid,target,bank,1/64,config,interrupted,7)
    assert first_epoch['complete']
    # Extend a completed first epoch: Adam moments and shuffled-data RNG must resume exactly.
    config = {**config, 'epochs': 2}
    resumed = fit_models(train,valid,target,bank,1/64,config,interrupted,7)
    fresh = fit_models(train,valid,target,bank,1/64,config,tmp_path/'fresh',7)
    assert resumed['complete'] and fresh['complete']
    for loader, name in ((load_a,'a_seed7.pt'),(load_b,'b_seed7.pt')):
        one, two = loader(interrupted/name),loader(tmp_path/'fresh'/name)
        expected_center = torch.tensor(np.stack([c.features() for c, _ in train]).mean(0),dtype=torch.float32)
        torch.testing.assert_close(one.normalization.center,expected_center,rtol=0,atol=0)
        for key,value in one.state_dict().items():
            torch.testing.assert_close(value,two.state_dict()[key],rtol=0,atol=0)
    prediction = load_b(interrupted/'b_seed7.pt').predict(train[0][0],bank)
    assert np.isfinite(prediction.scaled_covariance).all()
    fit_baselines(train,valid,target,bank,1/64,tmp_path/'baselines')
    baselines = load_baselines(tmp_path/'baselines')
    for name in ('polynomial_ridge','kernel_ridge'):
        result = baselines[name].predict(valid[0][0],bank)
        np.testing.assert_allclose(result.scaled_covariance, result.scaled_covariance.T, atol=1e-12)
        assert "covariance_minimum_eigenvalue" in result.diagnostics
    sample = baselines['autoregression'].sample(valid[0][0],[0.,1/64],4,3)
    np.testing.assert_allclose(sample.states[:,0].sum(-1),1.,atol=1e-14)


def test_ridge_indefinite_covariance_is_reported_without_projection():
    target = TargetSpec(n_cells=8)
    bank = QueryBank(times=(1/64,))
    d = bank.dimension
    coefficient = np.zeros((66, d+d*d))
    coefficient[0,d:] = -np.eye(d).ravel()
    model = RidgeMomentRegressor(target,bank,"polynomial_ridge",np.zeros(10),np.ones(10),
                                 np.zeros((1,10)),coefficient,(512,))
    prediction = model.predict(Context(),bank)
    np.testing.assert_array_equal(prediction.scaled_covariance,-np.eye(d))
    assert prediction.diagnostics['covariance_indefinite']
    assert prediction.diagnostics['covariance_minimum_eigenvalue'] == -1.
