"""Optional, explicitly identified acceleration of a fixed causal path operator.

These routines change neither the fitted kernel nor physical forcing.  Cached
dense contractions retain the legacy triangular map.  The block FFT backend
only transforms forcing in a block strictly preceding every output it updates;
it never relies on cancellation of future inputs in a full-horizon FFT.
Floating-point operation order differs from the legacy implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as functional

from .native_types import NativeDKSpec

RESPONSE_BACKENDS = {
    "dense-cached": "causal-cached-triangular-response-v1",
    "causal-block-fft": "causal-past-only-dyadic-fft-response-v1",
}
STOPPING_BACKEND = "absolute-proposal-first-crossing-vectorized-v1"


@dataclass(frozen=True)
class CausalResponsePlan:
    """Batch-independent transforms of one particular, unchanged learned kernel.

    Keep gradient tracking when preparing a training plan.  An inference cache
    must be discarded whenever its kernel, weights, step or horizon changes.
    Preparing a plan does not detach its kernel or alter its parameter values.
    """

    mode: str
    n_cells: int
    kernel: torch.Tensor
    dense: torch.Tensor | None
    fft_kernels: tuple[torch.Tensor, ...]
    padded_steps: int

    @property
    def backend_id(self) -> str:
        return RESPONSE_BACKENDS[self.mode]

    def apply(self, initial: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
        return accelerated_response(initial, forcing, plan=self)


def prepare_response(kernel: torch.Tensor, n_cells: int, *,
                     mode: str = "dense-cached") -> CausalResponsePlan:
    """Prepare a differentiable response plan, without fitting or approximation."""
    if mode not in RESPONSE_BACKENDS:
        raise ValueError(f"Use one of {tuple(RESPONSE_BACKENDS)}")
    if (type(n_cells) is not int or n_cells < 2 or not isinstance(kernel, torch.Tensor)
            or kernel.dtype != torch.float64 or kernel.ndim != 2
            or kernel.shape[0] < 2 or kernel.shape[1] != n_cells // 2 + 1
            or not bool(torch.isfinite(kernel).all())):
        raise ValueError("A finite double lag kernel [steps+1, rFFT modes] is required")
    steps = kernel.shape[0] - 1
    padded = 1 << (steps - 1).bit_length()
    if mode == "dense-cached":
        indices = torch.arange(steps, device=kernel.device)
        lags = indices[:, None] - indices[None]
        weights = kernel[lags.clamp_min(0)] * (lags >= 0)[..., None]
        dense = weights.permute(2, 0, 1).to(torch.complex128).contiguous()
        spectra = ()
    else:
        dense = None
        values = functional.pad(kernel[:steps].T.contiguous(), (0, padded - steps))
        # A circular convolution of length 2w can alias only into indices below
        # w-1.  Its right half therefore equals the linear cross-block response.
        spectra = tuple(torch.fft.fft(values[:, :2 * width], dim=-1)
                        for width in (1 << i for i in range((padded - 1).bit_length())))
    return CausalResponsePlan(mode, n_cells, kernel, dense, spectra, padded)


def accelerated_response(initial: torch.Tensor, forcing: torch.Tensor,
                         kernel: torch.Tensor | None = None, *,
                         mode: str = "dense-cached",
                         plan: CausalResponsePlan | None = None) -> torch.Tensor:
    """Apply the learned spatial-spectral Volterra response to physical forcing.

    Shapes are ``initial[paths,cells]``, ``forcing[paths,steps,cells]`` and
    ``kernel[steps+1,cells//2+1]``.  Supply either a kernel or a prepared plan.
    The initial state is retained exactly and the conserved mode is fixed.
    """
    if (not isinstance(initial, torch.Tensor) or not isinstance(forcing, torch.Tensor)
            or initial.dtype != torch.float64 or forcing.dtype != torch.float64
            or initial.ndim != 2 or forcing.ndim != 3 or initial.shape[0] < 1
            or forcing.shape[0] != initial.shape[0] or forcing.shape[1] < 1
            or forcing.shape[-1] != initial.shape[-1] or initial.device != forcing.device
            or not bool(torch.isfinite(initial).all() and torch.isfinite(forcing).all())):
        raise ValueError("Finite double initial [paths,cells] and forcing [paths,steps,cells] required")
    if plan is not None and kernel is not None:
        raise ValueError("Supply a kernel or its prepared plan, not both")
    if plan is None:
        if kernel is None:
            raise ValueError("Supply a learned kernel or its prepared response plan")
        plan = prepare_response(kernel, initial.shape[-1], mode=mode)
    if (not isinstance(plan, CausalResponsePlan) or plan.n_cells != initial.shape[-1]
            or plan.kernel.shape[0] != forcing.shape[1] + 1
            or plan.kernel.device != initial.device):
        raise ValueError("Response plan grid, step count and device must match the inputs")
    steps = forcing.shape[1]
    initial_hat = torch.fft.rfft(initial, dim=-1)
    nonzero = torch.arange(initial_hat.shape[-1], device=initial.device) != 0
    forcing_hat = torch.fft.rfft(forcing, dim=-1) * nonzero
    if plan.mode == "dense-cached":
        stochastic = (plan.dense @ forcing_hat.permute(2, 1, 0)).permute(2, 1, 0)
    else:
        # Time is contiguous for the dyadic batched FFTs.  Each left half is
        # strictly earlier than every output in its paired right half.
        values = functional.pad(forcing_hat.transpose(1, 2).contiguous(),
                                (0, plan.padded_steps - steps))
        stochastic = values * plan.kernel[0, :, None]
        for level, spectrum in enumerate(plan.fft_kernels):
            width = 1 << level
            n_blocks = plan.padded_steps // (2 * width)
            past = values.reshape(values.shape[0], values.shape[1], n_blocks,
                                  2 * width)[..., :width]
            transformed = torch.fft.fft(past, n=2 * width, dim=-1)
            cross = torch.fft.ifft(transformed * spectrum[None, :, None], dim=-1)[..., width:]
            contribution = torch.cat((torch.zeros_like(cross), cross), dim=-1)
            stochastic = stochastic + contribution.reshape_as(values)
        stochastic = stochastic[..., :steps].transpose(1, 2)
    update_hat = (((plan.kernel[1:] - 1)[None] * initial_hat[:, None]
                   + stochastic) * nonzero)
    subsequent = initial[:, None] + torch.fft.irfft(update_hat, n=plan.n_cells, dim=-1)
    return torch.cat((initial[:, None], subsequent), dim=1)


def vectorized_stop_paths(initial, raw_paths, spec: NativeDKSpec, dt: float, *,
                          stopping: bool = True) -> dict[str, torch.Tensor]:
    """Freeze absolute proposals at their first polygonal boundary crossing.

    Only the first exiting proposal and its predecessor determine the crossing.
    All paths and original proposals are retained, including initially stopped
    paths.  This is the legacy stopping map with a different floating-point
    evaluation order, not a density clipping or renormalization operation.
    """
    if not isinstance(spec, NativeDKSpec) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Native grid and a finite positive timestep are required")
    initial = torch.as_tensor(initial, dtype=torch.float64)
    raw_paths = torch.as_tensor(raw_paths, dtype=torch.float64, device=initial.device)
    if (initial.ndim != 2 or initial.shape[-1] != spec.n_cells or initial.shape[0] < 1
            or raw_paths.ndim != 3 or raw_paths.shape[0] != initial.shape[0]
            or raw_paths.shape[-1] != spec.n_cells or raw_paths.shape[1] < 1
            or not bool(torch.isfinite(initial).all() and torch.isfinite(raw_paths).all())):
        raise ValueError("Finite initial [paths,cells] and raw proposals [paths,steps,cells] required")
    if stopping and bool((initial < 0).any()):
        raise ValueError("Stopped DK initial states must be nonnegative")
    paths, steps, _ = raw_paths.shape
    if not stopping:
        return {"states": torch.cat((initial[:, None], raw_paths), dim=1),
                "raw_proposals": raw_paths, "proposals": raw_paths,
                "active": torch.ones((paths, steps), dtype=torch.bool, device=initial.device),
                "stopped_at": torch.full_like(initial[:, 0], torch.inf),
                "crossing_fraction": torch.ones_like(raw_paths[..., 0])}
    indices = torch.arange(steps, device=initial.device)[None]
    initial_stopped = (initial <= spec.mass_stop).any(-1)
    crossings = (raw_paths <= spec.mass_stop).any(-1)
    first = torch.where(crossings, indices, steps).amin(-1)
    first = torch.where(initial_stopped, torch.zeros_like(first), first)
    selected = first.clamp_max(steps - 1)
    row = torch.arange(paths, device=initial.device)
    preceding = torch.cat((initial[:, None], raw_paths[:, :-1]), dim=1)[row, selected]
    proposal = raw_paths[row, selected]
    descending = (proposal <= spec.mass_stop) & (proposal < preceding)
    denominator = torch.where(descending, preceding - proposal, torch.ones_like(preceding))
    ratios = torch.where(descending, (preceding - spec.mass_stop) / denominator,
                         torch.ones_like(preceding))
    # Match the legacy first-cell subgradient at a tied first crossing.
    fraction = ratios.min(-1).values.clamp(0, 1)
    fraction = torch.where(initial_stopped, torch.zeros_like(fraction), fraction)
    boundary = preceding + fraction[:, None] * (proposal - preceding)
    boundary = torch.where(initial_stopped[:, None], initial, boundary)
    subsequent = torch.where((indices < first[:, None])[..., None], raw_paths, boundary[:, None])
    active = (indices <= first[:, None]) & ~initial_stopped[:, None]
    fractions = torch.where(indices == first[:, None], fraction[:, None],
                            torch.ones_like(raw_paths[..., 0]))
    fractions = torch.where(active, fractions, torch.zeros_like(fractions))
    stop_times = torch.where(first < steps, (first + fraction) * dt,
                             torch.full_like(fraction, torch.inf))
    return {"states": torch.cat((initial[:, None], subsequent), dim=1),
            "raw_proposals": raw_paths, "proposals": raw_paths, "active": active,
            "stopped_at": stop_times, "crossing_fraction": fractions}
