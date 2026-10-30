"""A successor study must not mutate a historical study's module globals."""

from pathlib import Path
from types import ModuleType

import pytest

from analysis import run_payoff_regimes as historical
from analysis.payoff_study_adapter import bind_analysis_dependencies, bind_study, isolated_module


def test_isolated_runner_has_separate_globals_and_dataclass_registration():
    module = isolated_module("run_payoff_regimes.py", "analysis._test_boundary_isolated_runner")
    original_stream = historical.STREAM
    original_workflow = historical.WORKFLOW
    bind_study(module, Path(__file__).parents[1]/"analysis/payoff_study_adapter.py",
               STREAM="fresh-boundary-test-stream", WORKFLOW="new-boundary-test")
    assert historical.STREAM == original_stream
    assert historical.WORKFLOW == original_workflow
    assert module.STREAM == "fresh-boundary-test-stream"
    assert module._family.__globals__ is vars(module)
    assert module.StudyPlan.from_dict(module.StudyPlan().to_dict()) == module.StudyPlan()
    case = module.case_names()[0]
    config = module.configuration(case)
    assert module._family(config, "train", "uniform", 0, 0) != historical._family(
        historical.configuration(case), "train", "uniform", 0, 0)


def test_private_cache_rejects_changed_source_identity(monkeypatch):
    namespace = "analysis._test_boundary_identity"
    module = isolated_module("run_payoff_regimes.py", namespace)
    assert isolated_module("run_payoff_regimes.py", namespace) is module
    monkeypatch.setattr(module, "_adapter_origin_sha256", "changed")
    with pytest.raises(ValueError, match="source or namespace changed"):
        isolated_module("run_payoff_regimes.py", namespace)


def test_binding_rejects_typos_canonical_modules_and_nonanalysis_files():
    module = isolated_module("run_payoff_regimes.py", "analysis._test_boundary_hooks")
    path = Path(__file__).parents[1]/"analysis/payoff_study_adapter.py"
    with pytest.raises(ValueError, match="Unknown study hooks"):
        bind_study(module, path, TYPO_GRAPHIC_STREAM="wrong")
    with pytest.raises(ValueError, match="Only an isolated"):
        bind_study(historical, path, STREAM="wrong")
    with pytest.raises(ValueError, match="existing analysis file"):
        bind_study(module, __file__, STREAM="wrong")


@pytest.mark.parametrize("filename, namespace", [
    ("../src/nn_dk/payoff_repair.py", "analysis._bad"),
    ("run_payoff_regimes.py", "analysis.run_payoff_regimes"),
    ("run_payoff_regimes.py", "analysis._invalid-name"),
])
def test_loader_limits_source_and_namespace(filename, namespace):
    with pytest.raises(ValueError):
        isolated_module(filename, namespace)


def test_function_local_analysis_imports_redirect_only_in_private_module():
    module = isolated_module("run_payoff_regimes.py", "analysis._test_boundary_imports")
    replacement = ModuleType("analysis._successor_runner_fixture")
    replacement.STREAM = "fresh-private-stream"
    # Define a tiny local-import probe in the isolated namespace, as the real
    # historical audit helpers do inside their entry-point functions.
    exec("def probe():\n    from analysis import run_payoff_regimes\n"
         "    from pathlib import Path\n    return run_payoff_regimes, Path\n", vars(module))
    assert module.probe()[0] is historical
    bind_analysis_dependencies(module, run_payoff_regimes=replacement)
    assert module.probe() == (replacement, Path)
    # Later functions also inherit the same private mapping.
    exec("def private_probe():\n    from analysis import run_payoff_regimes\n"
         "    from pathlib import Path\n    return run_payoff_regimes, Path\n", vars(module))
    assert module.private_probe() == (replacement, Path)
    from analysis import run_payoff_regimes

    assert run_payoff_regimes is historical
    assert historical.STREAM != replacement.STREAM
    with pytest.raises(ValueError, match="Only an isolated"):
        bind_analysis_dependencies(historical, run_payoff_regimes=replacement)
    with pytest.raises(ValueError, match="module dependencies"):
        bind_analysis_dependencies(module, run_payoff_regimes="invalid")
