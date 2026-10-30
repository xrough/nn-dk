"""Explicit historical-source validation for derived inference experiments.

Legacy loaders remain source-strict. This adapter executes the recorded Git
package in a separate module namespace, validates the original checkpoint
there, and records its identity. It never silently ignores a source mismatch.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import subprocess
import sys
import tempfile
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from .data import source_hash
from .types import Context, stable_hash


def _git(*args):
    repo = Path(__file__).resolve().parents[2]
    return subprocess.check_output(["git", "-C", str(repo), *args])


def frozen_package(commit):
    """Materialize exactly a committed Python package, including its old guards."""
    revision = _git("rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    return _frozen_package(revision)


@lru_cache(maxsize=None)
def _frozen_package(revision):
    # A canonical revision owns exactly one import namespace and directory.
    # Cached aliases must not outlive evicted snapshots with the same alias.
    names = _git("ls-tree", "-r", "--name-only", revision, "src/nn_dk").decode().splitlines()
    texts = {str(Path(name).relative_to("src/nn_dk")): _git("show", f"{revision}:{name}").decode()
             for name in names if name.endswith(".py")}
    if "path_model.py" not in texts or "path_training.py" not in texts:
        raise ValueError("Recorded source revision has no path solver")
    root = Path(tempfile.mkdtemp(prefix=f"nn-dk-frozen-{revision[:10]}-"))/"nn_dk"
    root.mkdir()
    for relative, content in texts.items():
        target = root/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    alias = f"_nn_dk_frozen_{revision}"
    spec = importlib.util.spec_from_file_location(alias, root/"__init__.py",
                                                 submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return alias, revision, stable_hash(texts), root, texts


def load_frozen_model(path, commit, *, kind="path"):
    """Validate weights at their source freeze before deriving a new runtime."""
    path = Path(path).resolve()
    alias, revision, expected_source, root, texts = frozen_package(commit)
    # Frozen files are themselves checked on every load, including cache reuse.
    if stable_hash({str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*.py"))}) != expected_source:
        raise ValueError("Historical package snapshot changed")
    if kind == "path":
        module = importlib.import_module(f"{alias}.path_training")
        model = module.load_path_model(path)
    elif kind == "moments":
        module = importlib.import_module(f"{alias}.interacting_training")
        model = module.load_interacting_model(path)
    else:
        raise ValueError("Expected a path or moment checkpoint")
    if model.provenance["source_hash"] != expected_source:
        raise ValueError("Checkpoint does not belong to the declared historical Git source")
    # Derivation assumes unchanged physical equations, not unchanged training
    # configuration code. Old source is still used for all original validation.
    numerical_files = ("path_model.py", "path_numerics.py", "native.py", "native_types.py", "types.py")
    current = Path(__file__).parent
    if any((current/name).read_text() != texts[name] for name in numerical_files):
        raise ValueError("Historical/current dynamics differ; an explicit migration audit is required")
    record = {"checkpoint_path": str(path), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "historical_commit": revision, "historical_source_hash": expected_source,
              "current_source_hash": source_hash(), "kind": kind,
              "original_target_id": model.target_id,
              "validation": "original source-strict loader in an isolated Git package namespace",
              "unchanged_numerical_files": list(numerical_files)}
    model.frozen_derivation = record
    return model, record


def current_context(frozen_context):
    return Context(**asdict(frozen_context))
