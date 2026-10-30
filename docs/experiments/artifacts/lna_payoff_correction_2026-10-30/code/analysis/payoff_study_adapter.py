"""Isolated reuse of immutable study machinery for a separately named study.

The original analysis modules stay importable with their original globals.
Successor studies can supply explicit model and stream hooks to a *private*
module loaded from the same frozen source. Both the adapter and original source
must appear in the successor study's preregistered helper inventory.
"""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any


def isolated_module(filename: str, namespace: str) -> ModuleType:
    """Load one analysis file without altering its canonical import namespace.

    Registering the private module before execution is required by dataclasses.
    A second import reuses the same private globals and rejects a source edit
    during the current process instead of silently mixing implementations.
    """
    directory = Path(__file__).resolve().parent
    path = (directory / filename).resolve()
    if path.parent != directory or path.suffix != ".py":
        raise ValueError("An analysis Python basename is required")
    if not namespace.startswith("analysis._") or not all(
            part.isidentifier() for part in namespace.split(".")):
        raise ValueError("Use a private analysis module namespace")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    previous = sys.modules.get(namespace)
    if previous is not None:
        if (getattr(previous, "_adapter_origin", None) != str(path)
                or getattr(previous, "_adapter_origin_sha256", None) != digest):
            raise ValueError("Private study source or namespace changed in this process")
        return previous
    specification = importlib.util.spec_from_file_location(namespace, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load study source: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[namespace] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(namespace, None)
        raise
    module._adapter_origin = str(path)
    module._adapter_origin_sha256 = digest
    return module


def bind_study(module: ModuleType, adapter_path: str | Path, **hooks: Any) -> ModuleType:
    """Bind only declared existing hooks in an isolated implementation.

    ``__file__`` identifies the thin adapter in the original comparison-plan
    code. The immutable source path and its digest remain independently stored
    in ``_adapter_origin`` and ``_adapter_origin_sha256``.
    """
    if (not module.__name__.startswith("analysis._")
            or not getattr(module, "_adapter_origin_sha256", None)):
        raise ValueError("Only an isolated study implementation may be rebound")
    unknown = set(hooks) - set(vars(module))
    if unknown:
        raise ValueError(f"Unknown study hooks: {sorted(unknown)}")
    path = Path(adapter_path).resolve()
    if path.parent != Path(__file__).resolve().parent or not path.is_file():
        raise ValueError("The study adapter must be an existing analysis file")
    vars(module).update(hooks)
    module.__file__ = str(path)
    module._adapter_path = str(path)
    return module


def bind_analysis_dependencies(module: ModuleType, **aliases: ModuleType) -> ModuleType:
    """Redirect local ``from analysis import ...`` inside one private module.

    Several historical audit functions import the runner inside their body.
    A copied builtins mapping changes those function-local dependency imports
    without rewriting their source or touching process-wide module aliases.
    Every ordinary import delegates to Python's original importer.
    """
    if (not module.__name__.startswith("analysis._")
            or not getattr(module, "_adapter_origin_sha256", None)):
        raise ValueError("Only an isolated study implementation may bind dependencies")
    if not aliases or any(not name.isidentifier() or not isinstance(value, ModuleType)
                          for name, value in aliases.items()):
        raise ValueError("Named analysis module dependencies are required")
    canonical = builtins.__import__("analysis")
    proxy = ModuleType("analysis")

    def resolve(name):
        return aliases[name] if name in aliases else getattr(canonical, name)

    proxy.__getattr__ = resolve

    def private_import(name, globals=None, locals=None, fromlist=(), level=0):
        imported = builtins.__import__(name, globals, locals, fromlist, level)
        if (name == "analysis" and level == 0 and fromlist
                and any(item in aliases for item in fromlist)):
            return proxy
        return imported

    mapping = module.__builtins__
    mapping = dict(mapping if isinstance(mapping, dict) else vars(mapping))
    mapping["__import__"] = private_import
    module.__builtins__ = mapping
    # CPython captures a function's builtins mapping at definition time. A new
    # function object with identical code/globals/defaults/closure picks up this
    # private mapping. No bytecode, source, canonical globals or class changes.
    for name, value in tuple(vars(module).items()):
        if isinstance(value, FunctionType) and value.__globals__ is vars(module):
            rebound = FunctionType(value.__code__, vars(module), value.__name__,
                                   value.__defaults__, value.__closure__)
            rebound.__kwdefaults__ = value.__kwdefaults__
            rebound.__annotations__ = value.__annotations__
            rebound.__dict__.update(value.__dict__)
            rebound.__qualname__ = value.__qualname__
            rebound.__module__ = value.__module__
            rebound.__doc__ = value.__doc__
            setattr(module, name, rebound)
    module._adapter_analysis_dependencies = {name: value.__name__ for name, value in aliases.items()}
    return module
