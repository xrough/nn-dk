"""Optional compiled native Euler/JVP/adjoint benchmark control.

The FP64 C++ loop is serial, uses no fast-math and never clips or renormalizes.
Compiler/source/flag hashes identify each cached library. Compilation is charged
separately from warm solver calls; absence of a compiler disables this optional
analysis control rather than changing the Python package requirements.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np


def compiler_available(compiler=None):
    return shutil.which(compiler or "clang++") is not None


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CompiledNative:
    """Validated C-ABI native full paths and one exact initial/physics tangent."""

    def __init__(self, library, metadata):
        self.metadata = metadata
        self.library = ctypes.CDLL(str(library))
        self.library.nn_dk_control_abi.restype = ctypes.c_int
        if self.library.nn_dk_control_abi() != 2:
            raise ValueError("Unexpected compiled native-control ABI")
        pointer = ctypes.POINTER(ctypes.c_double)
        self.function = self.library.nn_dk_native_euler
        self.function.argtypes = ([ctypes.c_int] * 3 + [ctypes.c_double] * 7
            + [pointer] * 4 + [ctypes.c_int] + [pointer] * 4)
        self.function.restype = ctypes.c_int
        self.reverse_function = self.library.nn_dk_native_adjoint
        self.reverse_function.argtypes = ([ctypes.c_int] * 3 + [ctypes.c_double] * 7 + [pointer] * 6)
        self.reverse_function.restype = ctypes.c_int

    def __call__(self, initial, increments, context, spec, dt, *,
                 initial_tangent=None, physics_tangent=None):
        # Same public validation as the NumPy Euler control, without importing
        # Torch or the neural package into this independent analysis backend.
        initial, increments = np.asarray(initial, dtype=float), np.asarray(increments, dtype=float)
        if (initial.ndim != 2 or initial.shape[-1] != spec.n_cells or initial.shape[0] < 1
                or increments.ndim != 3 or increments.shape[0] != initial.shape[0]
                or increments.shape[-1] != spec.n_cells or increments.shape[1] < 1
                or not np.isfinite(dt) or dt <= 0 or not np.isfinite(initial).all()
                or not np.isfinite(increments).all() or np.any(initial < 0)
                or not np.allclose(initial.sum(-1), 1, atol=1e-10, rtol=0)):
            raise ValueError("Finite nonnegative unit-mass initial [B,G] and noise [B,K,G] required")
        velocity = abs(context.kappa) + 2 * abs(context.a2)
        if (dt * (2 * context.diffusion / spec.h ** 2 + velocity / spec.h) > 1 + 1e-12
                or velocity * spec.h > 2 * context.diffusion + 1e-12):
            raise ValueError("Native centered Euler stability condition violated")
        initial = np.ascontiguousarray(initial, dtype=np.float64)
        increments = np.ascontiguousarray(increments, dtype=np.float64)
        tangent = initial_tangent is not None or physics_tangent is not None
        if tangent:
            v = (np.zeros_like(initial) if initial_tangent is None else
                 np.broadcast_to(np.asarray(initial_tangent, dtype=float), initial.shape).copy())
            dp = (np.zeros(3) if physics_tangent is None else np.asarray(physics_tangent, dtype=float))
            if (dp.shape != (3,) or not np.isfinite(dp).all() or not np.isfinite(v).all()
                    or not np.allclose(v.sum(-1), 0, atol=1e-12, rtol=0)):
                raise ValueError("Finite conservative initial tangent and physics tangent [D,kappa,a2] required")
            v, dp = np.ascontiguousarray(v), np.ascontiguousarray(dp)
        else:
            v, dp = np.empty(0), np.empty(0)
        states = np.empty((len(initial), increments.shape[1] + 1, spec.n_cells), dtype=np.float64)
        stops = np.empty(len(initial), dtype=np.float64)
        derivatives = np.empty_like(states) if tangent else np.empty(0)
        stop_derivatives = np.empty_like(stops) if tangent else np.empty(0)

        def pointer(value):
            return value.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        status = self.function(len(initial), increments.shape[1], spec.n_cells,
            spec.h, spec.mass_stop, context.diffusion, context.kappa, context.a2,
            float(context.n_particles), dt, pointer(initial), pointer(increments),
            pointer(v), pointer(dp), int(tangent), pointer(states), pointer(stops),
            pointer(derivatives), pointer(stop_derivatives))
        if status or not np.isfinite(states).all() or (tangent and not np.isfinite(derivatives).all()):
            raise FloatingPointError(f"Compiled native-control failure status {status}; no path may be omitted")
        result = {"states": states, "stopped_at": stops}
        if tangent:
            result.update(tangent=derivatives, stopped_at_tangent=stop_derivatives)
        return result

    def adjoint(self, initial, increments, context, spec, dt, *,
                path_gradient=None, coefficients=(1., 1., 1.), temperature=.05):
        """Complete forward + NumPy workload cotangent + compiled reverse pass.

        Return the unconstrained initial gradient [B,G] and physical gradients
        [B,3].  Project the former to mass-zero directions for interpretation;
        sum the latter over B for derivatives of shared physical parameters.
        Every initially absorbed or subsequently stopped path is retained.
        """
        packet = self(initial, increments, context, spec, dt)
        states = packet["states"]
        gradient = (workload_state_gradient(states, spec, dt, coefficients=coefficients,
                                            temperature=temperature)
                    if path_gradient is None else np.asarray(path_gradient, dtype=np.float64))
        if gradient.shape != states.shape or not np.isfinite(gradient).all():
            raise ValueError("Finite complete path-state cotangent [B,K+1,G] required")
        increments = np.ascontiguousarray(increments, dtype=np.float64)
        gradient = np.ascontiguousarray(gradient)
        initial_gradient = np.empty((len(states), spec.n_cells), dtype=np.float64)
        physics_gradient = np.empty((len(states), 3), dtype=np.float64)

        def pointer(value):
            return value.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        status = self.reverse_function(len(states), increments.shape[1], spec.n_cells,
            spec.h, spec.mass_stop, context.diffusion, context.kappa, context.a2,
            float(context.n_particles), dt, pointer(states), pointer(increments),
            pointer(packet["stopped_at"]), pointer(gradient), pointer(initial_gradient), pointer(physics_gradient))
        if status or not np.isfinite(initial_gradient).all() or not np.isfinite(physics_gradient).all():
            raise FloatingPointError(f"Compiled adjoint failure status {status}; no path may be omitted")
        return {**packet, "initial_gradient": initial_gradient, "physics_gradient": physics_gradient}


def workload_state_gradient(states, spec, dt, *, coefficients=(1., 1., 1.), temperature=.05):
    """Identical scalarization/cotangent to the NumPy classical workload probe."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if (coefficients.shape != (3,) or not np.isfinite(coefficients).all()
            or not np.isfinite(temperature) or temperature <= 0):
        raise ValueError("Three finite coefficients and positive smooth-maximum temperature required")
    centers = (np.arange(spec.n_cells) + .5) * spec.h
    weights = np.sqrt(2) * np.sinc(3 * spec.h / (2 * np.pi)) * np.cos(3 * centers)
    mode = states @ weights
    softweights = np.exp((mode - mode.max(-1, keepdims=True)) / temperature)
    softweights /= softweights.sum(-1, keepdims=True)
    gradient = coefficients[2] * softweights[..., None] * weights
    quadrature = np.full(states.shape[1], dt)
    quadrature[[0, -1]] *= .5
    gradient += coefficients[1] * 2 * states / spec.h * quadrature[None, :, None]
    gradient[:, -1] += coefficients[0] * weights
    return gradient


def compile_native(cache_directory=None, *, compiler=None):
    """Build/reuse a source-and-toolchain-addressed optional analysis library."""
    executable = shutil.which(compiler or "clang++")
    if executable is None:
        raise RuntimeError("Optional compiled analysis control requires a C++ compiler")
    executable = str(Path(executable).resolve())
    source = Path(__file__).with_name("native_euler_control.cpp")
    compiler_version = subprocess.check_output([executable, "--version"], text=True)
    flags = ["-O3", "-std=c++17", "-ffp-contract=off"]
    flags += ["-dynamiclib"] if platform.system() == "Darwin" else ["-shared", "-fPIC"]
    identity = {"cpp_sha256": _sha(source), "compiler": executable,
                "compiler_version": compiler_version, "flags": flags,
                "platform": platform.platform(), "machine": platform.machine(), "abi": 2}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_directory or Path(tempfile.gettempdir()) / "nn-dk-compiled-native")
    cache.mkdir(parents=True, exist_ok=True)
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    library = cache / ("native_euler_" + fingerprint[:24] + suffix)
    manifest = library.with_suffix(library.suffix + ".json")
    elapsed = 0.
    cache_hit = library.exists() and manifest.exists()
    if cache_hit:
        saved = json.loads(manifest.read_text())
        if saved["fingerprint"] != fingerprint or saved["library_sha256"] != _sha(library):
            raise ValueError("Compiled control cache checksum mismatch")
    else:
        temporary = library.with_name(library.name + f".{os.getpid()}.tmp")
        command = [executable, *flags, str(source), "-o", str(temporary)]
        started = time.perf_counter()
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            elapsed = time.perf_counter() - started
            temporary.replace(library)
        finally:
            temporary.unlink(missing_ok=True)
        saved = {**identity, "fingerprint": fingerprint, "command": command,
                 "compile_seconds": elapsed, "library_sha256": _sha(library),
                 "compiler_output": completed.stderr}
        manifest.write_text(json.dumps(saved, indent=2) + "\n")
    return CompiledNative(library.resolve(), {**saved, "library_path": str(library.resolve()),
        "compile_seconds_this_process": elapsed, "cache_hit": cache_hit,
        "dtype": "float64", "cpu_threads": 1, "fast_math": False,
        "optional_analysis_only": True, "runtime_dependencies": "NumPy and ctypes; no PyTorch",
        "python_wrapper_sha256": _sha(__file__)})
