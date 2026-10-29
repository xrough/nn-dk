#!/usr/bin/env python3
"""Audit saved BSDE reference NPZ banks without loading checkpoints or simulating.

Usage: .venv/bin/python analysis/audit_bsde_references.py RUN --case all

This reads one small completed cache chunk at a time. Full cell-mass states and
Brownian increments are absent from these caches: their diagnostics are checked
as recorded simulator metadata, never independently reconstructed observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(131072):
            digest.update(block)
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _allocation(config, kind):
    if kind == "conditional":
        return config["conditional_anchors"], config["conditional_paths"]
    return config["outer_paths"], 1


def _preflight(directory, config, cases):
    """Reject an incomplete requested allocation before writing any audit files."""
    for case in cases:
        _require(case in config["cases"], f"Case {case} is absent from the saved configuration")
        for kind in ("conditional", "ensemble"):
            anchors, children = _allocation(config, kind)
            total = anchors * children
            starts = range(0, total, config["reference_batch_size"])
            expected = {f"chunk_{start:08d}.npz" for start in starts}
            path = directory / "evaluation_cache" / case / kind
            actual = {p.name for p in path.glob("*.npz")}
            _require(actual == expected,
                     f"Incomplete or unexpected {case}/{kind} cache: "
                     f"found {len(actual)}/{len(expected)} chunks; "
                     f"missing {len(expected - actual)}, unexpected {len(actual - expected)}")


def _audit_bank(directory, config, expected, case, kind):
    anchors, children = _allocation(config, kind)
    total, batch = anchors * children, config["reference_batch_size"]
    path = directory / "evaluation_cache" / case / kind
    families, forcing_seeds, initial_seeds = set(), set(), set()
    counts, cache_digest = Counter(), hashlib.sha256()
    fine_stops, coarse_stops = 0, 0
    min_mass, max_mass_error, seconds = math.inf, 0., 0.
    maximum_observable, maximum_paired_difference = 0., 0.
    n_chunks = 0
    for start in range(0, total, batch):
        end, n_chunks = min(start + batch, total), n_chunks + 1
        chunk = path / f"chunk_{start:08d}.npz"
        cache_digest.update(f"{chunk.name}:{_file_hash(chunk)}\n".encode())
        with np.load(chunk, allow_pickle=False) as store:
            metadata = json.loads(str(store["metadata"]))
            _require(all(metadata.get(key) == value for key, value in expected.items()),
                     f"Frozen provenance mismatch in {chunk}")
            _require((metadata.get("case"), metadata.get("kind"), metadata.get("start"),
                      metadata.get("end")) == (case, kind, start, end),
                     f"Chunk allocation metadata mismatch in {chunk}")
            _require(metadata.get("dt") == config["dt"]
                     and metadata.get("reference_dt") == config["reference_dt"],
                     f"Integration-grid metadata mismatch in {chunk}")
            _require(metadata.get("all_stopped_paths_included") is True,
                     f"Missing all-path inclusion declaration in {chunk}")
            wanted = [f"bsde-test/{expected['config_hash']}/{case}/{kind}/"
                      f"anchor{j // children}/child{j % children}" for j in range(start, end)]
            ids = store["family_ids"].tolist()
            _require(ids == wanted, f"Trajectory-family identities mismatch in {chunk}")
            _require(not families.intersection(ids), f"Repeated trajectory families in {chunk}")
            families.update(ids)
            selected = store["anchor_index"]
            _require(np.array_equal(selected, np.arange(start, end) // children),
                     f"Initial-anchor assignment mismatch in {chunk}")
            counts.update(selected.tolist())
            for key in ("fine", "coarse"):
                values = store[key]
                _require(values.shape == (end - start, len(config["query_times"]), 4)
                         and values.dtype == np.float64 and np.isfinite(values).all(),
                         f"Nonfinite or incompatible observable data in {chunk}/{key}")
                maximum_observable = max(maximum_observable, float(np.abs(values).max()))
            maximum_paired_difference = max(maximum_paired_difference,
                float(np.abs(store["fine"] - store["coarse"]).max()))
            for key in ("fine_stopped_at", "coarse_stopped_at"):
                stop = store[key]
                _require(stop.shape == (end - start,) and not np.isnan(stop).any()
                         and not (stop < 0).any()
                         and not ((stop > config["horizon"]) & np.isfinite(stop)).any(),
                         f"Invalid stopping times in {chunk}/{key}")
            fine_stops += int(np.isfinite(store["fine_stopped_at"]).sum())
            coarse_stops += int(np.isfinite(store["coarse_stopped_at"]).sum())
            for key in ("minimum_saved_mass", "maximum_mass_error", "seconds"):
                _require(isinstance(metadata.get(key), (int, float))
                         and np.isfinite(metadata[key]) and metadata[key] >= 0,
                         f"Invalid recorded simulator diagnostic {key} in {chunk}")
            min_mass = min(min_mass, metadata["minimum_saved_mass"])
            max_mass_error = max(max_mass_error, metadata["maximum_mass_error"])
            seconds += metadata["seconds"]
            _require(metadata.get("forcing_seed") not in forcing_seeds,
                     f"Repeated chunk forcing seed in {chunk}")
            _require(isinstance(metadata.get("forcing_seed"), int)
                     and isinstance(metadata.get("initial_seed"), int),
                     f"Missing reproducible initial/forcing stream identifiers in {chunk}")
            forcing_seeds.add(metadata["forcing_seed"])
            initial_seeds.add(metadata["initial_seed"])
    _require(len(families) == total and len(counts) == anchors
             and set(counts) == set(range(anchors)) and set(counts.values()) == {children},
             f"Actual trajectory allocation differs from {case}/{kind} configuration")
    _require(len(initial_seeds) == 1, f"Initial pool stream changed across {case}/{kind} chunks")
    no_stops = fine_stops == coarse_stops == 0
    return {
        "complete": True, "chunks": n_chunks, "independent_initial_anchors": anchors,
        "continuations_per_anchor": children, "independent_continuation_families": total,
        "family_ids_unique": True, "anchor_assignment_matches_ids": True,
        "unique_chunk_forcing_seeds": len(forcing_seeds), "initial_seed": next(iter(initial_seeds)),
        "observation_shape_per_path": [len(config["query_times"]), 4],
        "all_observations_finite": True, "maximum_absolute_observable": maximum_observable,
        "maximum_absolute_paired_coarse_fine_observable_difference": maximum_paired_difference,
        "fine_stopped_paths": fine_stops, "coarse_stopped_paths": coarse_stops,
        "zero_stop_one_sided_95_upper_ensemble": 1 - .05**(1 / total)
            if no_stops and kind == "ensemble" else None,
        "zero_stop_one_sided_95_upper_each_fixed_anchor": 1 - .05**(1 / children)
            if no_stops and kind == "conditional" else None,
        "minimum_saved_mass_from_metadata": min_mass,
        "maximum_mass_error_from_metadata": max_mass_error,
        "recorded_reference_generation_seconds": seconds,
        "dt": config["dt"], "reference_dt": config["reference_dt"],
        "fine_increments_per_coarse_step": round(config["dt"] / config["reference_dt"]),
        "all_paths_included": True, "frozen_provenance_matches_all_chunks": True,
        "ordered_cache_content_sha256": cache_digest.hexdigest(),
    }, families, forcing_seeds


def audit_references(directory, cases):
    """Return verified per-case dictionaries; a failed allocation writes nothing."""
    directory = Path(directory)
    manifest_path, config_path = directory / "manifest.json", directory / "config.json"
    manifest_bytes, config_bytes = manifest_path.read_bytes(), config_path.read_bytes()
    manifest, config = json.loads(manifest_bytes), json.loads(config_bytes)
    _require(manifest.get("workflow") == "conditional-bsde-moments-v1", "Wrong experiment workflow")
    _require(config == manifest["config"] and _hash_json(config) == manifest["config_hash"],
             "Saved configuration/manifest hash mismatch")
    ratio = config["dt"] / config["reference_dt"]
    _require(np.isfinite(ratio) and ratio >= 1
             and np.isclose(ratio, round(ratio), rtol=0, atol=1e-12),
             "Coarse integration grid does not nest within the reference grid")
    _preflight(directory, config, cases)
    expected = {"config_hash": manifest["config_hash"], "source_hash": manifest["source_hash"],
                "target_id": _hash_json(manifest["native_target"])}
    reports, known_ids, known_forcing = {}, set(), set()
    for case in cases:
        banks = {}
        for kind in ("conditional", "ensemble"):
            bank, ids, forcing = _audit_bank(directory, config, expected, case, kind)
            _require(not ids & known_ids, "Reference families overlap across requested banks")
            _require(not forcing & known_forcing, "Forcing seeds overlap across requested banks")
            known_ids.update(ids)
            known_forcing.update(forcing)
            banks[kind] = bank
        reports[case] = {
            "audit_kind": "completed-reference-banks-only", "overall_study_completion_assessed": False,
            "run_status_at_audit": manifest.get("status"), "created_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_script": str(Path(__file__).resolve()), "analysis_code_sha256": _file_hash(Path(__file__)),
            "input_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "input_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "provenance": expected, "case": case, "banks": banks,
            "conditional_and_ensemble_family_ids_disjoint": True,
            "forcing_stream_seeds_disjoint_across_requested_banks": True,
            "passed": True,
            "limitations": [
                "NPZ caches contain observed modes rather than full cell-mass states. Conservation "
                "and minimum-mass checks aggregate recorded simulator metadata.",
                "Brownian increments are absent. Shared coarse/fine forcing provenance and the "
                "integration-grid ratio are checked; increment summation itself is covered by code/tests.",
                "Conditional children are independent continuations of fixed initial anchors, not "
                "independent draws from the full initial-histogram law. Stopping bounds are pointwise per anchor.",
                "This reference-bank integrity audit does not certify neural accuracy or reference bias.",
            ],
        }
    joint = {
        "requested_cases": list(cases), "reference_banks": 2 * len(cases),
        "independent_continuation_families": len(known_ids),
        "family_ids_disjoint_across_all_requested_banks": True,
        "chunk_forcing_seeds_disjoint_across_all_requested_banks": True,
        "chunks": len(known_forcing),
        "interpretation": "Coarse/fine levels share each trajectory family; they are not independent replicates.",
    }
    for report in reports.values():
        report["joint_requested_bank_audit"] = joint
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--case", choices=("uniform", "nonuniform", "all"), default="all")
    args = parser.parse_args()
    cases = ("uniform", "nonuniform") if args.case == "all" else (args.case,)
    try:
        reports = audit_references(args.directory, cases)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    output = args.directory / "report"
    output.mkdir(parents=True, exist_ok=True)
    for case, report in reports.items():
        path = output / f"{case}_cache_audit.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(path)
        print(f"{case}: reference bank integrity passed; saved {path}")


if __name__ == "__main__":
    main()
