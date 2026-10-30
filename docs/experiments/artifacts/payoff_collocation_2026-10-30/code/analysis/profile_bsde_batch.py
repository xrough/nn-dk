"""Profile one DK BSDE epoch at the scientific width, horizon and batch size.

The reduced pool measures execution cost only. It is not the scientific first
batch and does not produce accuracy estimates or a speedup comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from nn_dk.bsde import prepare_bsde_run
from nn_dk.bsde_config import BSDEConfig
from nn_dk.bsde_data import generate_bsde_pool, load_bsde_pool
from nn_dk.bsde_training import fit_bsde, load_bsde_model
from nn_dk.data import write_json


def profile(directory: Path) -> dict:
    config = replace(
        BSDEConfig(), name="bsde-batch-profile", optimizer_seeds=(0,),
        train_anchors=64, train_continuations=2,
        validation_anchors=16, validation_continuations=2,
        training={"epochs": 1, "patience": 1, "batch_size": 32},
    )
    manifest = prepare_bsde_run(directory, config)
    output = directory / "profile.json"
    if output.exists():
        return json.loads(output.read_text())
    torch.set_num_threads(1)
    started = time.perf_counter()
    generated = generate_bsde_pool(directory, config, "dk")
    if not generated["complete"]:
        raise RuntimeError("Performance pool generation did not complete")
    tick = time.perf_counter()
    training = load_bsde_pool(directory, config, "dk", "train")
    validation = load_bsde_pool(directory, config, "dk", "validation")
    pool_loading_seconds = time.perf_counter() - tick
    path = directory / "checkpoints" / "dk" / "seed_0.pt"
    metadata = fit_bsde(training, validation, config, "dk", 0, path)
    if not metadata["complete"]:
        raise RuntimeError("Performance fit did not complete")
    metadata["shared_forward_pool_generation_seconds"] = generated["seconds_this_call"]
    write_json(path.with_suffix(".json"), metadata)
    tick = time.perf_counter()
    model = load_bsde_model(path)
    checkpoint_loading_seconds = time.perf_counter() - tick
    # One representative initial state per anchor; continuations repeat states.
    _, indices = np.unique(np.asarray(training.anchor_ids), return_index=True)
    anchors = training.initial[np.sort(indices)]
    model.predict_conditional(anchors, config.query_times)  # untimed warm-up
    inference = []
    for count in (1, len(anchors)):
        costs = []
        for _ in range(3):
            tick = time.perf_counter()
            model.predict_conditional(anchors[:count], config.query_times)
            costs.append(time.perf_counter() - tick)
        inference.append({
            "initial_states": count, "horizons_per_state": len(config.query_times),
            "seconds": costs, "median_seconds": float(np.median(costs)),
            "median_seconds_per_initial_state": float(np.median(costs)) / count,
            "scope": "Precomputed initial masses; validation and value-head inference included",
        })
    result = {
        "scope": "Performance diagnostic only: reduced pool, one epoch, no accuracy conclusion",
        "config_hash": config.config_hash, "source_hash": manifest["source_hash"],
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "device": "cpu", "torch_threads": 1, "dtype": "float64",
        "state_dimension": config.n_cells, "horizon": config.horizon,
        "dt": config.dt, "width": config.width,
        "training_paths": training.n_paths, "validation_paths": validation.n_paths,
        "training_anchors": len(set(training.anchor_ids)),
        "validation_anchors": len(set(validation.anchor_ids)),
        "parameters": sum(p.numel() for p in model.parameters()),
        "value_parameters": sum(p.numel() for p in model.value_network.parameters()),
        "integrand_parameters": sum(p.numel() for p in model.integrand_network.parameters()),
        "pool_generation_seconds": generated["seconds_this_call"],
        "pool_loading_seconds": pool_loading_seconds,
        "checkpoint_loading_seconds": checkpoint_loading_seconds,
        "training": metadata, "conditional_inference": inference,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output, result)
    manifest["status"] = "performance-profile-complete"
    manifest["elapsed_seconds"] += result["elapsed_seconds"]
    write_json(directory / "manifest.json", manifest)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Separate output directory for this diagnostic")
    arguments = parser.parse_args()
    record = profile(arguments.output)
    print(json.dumps({
        "output": str((arguments.output / "profile.json").resolve()),
        "first_epoch_seconds": record["training"]["first_epoch_seconds"],
        "inference": record["conditional_inference"],
    }, indent=2))
