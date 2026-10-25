"""Show an actual held-out density realization, not only its statistics.

Select the first declared optimizer seed, coarsest declared step and trajectory
zero before reading any values. This is a display of saved, locked results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

from nn_dk.data import source_hash
from nn_dk.path_config import PathConfig
from nn_dk.path_data import load_path_bank
from nn_dk.path_numerics import torch_euler
from nn_dk.path_study import _artifacts


def plot_paths(directory, output=None):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory).resolve()
    config = PathConfig.load(directory/"config.json")
    manifest = json.loads((directory/"manifest.json").read_text())
    if (manifest.get("status") != "complete" or manifest.get("source_hash") != source_hash()
            or manifest.get("config_hash") != config.config_hash
            or manifest.get("artifact_lock") != _artifacts(directory)):
        raise ValueError("Plot actual paths only from complete unchanged fitted results")
    records = json.loads((directory/"evaluations.json").read_text())
    seed, dt = config.optimizer_seeds[0], max(config.step_sizes)
    selected = [record for record in records if record["problem"] == "dk"
                and record["seed"] == seed and record["dt"] == dt]
    if not selected:
        raise ValueError("This display requires a completed DK path experiment")
    figure, axes = plt.subplots(len(selected), 3, figsize=(13, 3.3*len(selected)),
                                squeeze=False, constrained_layout=True)
    for row, record in enumerate(selected):
        case = record["case"]
        bank = load_path_bank(directory, config, "test", case, dt)
        with h5py.File(record["prediction_path"], "r") as stored:
            neural = stored["states"][0]
            times = stored["times"][:]
        with torch.no_grad():
            exact = torch_euler(bank["initial"][:1], bank["brownian"][:1],
                                config.context(case), config.spec, dt, "dk")["states"][0].numpy()
        density, reference = neural/config.spec.h, exact/config.spec.h
        low, high = min(density.min(), reference.min()), max(density.max(), reference.max())
        extent = (times[0], times[-1], 0., config.spec.length)
        for column, field in enumerate((density, reference)):
            displayed = axes[row, column].imshow(field.T, origin="lower", aspect="auto",
                                                extent=extent, vmin=low, vmax=high,
                                                interpolation="nearest", cmap="viridis")
            axes[row, column].set(xlabel="Time", ylabel="Position x",
                                  title=f"{case}: {'neural path' if column == 0 else 'same-noise Euler path'}")
        figure.colorbar(displayed, ax=list(axes[row, :2]), label="Density")
        error = density-reference
        limit = max(float(np.max(np.abs(error))), 1e-15)
        image = axes[row, 2].imshow(error.T, origin="lower", aspect="auto", extent=extent,
                                    vmin=-limit, vmax=limit, cmap="RdBu_r", interpolation="nearest")
        axes[row, 2].set(xlabel="Time", ylabel="Position x", title="Neural − same-step Euler")
        figure.colorbar(image, ax=axes[row, 2], label="Density error")
    figure.suptitle(f"Full-grid held-out density realizations: trajectory 0, seed {seed}, Δ={dt:g}\n"
                   "Same initial histograms and physical Brownian increments; all cells retained")
    output = Path(output) if output else directory/"report"/"density_realizations.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(plot_paths(args.directory, args.output))
