from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_history(case_dir: Path, output: Path) -> None:
    history_path = case_dir / "history.csv"
    with history_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    epochs = np.asarray([int(row["epoch"]) for row in rows])
    energy = np.asarray([float(row["energy_per_a_no_com_mev"]) for row in rows])
    residual = np.asarray([float(row["normalized_commutator"]) for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, energy, color="#1565c0", lw=1.8)
    axes[0].set(xlabel="Epoch", ylabel="E/A without CoM (MeV)", title="Variational energy")
    axes[0].grid(alpha=0.25)
    axes[1].semilogy(epochs, residual, color="#c62828", lw=1.8, label="stationarity")
    axes[1].axhline(1.0e-2, color="#2e7d32", ls="--", lw=1.2, label="1% gate")
    axes[1].set(xlabel="Epoch", ylabel="Normalized residual", title="Euler-Lagrange stationarity")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_vs(case_dir: Path, output: Path) -> None:
    data_path = case_dir / "scf_comparison" / "vs_dpl_scf_data.npz"
    data = np.load(data_path)
    r = data["r"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, field in zip(axes, ("S", "V")):
        for species, color in (("n", "#1565c0"), ("p", "#c62828")):
            ax.plot(r, data[f"scf_{field}_{species}"], color=color, lw=2.0, label=f"SCF {species}")
            ax.plot(r, data[f"dpl_{field}_{species}"], color=color, lw=1.4, ls="--", label=f"DPL {species}")
        ax.set(xlabel="r (fm)", ylabel=f"{field} (MeV)", title=f"{field} potential")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate DPL-RMF convergence and DPL/SCF figures")
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.case_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plot_history(args.case_dir, out / "training_convergence.png")
    comparison = args.case_dir / "scf_comparison" / "vs_dpl_scf_data.npz"
    if comparison.exists():
        plot_vs(args.case_dir, out / "vs_dpl_scf.png")
    print(out.resolve())


if __name__ == "__main__":
    main()
