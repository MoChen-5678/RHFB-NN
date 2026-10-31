#!/usr/bin/env python3
"""Paper figures and tables from saved neural RHFB and Fortran reference results.

Reads only: outputs/rhfb_spherical/comparison.json, production/<nuc>/history.csv,
production/<nuc>/summary.json, production/<nuc>/profiles.npz.  Writes PDF figures
to paper/figures and LaTeX tables to paper/tables.  No calculation is run.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Validated categorical slots (light surface): NN blue, reference orange, third aqua.
NN, REF, THIRD = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#1f1f1e", "#6b6a64"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e4e3de",
    "grid.linewidth": 0.5, "lines.linewidth": 1.5, "savefig.bbox": "tight",
})


def label(nucleus):
    """'Pb208' -> '$^{208}$Pb' for any element symbol."""
    match = re.match(r"([A-Za-z]+)(\d+)$", nucleus)
    return rf"$^{{{match.group(2)}}}${match.group(1)}" if match else nucleus


def _save(fig, path):
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def _history(path):
    rows = list(csv.DictReader(path.open()))
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}


def figure_training(prod, comparison, nucleus, out):
    history = _history(prod / nucleus / "history.csv")
    ref = next((r for r in comparison if r["nucleus"] == nucleus), None)
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 4.6), sharex=True)
    epoch = history["epoch"]
    axes[0].plot(epoch, history["energy_per_a_no_com_mev"], color=NN, label="NN-RHFB")
    axes[1].plot(epoch, history["pairing_energy_n_mev"], color=NN)
    if ref:
        axes[0].axhline(ref["ft_E_per_A"], color=REF, ls="--", lw=1.2, label="Reference RHFB")
        axes[1].axhline(ref["ft_E_pair_n"], color=REF, ls="--", lw=1.2)
    axes[2].semilogy(epoch, history["orbital_residual_rms_mev"], color=NN, label=r"orbital $R_Q$")
    axes[2].semilogy(epoch, history["occupation_residual_rms_mev"], color=THIRD, ls="-.",
                     label=r"occupation $g_\nu$")
    axes[0].set_ylabel(r"$E/A$ (MeV)")
    axes[1].set_ylabel(r"$E_{\rm pair}^{n}$ (MeV)")
    axes[2].set_ylabel("residual rms (MeV)")
    axes[2].set_xlabel("epoch")
    axes[0].legend(frameon=False, loc="upper right")
    axes[2].legend(frameon=False, loc="upper right")
    axes[0].set_title(label(nucleus),
                      fontsize=8, color=INK)
    _save(fig, out / f"training_{nucleus}.pdf")


def figure_systematics(comparison, out):
    labels = [r["nucleus"] for r in comparison]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 4.8), sharex=True)
    axes[0].plot(x, [1e3 * r["dE_per_A"] for r in comparison], "o", color=NN, ms=5)
    axes[0].axhline(0.0, color=MUTED, lw=0.8)
    axes[0].set_ylabel(r"$\Delta(E/A)$ (keV)")
    axes[1].plot(x, [r["ft_E_pair_n"] for r in comparison], "s", mfc="none", color=REF, ms=6,
                 label="Reference RHFB")
    axes[1].plot(x, [r["nn_E_pair_n"] for r in comparison], "o", color=NN, ms=4, label="NN-RHFB")
    axes[1].set_ylabel(r"$E_{\rm pair}^{n}$ (MeV)")
    axes[1].legend(frameon=False)
    with_lambda = [(i, r["ft_lambda_n"]) for i, r in zip(x, comparison) if r["ft_lambda_n"] is not None]
    axes[2].plot([i for i, _ in with_lambda], [v for _, v in with_lambda], "s", mfc="none",
                 color=REF, ms=6, label="Reference RHFB")
    axes[2].plot(x, [r["nn_lambda_n"] for r in comparison], "o", color=NN, ms=4, label="NN-RHFB")
    axes[2].set_ylabel(r"$\lambda_n$ (MeV)")
    axes[2].set_xticks(x, [label(l) for l in labels])
    _save(fig, out / "systematics.pdf")


def figure_levels(comparison, nucleus, out):
    row = next(r for r in comparison if r["nucleus"] == nucleus)
    levels = [l for l in row["levels"] if l["species"] == "n"]
    levels.sort(key=lambda l: l["ft_e"])
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.6), sharey=True)
    for l in levels:
        axes[0].hlines(l["ft_e"], 0.05, 0.45, color=REF, lw=1.5)
        axes[0].hlines(l["nn_e"], 0.55, 0.95, color=NN, lw=1.5)
        axes[0].text(1.0, l["nn_e"], l["level"], fontsize=6, va="center", color=INK)
        axes[1].plot(l["ft_v2"], l["ft_e"], "s", mfc="none", color=REF, ms=5)
        axes[1].plot(l["nn_v2"], l["nn_e"], "o", color=NN, ms=3.5)
    axes[0].set_xlim(0, 1.35)
    axes[0].set_xticks([0.25, 0.75], ["Ref.", "NN"])
    axes[0].set_ylabel(r"canonical $h_{\nu\nu}$ (MeV)")
    axes[1].set_xlabel(r"$v^2$")
    axes[1].set_xlim(-0.05, 1.05)
    _save(fig, out / f"levels_{nucleus}.pdf")


def figure_profiles(prod, nuclei, out):
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.8), sharex=True)
    colors = [NN, REF, THIRD]
    styles = ["-", "--", "-."]
    for i, nucleus in enumerate(nuclei):
        path = prod / nucleus / "profiles.npz"
        if not path.exists():
            continue
        data = np.load(path)
        r = data["r"]
        name = label(nucleus)
        axes[0].plot(r, data["rho_v_n"], color=colors[i], ls=styles[i], label=name)
        if "kappa_local_n" in data:
            inner = r[1:-1]
            axes[1].plot(inner, data["kappa_local_n"] / (4 * np.pi * inner**2),
                         color=colors[i], ls=styles[i], label=name)
    axes[0].set_ylabel(r"$\rho_{v}^{n}(r)$ (fm$^{-3}$)")
    axes[1].set_ylabel(r"$\kappa^{n}(r,r)$ (fm$^{-3}$)")
    axes[1].set_xlabel(r"$r$ (fm)")
    axes[1].set_xlim(0, 8)
    axes[0].legend(frameon=False)
    _save(fig, out / "profiles.pdf")


def _num(value, digits):
    """Fixed-point text without a negative zero."""
    text = f"{value:.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0.0 else text


def table_comparison(comparison, out):
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{$E/A$ (MeV)} & \multicolumn{2}{c}{$E^n_{\rm pair}$ (MeV)}"
             r" & \multicolumn{2}{c}{$\lambda_n$ (MeV)} & \multicolumn{2}{c}{$r_{\rm ch}$ (fm)}\\",
             r" & NN & Ref. & NN & Ref. & NN & Ref. & NN & Ref.\\", r"\midrule"]
    for r in comparison:
        nucleus = r["nucleus"]
        name = label(nucleus)
        lines.append(
            f"{name} & {r['nn_E_per_A']:.4f} & {r['ft_E_per_A']:.4f} & "
            f"{_num(r['nn_E_pair_n'], 3)} & {_num(r['ft_E_pair_n'], 3)} & "
            f"{r['nn_lambda_n']:.3f} & "
            f"{'--' if r['ft_lambda_n'] is None else format(r['ft_lambda_n'], '.3f')} & "
            f"{r['nn_r_ch']:.4f} & {r['ft_r_ch']:.4f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "comparison.tex").write_text("\n".join(lines) + "\n")


LEVEL = {-1: "s_{1/2}", 1: "p_{1/2}", -2: "p_{3/2}", 2: "d_{3/2}", -3: "d_{5/2}",
         3: "f_{5/2}", -4: "f_{7/2}", 4: "g_{7/2}", -5: "g_{9/2}"}


def table_parameters(prod, nuclei, out):
    lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
             r" & $C_n$ & $C_p$ & $\kappa$ range & blocks & $n_c$ & epochs & start\\",
             r"\midrule"]
    for nucleus in nuclei:
        path = prod / nucleus / "summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        space = summary["canonical_space"]
        caps = space.get("capacities") or {}
        extra = space["extra_capacity"]
        z, n = summary["nucleus"]["z"], summary["nucleus"]["n"]
        cap_n = caps.get("n") or n + extra
        cap_p = caps.get("p") or z + extra
        neg, pos = space["kappa_limits"]
        blocks = space["blocks"]
        columns = sum(len(b["principals"]) for b in blocks)
        init = None
        for train_dir in ("production", "production_remote"):
            train = prod.parent / train_dir / nucleus / "summary.json"
            if train.exists():
                init = json.loads(train.read_text())["config"].get("initial_model")
                break
        start = "osc." if not init else label(Path(init).parent.name)
        blocked = summary.get("blocking", {}).get("n")
        name = label(nucleus)
        if blocked is not None:
            name += rf" ($\nu 1{LEVEL.get(blocked, blocked)}$)"
        lines.append(f"{name} & {cap_n} & {cap_p} & $[-{neg},{pos}]$ & {len(blocks)} & {columns} & "
                     f"{summary['epochs_completed']} & {start}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "parameters.tex").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prod", default=str(ROOT / "outputs/rhfb_spherical/production"))
    parser.add_argument("--comparison", default=str(ROOT / "outputs/rhfb_spherical/comparison.json"))
    parser.add_argument("--training-nuclei", nargs="+", default=["O18"])
    parser.add_argument("--level-nucleus", default="O18")
    parser.add_argument("--profile-nuclei", nargs="+", default=["O18", "O20"])
    args = parser.parse_args()
    prod = Path(args.prod)
    figures, tables = ROOT / "paper/figures", ROOT / "paper/tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    comparison = json.loads(Path(args.comparison).read_text())
    for nucleus in args.training_nuclei:
        figure_training(prod, comparison, nucleus, figures)
    figure_systematics(comparison, figures)
    figure_levels(comparison, args.level_nucleus, figures)
    figure_profiles(prod, args.profile_nuclei, figures)
    table_comparison(comparison, tables)
    table_parameters(prod, [r["nucleus"] for r in comparison], tables)
    print("figures:", sorted(p.name for p in figures.iterdir()))


if __name__ == "__main__":
    main()
