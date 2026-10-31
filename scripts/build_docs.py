from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DEFAULT_EXAMPLE = ROOT / "outputs" / "pkdd_o16_density_matrix_refresh_cpu_500"
PAGES = [
    "INSTALLATION.md", "USER_GUIDE.md", "PHYSICS_AND_WORKFLOW.md",
    "EXPERIMENT_DESIGN.md", "RESULTS_PKDD_O16.md", "OUTPUT_SCHEMA.md",
    "DEVELOPER_GUIDE.md", "TROUBLESHOOTING.md", "REFERENCES.md",
]


def _box(ax, xy, width, height, title, body, color):
    ax.add_patch(Rectangle(xy, width, height, facecolor=color, edgecolor="#263238", lw=1.2))
    x, y = xy
    ax.text(x + width / 2, y + height * 0.66, title, ha="center", va="center", fontsize=11, weight="bold")
    ax.text(x + width / 2, y + height * 0.32, body, ha="center", va="center", fontsize=8.5)


def architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.5); ax.axis("off")
    boxes = [
        ((0.3, 3.1), "Nucleus", "model, Z, N\nradial grid", "#e3f2fd"),
        ((2.7, 3.1), "SiLU network", "radial features\nquantum embeddings", "#fff3e0"),
        ((5.1, 3.1), "Hard projection", "positive space\nQR orthonormality", "#e8f5e9"),
        ((7.5, 3.1), "RMF functional", "densities, fields\nS/V Hamiltonian", "#f3e5f5"),
        ((9.9, 3.1), "Variational loss", "energy + projected\nstationarity", "#ffebee"),
    ]
    for xy, title, body, color in boxes:
        _box(ax, xy, 2.0, 1.35, title, body, color)
    for x in (2.3, 4.7, 7.1, 9.5):
        ax.add_patch(FancyArrowPatch((x, 3.78), (x + 0.35, 3.78), arrowstyle="-|>", mutation_scale=14, lw=1.3))
    _box(ax, (3.3, 0.7), 2.4, 1.25, "Sparse spectral refresh", "every N epochs only\nno eigensolve in ordinary steps", "#e0f7fa")
    _box(ax, (7.3, 0.7), 2.4, 1.25, "Final validation", "spectrum and density check\nCore-1204 external SCF", "#f1f8e9")
    ax.add_patch(FancyArrowPatch((10.9, 3.05), (8.6, 2.0), arrowstyle="-|>", connectionstyle="arc3,rad=.15", mutation_scale=14))
    ax.add_patch(FancyArrowPatch((7.25, 1.3), (5.75, 1.3), arrowstyle="<|-", mutation_scale=14))
    ax.text(6.5, 5.05, "DPL-RMF density-matrix neural variational workflow", ha="center", fontsize=16, weight="bold")
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def constraints(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 6)
    ax.text(5.5, 5.55, "Physical manifold embedded in the forward pass", ha="center", fontsize=16, weight="bold")
    labels = [
        ("Positive-energy basis U", "No-sea subspace", "#e3f2fd"),
        ("Neural coordinates C_raw", "SiLU radial correction", "#fff3e0"),
        ("QR: C^T C = I", "Exact orthonormality", "#e8f5e9"),
        ("P = U C n C^T U^T", "Hermitian, fixed N/Z, idempotent", "#f3e5f5"),
        ("Projected EL residual", "Stationarity on physical tangent", "#ffebee"),
    ]
    for i, (title, body, color) in enumerate(labels):
        x = 0.55 + i * 2.1
        _box(ax, (x, 2.35), 1.75, 1.45, title, body, color)
        if i < len(labels) - 1:
            ax.add_patch(FancyArrowPatch((x + 1.75, 3.08), (x + 2.06, 3.08), arrowstyle="-|>", mutation_scale=13))
    ax.text(5.5, 1.2, "Penalty-free invariants: particle number | orthogonality | Hermiticity | no-sea", ha="center", fontsize=12)
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def convergence(example: Path, path: Path) -> None:
    with (example / "history.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    epoch = np.array([int(r["epoch"]) for r in rows])
    energy = np.array([float(r["energy_per_a_no_com_mev"]) for r in rows])
    residual = np.array([float(r["normalized_commutator"]) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epoch, energy, color="#1565c0", lw=1.8)
    axes[0].set(xlabel="Epoch", ylabel="E/A no CoM (MeV)", title="Variational energy"); axes[0].grid(alpha=.25)
    axes[1].semilogy(epoch, residual, color="#c62828", lw=1.8)
    axes[1].axhline(.01, color="#2e7d32", ls="--", label="1% gate")
    axes[1].set(xlabel="Epoch", ylabel="Normalized residual", title="Stationarity"); axes[1].grid(alpha=.25); axes[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def dashboard(example: Path, path: Path) -> None:
    summary = json.loads((example / "summary.json").read_text(encoding="utf-8"))
    comparison = json.loads((example / "scf_comparison" / "detailed_comparison.json").read_text(encoding="utf-8"))
    diff = comparison["differences"]
    values = [
        100 * summary["final_normalized_commutator"],
        100 * summary["spectral_validation"]["max_density_relative_l2"],
        100 * abs(diff["energy_per_a_no_com_mev"] / comparison["core1204_scf"]["energy"]["e_per_A_no_com"]),
        diff["occupied_level_mae_mev"],
    ]
    labels = ["Stationarity\n(%)", "Density L2\n(%)", "E/A error\n(%)", "Level MAE\n(MeV)"]
    limits = [1.0, 1.0, 1.0, 1.0]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, label, value, limit in zip(axes, labels, values, limits):
        ax.bar([0], [value], width=.55, color="#2e7d32" if value <= limit else "#c62828")
        ax.axhline(limit, color="#37474f", ls="--", lw=1)
        ax.set_xticks([0], [label]); ax.set_ylim(0, max(limit * 1.15, value * 1.25)); ax.grid(axis="y", alpha=.2)
        ax.text(0, value, f"{value:.4g}", ha="center", va="bottom", fontsize=10)
    fig.suptitle("PKDD/O16 internal and external validation", fontsize=14, weight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def markdown_to_html(text: str) -> str:
    lines = text.splitlines(); out = []; in_code = False; in_list = False
    for raw in lines:
        if raw.startswith("```"):
            if in_list: out.append("</ul>"); in_list = False
            out.append("</code></pre>" if in_code else "<pre><code>"); in_code = not in_code; continue
        if in_code:
            out.append(html.escape(raw)); continue
        if raw.startswith("#"):
            if in_list: out.append("</ul>"); in_list = False
            level = len(raw) - len(raw.lstrip("#")); out.append(f"<h{level}>{html.escape(raw[level:].strip())}</h{level}>"); continue
        if raw.startswith("- "):
            if not in_list: out.append("<ul>"); in_list = True
            out.append(f"<li>{html.escape(raw[2:])}</li>"); continue
        if in_list: out.append("</ul>"); in_list = False
        image = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", raw.strip())
        if image: out.append(f'<figure><img src="{html.escape(image.group(2))}" alt="{html.escape(image.group(1))}"></figure>'); continue
        if raw.strip(): out.append(f"<p>{html.escape(raw)}</p>")
    if in_list: out.append("</ul>")
    return "\n".join(out)


def html_site() -> None:
    nav = "".join(f'<a href="#{Path(page).stem}">{Path(page).stem.replace("_", " ")}</a>' for page in PAGES)
    sections = []
    for page in PAGES:
        content = (DOCS / page).read_text(encoding="utf-8")
        sections.append(f'<section id="{Path(page).stem}">{markdown_to_html(content)}</section>')
    css = """body{margin:0;font:16px/1.65 system-ui;color:#202124;background:#fff}nav{position:fixed;width:245px;height:100vh;overflow:auto;background:#f5f7f8;padding:24px;box-sizing:border-box}nav a{display:block;color:#174ea6;text-decoration:none;padding:5px 0}main{margin-left:285px;max-width:1040px;padding:36px 48px}h1{border-bottom:2px solid #263238;padding-bottom:8px}h2{margin-top:36px}pre{background:#202124;color:#f5f5f5;padding:16px;overflow:auto}img{max-width:100%;height:auto;border:1px solid #ddd}p{white-space:pre-wrap}section{margin-bottom:70px}@media(max-width:800px){nav{position:static;width:auto;height:auto}main{margin:0;padding:20px}}"""
    page = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>DPL-RMF v1.0.0 Documentation</title><style>{css}</style></head><body><nav><h2>DPL-RMF v1.0.0</h2>{nav}</nav><main>{"".join(sections)}</main></body></html>'
    (DOCS / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline DPL-RMF figures and HTML documentation")
    parser.add_argument("--example", type=Path, default=DEFAULT_EXAMPLE)
    args = parser.parse_args(); images = DOCS / "images"; images.mkdir(parents=True, exist_ok=True)
    architecture(images / "architecture_workflow.png")
    constraints(images / "density_matrix_constraints.png")
    convergence(args.example, images / "training_convergence.png")
    dashboard(args.example, images / "validation_dashboard.png")
    shutil.copy2(args.example / "scf_comparison" / "vs_dpl_scf.png", images / "vs_dpl_scf.png")
    html_site(); print(DOCS / "index.html")


if __name__ == "__main__":
    main()
