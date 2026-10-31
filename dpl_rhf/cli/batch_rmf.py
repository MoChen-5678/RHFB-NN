from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Iterable

import numpy as np

from dpl_rhf.functionals.pkdd_action import (
    RMF_PARAMETER_DATA,
    canonical_rmf_model,
)


NUCLEI = {
    "O16": (8, 8, 16),
    "Ca40": (20, 20, 40),
    "Ca48": (20, 28, 48),
    "Sn132": (50, 82, 132),
    "Pb208": (82, 126, 208),
}
DEFAULT_MODELS = tuple(RMF_PARAMETER_DATA)
DEFAULT_NUCLEI = tuple(NUCLEI)
PRINT_LOCK = threading.Lock()


def _message(text: str) -> None:
    with PRINT_LOCK:
        print(text, flush=True)


def _run(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        result = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=Path(__file__).resolve().parents[2],
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}; see {log_path}"
        )


def _completed_training(case_dir: Path, epochs: int, mode: str) -> bool:
    summary_path = case_dir / "summary.json"
    if not summary_path.exists() or not (case_dir / "dpl_hamiltonian.npz").exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        int(summary.get("epochs", -1)) == epochs
        and summary.get("variation_mode") == mode
    )


def _archive_incomplete(case_dir: Path) -> None:
    if not case_dir.exists() or not any(case_dir.iterdir()):
        return
    suffix = time.strftime("%Y%m%d-%H%M%S")
    archived = case_dir.with_name(case_dir.name + f".incomplete-{suffix}")
    case_dir.rename(archived)


def _case_commands(
    *,
    model: str,
    nucleus: str,
    case_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[str], list[str], list[str]]:
    z, n, a = NUCLEI[nucleus]
    train = [
        args.python,
        "-m",
        "dpl_rhf.cli.prl_train",
        "--model",
        model,
        "--functional",
        "rmf-dd",
        "--z",
        str(z),
        "--n",
        str(n),
        "--a",
        str(a),
        "--mode",
        args.mode,
        "--backend",
        "torch-rmf",
        "--epochs",
        str(args.epochs),
        "--energy-gradient-weight",
        "1.0",
        "--derivative-order",
        "7",
        "--activation",
        "silu",
        "--checkpoint-policy",
        "physics",
        "--density-refresh-interval",
        str(args.refresh_interval),
        "--print-every",
        str(args.print_every),
        "--device",
        "cuda",
        "--out",
        str(case_dir),
    ]
    if args.mode in {"network", "density-matrix"}:
        train.extend([
            "--hidden", str(args.hidden),
            "--network-depth", str(args.network_depth),
            "--radial-basis-size", str(args.radial_basis_size),
            "--lr", str(args.lr),
            "--scheduler-patience", str(args.scheduler_patience),
            "--scheduler-cooldown", str(args.scheduler_cooldown),
        ])
        if args.mode == "network":
            train.extend([
                "--message-neighbors", str(args.message_neighbors),
                "--lambda-mode", "balanced",
                "--allow-unvalidated-network",
            ])
    else:
        train.extend([
            "--direct-order", str(args.direct_order),
            "--lambda-mode", "fixed",
            "--lambda-reconstruct", "0.001",
        ])
    comparison_dir = case_dir / "scf_comparison"
    compare = [
        args.python,
        "-m",
        "dpl_rhf.cli.compare_core1204",
        "--dpl-dir",
        str(case_dir),
        "--model",
        model,
        "--z",
        str(z),
        "--n",
        str(n),
        "--a",
        str(a),
        "--tolerance",
        "1e-10",
        "--max-iterations",
        "500",
        "--out",
        str(comparison_dir),
    ]
    if args.core_dir is not None:
        compare.extend(["--core-dir", str(args.core_dir)])
    plot = [
        args.python,
        "-m",
        "dpl_rhf.cli.plot_vs_comparison",
        "--dpl",
        str(case_dir / "dpl_hamiltonian.npz"),
        "--scf",
        str(comparison_dir / "core1204_scf_profiles.npz"),
        "--model",
        model,
        "--nucleus",
        nucleus,
        "--out",
        str(comparison_dir / "vs_dpl_scf.png"),
        "--r-max",
        "12",
    ]
    return train, compare, plot


def _run_case(
    *,
    gpu: str,
    slot: int,
    model: str,
    nucleus: str,
    args: argparse.Namespace,
) -> None:
    slug = model.lower().replace("-", "")
    case_dir = args.out / slug / nucleus
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("OMP_NUM_THREADS", str(args.torch_threads))
    env.setdefault("MKL_NUM_THREADS", str(args.torch_threads))
    env.setdefault("OPENBLAS_NUM_THREADS", str(args.torch_threads))
    env.setdefault("NUMEXPR_NUM_THREADS", str(args.torch_threads))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    train, compare, plot = _case_commands(
        model=model,
        nucleus=nucleus,
        case_dir=case_dir,
        args=args,
    )
    worker = f"GPU {gpu}/slot {slot}"
    _message(f"[{worker}] START {model}/{nucleus}")
    if not _completed_training(case_dir, args.epochs, args.mode):
        _archive_incomplete(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        _run(train, case_dir / "train.log", env)
    else:
        _message(f"[{worker}] RESUME training already complete: {model}/{nucleus}")

    comparison_dir = case_dir / "scf_comparison"
    comparison_json = comparison_dir / "detailed_comparison.json"
    comparison_complete = (
        comparison_json.exists()
        and (comparison_dir / "core1204_scf_profiles.npz").exists()
        and (comparison_dir / "occupied_levels.csv").exists()
        and (comparison_dir / "all_observables.csv").exists()
        and (comparison_dir / "v_plus_minus_s_dpl_scf.png").exists()
    )
    if args.refresh_scf or not comparison_complete:
        _run(compare, case_dir / "compare.log", env)
    plot_complete = (
        (comparison_dir / "vs_dpl_scf.png").exists()
        and (comparison_dir / "vs_dpl_scf.pdf").exists()
        and (comparison_dir / "vs_dpl_scf_data.npz").exists()
    )
    if args.refresh_scf or not plot_complete:
        _run(plot, case_dir / "plot.log", env)
    _message(f"[{worker}] DONE  {model}/{nucleus}")


def _worker(
    gpu: str,
    slot: int,
    tasks: Iterable[tuple[str, str]],
    args: argparse.Namespace,
) -> None:
    for model, nucleus in tasks:
        try:
            _run_case(
                gpu=gpu,
                slot=slot,
                model=model,
                nucleus=nucleus,
                args=args,
            )
        except Exception as exc:
            _message(f"[GPU {gpu}/slot {slot}] FAIL  {model}/{nucleus}: {exc}")
            failure = args.out / "failures.jsonl"
            with PRINT_LOCK:
                with failure.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "gpu": gpu, "slot": slot, "model": model, "nucleus": nucleus,
                        "error": str(exc),
                    }, ensure_ascii=False) + "\n")


def _vs_metrics(path: Path) -> dict[str, float]:
    archive = np.load(path)
    r = np.asarray(archive["r"])
    mask = (r >= 0.1) & (r <= 12.0)
    result = {}
    for field in ("V_n", "S_n", "V_p", "S_p"):
        delta = np.asarray(archive[f"delta_{field}"])[mask]
        result[f"{field}_mae_mev"] = float(np.mean(np.abs(delta)))
        result[f"{field}_rmse_mev"] = float(np.sqrt(np.mean(delta * delta)))
        result[f"{field}_max_mev"] = float(np.max(np.abs(delta)))
    return result


def _plot_model_chain(model: str, rows: list[dict], model_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = {name: index for index, name in enumerate(DEFAULT_NUCLEI)}
    rows = sorted(rows, key=lambda row: order[row["nucleus"]])
    labels = [row["nucleus"] for row in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), constrained_layout=True)
    panels = (
        ("e_per_a", "E/A no CoM (MeV)"),
        ("matter_radius", "Matter radius (fm)"),
        ("charge_radius", "Charge radius (fm)"),
    )
    for axis, (prefix, ylabel) in zip(axes.flat[:3], panels):
        axis.plot(
            x,
            [row[f"dpl_{prefix}_mev"] if prefix == "e_per_a" else row[f"dpl_{prefix}_fm"] for row in rows],
            marker="o",
            linewidth=2,
            label="DPL",
        )
        axis.plot(
            x,
            [row[f"scf_{prefix}_mev"] if prefix == "e_per_a" else row[f"scf_{prefix}_fm"] for row in rows],
            marker="s",
            linestyle="--",
            linewidth=1.8,
            label="Core SCF",
        )
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, labels)
        axis.grid(True, color="#dddddd")
        axis.legend(frameon=False)
    axis = axes.flat[3]
    axis.semilogy(
        x,
        [max(row["occupied_level_mae_mev"], 1.0e-12) for row in rows],
        marker="o",
        linewidth=2,
        color="#7a3e9d",
    )
    axis.set_ylabel("Occupied-level MAE (MeV)")
    axis.set_xticks(x, labels)
    axis.grid(True, which="both", color="#dddddd")
    fig.suptitle(f"{model}: DPL versus Core SCF along the nuclear chain")
    fig.savefig(model_dir / "nuclear_chain_observables.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    colors = {"V_n": "#1769aa", "S_n": "#c43d3d", "V_p": "#2d8f5b", "S_p": "#b06a17"}
    for field, color in colors.items():
        axes[0].semilogy(
            x,
            [max(row[f"{field}_rmse_mev"], 1.0e-12) for row in rows],
            marker="o",
            linewidth=2,
            color=color,
            label=field,
        )
        axes[1].semilogy(
            x,
            [max(row[f"{field}_max_mev"], 1.0e-12) for row in rows],
            marker="o",
            linewidth=2,
            color=color,
            label=field,
        )
    axes[0].set_ylabel("V/S RMSE (MeV)")
    axes[1].set_ylabel("V/S maximum error (MeV)")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(True, which="both", color="#dddddd")
        axis.legend(frameon=False, ncol=2)
    fig.suptitle(f"{model}: scalar/vector potential errors by nucleus")
    fig.savefig(model_dir / "nuclear_chain_vs_errors.png", dpi=220)
    plt.close(fig)


def _write_model_chain(model: str, rows: list[dict], out: Path) -> None:
    model_dir = out / model.lower().replace("-", "")
    model_dir.mkdir(parents=True, exist_ok=True)
    order = {name: index for index, name in enumerate(DEFAULT_NUCLEI)}
    rows = sorted(rows, key=lambda row: order[row["nucleus"]])
    with (model_dir / "nuclear_chain_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# {model} 五核素 DPL/Core-1204 SCF 对比",
        "",
        "| 核素 | 最佳轮 | DPL E/A | SCF E/A | 差值 | DPL Rm | SCF Rm | Rm差 | 能级MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['nucleus']} | {row['best_epoch']} | "
            f"{row['dpl_e_per_a_mev']:.8f} | {row['scf_e_per_a_mev']:.8f} | "
            f"{row['delta_e_per_a_mev']:+.6g} | "
            f"{row['dpl_matter_radius_fm']:.8f} | "
            f"{row['scf_matter_radius_fm']:.8f} | "
            f"{row['delta_matter_radius_fm']:+.6g} | "
            f"{row['occupied_level_mae_mev']:.6g} |"
        )
    lines += [
        "",
        "## V/S 势误差",
        "",
        "| 核素 | Vn RMSE | Sn RMSE | Vp RMSE | Sp RMSE | V/S 径向图 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        plot = Path(row["vs_plot"]).relative_to(model_dir)
        lines.append(
            f"| {row['nucleus']} | {row['V_n_rmse_mev']:.6g} | "
            f"{row['S_n_rmse_mev']:.6g} | {row['V_p_rmse_mev']:.6g} | "
            f"{row['S_p_rmse_mev']:.6g} | [{row['nucleus']}]({plot.as_posix()}) |"
        )
    lines += [
        "",
        "![核素链总体物理量](nuclear_chain_observables.png)",
        "",
        "![核素链 V/S 误差](nuclear_chain_vs_errors.png)",
    ]
    (model_dir / "NUCLEAR_CHAIN_COMPARISON.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    _plot_model_chain(model, rows, model_dir)


def aggregate(out: Path, models: tuple[str, ...], nuclei: tuple[str, ...]) -> list[dict]:
    rows = []
    all_observables = []
    all_levels = []
    for model in models:
        slug = model.lower().replace("-", "")
        for nucleus in nuclei:
            case_dir = out / slug / nucleus
            summary_path = case_dir / "summary.json"
            comparison_path = case_dir / "scf_comparison" / "detailed_comparison.json"
            vs_path = case_dir / "scf_comparison" / "vs_dpl_scf_data.npz"
            if not (summary_path.exists() and comparison_path.exists() and vs_path.exists()):
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            dpl = comparison["dpl_differentiable_rmf"]
            scf = comparison["core1204_scf"]
            differences = comparison["differences"]
            dpl_energy = dpl["energy_terms_mev"]
            scf_energy = scf["energy"]
            diagnostics = dpl["diagnostics"]
            rows.append({
                "model": model,
                "nucleus": nucleus,
                "z": NUCLEI[nucleus][0],
                "n": NUCLEI[nucleus][1],
                "a": NUCLEI[nucleus][2],
                "epochs": summary["epochs"],
                "best_epoch": summary["best_epoch"],
                "best_monitor": summary["best_monitor"],
                "internal_gate_passed": summary["internal_physics_gate_passed"],
                "formula_invariants_passed": summary["formula_invariants"]["passed"],
                "dpl_e_per_a_mev": dpl_energy["per_a_no_com"],
                "scf_e_per_a_mev": scf_energy["e_per_A_no_com"],
                "delta_e_per_a_mev": differences["energy_per_a_no_com_mev"],
                "dpl_total_energy_mev": dpl_energy["total_no_com"],
                "scf_total_energy_mev": scf_energy["e_total_no_com"],
                "dpl_kinetic_energy_mev": dpl_energy["kinetic"],
                "scf_kinetic_energy_mev": scf_energy["e_kin"],
                "dpl_direct_energy_mev": dpl_energy["direct_total"],
                "scf_direct_energy_mev": scf_energy["e_dir"],
                "dpl_neutron_radius_fm": diagnostics["rms_n_no_com"],
                "scf_neutron_radius_fm": scf_energy["rms_n_no_com"],
                "dpl_proton_radius_fm": diagnostics["rms_p_no_com"],
                "scf_proton_radius_fm": scf_energy["rms_p_no_com"],
                "dpl_matter_radius_fm": diagnostics["rms_matter_no_com"],
                "scf_matter_radius_fm": scf_energy["rms_matter_no_com"],
                "delta_matter_radius_fm": differences["matter_radius_no_com_fm"],
                "dpl_charge_radius_fm": diagnostics["charge_radius_no_com"],
                "scf_charge_radius_fm": scf_energy["charge_radius_no_com"],
                "delta_charge_radius_fm": differences["charge_radius_no_com_fm"],
                "occupied_level_mae_mev": differences["occupied_level_mae_mev"],
                "occupied_level_rmse_mev": differences["occupied_level_rmse_mev"],
                **_vs_metrics(vs_path),
                "case_dir": str(case_dir),
                "vs_plot": str(case_dir / "scf_comparison" / "vs_dpl_scf.png"),
                "v_plus_minus_s_plot": str(
                    case_dir / "scf_comparison" / "v_plus_minus_s_dpl_scf.png"
                ),
            })
            with (case_dir / "scf_comparison" / "all_observables.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                for item in csv.DictReader(handle):
                    all_observables.append({"model": model, "nucleus": nucleus, **item})
            with (case_dir / "scf_comparison" / "occupied_levels.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                for item in csv.DictReader(handle):
                    all_levels.append({"model": model, "nucleus": nucleus, **item})

    out.mkdir(parents=True, exist_ok=True)
    (out / "batch_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if rows:
        with (out / "batch_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        with (out / "all_observables.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=all_observables[0].keys())
            writer.writeheader()
            writer.writerows(all_observables)
        with (out / "all_single_particle_levels.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=all_levels[0].keys())
            writer.writeheader()
            writer.writerows(all_levels)
        for model in models:
            model_rows = [row for row in rows if row["model"] == model]
            if model_rows:
                _write_model_chain(model, model_rows, out)
    lines = [
        "# 五参数集五核素 DPL/Core-1204 SCF 对比",
        "",
        "| 参数集 | 核素 | 最佳轮 | DPL E/A | SCF E/A | 差值 (MeV) | Rm 差 (fm) | 能级 MAE (MeV) | V/S 图 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        plot = Path(row["vs_plot"]).relative_to(out)
        lines.append(
            f"| {row['model']} | {row['nucleus']} | {row['best_epoch']} | "
            f"{row['dpl_e_per_a_mev']:.8f} | "
            f"{row['scf_e_per_a_mev']:.8f} | "
            f"{row['delta_e_per_a_mev']:+.6g} | "
            f"{row['delta_matter_radius_fm']:+.6g} | "
            f"{row['occupied_level_mae_mev']:.6g} | [{plot.name}]({plot.as_posix()}) |"
        )
    lines += [
        "",
        f"已完成 {len(rows)}/{len(models) * len(nuclei)} 组。训练不使用 SCF 标签；表中 SCF 只在最佳点确定后计算。",
    ]
    (out / "BATCH_COMPARISON.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run five differentiable RMF models for five nuclei on multiple GPUs"
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument(
        "--mode", choices=["network", "direct", "density-matrix"], default="density-matrix"
    )
    parser.add_argument("--direct-order", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--network-depth", type=int, default=3)
    parser.add_argument("--radial-basis-size", type=int, default=16)
    parser.add_argument("--message-neighbors", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--scheduler-patience", type=int, default=20)
    parser.add_argument("--scheduler-cooldown", type=int, default=10)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument(
        "--refresh-interval",
        type=int,
        default=0,
        help="must remain zero; spectral subspace refresh is forbidden",
    )
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument(
        "--slots-per-gpu",
        type=int,
        default=3,
        help="independent processes sharing each GPU",
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--nuclei", nargs="+", default=list(DEFAULT_NUCLEI))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--core-dir",
        type=Path,
        default=None,
        help="Core-1204 directory; defaults to DPL_RHF_CORE_DIR or bundled sibling",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/rmf_five_by_five_prl_network_gpu_2000"),
    )
    parser.add_argument("--refresh-scf", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    args.models = tuple(canonical_rmf_model(model) for model in args.models)
    unknown = set(args.nuclei) - set(NUCLEI)
    if unknown:
        parser.error(f"unknown nuclei: {', '.join(sorted(unknown))}")
    args.nuclei = tuple(args.nuclei)
    args.gpus = tuple(item.strip() for item in args.gpus.split(",") if item.strip())
    if not args.gpus and not args.aggregate_only:
        parser.error("at least one GPU is required")
    if args.slots_per_gpu < 1:
        parser.error("--slots-per-gpu must be positive")
    if args.refresh_interval != 0:
        parser.error("the no-refresh release requires --refresh-interval 0")
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not args.aggregate_only:
        tasks = [
            (model, nucleus)
            for model in args.models
            for nucleus in args.nuclei
        ]
        workers = [
            (gpu, slot)
            for gpu in args.gpus
            for slot in range(args.slots_per_gpu)
        ]
        partitions = [tasks[index::len(workers)] for index in range(len(workers))]
        threads = [
            threading.Thread(
                target=_worker,
                args=(gpu, slot, partition, args),
                name=f"gpu-{gpu}-slot-{slot}",
            )
            for (gpu, slot), partition in zip(workers, partitions)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    rows = aggregate(args.out, args.models, args.nuclei)
    _message(f"aggregate complete: {len(rows)}/{len(args.models) * len(args.nuclei)} cases")


if __name__ == "__main__":
    main()
