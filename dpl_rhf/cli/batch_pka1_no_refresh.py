from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


NUCLEI = {
    "O16": (8, 8, 16),
    "Ca40": (20, 20, 40),
    "Ca48": (20, 28, 48),
    "Sn132": (50, 82, 132),
    "Pb208": (82, 126, 208),
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _training_complete(case_dir: Path, epochs: int) -> bool:
    summary_path = case_dir / "summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = _read_json(summary_path)
        completed_epochs = int(summary["training_config"]["epochs"])
        counts = summary["solver_counts"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        completed_epochs >= epochs
        and int(counts["eigensolve_count"]) == 0
        and int(counts["refresh_count"]) == 0
        and int(counts["training_dirac_solves"]) == 0
    )


def _run(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{time.strftime('%F %T')}] COMMAND: {' '.join(command)}\n")
        handle.flush()
        subprocess.run(
            command,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )


def _write_status(out: Path, rows: list[dict]) -> None:
    temporary = out / "RUN_STATUS.json.tmp"
    temporary.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out / "RUN_STATUS.json")


def _train_and_compare(nucleus: str, args: argparse.Namespace) -> dict:
    z, n, a = NUCLEI[nucleus]
    case_dir = args.out / nucleus
    case_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("OMP_NUM_THREADS", str(args.torch_threads))
    env.setdefault("MKL_NUM_THREADS", str(args.torch_threads))
    env.setdefault("OPENBLAS_NUM_THREADS", str(args.torch_threads))
    env.setdefault("NUMEXPR_NUM_THREADS", str(args.torch_threads))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if not _training_complete(case_dir, args.epochs):
        train = [
            args.python,
            "-m",
            "dpl_rhf.cli.no_refresh",
            "train-rhf",
            "--model",
            "PKA1",
            "--z",
            str(z),
            "--n",
            str(n),
            "--a",
            str(a),
            "--epochs",
            str(args.epochs),
            "--hidden",
            str(args.hidden),
            "--depth",
            str(args.depth),
            "--radial-basis-size",
            str(args.radial_basis_size),
            "--lr",
            str(args.lr),
            "--energy-weight",
            str(args.energy_weight),
            "--inverse-weight",
            str(args.inverse_weight),
            "--refresh-interval",
            "0",
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--print-every",
            str(args.print_every),
            "--torch-threads",
            str(args.torch_threads),
            "--graph-offload",
            args.graph_offload,
            "--graph-offload-dir",
            str(args.graph_offload_dir),
            "--graph-offload-min-mib",
            str(args.graph_offload_min_mib),
            "--resume",
            "--device",
            "cuda",
            "--out",
            str(case_dir),
        ]
        if args.stationarity_weight is not None:
            train[train.index("--inverse-weight"):train.index("--inverse-weight")] = [
                "--stationarity-weight",
                str(args.stationarity_weight),
            ]
        _run(train, case_dir / "train.log", env)

    comparison_path = case_dir / "core_comparison" / "detailed_comparison.json"
    if args.force_compare or not comparison_path.exists():
        compare = [
            args.python,
            "-m",
            "dpl_rhf.cli.no_refresh",
            "compare-core",
            "--kind",
            "rhf",
            "--result",
            str(case_dir),
            "--model",
            "PKA1",
            "--z",
            str(z),
            "--n",
            str(n),
            "--a",
            str(a),
            "--core-dir",
            str(args.core_dir),
            "--out",
            str(case_dir / "core_comparison"),
        ]
        _run(compare, case_dir / "compare.log", env)
    return {"nucleus": nucleus, "status": "completed", "epochs": args.epochs}


def _aggregate(out: Path, nuclei: list[str]) -> None:
    rows: list[dict] = []
    for nucleus in nuclei:
        path = out / nucleus / "core_comparison" / "detailed_comparison.json"
        if not path.exists():
            continue
        data = _read_json(path)
        difference = data["differences"]
        local = difference.get(
            "local_direct_rearrangement", difference.get("local_hartree_rearrangement")
        )
        nonlocal_r = difference["nonlocal_density_R"]
        rows.append(
            {
                "nucleus": nucleus,
                "dpl_e_per_a_mev": data["dpl"]["energy_per_a_no_com_mev"],
                "core_e_per_a_mev": data["core"]["e_per_A_no_com"],
                "delta_e_per_a_mev": difference["energy_per_a_no_com_mev"],
                "delta_matter_radius_fm": difference["rms_matter_no_com_fm"],
                "delta_charge_radius_fm": difference["charge_radius_no_com_fm"],
                "occupied_level_mae_mev": difference["occupied_level_mae_mev"],
                "local_s_n_rmse_mev": local["S_n"]["rmse_mev"],
                "local_s_p_rmse_mev": local["S_p"]["rmse_mev"],
                "local_v_n_rmse_mev": local["V_n"]["rmse_mev"],
                "local_v_p_rmse_mev": local["V_p"]["rmse_mev"],
                "nonlocal_r_relative_l2_percent": nonlocal_r[
                    "aggregate_relative_l2_percent"
                ],
                "nonlocal_r_rmse_fm_inverse": nonlocal_r[
                    "aggregate_rmse_fm_inverse"
                ],
                "eigensolve_count": data["dpl"]["eigensolve_count"],
                "refresh_count": data["dpl"]["refresh_count"],
                "training_dirac_solves": data["dpl"]["training_dirac_solves"],
            }
        )
    (out / "batch_summary.json").write_text(
        json.dumps({"functional": "PKA1", "solver": "no-refresh", "results": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    if rows:
        with (out / "batch_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sequential, resumable five-nucleus PKA1 no-refresh benchmark"
    )
    parser.add_argument("--nuclei", nargs="+", choices=tuple(NUCLEI), default=list(NUCLEI))
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--radial-basis-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--energy-weight", type=float, default=1.0e-2)
    parser.add_argument("--stationarity-weight", type=float, default=1.0)
    parser.add_argument("--inverse-weight", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--graph-offload", choices=("memory", "recompute"), default="memory")
    parser.add_argument("--graph-offload-dir", type=Path, default=Path("outputs/.pka1_graph_cache"))
    parser.add_argument("--graph-offload-min-mib", type=float, default=1.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--core-dir", type=Path, default=Path("../Core-1204"))
    parser.add_argument("--out", type=Path, default=Path("outputs/no_refresh/PKA1_five"))
    parser.add_argument("--force-compare", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs < 0 or args.torch_threads < 1 or args.checkpoint_every < 0:
        raise ValueError("epochs/checkpoint interval must be non-negative and threads positive")
    args.out.mkdir(parents=True, exist_ok=True)
    status: list[dict] = []
    for nucleus in args.nuclei:
        running = {"nucleus": nucleus, "status": "running", "epochs": args.epochs}
        status.append(running)
        _write_status(args.out, status)
        try:
            status[-1] = _train_and_compare(nucleus, args)
        except Exception as exc:
            status[-1] = {
                "nucleus": nucleus,
                "status": "failed",
                "epochs": args.epochs,
                "error": str(exc),
            }
            _write_status(args.out, status)
            _aggregate(args.out, args.nuclei)
            raise
        _write_status(args.out, status)
        _aggregate(args.out, args.nuclei)


if __name__ == "__main__":
    main()
