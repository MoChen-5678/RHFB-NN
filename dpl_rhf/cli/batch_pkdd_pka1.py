from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _training_complete(case_dir: Path, epochs: int) -> bool:
    path = case_dir / "summary.json"
    if not path.exists():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    configured_epochs = summary.get("epochs")
    if configured_epochs is None:
        configured_epochs = summary.get("training_config", {}).get("epochs")
    return int(configured_epochs if configured_epochs is not None else -1) == epochs


def _archive_incomplete(case_dir: Path) -> None:
    if (case_dir / "training_checkpoint.pt").exists():
        return
    if not case_dir.exists() or not any(case_dir.iterdir()):
        return
    archived = case_dir.with_name(
        case_dir.name
        + ".incomplete-"
        + time.strftime("%Y%m%d-%H%M%S")
        + f"-{time.time_ns()}"
    )
    case_dir.rename(archived)


def _run(command: list[str], log: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT, env=env)


def _case(model: str, nucleus: str, gpu: str, args: argparse.Namespace) -> dict:
    z, n, a = NUCLEI[nucleus]
    case_dir = args.out / model / nucleus
    training_complete = _training_complete(case_dir, args.epochs)
    if not training_complete:
        _archive_incomplete(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("OMP_NUM_THREADS", str(args.torch_threads))
    env.setdefault("MKL_NUM_THREADS", str(args.torch_threads))
    env.setdefault("OPENBLAS_NUM_THREADS", str(args.torch_threads))
    env.setdefault("NUMEXPR_NUM_THREADS", str(args.torch_threads))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if model == "PKDD":
        train = [
            args.python, "-m", "dpl_rhf.cli.prl_train", "--mode", "density-matrix",
            "--model", model, "--z", str(z), "--n", str(n), "--a", str(a),
            "--epochs", str(args.epochs), "--device", "cuda", "--hidden", str(args.hidden),
            "--network-depth", str(args.depth), "--radial-basis-size", str(args.radial_basis_size),
            "--lr", str(args.lr), "--density-refresh-interval", str(args.refresh_interval), "--print-every", str(args.print_every),
            "--out", str(case_dir),
        ]
        compare = [
            args.python, "-m", "dpl_rhf.cli.compare_core1204", "--dpl-dir", str(case_dir),
            "--model", model, "--z", str(z), "--n", str(n), "--a", str(a),
            "--core-dir", str(args.core_dir), "--out", str(case_dir / "core_comparison"),
        ]
    else:
        graph_offload = (
            args.pka1_pb208_graph_offload
            if nucleus == "Pb208" and args.pka1_pb208_graph_offload is not None
            else args.pka1_graph_offload
        )
        train = [
            args.python, "-m", "dpl_rhf.cli.rhf_train", "--model", "PKA1",
            "--z", str(z), "--n", str(n), "--a", str(a), "--epochs", str(args.epochs),
            "--device", "cuda", "--hidden", str(args.hidden), "--depth", str(args.depth),
            "--radial-basis-size", str(args.radial_basis_size), "--lr", str(args.lr),
            "--refresh-interval", str(args.refresh_interval), "--torch-threads", str(args.torch_threads),
            "--checkpoint-every", str(args.checkpoint_every), "--resume",
            "--graph-offload", graph_offload,
            "--graph-offload-dir", str(args.pka1_graph_offload_dir),
            "--graph-offload-min-mib", str(args.pka1_graph_offload_min_mib),
            "--print-every", str(args.print_every), "--out", str(case_dir),
        ]
        if nucleus == "Pb208" and args.pka1_pb208_stationarity_weight is not None:
            train += [
                "--stationarity-weight",
                str(args.pka1_pb208_stationarity_weight),
            ]
        compare = [
            args.python, "-m", "dpl_rhf.cli.compare_rhf_core1204", "--result-dir", str(case_dir),
            "--z", str(z), "--n", str(n), "--a", str(a), "--core-dir", str(args.core_dir),
            "--out", str(case_dir / "core_comparison"),
        ]
        if args.pka1_memory_max:
            properties = [
                "--slice", args.pka1_slice,
                "--setenv", f"CUDA_VISIBLE_DEVICES={gpu}",
                "--setenv", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                "-p", f"MemoryMax={args.pka1_memory_max}",
            ]
            if args.pka1_memory_high:
                properties += ["-p", f"MemoryHigh={args.pka1_memory_high}"]
            train = ["systemd-run", "--user", "--scope", "--quiet", *properties, *train]
    if not training_complete:
        _run(train, case_dir / "train.log", env)
    comparison = case_dir / "core_comparison" / "detailed_comparison.json"
    if not comparison.exists():
        _run(compare, case_dir / "compare.log", env)
    return {"model": model, "nucleus": nucleus, "gpu": gpu, "status": "completed"}


def _aggregate(out: Path) -> None:
    rows = []
    for model in ("PKDD", "PKA1"):
        for nucleus in NUCLEI:
            path = out / model / nucleus / "core_comparison" / "detailed_comparison.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            diff = data["differences"]
            rows.append({
                "model": model,
                "nucleus": nucleus,
                "delta_e_per_a_mev": diff["energy_per_a_no_com_mev"],
                "delta_matter_radius_fm": diff.get("matter_radius_no_com_fm", diff.get("rms_matter_no_com_fm")),
                "delta_charge_radius_fm": diff.get("charge_radius_no_com_fm"),
                "occupied_level_mae_mev": diff.get("occupied_level_mae_mev"),
            })
    if rows:
        with (out / "batch_observables.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    (out / "BATCH_STATUS.json").write_text(json.dumps({"completed": rows}, indent=2) + "\n", encoding="utf-8")


def _run_lane(model: str, gpu: str, nuclei: list[str], args: argparse.Namespace) -> list[dict]:
    results = []
    for nucleus in nuclei:
        try:
            result = _case(model, nucleus, gpu, args)
        except Exception as exc:
            result = {"model": model, "nucleus": nucleus, "gpu": gpu, "status": "failed", "error": str(exc)}
        results.append(result)
        print(json.dumps(result), flush=True)
    return results


def _run_model(model: str, gpus: list[str], slots_per_gpu: int, args: argparse.Namespace) -> list[dict]:
    workers = [gpu for gpu in gpus for _ in range(slots_per_gpu)]
    if model == "PKA1":
        workers = workers[: args.pka1_total_workers]
    lanes = [[] for _ in workers]
    for index, nucleus in enumerate(NUCLEI):
        lanes[index % len(workers)].append(nucleus)
    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        futures = [
            pool.submit(_run_lane, model, gpu, lane, args)
            for gpu, lane in zip(workers, lanes)
            if lane
        ]
        return [item for future in as_completed(futures) for item in future.result()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel PKDD/PKA1 five-nucleus neural/Core benchmark")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--radial-basis-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--refresh-interval", type=int, default=0)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--pkdd-slots-per-gpu", type=int, default=2)
    parser.add_argument("--pka1-slots-per-gpu", type=int, default=1)
    parser.add_argument(
        "--pka1-total-workers", type=int, default=1,
        help="global PKA1 concurrency; keep at 1 because all GPUs share pinned/host memory",
    )
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--pka1-memory-high", default="105G")
    parser.add_argument("--pka1-memory-max", default="120G")
    parser.add_argument(
        "--pka1-graph-offload",
        choices=["memory", "disk", "recompute", "cpu"],
        default="memory",
        help="RHF saved-tensor mode; cpu is a legacy alias for memory",
    )
    parser.add_argument(
        "--pka1-pb208-graph-offload",
        choices=["memory", "disk", "recompute", "cpu"],
        default=None,
        help="optional Pb208-only override, useful until its recompute GPU peak is measured",
    )
    parser.add_argument(
        "--pka1-pb208-stationarity-weight",
        type=float,
        default=None,
        help=(
            "optional Pb208-only fixed stationarity weight; use an exact value "
            "from an identical initial-state calibration to avoid repeating the "
            "memory-intensive calibration gradients"
        ),
    )
    parser.add_argument(
        "--pka1-graph-offload-dir",
        type=Path,
        default=Path("outputs/.pka1_graph_cache"),
    )
    parser.add_argument("--pka1-graph-offload-min-mib", type=float, default=0.25)
    parser.add_argument(
        "--pka1-slice", default="dpl-pka1.slice",
        help="shared systemd slice used to enforce an aggregate host-memory limit",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--core-dir", type=Path, default=Path("../Core-1204"))
    parser.add_argument("--out", type=Path, default=Path("outputs/pkdd_pka1_five_nuclei_refresh4_500"))
    args = parser.parse_args()
    if (
        args.pka1_total_workers < 1
        or args.torch_threads < 1
        or args.pka1_graph_offload_min_mib < 0.0
    ):
        parser.error("PKA1 workers and torch threads must be positive")
    if args.refresh_interval != 0:
        parser.error("the no-refresh release requires --refresh-interval 0")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as model_pool:
        model_futures = [
            model_pool.submit(_run_model, "PKDD", gpus, args.pkdd_slots_per_gpu, args),
            model_pool.submit(_run_model, "PKA1", gpus, args.pka1_slots_per_gpu, args),
        ]
        results = [item for future in model_futures for item in future.result()]
    (args.out / "RUN_STATUS.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    _aggregate(args.out)


if __name__ == "__main__":
    main()
