#!/usr/bin/env python3
"""Continue every RHFB run until Sigma_+- are converged to a field tolerance.

Each run resumes from its own training_checkpoint.pt (copied into a new
directory, originals untouched) with the settings recorded in its
summary.json, restarts the learning-rate schedule, trains until
max |Sigma_+-(k) - Sigma_+-(k-window)| < --tolerance-kev and is then
re-evaluated self-consistently.  One guarded job per GPU lane.

Start: setsid nohup python scripts/remote_converge_sigma.py > converge.log 2>&1 < /dev/null &

--tag v2 resumes from the previous stage (--previous-tag, default the untagged
first stage) where it exists, so no epoch already trained is lost; only runs
that start from an original checkpoint restart their learning-rate schedule.
"""

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs/rhfb_spherical"


def train_arguments(summary: dict, opts=None) -> list[str]:
    cfg, space, nucleus = summary["config"], summary["canonical_space"], summary["nucleus"]
    kappa_limits = list(space["kappa_limits"])
    if opts is not None and opts.extend_kappa and nucleus["z"] >= opts.extend_kappa_min_z:
        kappa_limits = [max(a, b) for a, b in zip(kappa_limits, opts.extend_kappa)]
    args = ["--z", str(nucleus["z"]), "--n", str(nucleus["n"]),
            "--pairing", "d1s" if summary["pairing"] else "none",
            "--capacity-n", str(space["capacities"]["n"]), "--capacity-p", str(space["capacities"]["p"]),
            "--extra-capacity", str(space["extra_capacity"]),
            "--kappa-limits", *(str(k) for k in kappa_limits),
            "--lr", str(cfg["learning_rate"]), "--logit-lr", str(cfg["logit_learning_rate"]),
            "--energy-weight", str(cfg["energy_weight"]),
            "--stationarity-weight", str(cfg["stationarity_weight"]),
            "--inverse-weight", str(cfg["inverse_weight"]), "--branch-floor", str(cfg["branch_floor"]),
            "--hidden", str(cfg["hidden"]), "--depth", str(cfg["depth"]), "--seed", str(cfg["seed"])]
    if opts is not None and opts.min_states_per_kappa:
        args += ["--min-states-per-kappa", str(opts.min_states_per_kappa), "--pair-sign-init", "phase"]
    for species in ("n", "p"):
        if summary["blocking"].get(species) is not None:
            args += [f"--block-{species}", str(summary["blocking"][species])]
    return args


def jobs():
    found = []
    for set_name in ("production", "dripline"):
        for run in sorted((RUNS / set_name).iterdir()):
            if (run / "training_checkpoint.pt").exists() and (run / "summary.json").exists():
                found.append((set_name, run))
    # Heaviest first so the lanes finish together.
    return sorted(found, key=lambda item: -json.loads((item[1] / "summary.json").read_text())["nucleus"]["a"])


def stage_dir(tag):
    return "converged" + (f"_{tag}" if tag else "")


def run_job(args, gpu, set_name, run, status):
    summary = json.loads((run / "summary.json").read_text())
    name = run.name
    out = RUNS / stage_dir(args.tag) / set_name / name
    evaluation = RUNS / (stage_dir(args.tag) + "_eval") / set_name / name
    if (evaluation / "summary.json").exists():
        return
    previous = RUNS / stage_dir(args.previous_tag) / set_name / name
    source = previous if args.tag and (previous / "training_checkpoint.pt").exists() else run
    out.mkdir(parents=True, exist_ok=True)
    warm_model = None
    if args.warm_from_tag is not None:
        # A changed canonical space cannot resume a checkpoint; start a new
        # optimization from the previous stage's final state instead.
        warm_model = RUNS / (stage_dir(args.warm_from_tag) + "_eval") / set_name / name / "model_final.pt"
        if not warm_model.exists():
            status(f"{set_name}/{name} skipped: no warm-start model {warm_model}")
            return
    elif not (out / "training_checkpoint.pt").exists():
        shutil.copy2(source / "training_checkpoint.pt", out / "training_checkpoint.pt")
        if (source / "history.csv").exists():
            shutil.copy2(source / "history.csv", out / "history.csv")
    common = train_arguments(summary, args) + ["--device", "cuda", "--cuda-budget-mib", str(args.gpu_mib)]
    guard = [args.python, "scripts/run_rhfb_procguard.py", "--memory-mib", str(args.host_mib),
             "--gpu-indices", str(gpu), "--gpu-memory-mib", str(args.gpu_mib),
             "--timeout-seconds", str(args.timeout)]
    env = {**os.environ, "RHFB_AUTHORIZED_GPUS": args.gpus,
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    execution = ROOT / "outputs/execution"
    execution.mkdir(parents=True, exist_ok=True)
    if not (out / "summary.json").exists():
        if warm_model is not None:
            start_epoch = 0
            # Resume only an interrupted run of this stage.
            start = (["--resume"] if (out / "training_checkpoint.pt").exists()
                     else ["--initial-model", str(warm_model)])
        else:
            with open(source / "history.csv") as handle:
                start_epoch = sum(1 for _ in handle) - 1
            start = ["--resume"] + (["--reset-lr"] if source == run else [])
        train = guard + ["--report", str(execution / f"{stage_dir(args.tag)}_train_{set_name}_{name}.json"), "--",
                         args.python, "-u", "-m", "dpl_rhf.cli.rhfb_spherical_train", *common,
                         *start, "--min-lr", str(args.min_lr),
                         "--converge-sigma-kev", str(args.tolerance_kev),
                         "--convergence-window", str(args.window),
                         "--convergence-patience", str(args.patience),
                         "--sigma-lr-patience", str(args.sigma_lr_patience),
                         "--epochs", str(start_epoch + args.max_extra_epochs), "--out", str(out)]
        with open(out.parent / f"{name}.log", "a") as log:
            code = subprocess.call(train, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        status(f"{set_name}/{name} trained [gpu {gpu}] exit={code}")
        if code != 0 or not (out / "summary.json").exists():
            return
    evaluation.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out / "history.csv", evaluation / "history.csv")
    evaluate = guard + ["--report", str(execution / f"{stage_dir(args.tag)}_eval_{set_name}_{name}.json"), "--",
                        args.python, "-u", "-m", "dpl_rhf.cli.rhfb_spherical_train", *common,
                        "--evaluate-only", str(out / "training_checkpoint.pt"),
                        "--evaluate-state", "last", "--out", str(evaluation)]
    with open(evaluation.parent / f"{name}.log", "a") as log:
        code = subprocess.call(evaluate, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    conv = json.loads((out / "summary.json").read_text()).get("convergence") or {}
    status(f"{set_name}/{name} evaluated [gpu {gpu}] exit={code} converged={conv.get('converged')} "
           f"dSigma={conv.get('final_change_kev')} keV epochs={conv.get('epochs_completed')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--tolerance-kev", type=float, default=10.0)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--patience", type=int, default=1,
                        help="consecutive epochs the tolerance must hold")
    parser.add_argument("--sigma-lr-patience", type=int, default=0)
    parser.add_argument("--tag", default="", help="stage tag of the output directories")
    parser.add_argument("--warm-from-tag", default=None,
                        help="start from <stage>_eval/<set>/<nucleus>/model_final.pt (new space)")
    parser.add_argument("--min-states-per-kappa", type=int, default=0)
    parser.add_argument("--extend-kappa", type=int, nargs=2, default=None, metavar=("NEG", "POS"))
    parser.add_argument("--extend-kappa-min-z", type=int, default=50)
    parser.add_argument("--previous-tag", default="", help="stage resumed by --tag")
    parser.add_argument("--min-lr", type=float, default=2.0e-5)
    parser.add_argument("--max-extra-epochs", type=int, default=20000)
    parser.add_argument("--gpu-mib", type=int, default=12288)
    parser.add_argument("--host-mib", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=604800)
    parser.add_argument("--python", default=str(Path.home() / "rhfb-nn/env/bin/python"))
    args = parser.parse_args()

    queue = jobs()
    lock = threading.Lock()
    status_file = RUNS / (stage_dir(args.tag) + ".status")

    def status(message):
        with lock, open(status_file, "a") as handle:
            handle.write(f"{time.strftime('%F %T')} {message}\n")

    def lane(gpu):
        while True:
            with lock:
                if not queue:
                    return
                set_name, run = queue.pop(0)
            run_job(args, gpu, set_name, run, status)

    status(f"start: {len(queue)} runs, tolerance {args.tolerance_kev} keV over {args.window} epochs, "
           f"{args.patience} consecutive, sigma lr patience {args.sigma_lr_patience}")
    threads = [threading.Thread(target=lane, args=(int(g),)) for g in args.gpus.split(",")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    status("CONVERGE_DONE")


if __name__ == "__main__":
    main()
