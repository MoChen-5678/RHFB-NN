from __future__ import annotations

import argparse
import json

import torch

from dpl_rhf.backends.base import NucleusCase
from dpl_rhf.rhf.hfb_backend import SphericalRHFBBackend
from dpl_rhf.rhf.hfb_trainer import RHFBTrainingConfig, evaluate_saved, train_rhfb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spherical no-refresh neural RHFB (PKA1 + Gogny D1S pairing)")
    parser.add_argument("--model", choices=["PKA1", "PKO2"], default="PKA1")
    parser.add_argument("--z", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--pairing", choices=["d1s", "none"], default="d1s")
    parser.add_argument("--block-n", type=int, default=None,
                        help="kappa of the equal-filling blocked neutron level (odd N)")
    parser.add_argument("--block-p", type=int, default=None,
                        help="kappa of the equal-filling blocked proton level (odd Z)")
    parser.add_argument("--extra-capacity", type=int, default=20,
                        help="canonical states beyond the particle number per species")
    parser.add_argument("--capacity-n", type=int, default=None,
                        help="absolute neutron canonical capacity (overrides --extra-capacity)")
    parser.add_argument("--capacity-p", type=int, default=None)
    parser.add_argument("--initial-model", default=None,
                        help="warm start from model_final.pt with identical canonical blocks")
    parser.add_argument("--kappa-limits", type=int, nargs=2, default=(5, 4),
                        metavar=("NEG", "POS"), help="keep -NEG <= kappa <= POS")
    parser.add_argument("--epochs", type=int, default=600,
                        help="fixed epochs, or the safety cap with --converge-sigma-kev")
    parser.add_argument("--converge-sigma-kev", type=float, default=None,
                        help="stop when Sigma_+ and Sigma_- (n, p) change by less than this "
                             "many keV over --convergence-window epochs")
    parser.add_argument("--convergence-window", type=int, default=10)
    parser.add_argument("--convergence-patience", type=int, default=1,
                        help="consecutive epochs the Sigma tolerance must hold")
    parser.add_argument("--sigma-lr-patience", type=int, default=0,
                        help="halve the learning rates after this many epochs without a 10%% "
                             "improvement of the windowed Sigma change (0: energy plateau)")
    parser.add_argument("--min-lr", type=float, default=0.0,
                        help="floor of the plateau learning-rate schedule")
    parser.add_argument("--reset-lr", action="store_true",
                        help="on --resume, restart from --lr/--logit-lr with a fresh schedule")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--logit-lr", type=float, default=2.0e-2)
    parser.add_argument("--energy-weight", type=float, default=2.0)
    parser.add_argument("--stationarity-weight", type=float, default=0.0,
                        help="0: energy minimization with residual as diagnostic")
    parser.add_argument("--inverse-weight", type=float, default=10.0)
    parser.add_argument("--branch-floor", type=float, default=0.5)
    parser.add_argument("--alignment-weight", type=float, default=0.0,
                        help="annealed inverse-Hamiltonian alignment warm-up weight")
    parser.add_argument("--alignment-epochs", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--graph-offload", choices=["none", "cpu", "recompute"], default="none")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--fock-stream-devices", default=None,
                        help="stream Fock exchange per kappa pair over these process-local "
                             "devices, e.g. cuda:0,cuda:1 (pairs split across GPUs)")
    parser.add_argument("--cuda-budget-mib", type=int, default=None,
                        help="required for CUDA; must equal the protected launcher budget")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--evaluate-only", default=None, metavar="MODEL_FINAL_PT",
                        help="re-evaluate a saved model with the current code; no training")
    parser.add_argument("--evaluate-state", choices=["best", "last"], default="best",
                        help="state of a training checkpoint used by --evaluate-only")
    parser.add_argument("--out", required=True)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(args.torch_threads)
    torch.set_default_dtype(torch.float64)
    from dpl_rhf.rhfb.execution_limits import require_bounded_cgroup
    require_bounded_cgroup()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda":
        from dpl_rhf.rhfb.cuda_resources import configure_cuda
        if args.cuda_budget_mib is None:
            raise SystemExit("--cuda-budget-mib is required with --device cuda")
        configure_cuda("cuda:0", budget_mib=args.cuda_budget_mib)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    case = NucleusCase(args.model, args.z, args.n, args.z + args.n)
    backend = SphericalRHFBBackend(
        case, device=device, pairing=args.pairing == "d1s",
        extra_capacity=args.extra_capacity, kappa_limits=tuple(args.kappa_limits),
        blocking={"n": args.block_n, "p": args.block_p},
        capacities={"n": args.capacity_n, "p": args.capacity_p},
    )
    if args.fock_stream_devices:
        devices = [torch.device(d) for d in args.fock_stream_devices.split(",")]
        if any(d.type != device.type for d in devices) or devices[0] != device:
            raise SystemExit("--fock-stream-devices must start with the primary device")
        backend.functional.configure_fock_streaming(devices)
    config = RHFBTrainingConfig(
        epochs=args.epochs, hidden=args.hidden, depth=args.depth,
        learning_rate=args.lr, logit_learning_rate=args.logit_lr,
        energy_weight=args.energy_weight, stationarity_weight=args.stationarity_weight,
        inverse_weight=args.inverse_weight, branch_floor=args.branch_floor,
        print_every=args.print_every, checkpoint_every=args.checkpoint_every,
        seed=args.seed, resume=args.resume, graph_offload=args.graph_offload,
        alignment_weight=args.alignment_weight, alignment_epochs=args.alignment_epochs,
        initial_model=args.initial_model,
        convergence_sigma_kev=args.converge_sigma_kev,
        convergence_window=args.convergence_window,
        convergence_patience=args.convergence_patience,
        sigma_lr_patience=args.sigma_lr_patience,
        min_learning_rate=args.min_lr,
        reset_learning_rate=args.reset_lr,
    )
    if args.evaluate_only:
        summary = evaluate_saved(backend, config, args.evaluate_only, args.out,
                                 state=args.evaluate_state)
    else:
        _, _, summary = train_rhfb(backend, config, args.out)
    printed = {key: summary[key] for key in (
        "energy_total_no_com_mev", "energy_per_a_no_com_mev", "chemical_potentials_mev",
        "average_canonical_gaps_mev", "particle_numbers", "radii_fm", "stationarity", "counts")}
    printed["E_pair"] = summary["energy_terms_mev"]["E_pair"]
    print(json.dumps(printed, indent=2))


if __name__ == "__main__":
    main()
