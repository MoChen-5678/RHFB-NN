from __future__ import annotations

import argparse

import torch

from dpl_rhf.backends.base import NucleusCase
from dpl_rhf.rhf.backend import DensityMatrixRHFBackend
from dpl_rhf.rhf.trainer import RHFTrainingConfig, train_rhf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the strict h_kin+h_D+h_E density-matrix PKO2/PKA1 RHF solver"
    )
    parser.add_argument(
        "--solver", choices=["density-matrix"], default="density-matrix"
    )
    parser.add_argument("--model", choices=["PKO2", "PKA1"], default="PKO2")
    parser.add_argument("--z", type=int, default=8)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--a", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--radial-basis-size", type=int, default=16)
    parser.add_argument("--correction-scale", type=float, default=1.0e-2)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--energy-weight", type=float, default=1.0e-2,
        help="explicit RHF total-energy weight in the joint physics loss",
    )
    parser.add_argument(
        "--stationarity-weight",
        type=float,
        default=1.0,
        help="weight of the squared full-space RHF Euler-Lagrange residual",
    )
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--refresh-interval", type=int, default=0,
        help="must remain zero; subspace refresh is forbidden",
    )
    parser.add_argument("--inverse-weight", type=float, default=1.0)
    parser.add_argument(
        "--channel-balance-weight",
        type=float,
        default=0.0,
        help="annealed optimizer preconditioner for kinetic/direct/Fock gradients",
    )
    parser.add_argument(
        "--channel-balance-epochs",
        type=int,
        default=0,
        help="number of stage-local epochs over which channel balancing decays to zero",
    )
    parser.add_argument(
        "--branch-floor",
        type=float,
        default=1.0,
        help="strict inverse-Hamiltonian lower bound for occupied bound states",
    )
    parser.add_argument(
        "--no-sea-projection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "restrict occupied factors to the positive-energy space of the "
            "current mean field (no-sea approximation); disabling it allows "
            "variational collapse into the Dirac sea"
        ),
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--initial-checkpoint",
        default=None,
        help="start a fresh optimizer from a compatible trained density state",
    )
    parser.add_argument(
        "--initial-dilation-factor",
        type=float,
        default=1.0,
        help="scale checkpoint radial-dilation coordinates before fresh training",
    )
    parser.add_argument(
        "--initial-uniform-dilation",
        type=float,
        default=None,
        help="fresh-run radial dilation coordinate shared by all occupied states",
    )
    parser.add_argument(
        "--graph-offload",
        choices=["none", "memory", "disk", "recompute", "cpu"],
        default="none",
        help=(
            "saved-tensor mode for A>=48: none keeps the graph on the device "
            "(batched Fock fits Pb208 in about 8 GiB); memory keeps it in host "
            "RAM, disk stores it on local storage, recompute regenerates exact "
            "Fock activations during backward; cpu is a legacy alias for memory"
        ),
    )
    parser.add_argument(
        "--graph-offload-dir",
        default="/tmp/dpl-rhf-graph-cache",
        help="temporary disk directory used only by --graph-offload disk",
    )
    parser.add_argument("--graph-offload-min-mib", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--fock-stream-devices", default=None,
        help="stream Fock exchange per kappa pair over these process-local devices, "
             "e.g. cuda:0,cuda:1 (memory-bounded; pairs split across GPUs)",
    )
    parser.add_argument(
        "--torch-threads", type=int, default=2,
        help="host intra-op threads; bounded by default to keep heavy PKA1 responsive",
    )
    parser.add_argument("--out", default="outputs/rhf/PKO2/O16")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (
        args.torch_threads < 1
        or args.checkpoint_every < 0
        or args.graph_offload_min_mib < 0.0
        or args.channel_balance_weight < 0.0
        or args.channel_balance_epochs < 0
        or args.initial_dilation_factor < 0.0
    ):
        raise ValueError("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    a = args.a if args.a is not None else args.z + args.n
    if a != args.z + args.n:
        raise ValueError("A must equal Z+N")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        import os
        if os.environ.get("RHFB_CUDA_UUID"):
            # Under a resource guard: enforce its per-device allocator limit.
            from dpl_rhf.rhfb.cuda_resources import configure_cuda
            configure_cuda("cuda:0", budget_mib=int(os.environ["RHFB_CUDA_BUDGET_MIB"]))
            device = torch.device("cuda:0")
    torch.set_default_dtype(torch.float64)
    case = NucleusCase(args.model, args.z, args.n, a)
    backend = DensityMatrixRHFBackend(case, device=device)
    if args.fock_stream_devices:
        devices = [torch.device(d) for d in args.fock_stream_devices.split(",")]
        if devices[0] != (torch.device("cuda:0") if device.type == "cuda" and device.index is None
                          else device):
            raise SystemExit("--fock-stream-devices must start with the primary device")
        backend.functional.configure_fock_streaming(devices)
    config = RHFTrainingConfig(
        epochs=args.epochs,
        hidden=args.hidden,
        depth=args.depth,
        radial_basis_size=args.radial_basis_size,
        correction_scale=args.correction_scale,
        learning_rate=1.0e-3 if args.lr is None else args.lr,
        energy_weight=args.energy_weight,
        stationarity_weight=args.stationarity_weight,
        grad_clip=args.grad_clip,
        print_every=args.print_every,
        seed=args.seed,
        refresh_interval=args.refresh_interval,
        inverse_weight=args.inverse_weight,
        branch_floor=args.branch_floor,
        channel_balance_weight=args.channel_balance_weight,
        channel_balance_epochs=args.channel_balance_epochs,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        graph_offload=args.graph_offload,
        graph_offload_dir=args.graph_offload_dir,
        graph_offload_min_bytes=int(args.graph_offload_min_mib * 1024**2),
        initial_checkpoint=args.initial_checkpoint,
        initial_dilation_factor=args.initial_dilation_factor,
        initial_uniform_dilation=args.initial_uniform_dilation,
        no_sea_projection=args.no_sea_projection,
    )
    _, result = train_rhf(backend, config, args.out)
    print(
        f"best {args.model} RHF no-CoM: E={result.energy_total_no_com:.8f} MeV, "
        f"E/A={result.energy_per_a_no_com:.8f} MeV",
        flush=True,
    )


if __name__ == "__main__":
    main()
