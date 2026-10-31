from __future__ import annotations

import argparse

from dpl_rhf.training.prl_hamiltonian_trainer import train_prl_hamiltonian
from dpl_rhf.training.density_matrix_trainer import train_density_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PRL/AI2DFT-style local-Hamiltonian RMF trainer")
    parser.add_argument("--model", default="PKDD")
    parser.add_argument(
        "--functional",
        choices=["rmf-dd", "rmf-pkdd"],
        default="rmf-dd",
        help="rmf-pkdd is retained as a compatibility alias for rmf-dd",
    )
    parser.add_argument(
        "--mode", choices=["direct", "network", "density-matrix"], default="density-matrix"
    )
    parser.add_argument("--direct-order", type=int, default=64)
    parser.add_argument("--direct-gate", default=None)
    parser.add_argument("--allow-unvalidated-network", action="store_true")
    parser.add_argument("--gate-residual", type=float, default=1.0e-3)
    parser.add_argument("--gate-energy-mev", type=float, default=0.5)
    parser.add_argument("--gate-radius-fm", type=float, default=0.1)
    parser.add_argument("--gate-profile-relative", type=float, default=0.1)
    parser.add_argument("--gate-level-mev", type=float, default=1.0)
    parser.add_argument("--gate-adf-mev", type=float, default=0.5)
    parser.add_argument("--checkpoint-policy", choices=["physics", "last"], default="physics")
    parser.add_argument("--z", type=int, default=8)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--a", type=int, default=None)
    parser.add_argument("--backend", choices=["torch-rmf", "fortran-fixed"], default="torch-rmf")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--network-depth", type=int, default=3)
    parser.add_argument("--radial-basis-size", type=int, default=16)
    parser.add_argument("--message-neighbors", type=int, default=3)
    parser.add_argument(
        "--density-correction-scale",
        type=float,
        default=1.0e-2,
        help="scale of the neural correction before positive-space projection",
    )
    parser.add_argument(
        "--density-spectral-gate",
        type=float,
        default=1.0e-2,
        help="maximum final neural/spectral density relative L2 error",
    )
    parser.add_argument(
        "--density-stationarity-gate",
        type=float,
        default=1.0e-2,
        help="maximum normalized projected Euler-Lagrange residual",
    )
    parser.add_argument(
        "--density-stationarity-weight",
        type=float,
        default=1.0,
        help="coefficient of the squared full-space Euler-Lagrange residual",
    )
    parser.add_argument(
        "--density-energy-weight",
        type=float,
        default=1.0e-2,
        help="total-energy weight combined with the projected relativistic stationarity constraint",
    )
    parser.add_argument(
        "--density-refresh-interval",
        type=int,
        default=0,
        help="must remain zero; subspace refresh is forbidden",
    )
    parser.add_argument(
        "--density-inverse-weight", type=float, default=1.0,
        help="weight of the differentiable inverse-Hamiltonian no-sea constraint",
    )
    parser.add_argument(
        "--density-branch-floor", type=float, default=0.0,
        help="minimum dimensionless inverse-Hamiltonian Rayleigh score",
    )
    parser.add_argument("--activation", choices=["silu"], default="silu")
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="default: 1e-3 for density-matrix, 6e-5 for network, 5e-4 for direct",
    )
    parser.add_argument("--scheduler-patience", type=int, default=120)
    parser.add_argument("--scheduler-cooldown", type=int, default=40)
    parser.add_argument("--backtrack-threshold", type=float, default=2.0)
    parser.add_argument("--lr-decay", type=float, default=0.5)
    parser.add_argument("--max-backtracks", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--lambda-reconstruct", type=float, default=1.0e-3,
        help="PRL Hamiltonian reconstruction coefficient in MeV^-1",
    )
    parser.add_argument(
        "--lambda-mode",
        choices=["auto", "fixed", "balanced"],
        default="auto",
        help="auto uses balanced PRL gradient terms for network mode and the fixed coefficient for direct mode",
    )
    parser.add_argument("--energy-gradient-weight", type=float, default=1.0)
    parser.add_argument("--derivative-order", type=int, choices=[1, 4, 5, 6, 7], default=7)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=20240623)
    parser.add_argument(
        "--initial-checkpoint", default=None,
        help="continue network optimization from a density-matrix model.pt without restoring any subspace",
    )
    parser.add_argument("--compare-scf", action="store_true")
    parser.add_argument(
        "--final-spectral-validation",
        action="store_true",
        help="optionally diagonalize the final Python Hamiltonian for diagnostics",
    )
    parser.add_argument("--out", default="outputs/prl_hamiltonian_rmf")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "density-matrix":
        train_density_matrix(args)
    else:
        train_prl_hamiltonian(args)


if __name__ == "__main__":
    main()
