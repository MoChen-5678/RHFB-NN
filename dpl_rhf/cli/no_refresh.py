from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-space neural RMF/RHF solver with zero subspace refreshes"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train-rmf", add_help=False)
    subparsers.add_parser("train-rhf", add_help=False)
    subparsers.add_parser("train-rhfb", add_help=False)
    subparsers.add_parser("validate-rhfb-prototype", add_help=False)
    compare = subparsers.add_parser("compare-core")
    compare.add_argument("--kind", choices=("rmf", "rhf"), required=True)
    compare.add_argument("--result", required=True)
    compare.add_argument("--model", required=True)
    compare.add_argument("--z", type=int, required=True)
    compare.add_argument("--n", type=int, required=True)
    compare.add_argument("--a", type=int, default=None)
    compare.add_argument("--core-dir", default="../Core-1204")
    compare.add_argument("--out", required=True)
    return parser


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else None
    if command == "train-rhfb":
        from dpl_rhf.cli.rhfb_train import main as train

        train(sys.argv[2:])
        return
    if command == "validate-rhfb-prototype":
        from dpl_rhf.cli.rhfb_prototype import main as validate

        validate(sys.argv[2:])
        return
    if command == "train-rmf":
        from dpl_rhf.cli.prl_train import main as train

        sys.argv = [sys.argv[0], "--mode", "density-matrix", *sys.argv[2:]]
        train()
        return
    if command == "train-rhf":
        from dpl_rhf.cli.rhf_train import main as train

        sys.argv = [sys.argv[0], *sys.argv[2:]]
        train()
        return

    args = build_parser().parse_args()
    a = args.a if args.a is not None else args.z + args.n
    if args.kind == "rmf":
        from dpl_rhf.cli.compare_core1204 import main as compare

        sys.argv = [
            sys.argv[0], "--dpl-dir", args.result, "--model", args.model,
            "--z", str(args.z), "--n", str(args.n), "--a", str(a),
            "--core-dir", args.core_dir, "--out", args.out,
        ]
    else:
        from dpl_rhf.cli.compare_rhf_core1204 import main as compare

        sys.argv = [
            sys.argv[0], "--result-dir", args.result, "--z", str(args.z),
            "--n", str(args.n), "--a", str(a), "--core-dir", args.core_dir,
            "--out", args.out,
        ]
    compare()


if __name__ == "__main__":
    main()
