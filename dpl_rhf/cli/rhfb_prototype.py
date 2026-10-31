from __future__ import annotations

import argparse
import json


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Independent small-model no-sea RHFB acceptance gate")
    parser.add_argument("--out", required=True, help="Path to the JSON acceptance report")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args(argv)
    if args.epochs < 10:
        parser.error("epochs must be at least 10")
    from dpl_rhf.rhfb.prototype import run_prototype_validation

    report = run_prototype_validation(args.out, epochs=args.epochs, seed=args.seed)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report.get("status") != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
