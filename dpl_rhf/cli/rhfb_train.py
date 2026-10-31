from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import uuid

from dpl_rhf.rhfb.config import BlockedRHFBConfig
from dpl_rhf.rhfb.result import append_jsonl, write_blocked_result
from dpl_rhf.rhfb.trainer import (
    PhysicalGateError,
    active_space_failures,
    functional_from_module,
    prepare_networks,
    train_rhfb,
    functional_preflight_from_module,
    validate_no_sea_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an explicitly blocked, no-refresh PKA1+D1S RHFB state"
    )
    parser.add_argument("--z", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--radius-fm", type=float, default=20.0)
    parser.add_argument("--spacing-fm", type=float, default=0.1)
    # The default 40-state complete-shell target for O-17 reaches 1g9/2
    # (|kappa|=5); kappa=4 fails before it can build the requested active space.
    parser.add_argument("--kappa-max", type=int, default=5)
    parser.add_argument("--active-n", type=int)
    parser.add_argument("--active-p", type=int)
    parser.add_argument("--block-n", default="auto",
                        help="auto or kappa=<integer>,m2=<integer> (m is doubled)")
    parser.add_argument("--block-p", default="auto")
    parser.add_argument("--candidate-shape", choices=("sphere", "prolate", "oblate", "triaxial"),
                        default="sphere",
                        help="initial unrestricted m-scheme seed for a shape-search start")
    parser.add_argument("--candidate-seed-amplitude", type=float, default=0.5,
                        help="relative symmetry-breaking Q-coordinate seed before optimization")
    parser.add_argument("--chunk-size", type=int, default=128,
                        help="spatial-point chunk for exact field evaluation")
    parser.add_argument("--pair-chunk", type=int, default=4,
                        help="active-orbital pair tile for exact Fock contractions")
    parser.add_argument("--lmax", type=int,
                        help="Green-function angular expansion cutoff; default selects full basis support")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--torch-threads", type=int, default=2,
                        help="CPU intra-op thread count")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--refresh-interval", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--number-penalty", type=float, default=50.0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", help="resume from a blocked-RHFB checkpoint")
    parser.add_argument("--no-sea-report", default="outputs/validation/prototype_gate.json")
    parser.add_argument("--out", required=True)
    return parser


def _config(args) -> BlockedRHFBConfig:
    values = {key: value for key, value in vars(args).items()
              if key not in {"out", "learning_rate", "number_penalty",
                             "checkpoint_every", "no_sea_report", "resume",
                             "device", "torch_threads"}}
    return BlockedRHFBConfig(**values)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    import torch
    if args.torch_threads < 1:
        raise SystemExit("--torch-threads must be a positive integer")
    torch.set_num_threads(args.torch_threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    config = _config(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    no_sea_ok, no_sea_evidence = validate_no_sea_report(args.no_sea_report)
    evaluate = functional_from_module()
    functional_ready, functional_readiness = functional_preflight_from_module()
    functional_info = {"available": evaluate is not None,
                       "module": "dpl_rhf.rhfb.functional",
                       "entry_point": "evaluate_rhfb_energy"}
    failures = []
    if not no_sea_ok:
        failures.append("independent moving-projector versus conventional odd-RHFB gate did not pass")
    if evaluate is None:
        failures.append("complete differentiable PKA1 RHFB functional is unavailable")
    if not functional_ready:
        failures.append("complete PKA1 functional channel closures are not ready")
    failures.extend(active_space_failures(config))

    if failures:
        gates = {
            "complete_pka1_energy_and_field": {"status": "ready" if functional_ready else "blocked",
                                                  **functional_info,
                                                  "readiness": functional_readiness},
            "moving_no_sea_and_odd_stationarity": no_sea_evidence,
            "unresolved": failures,
        }
        (out / "history.jsonl").unlink(missing_ok=True)
        report = write_blocked_result(
            out / "summary.json", config=config, reason="; ".join(failures),
            gates=gates,
            initialization={
                "network_constructed": False,
                "requested_block_n": config.block_n,
                "resolved_default_block_n": (
                    "1d5/2:kappa=-3:m2=5" if config.z == 8 and config.n == 9
                    and config.block_n == "auto" else None
                ),
                "requested_block_p": config.block_p,
                "dtype": "complex128", "refresh_interval": 0,
                "candidate_shape": config.candidate_shape,
                "device": str(device), "torch_threads": args.torch_threads,
                "functional_readiness": functional_readiness,
            },
            run_id=run_id,
        )
        append_jsonl(out / "history.jsonl", {
            "event": "physical-training-blocked", "status": "blocked",
            "reason": failures, "gates": gates,
        })
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    # A costly spherical-grid/network build starts only after both independent
    # physics prerequisites pass. No reduced or toy energy is substituted.
    from dpl_rhf.rhfb.execution_limits import require_bounded_cgroup
    require_bounded_cgroup()
    from dpl_rhf.rhfb.basis import AngularRadialBasis
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    basis = AngularRadialBasis(radius_fm=config.radius_fm,
                               spacing_fm=config.spacing_fm,
                               kappa_max=config.kappa_max,
                               n_theta=max(2 * config.kappa_max + 1,
                                           (config.lmax or 2 * config.kappa_max) + 1),
                               n_phi=max(4 * config.kappa_max + 2,
                                         2 * (config.lmax or 2 * config.kappa_max) + 1))
    networks, blocks = prepare_networks(config, basis, device=device)
    try:
        result = train_rhfb(
            config, basis, networks, blocks, evaluate,
            output_dir=out, learning_rate=args.learning_rate,
            number_penalty=args.number_penalty,
            checkpoint_every=args.checkpoint_every,
            resume_from=args.resume,
        )
    except PhysicalGateError as exc:
        gates = {"functional_audit": exc.audit, "unresolved": exc.failures,
                 "moving_no_sea_and_odd_stationarity": no_sea_evidence}
        report = write_blocked_result(out / "summary.json", config=config,
                                      reason=str(exc), gates=gates,
                                      initialization={"network_constructed": True,
                                                     "candidate_shape": config.candidate_shape,
                                                     "device": str(device),
                                                     "torch_threads": args.torch_threads,
                                                     "blocking": {k: (None if b is None else asdict(b))
                                                                  for k, b in blocks.items()}},
                                      run_id=run_id)
        append_jsonl(out / "history.jsonl", {"event": "functional-audit-blocked",
                                               "status": "blocked", "reason": exc.failures})
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
