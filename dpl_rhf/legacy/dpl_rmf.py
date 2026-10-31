from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from dpl_rmf_core import MODEL_INDEX, RHFCore, require_rmf_model, write_json


def import_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for `run`: install torch in this environment") from exc
    return torch


def ensure_out(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_core_arrays(core: RHFCore, out: Path) -> None:
    np.savez(out / "final_potentials.npz", r=core.r, **core.local_potentials(), **core.fields())
    np.savez(out / "final_densities.npz", r=core.r, **core.densities())
    np.savez(out / "final_wavefunctions.npz", **core.wavefunctions())


def reported_energy(energy, mode: str) -> dict[str, float | str]:
    if mode == "with-com":
        return {
            "mode": "with-com",
            "e_total": energy.e_total_with_com,
            "e_per_A": energy.e_per_A_with_com,
            "charge_radius": energy.charge_radius_with_com,
            "rms_matter": energy.rms_matter_with_com,
        }
    return {
        "mode": "no-com",
        "e_total": energy.e_total_no_com,
        "e_per_A": energy.e_per_A_no_com,
        "charge_radius": energy.charge_radius_no_com,
        "rms_matter": energy.rms_matter_no_com,
    }


def energy_row(energy, mode: str = "no-com") -> dict[str, float]:
    selected = reported_energy(energy, mode)
    return {
        "e_total": float(selected["e_total"]),
        "e_per_A": float(selected["e_per_A"]),
        "e_total_no_com": energy.e_total_no_com,
        "e_per_A_no_com": energy.e_per_A_no_com,
        "e_total_with_com": energy.e_total_with_com,
        "e_per_A_with_com": energy.e_per_A_with_com,
        "e_cm": energy.e_cm,
        "e_kin": energy.e_kin,
        "e_dir": energy.e_dir,
        "e_exc": energy.e_exc,
        "e_rearr": energy.e_rearr,
        "rms_n_no_com": energy.rms_n_no_com,
        "rms_p_no_com": energy.rms_p_no_com,
        "rms_matter_no_com": energy.rms_matter_no_com,
        "charge_radius_no_com": energy.charge_radius_no_com,
        "rms_n_with_com": energy.rms_n_with_com,
        "rms_p_with_com": energy.rms_p_with_com,
        "rms_matter_with_com": energy.rms_matter_with_com,
        "charge_radius_with_com": energy.charge_radius_with_com,
    }


def inspect_core(args: argparse.Namespace) -> None:
    core = RHFCore(args.core_dir)
    print(f"Core directory: {core.core_dir}")
    print("Available models:")
    for name, idx in sorted(MODEL_INDEX.items(), key=lambda item: (item[1], item[0])):
        tag = "RMF v1" if idx in {4, 5, 6, 7, 8} else "RHF v2"
        print(f"  {idx}: {name:<7} {tag}")


def scf_baseline(args: argparse.Namespace) -> None:
    out = ensure_out(args.out)
    core = RHFCore(args.core_dir)
    core.init(args.model, args.z, args.n, args.a)

    rows = []
    converged = False
    for epoch in range(1, args.max_iter + 1):
        si = core.step()
        energy = core.energy()
        rows.append(
            {
                "iter": epoch,
                "si": si,
                **energy_row(energy, args.energy_mode),
            }
        )
        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"{epoch:5d}  si={si:.6e}  "
                f"E/A(no CoM)={energy.e_per_A_no_com:+.6f}  "
                f"E/A(with CoM)={energy.e_per_A_with_com:+.6f}"
            )
        if abs(si) <= args.tol:
            converged = True
            break

    with (out / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_core_arrays(core, out)
    final_energy = core.energy()
    write_json(out / "observables.json", final_energy.__dict__)
    write_json(
        out / "summary.json",
        {
            "mode": "scf-baseline",
            "model": args.model,
            "z": args.z,
            "n": args.n,
            "a": args.a,
            "iterations": len(rows),
            "converged": converged,
            "tol": args.tol,
            "reported_energy_mode": args.energy_mode,
            "energy_convention": "energy contains both no_com and with_com; reported_energy follows --energy-mode",
            "reported_energy": reported_energy(final_energy, args.energy_mode),
            "energy": final_energy.__dict__,
        },
    )
    print(f"wrote {out}")


def dpl_run(args: argparse.Namespace) -> None:
    require_rmf_model(args.model)
    torch = import_torch()
    from dpl_model import PotentialNet, RMFPhysicsLoss

    out = ensure_out(args.out)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    print(f"[setup] torch device: {device}")

    core = RHFCore(args.core_dir)
    core.init(args.model, args.z, args.n, args.a)
    base = core.local_stack()

    model = PotentialNet(
        core.r,
        base,
        z=args.z,
        n=args.n,
        hidden=args.hidden,
        max_delta=args.max_delta,
    ).to(device)
    physics_loss = RMFPhysicsLoss(
        base,
        smooth_weight=args.smooth_weight,
        boundary_weight=args.boundary_weight,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    rows = []
    best = {"loss": float("inf"), "epoch": 0, "state": None}
    converged = False

    for epoch in range(1, args.epochs + 1):
        pred = model()
        pred_np = pred.detach().cpu().numpy()

        core.set_local_stack(pred_np)
        core.solve_fixed_potential()
        core.rebuild_rmf_potentials()
        target_np = core.local_stack()
        target = torch.as_tensor(target_np, dtype=torch.float64, device=device)

        components = physics_loss(pred, target)
        loss = components["loss"]

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        residual = float(components["residual"].detach())
        energy = core.energy()
        rows.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach()),
                "residual": residual,
                "fixed_point_loss": float(components["fixed_point_loss"].detach()),
                "smooth_loss": float(components["smooth_loss"].detach()),
                "boundary_loss": float(components["boundary_loss"].detach()),
                **energy_row(energy, args.energy_mode),
            }
        )
        if float(loss.detach()) < best["loss"]:
            best = {
                "loss": float(loss.detach()),
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"{epoch:5d}  residual={residual:.6e}  loss={float(loss.detach()):.6e}  "
                f"E/A(no CoM)={energy.e_per_A_no_com:+.6f}  "
                f"E/A(with CoM)={energy.e_per_A_with_com:+.6f}"
            )
        if residual <= args.tol:
            converged = True
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    final_pred = model().detach().cpu().numpy()
    core.set_local_stack(final_pred)
    core.solve_fixed_potential()
    core.rebuild_rmf_potentials()

    with (out / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "physics_loss_state": {k: v.detach().cpu() for k, v in physics_loss.state_dict().items()},
            "physics_loss": {
                "type": "RMFPhysicsLoss",
                "smooth_weight": args.smooth_weight,
                "boundary_weight": args.boundary_weight,
                "residual": "sqrt(mean(((P_theta - P_RMF[rho(P_theta)]) / scale)^2))",
            },
            "model": args.model,
            "z": args.z,
            "n": args.n,
            "a": args.a,
            "r": core.r,
            "base_potentials": base,
        },
        out / "model.pt",
    )
    np.savez(out / "dpl_prediction.npz", r=core.r, vps_n=final_pred[0], vms_n=final_pred[1],
             vps_p=final_pred[2], vms_p=final_pred[3])
    save_core_arrays(core, out)
    final_energy = core.energy()
    write_json(out / "observables.json", final_energy.__dict__)
    write_json(
        out / "summary.json",
        {
            "mode": "dpl-rmf",
            "model": args.model,
            "z": args.z,
            "n": args.n,
            "a": args.a,
            "epochs": len(rows),
            "best_epoch": best["epoch"],
            "best_loss": best["loss"],
            "converged": converged,
            "tol": args.tol,
            "reported_energy_mode": args.energy_mode,
            "energy_convention": "energy contains both no_com and with_com; reported_energy follows --energy-mode",
            "reported_energy": reported_energy(final_energy, args.energy_mode),
            "energy": final_energy.__dict__,
        },
    )
    print(f"wrote {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DPL-RMF driver for Core-1204")
    parser.add_argument("--core-dir", default="../Core-1204", help="path to Core-1204")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_cmd = sub.add_parser("inspect-core", help="show available Core models")
    inspect_cmd.set_defaults(func=inspect_core)

    baseline = sub.add_parser("scf-baseline", help="run the existing Core RMF SCF loop from Python")
    add_case_args(baseline)
    baseline.add_argument("--max-iter", type=int, default=200)
    baseline.add_argument("--tol", type=float, default=1.0e-5)
    baseline.add_argument("--print-every", type=int, default=10)
    baseline.add_argument("--out", default="outputs/scf_baseline")
    baseline.set_defaults(func=scf_baseline)

    run = sub.add_parser("run", help="run DPL replacement for the RMF outer iteration")
    add_case_args(run)
    run.add_argument("--epochs", type=int, default=500)
    run.add_argument("--tol", type=float, default=1.0e-4)
    run.add_argument("--lr", type=float, default=2.0e-3)
    run.add_argument("--hidden", type=int, default=64)
    run.add_argument("--max-delta", type=float, default=0.75)
    run.add_argument("--smooth-weight", type=float, default=1.0e-3)
    run.add_argument("--boundary-weight", type=float, default=1.0e-2)
    run.add_argument("--weight-decay", type=float, default=1.0e-6)
    run.add_argument("--grad-clip", type=float, default=10.0)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    run.add_argument("--print-every", type=int, default=10)
    run.add_argument("--out", default="outputs/dpl_rmf")
    run.set_defaults(func=dpl_run)

    return parser


def add_case_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="DD-ME2", help="RMF model: DD-ME1, DD-ME2, PKDD, TW99, DD-LZ1")
    parser.add_argument("--z", type=int, default=8, help="proton number")
    parser.add_argument("--n", type=int, default=8, help="neutron number")
    parser.add_argument("--a", type=int, default=None, help="mass number; defaults to Z+N")
    parser.add_argument(
        "--energy-mode",
        default="no-com",
        choices=["no-com", "with-com"],
        help="which convention to use for reported e_total/e_per_A; both are always saved",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    try:
        args.func(args)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
