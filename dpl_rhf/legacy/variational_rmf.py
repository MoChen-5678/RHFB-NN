from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

from dpl_rmf_core import RHFCore, write_json
from rmf_functional import PKDDRMFFunctional
from variational_model import StrictSphericalRMFNet


PKDD_SIX = [
    ("O16", 8, 8, 16),
    ("Ca40", 20, 20, 40),
    ("Ca48", 20, 28, 48),
    ("Zr90", 40, 50, 90),
    ("Sn132", 50, 82, 132),
    ("Pb208", 82, 126, 208),
]


def import_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required") from exc
    return torch


def ensure_out(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_scf(model: str, z: int, n: int, a: int, out: Path, max_iter: int = 500, min_iter: int = 5) -> dict:
    core = RHFCore()
    core.init(model, z, n, a)
    converged = False
    rows = []
    for i in range(1, max_iter + 1):
        si = core.step()
        energy = core.energy()
        rows.append({"iter": i, "si": si, "e_per_A_no_com": energy.e_per_A_no_com,
                     "e_per_A_with_com": energy.e_per_A_with_com})
        if i >= min_iter and abs(si) <= 1.0e-5:
            converged = True
            break
    np.savez(out / "scf_wavefunctions.npz", **core.wavefunctions())
    np.savez(out / "scf_potentials.npz", r=core.r, **core.local_potentials(), **core.fields())
    np.savez(out / "scf_densities.npz", r=core.r, **core.densities())
    summary = {
        "model": model,
        "z": z,
        "n": n,
        "a": a,
        "iterations": len(rows),
        "converged": converged,
        "energy": core.energy().__dict__,
    }
    write_json(out / "scf_summary.json", summary)
    return summary


def evaluate_dpl_with_fortran(model: str, z: int, n: int, a: int, stack: np.ndarray, out: Path) -> dict:
    core = RHFCore()
    core.init(model, z, n, a)
    core.set_local_stack(stack)
    core.solve_fixed_potential()
    np.savez(out / "dpl_fortran_eval_wavefunctions.npz", **core.wavefunctions())
    summary = {
        "status": "ok",
        "interpretation": "non-self-consistent fixed-potential diagnostic; not the DPL variational total energy",
        "energy": core.energy().__dict__,
    }
    write_json(out / "dpl_fortran_eval_summary.json", summary)
    return summary


def compare_single_particle(out: Path) -> dict:
    scf = np.load(out / "scf_wavefunctions.npz")
    dpl = np.load(out / "dpl_fortran_eval_wavefunctions.npz")
    scf_map = {}
    for i in range(len(scf["energy"])):
        key = (str(scf["species"][i]), str(scf["name"][i]).strip(), int(scf["kappa"][i]))
        scf_map[key] = float(scf["energy"][i])

    rows = []
    diffs = []
    for i in range(len(dpl["energy"])):
        key = (str(dpl["species"][i]), str(dpl["name"][i]).strip(), int(dpl["kappa"][i]))
        if key not in scf_map:
            continue
        dpl_e = float(dpl["energy"][i])
        scf_e = scf_map[key]
        occ = float(dpl["occupancy"][i])
        diff = dpl_e - scf_e
        rows.append({
            "species": key[0],
            "name": key[1],
            "kappa": key[2],
            "occupancy": occ,
            "epsilon_dpl": dpl_e,
            "epsilon_scf": scf_e,
            "delta": diff,
        })
        if occ > 1.0e-8:
            diffs.append(diff)

    with (out / "single_particle_compare.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["species", "name", "kappa"])
        writer.writeheader()
        writer.writerows(rows)
    arr = np.asarray(diffs, dtype=np.float64)
    return {
        "occupied_count": int(arr.size),
        "occupied_mae": float(np.mean(np.abs(arr))) if arr.size else None,
        "occupied_rmse": float(np.sqrt(np.mean(arr * arr))) if arr.size else None,
    }


def tensor_to_numpy_dict(state: dict) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in state.items()}


SHELLS = [
    ("1s1/2", -1, 0, 1, 2), ("1p3/2", -2, 1, 1, 4), ("1p1/2", 1, 1, 1, 2),
    ("1d5/2", -3, 2, 1, 6), ("2s1/2", -1, 0, 2, 2), ("1d3/2", 2, 2, 1, 4),
    ("1f7/2", -4, 3, 1, 8), ("2p3/2", -2, 1, 2, 4), ("1f5/2", 3, 3, 1, 6),
    ("2p1/2", 1, 1, 2, 2), ("1g9/2", -5, 4, 1, 10), ("1g7/2", 4, 4, 1, 8),
    ("2d5/2", -3, 2, 2, 6), ("2d3/2", 2, 2, 2, 4), ("3s1/2", -1, 0, 3, 2),
    ("1h11/2", -6, 5, 1, 12), ("2f7/2", -4, 3, 2, 8), ("1h9/2", 5, 5, 1, 10),
    ("1i13/2", -7, 6, 1, 14), ("3p3/2", -2, 1, 3, 4), ("2f5/2", 3, 3, 2, 6),
    ("3p1/2", 1, 1, 3, 2),
]


def generalized_laguerre(order: int, alpha: float, x: np.ndarray) -> np.ndarray:
    if order == 0:
        return np.ones_like(x)
    previous = np.ones_like(x)
    current = 1.0 + alpha - x
    for k in range(2, order + 1):
        previous, current = current, ((2 * k - 1 + alpha - x) * current - (k - 1 + alpha) * previous) / k
    return current


def independent_orbital_data(z: int, n: int, r: np.ndarray) -> tuple[dict, dict]:
    """Nucleus-only HO orbitals and textbook Dirac Woods-Saxon fields."""
    records = []
    a = z + n
    hbar_omega = 41.0 * a ** (-1.0 / 3.0)
    oscillator_length = np.sqrt(197.328284**2 / (938.9 * hbar_omega))
    for species, number, sign in (("n", n, 1.0), ("p", z, -1.0)):
        remaining = number
        for name, kappa, angular, principal, degeneracy in SHELLS:
            if remaining <= 0:
                break
            particles = min(remaining, degeneracy)
            occupation = particles / degeneracy
            q = (r / oscillator_length) ** 2
            radial = generalized_laguerre(principal - 1, angular + 0.5, q)
            large = r ** (angular + 1) * np.exp(-0.5 * q) * radial
            large /= np.sqrt(np.trapezoid(large * large, r))
            records.append((species, sign, name, kappa, angular, principal, occupation, degeneracy, large))
            remaining -= particles
        if remaining:
            raise ValueError(f"shell table does not cover {species}={number}")

    # Nuclear Physics Practice, Eqs. (3.75)-(3.77), isoscalar part. This is an
    # analytic physics initial condition, not an SCF field or training label.
    v0, av = -71.28, 11.1175
    radius_plus = 0.5 * (1.2334 + 1.2496) * a ** (1.0 / 3.0)
    radius_minus = 0.5 * (1.1443 + 1.1400) * a ** (1.0 / 3.0)
    diffuseness_plus = 0.5 * (0.6150 + 0.6124)
    diffuseness_minus = 0.5 * (0.6476 + 0.6469)
    sigma_plus = v0 / (1.0 + np.exp((r - radius_plus) / diffuseness_plus))
    sigma_minus = -v0 * av / (1.0 + np.exp((r - radius_minus) / diffuseness_minus))
    scalar_potential = 0.5 * (sigma_plus - sigma_minus)
    vector_potential = 0.5 * (sigma_plus + sigma_minus)
    hbar_c, g_sigma, g_omega = 197.328284, 10.738508, 13.147623
    initial_sigma = scalar_potential / (hbar_c * g_sigma)
    initial_omega = vector_potential / (hbar_c * g_omega)
    charge_radius = 1.2496 * a ** (1.0 / 3.0)
    alpha = 1.0 / 137.03602
    initial_coul = np.where(
        r < charge_radius,
        alpha * z * (3.0 / (2.0 * charge_radius) - r * r / (2.0 * charge_radius**3)),
        alpha * z / r,
    )

    epsilon_initial = -0.5 * hbar_omega
    initial_large = np.stack([row[8] for row in records])
    initial_small = []
    for row, large in zip(records, initial_large):
        coulomb = initial_coul if row[0] == "p" else 0.0
        mass = 939.5731 if row[0] == "n" else 938.2796
        vms = sigma_minus / hbar_c + coulomb - 2.0 * mass / hbar_c
        derivative = np.gradient(large, r, edge_order=2)
        small = (derivative + row[3] * large / r) / (epsilon_initial / hbar_c - vms)
        norm = np.sqrt(np.trapezoid(large * large + small * small, r))
        large /= norm
        initial_small.append(small / norm)

    data = {
        "G": initial_large,
        "F": np.stack(initial_small),
        "energy": np.full(len(records), epsilon_initial),
        "kappa": np.asarray([row[3] for row in records]),
        "principal": np.asarray([row[5] for row in records]),
        "occupancy": np.asarray([row[6] for row in records]),
        "degeneracy": np.asarray([row[7] for row in records]),
        "species_sign": np.asarray([row[1] for row in records]),
        "sigma": initial_sigma, "omega": initial_omega,
        "rho": np.zeros_like(r), "coul": initial_coul,
    }
    metadata = {
        "species": np.asarray([row[0] for row in records]),
        "it": np.asarray([1 if row[0] == "n" else 2 for row in records]),
        "index": np.arange(1, len(records) + 1),
        "name": np.asarray([row[2] for row in records]),
        "l": np.asarray([row[4] for row in records]),
        "n": np.asarray([row[5] for row in records]),
    }
    return data, metadata


def optimize_case(args: argparse.Namespace, name: str | None = None) -> dict:
    if args.model.upper() != "PKDD":
        raise ValueError("variational v1 only supports PKDD")
    torch = import_torch()
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    out = ensure_out(Path(args.out) / name if name else args.out)
    torch.manual_seed(args.seed)
    r_np = np.concatenate(([0.001], np.arange(0.1, 20.0 + 0.05, 0.1)))
    r = torch.as_tensor(r_np, dtype=torch.float64, device=device)
    functional = PKDDRMFFunctional(r, args.z, args.n)
    orbital_data, orbital_metadata = independent_orbital_data(args.z, args.n, r_np)
    net = StrictSphericalRMFNet(r_np, args.z, args.n, orbital_data, hidden=args.hidden).to(device)
    parameters = list(net.parameters())
    adaptive_optimizer = (
        torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
        if args.optimizer == "adam"
        else None
    )
    vector_parameters = [
        parameter
        for name in ("omega", "rho", "coul")
        for parameter in net.field_nets[name].parameters()
    ]
    vector_parameter_ids = {id(parameter) for parameter in vector_parameters}

    def objectives(components: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        action = components["rmf_action_per_A"] / 50.0
        feasibility = (
            args.dirac_weight * components["dirac_residual_loss"]
            + args.field_weight * components["field_reconstruction_loss"]
            + args.orthogonality_weight * components["orthogonality_loss"]
            + args.boundary_weight * (components["orbital_boundary_loss"] + components["field_boundary_loss"])
        )
        return action, feasibility

    def saddle_gradients(action: torch.Tensor, feasibility: torch.Tensor) -> list[torch.Tensor]:
        action_gradients = torch.autograd.grad(action, parameters, retain_graph=True, allow_unused=True)
        feasibility_gradients = torch.autograd.grad(feasibility, parameters, allow_unused=True)
        result = []
        for parameter, action_gradient, feasibility_gradient in zip(parameters, action_gradients, feasibility_gradients):
            action_value = torch.zeros_like(parameter) if action_gradient is None else action_gradient
            feasibility_value = torch.zeros_like(parameter) if feasibility_gradient is None else feasibility_gradient
            if id(parameter) in vector_parameter_ids:
                action_value = -action_value
            result.append(action_value + feasibility_value)
        norm = torch.sqrt(sum(torch.sum(gradient.square()) for gradient in result))
        scale = torch.clamp(args.grad_clip / norm.clamp(min=1.0e-30), max=1.0)
        return [gradient * scale for gradient in result]
    best = {"loss": float("inf"), "epoch": 0, "state": None}
    rows = []
    total_epochs = args.outer_steps * args.inner_epochs
    for epoch in range(1, total_epochs + 1):
        orbital_state = net()
        comps, state, pot, orbital_state = functional.strict_loss(orbital_state, {})
        action, feasibility = objectives(comps)
        constrained_action = action + feasibility
        current_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        if adaptive_optimizer is not None:
            adaptive_optimizer.zero_grad(set_to_none=True)
            for parameter, gradient in zip(parameters, saddle_gradients(action, feasibility)):
                parameter.grad = gradient
            adaptive_optimizer.step()
        else:
            # Extragradient suppresses rotational instability near a saddle;
            # both evaluations use the same generalized RMF objective.
            base_parameters = [parameter.detach().clone() for parameter in parameters]
            first_gradients = saddle_gradients(action, feasibility)
            with torch.no_grad():
                for parameter, base, gradient in zip(parameters, base_parameters, first_gradients):
                    parameter.copy_(base * (1.0 - args.lr * args.weight_decay) - args.lr * gradient)

            lookahead_state = net()
            lookahead_comps, _, _, _ = functional.strict_loss(lookahead_state, {})
            lookahead_action, lookahead_feasibility = objectives(lookahead_comps)
            lookahead_gradients = saddle_gradients(lookahead_action, lookahead_feasibility)
            with torch.no_grad():
                for parameter, base, gradient in zip(parameters, base_parameters, lookahead_gradients):
                    parameter.copy_(base * (1.0 - args.lr * args.weight_decay) - args.lr * gradient)

        stationarity = (
            comps["dirac_residual_loss"]
            + comps["field_reconstruction_loss"]
            + comps["normalization_loss"]
            + comps["orthogonality_loss"]
        )
        loss_value = float(stationarity.detach().cpu())
        if loss_value < best["loss"]:
            best = {
                "loss": loss_value,
                "epoch": epoch,
                "state": current_state,
            }
        if epoch == 1 or epoch % args.print_every == 0 or epoch == total_epochs:
            row = {"epoch": epoch}
            for k, v in comps.items():
                row[k] = float(v.detach().cpu())
            row["stationarity"] = loss_value
            row["constrained_action"] = float(constrained_action.detach().cpu())
            rows.append(row)
            print(
                f"{name or 'case'} {epoch:5d}/{total_epochs} "
                f"stat={row['stationarity']:.4e} dirac={row['dirac_residual_loss']:.4e} "
                f"field={row['field_residual_loss']:.4e} recon={row['field_reconstruction_loss']:.4e} "
                f"S/A={row['rmf_action_per_A']:.3f} E/A={row['E_A_no_com_exact']:.3f} "
                f"N={row['n_number']:.3f} Z={row['z_number']:.3f}"
            )

    if best["state"] is not None:
        net.load_state_dict(best["state"])
    orbital_state = net()
    comps, state, pot, orbital_state = functional.strict_loss(orbital_state, {})
    state_np = tensor_to_numpy_dict(state)
    pot_np = tensor_to_numpy_dict(pot)
    orbital_np = tensor_to_numpy_dict(orbital_state)
    np.savez(out / "dpl_densities.npz", r=r_np, **state_np)
    np.savez(out / "dpl_potentials.npz", r=r_np, **pot_np)
    np.savez(out / "dpl_orbitals.npz", r=r_np, **orbital_np, **orbital_metadata)
    stack = np.stack([pot_np["vps_n"], pot_np["vms_n"], pot_np["vps_p"], pot_np["vms_p"]])
    np.savez(out / "dpl_local_stack.npz", r=r_np, stack=stack)
    with (out / "loss_history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    torch.save({"state_dict": net.state_dict(), "best_epoch": best["epoch"], "best_loss": best["loss"]}, out / "model.pt")

    obs = {k: float(v.detach().cpu()) for k, v in comps.items() if k != "loss"}
    summary = {
        "mode": "strict-variational-rmf",
        "constraint_mode": "strict-book-rmf",
        "physics_residual_embedded": True,
        "optimizer": f"{args.optimizer}-generalized-rmf-action",
        "energy_note": "No empirical energy target or Thomas-Fermi term is used; kinetic energy follows the occupied radial orbitals and book Eq. (3.35).",
        "initialization": "analytic-dirac-woods-saxon-no-scf-data",
        "generalized_physics_target": "rmf-action-plus-dirac-and-reconstructed-field-consistency",
        "model": "PKDD",
        "z": args.z,
        "n": args.n,
        "a": args.a,
        "epochs": total_epochs,
        "best_epoch": best["epoch"],
        "best_loss": best["loss"],
        "observables": obs,
    }
    write_json(out / "dpl_observables.json", summary)

    evaluate_dpl_with_fortran("PKDD", args.z, args.n, args.a, stack, out)
    scf_summary = run_scf("PKDD", args.z, args.n, args.a, out)
    metrics = compare_single_particle(out)
    compare = {
        "nucleus": name,
        "dpl": summary,
        "scf": scf_summary,
        "single_particle": metrics,
    }
    write_json(out / "compare_summary.json", compare)
    return compare


def batch_pkdd_six(args: argparse.Namespace) -> None:
    out = ensure_out(args.out)
    rows = []
    for name, z, n, a in PKDD_SIX:
        print(f"=== variational PKDD {name} ===")
        case_args = argparse.Namespace(**vars(args))
        case_args.z = z
        case_args.n = n
        case_args.a = a
        case_args.model = "PKDD"
        case_args.out = out
        compare = optimize_case(case_args, name=name)
        scf_e = compare["scf"]["energy"]
        dpl_obs = compare["dpl"]["observables"]
        sp = compare["single_particle"]
        rows.append({
            "nucleus": name,
            "best_loss": compare["dpl"]["best_loss"],
            "N_dpl": dpl_obs["n_number"],
            "Z_dpl": dpl_obs["z_number"],
            "Rch_dpl_no_com": dpl_obs["charge_radius_no_com"],
            "E_A_dpl_exact_no_com": dpl_obs["E_A_no_com_exact"],
            "E_A_scf_no_com": scf_e["e_per_A_no_com"],
            "E_A_scf_with_com": scf_e["e_per_A_with_com"],
            "dirac_residual_loss": dpl_obs["dirac_residual_loss"],
            "field_residual_loss": dpl_obs["field_residual_loss"],
            "normalization_loss": dpl_obs["normalization_loss"],
            "orthogonality_loss": dpl_obs["orthogonality_loss"],
            "sp_occ_mae": sp["occupied_mae"],
            "sp_occ_rmse": sp["occupied_rmse"],
        })
    with (out / "pkdd_six_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Variational PKDD RMF DPL with embedded physics residuals")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("optimize", "batch-pkdd-six"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--model", default="PKDD")
        cmd.add_argument("--z", type=int, default=8)
        cmd.add_argument("--n", type=int, default=8)
        cmd.add_argument("--a", type=int, default=16)
        cmd.add_argument("--outer-steps", type=int, default=20)
        cmd.add_argument("--inner-epochs", type=int, default=200)
        cmd.add_argument("--hidden", type=int, default=96)
        cmd.add_argument("--lr", type=float, default=1.0e-3)
        cmd.add_argument("--weight-decay", type=float, default=1.0e-6)
        cmd.add_argument("--grad-clip", type=float, default=10.0)
        cmd.add_argument("--optimizer", default="adam", choices=["adam", "extragradient"])
        cmd.add_argument("--dirac-weight", type=float, default=10.0)
        cmd.add_argument("--field-weight", type=float, default=10.0)
        cmd.add_argument("--variational-weight", type=float, default=1.0e-2)
        cmd.add_argument("--bound-spectrum-weight", type=float, default=1.0)
        cmd.add_argument("--normalization-weight", type=float, default=100.0)
        cmd.add_argument("--orthogonality-weight", type=float, default=100.0)
        cmd.add_argument("--boundary-weight", type=float, default=10.0)
        cmd.add_argument("--seed", type=int, default=42)
        cmd.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
        cmd.add_argument("--print-every", type=int, default=100)
        cmd.add_argument("--out", default="outputs/variational_rmf")
    sub.choices["optimize"].set_defaults(func=lambda args: optimize_case(args))
    sub.choices["batch-pkdd-six"].set_defaults(func=batch_pkdd_six)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
