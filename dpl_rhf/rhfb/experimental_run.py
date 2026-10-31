"""Uncertified neural O-17 experiment for inspecting implementation failures.

This is deliberately separate from the certified training entry point.  It
evaluates the implemented Cartesian PKA1 EDF, and does NOT certify the model,
the sea branch, or convergence.  A free-Dirac sea diagnostic is recorded but
does not replace the missing self-consistent no-sea constraint.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .basis import AngularRadialBasis
from .config import BlockedRHFBConfig
from .functional import evaluate_rhfb_energy, functional_readiness
from .manifold import generalized_density_matrix
from .parity_diagnostic import number_parity_from_density
from .operators import DensityFactors
from .result import append_jsonl, atomic_json, save_observables
from .trainer import (_TrainingEigenCounter, _blocking_spec, prepare_networks,
                      update_blocked_mode_tracker)


def free_sea_blocks(basis, species, device):
    """Compute free-Dirac negative projectors without extracting eigenvectors.

    A=h_free+M has A²>=M² by its off-diagonal Dirac form.  Polynomial sign
    iteration converges on that gapped spectrum.  Its idempotence and
    Hermiticity errors are saved; these are free-field diagnostics only.
    """
    result = {}
    mass = basis.mass_n_mev if species == "n" else basis.mass_p_mev
    for channel in basis.channels:
        if channel.kappa in result:
            continue
        h = basis._radial_operator(channel.kappa, device).to(torch.complex128).clone()
        nr = basis.n_radial
        h[nr:, nr:] = -2 * mass * torch.eye(nr, dtype=h.dtype, device=device)
        identity = torch.eye(2 * nr, dtype=h.dtype, device=device)
        a = h + mass * identity
        sign = a / torch.linalg.matrix_norm(a)
        for _ in range(40):
            sign = 0.5 * sign @ (3 * identity - sign @ sign)
            if float(torch.linalg.matrix_norm(sign @ sign - identity)) < 1e-12:
                break
        minus = 0.5 * (identity - sign)
        result[channel.kappa] = minus
    residual = max(float(torch.linalg.matrix_norm(p @ p - p))
                   for p in result.values())
    if residual > 1e-10:
        raise RuntimeError(f"free sign diagnostic did not converge: {residual}")
    return result, residual


def free_sea_support(basis, state, blocks):
    minus_q = torch.empty_like(state.q)
    for channel in basis.channels:
        start, stop = channel.offset, channel.offset + 2 * basis.n_radial
        minus_q[start:stop] = blocks[channel.kappa] @ state.q[start:stop]
    number = state.particle_number()
    sea_number = torch.trace(state.rho_active @ (minus_q.mH @ minus_q)).real
    chi = state.chi_active
    chi_norm2 = torch.sum(chi.abs().square())
    # ||P_- chi||_F² = Tr[chi† Q† P_- Q chi], with Q orthonormal.
    chi_sea2 = torch.sum((minus_q @ chi).abs().square())
    return {"free_dirac_sea_particle_fraction": float(sea_number / number),
            "free_dirac_sea_anomalous_fraction": (
                float(chi_sea2 / chi_norm2) if float(chi_norm2) > 1e-30 else 0.0)}


def run(output_dir, *, epochs=20, device="cpu", learning_rate=2e-5,
        radius_fm=8.0, spacing_fm=0.8, preserve_spatial_parity=False):
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(20261002)
    config = BlockedRHFBConfig(
        z=8, n=9, radius_fm=radius_fm, spacing_fm=spacing_fm, kappa_max=3,
        active_n=14, active_p=14, epochs=epochs, seed=20261002,
        chunk_size=128, pair_chunk=14, lmax=6,
    )
    basis = AngularRadialBasis(
        radius_fm=config.radius_fm, spacing_fm=config.spacing_fm,
        kappa_max=config.kappa_max, n_theta=7, n_phi=14, derivative_order=5,
    )
    from .spatial_parity import ParityConservingDensityNet, parity_adapted_density_diagnostic
    data, blocks = prepare_networks(config, basis, device=device,
        network_class=ParityConservingDensityNet if preserve_spatial_parity else None)
    modules = {s: d["network"] for s, d in data.items()}
    parameters = [p for net in modules.values() for p in net.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    counter = _TrainingEigenCounter(
        basis.dimension, {s: net.active_dimension for s, net in modules.items()})
    previous = {"n": None, "p": None}
    sources = sorted(Path(__file__).parent.glob("*.py"))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    snapshot = out / "source_snapshot" / "rhfb"
    snapshot.mkdir(parents=True)
    for source in sources:
        (snapshot / source.name).write_bytes(source.read_bytes())
    started = time.monotonic()
    metadata = {
        "status": "experimental_uncertified", "physical_result_accepted": False,
        "nucleus": "O17", "config": asdict(config),
        "backend": "Python/PyTorch "+type(modules['n']).__name__, "fortran_used": False,
        "network_spatial_symmetry": "conserved parity" if preserve_spatial_parity else "unrestricted",
        "device": str(device), "torch_version": torch.__version__,
        "derivative": "Zhang five-point asymmetric radial finite difference, fourth-order accuracy",
        "learning_rate": learning_rate, "number_penalty_mev": 200.0,
        "blocking": _blocking_spec(blocks),
        "primitive_dimension": basis.dimension, "spatial_points": basis.point_count,
        "functional_readiness_at_start": functional_readiness(),
        "source_sha256_at_start": hashes,
        "limitations": [
            "Full PKA1 implementation has not passed all independent physical closures.",
            "Self-consistent no-sea branch constraint is absent in this experiment.",
            "Free-Dirac sea support is a diagnostic, not an interacting-field certificate.",
            f"{radius_fm} fm box, {spacing_fm} fm radial spacing and 14 active orbitals are unconverged.",
            "Parameter gradient is not the full-space RHFB stationary residual.",
            "The energy below is an experimental functional value, not an accepted binding energy.",
        ],
    }
    atomic_json(out / "run_config.json", metadata)
    with torch.no_grad(), counter:
        projectors = {s: free_sea_blocks(basis, s, torch.device(device)) for s in modules}
    history = []
    updates = 0
    targets = {"n": 9, "p": 8}
    last_states = None
    last_network_state = None
    last_optimizer_state = None
    last_audit = None
    stop_reason = "requested_updates_completed"
    try:
        for step in range(epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            with counter:
                states, modes, structure = {}, {}, {}
                for s, net in modules.items():
                    q, gd, mode = net.forward_with_blocked_mode()
                    states[s] = DensityFactors(q, gd.rho, gd.chi)
                    modes[s] = mode
                    r = generalized_density_matrix(gd.rho, gd.chi)
                    parity = number_parity_from_density(
                        gd.rho.detach().cpu().numpy(), gd.chi.detach().cpu().numpy())
                    if parity["number_parity"] != net.reference_parity:
                        raise RuntimeError("independent density parity differs from blocked reference")
                    structure[s] = {
                        "orthogonality_max": float((q.mH @ q - torch.eye(
                            q.shape[1], dtype=q.dtype, device=q.device)).abs().max().detach()),
                        "generalized_idempotence_max": float((r @ r - r).abs().max().detach()),
                        "chi_antisymmetry_max": float((gd.chi + gd.chi.T).abs().max().detach()),
                        "fermion_parity": parity["number_parity"],
                        "reference_fermion_parity": net.reference_parity,
                        "independent_density_parity": parity,
                    }
                    if preserve_spatial_parity:
                        spatial = parity_adapted_density_diagnostic(basis, states[s])
                        if spatial['many_body_spatial_parity'] != 1:
                            raise RuntimeError('O17 positive spatial parity candidate changed parity')
                        structure[s]['independent_spatial_parity'] = spatial
                overlaps = update_blocked_mode_tracker(previous, modes)
                evaluation = evaluate_rhfb_energy(
                    basis, states, chunk_size=128, pair_chunk=14, lmax=6)
                numbers = {s: state.particle_number() for s, state in states.items()}
                penalty = 200 * sum((numbers[s] - targets[s]) ** 2 for s in states)
                loss = evaluation.total_energy + penalty
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite experiment loss")
                loss.backward()
                gradient_square = sum(float(torch.sum(p.grad.abs().square()))
                                      for p in parameters if p.grad is not None)
                neural_square = sum(float(torch.sum(p.grad.abs().square()))
                                    for net in modules.values() for name, p in net.named_parameters()
                                    if not name.startswith("raw_") and p.grad is not None)
                if not np.isfinite(gradient_square):
                    raise FloatingPointError("nonfinite parameter gradient")
                row = {
                    "step": step, "optimizer_updates_completed": updates,
                    "elapsed_seconds": time.monotonic() - started,
                    "experimental_total_energy_mev": evaluation.total_energy.detach(),
                    "energy_terms_mev": {k: v.detach() for k, v in evaluation.energy_terms.items()},
                    "loss_mev": loss.detach(), "number_penalty_mev": penalty.detach(),
                    "particle_numbers": {s: v.detach() for s, v in numbers.items()},
                    "particle_number_errors": {s: numbers[s].detach() - targets[s] for s in states},
                    "chi_frobenius": {s: float(torch.linalg.matrix_norm(
                        state.chi_active).detach()) for s, state in states.items()},
                    "parameter_gradient_l2": gradient_square ** 0.5,
                    "neural_layer_gradient_l2": neural_square ** 0.5,
                    "structure": structure, "adjacent_blocked_overlap": overlaps,
                    "sea_diagnostic": {s: free_sea_support(basis, state, projectors[s][0])
                                       for s, state in states.items()},
                    "solver_counts": dict(counter.counts),
                    "physical_result_accepted": False,
                }
                append_jsonl(out / "history.jsonl", row)
                history.append(row)
                print(f"step={step} E_exp={float(evaluation.total_energy.detach()):.9f} "
                      f"N={float(numbers['n'].detach()):.9f} Z={float(numbers['p'].detach()):.9f} "
                      f"grad={gradient_square**0.5:.5g}", flush=True)
                last_states = states
                last_audit = evaluation.audit
                last_network_state = {s: copy.deepcopy(net.state_dict())
                                      for s, net in modules.items()}
                last_optimizer_state = copy.deepcopy(optimizer.state_dict())
                if sum(counter.counts.values()):
                    raise RuntimeError("eigensolver detected in neural experiment")
                if step < epochs:
                    optimizer.step()
                    updates += 1
    except Exception as exc:
        stop_reason = f"{type(exc).__name__}: {exc}"
    # Keep the actual last evaluated network, not an unevaluated failed update.
    if last_network_state is not None:
        for s, net in modules.items():
            net.load_state_dict(last_network_state[s])
        optimizer.load_state_dict(last_optimizer_state)
        updates = history[-1]["optimizer_updates_completed"]
    torch.save({"format": "experimental_uncertified_network_only_v1",
                "physical_resume_compatible": False,
                "config": asdict(config), "network_state": {
                    s: net.state_dict() for s, net in modules.items()},
                "optimizer_state": optimizer.state_dict(),
                "network_spatial_symmetry": metadata['network_spatial_symmetry'],
                "updates_completed": updates, "stop_reason": stop_reason},
               out / "experimental_network.pt")
    observables = None
    if last_states is not None:
        with torch.no_grad():
            observables = save_observables(basis, last_states, out, targets=targets)
            np.savez_compressed(out / "density_factors.npz", **{
                f"{s}_{name}": value.detach().cpu().numpy()
                for s, state in last_states.items() for name, value in (
                    ("Q", state.q), ("rho_active", state.rho_active),
                    ("chi_active", state.chi_active))})
    summary = {
        **metadata, "stop_reason": stop_reason, "optimizer_updates_completed": updates,
        "elapsed_seconds": time.monotonic() - started,
        "initial": history[0] if history else None, "final": history[-1] if history else None,
        "observables": observables, "solver_counts": counter.counts,
        "functional_audit_last_evaluation": last_audit,
        "eigensolver_operations": counter.operations,
        "free_projector_idempotence": {s: projectors[s][1] for s in modules},
        "source_files_changed_during_run": [str(p) for p in sources
            if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p)]],
    }
    atomic_json(out / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--radius-fm", type=float, default=8.0)
    parser.add_argument("--spacing-fm", type=float, default=0.8)
    parser.add_argument("--preserve-spatial-parity", action="store_true",
                        help="add a positive spatial-parity O17 candidate; retain general m mixing")
    args = parser.parse_args()
    run(args.output, epochs=args.epochs, device=args.device, learning_rate=args.learning_rate,
        radius_fm=args.radius_fm, spacing_fm=args.spacing_fm,
        preserve_spatial_parity=args.preserve_spatial_parity)
