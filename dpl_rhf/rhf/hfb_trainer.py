"""Training loop of the spherical no-refresh neural RHFB solver."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import copy
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch

from dpl_rhf import __release_name__, __version__
from dpl_rhf.io.outputs import ensure_out, write_history
from dpl_rhf.models.canonical_hfb_net import CanonicalHFBNet
from dpl_rhf.rhf.hfb_backend import SphericalRHFBBackend, SphericalRHFBResult
from dpl_rhf.training.inverse_hamiltonian import inverse_hamiltonian_constraint


@dataclass(frozen=True)
class RHFBTrainingConfig:
    epochs: int = 600
    hidden: int = 64
    depth: int = 3
    radial_basis_size: int = 16
    correction_scale: float = 1.0e-2
    learning_rate: float = 1.0e-3
    logit_learning_rate: float = 2.0e-2
    energy_weight: float = 2.0
    # 0 minimizes the energy only; the HFB residual is then a convergence
    # diagnostic and no second-order (create_graph) Fock graph is built.
    # None calibrates a weight from the initial gradient norms.
    stationarity_weight: float | None = 0.0
    inverse_weight: float = 10.0
    branch_floor: float = 0.5
    grad_clip: float = 10.0
    print_every: int = 10
    seed: int = 20261026
    checkpoint_every: int = 10
    resume: bool = False
    graph_offload: str = "none"
    # Optional warm-up only: the inverse-Hamiltonian alignment of the canonical
    # span (exact for RHF, approximate for HFB) is annealed linearly to zero
    # over ``alignment_epochs`` so the final state is the unbiased minimum.
    alignment_weight: float = 0.0
    alignment_epochs: int = 0
    # Warm start from a converged state with the same canonical blocks (e.g. a
    # neighbouring isotope); the Fermi shift restores the particle numbers.
    initial_model: str | None = None


def build_network(backend: SphericalRHFBBackend, config: RHFBTrainingConfig) -> CanonicalHFBNet:
    return CanonicalHFBNet(
        torch.as_tensor(backend.r, dtype=torch.float64, device=backend.device),
        backend.block_specs,
        backend.initial_logits(),
        hidden=config.hidden,
        depth=config.depth,
        radial_basis_size=config.radial_basis_size,
        correction_scale=config.correction_scale,
    ).to(backend.device)


def _evaluate(backend, network, projectors, *, create_graph):
    return backend.evaluate_hfb(network(projectors), network.logits(), create_graph=create_graph)


def _self_consistent_evaluation(backend, network, projectors, *, max_passes=40,
                                tolerance=1e-9):
    """Evaluate with a no-sea projector consistent with the state's own field.

    The projector P[h] and the pair signs are iterated with the network state
    fixed until max|P_new - P_old| < tolerance and no sign changes.
    """
    change, passes = float("inf"), 0
    for passes in range(1, max_passes + 1):
        result = _evaluate(backend, network, projectors, create_graph=False)
        updated = backend.positive_projectors(
            {k: v.detach() for k, v in result.rhf.hamiltonians.items()})
        change = max(float((new - old).abs().max()) for new, old in zip(updated, projectors))
        flips = backend.update_pair_signs(result)
        projectors, result = updated, None
        if backend.device.type == "cuda":
            torch.cuda.empty_cache()
        if change < tolerance and flips == 0:
            break
    final = _evaluate(backend, network, projectors, create_graph=False)
    return final, projectors, {"passes": passes, "projector_max_change": change,
                               "converged": change < tolerance}


def _sea_leakage(backend, result) -> float:
    """max_b ||(1 - P[h_b]) Q_b||_F / ||Q_b||_F with h from the state itself."""
    projectors = backend.positive_projectors(
        {k: v.detach() for k, v in result.rhf.hamiltonians.items()})
    worst = 0.0
    for block, projector in zip(result.rhf.blocks, projectors):
        q = block["factors"].detach()
        worst = max(worst, float(torch.linalg.matrix_norm(q - projector @ q)
                                 / torch.linalg.matrix_norm(q)))
    return worst


def _inverse(backend, result, config, alignment=0.0):
    return inverse_hamiltonian_constraint(
        result.rhf.blocks, result.rhf.hamiltonians, backend.shifts,
        branch_floor=config.branch_floor,
        # Canonical HFB orbitals do not span an invariant subspace of h, so the
        # RHF alignment term is used at most as an annealed warm-up; the no-sea
        # projector enforces the branch and the barrier only monitors it.
        alignment_weight=alignment, inverse_variation_weight=0.0,
    )


def save_result(out: Path, backend: SphericalRHFBBackend, network: CanonicalHFBNet,
                result: SphericalRHFBResult, config: RHFBTrainingConfig,
                history: list[dict], inverse, projector_info: dict | None = None) -> dict:
    rhf = result.rhf
    a = backend.case.a
    terms = {name: float(value.detach().cpu()) for name, value in rhf.action_terms.items()}
    terms["E_pair_n"] = result.diagnostics["pairing_energy_n_mev"]
    terms["E_pair_p"] = result.diagnostics["pairing_energy_p_mev"]
    terms["E_pair"] = terms["E_pair_n"] + terms["E_pair_p"]
    terms["E_total_rhfb"] = result.energy_total
    summary = {
        "release": __release_name__, "version": __version__,
        "solver": "spherical no-refresh neural RHFB (PKA1 + D1S J=0 pairing)",
        "nucleus": {"z": backend.case.z, "n": backend.case.n, "a": a, "model": backend.case.model},
        "pairing": backend.pairing_enabled,
        "blocking": backend.blocking,
        "canonical_space": {"extra_capacity": backend.extra_capacity,
                            "capacities": backend.capacities,
                            "kappa_limits": list(backend.kappa_limits),
                            "blocks": [
                                {"species": "n" if s.species_sign > 0 else "p", "kappa": s.kappa,
                                 "principals": list(s.principals)} for s in backend.block_specs]},
        "mesh": {"r_max_fm": float(backend.r[-1]), "h_fm": float(backend.functional.h),
                 "points": int(len(backend.r))},
        "energy_total_no_com_mev": result.energy_total,
        "energy_per_a_no_com_mev": result.energy_total / a,
        "energy_terms_mev": terms,
        "chemical_potentials_mev": result.chemical_potentials,
        "average_canonical_gaps_mev": result.average_gaps,
        "particle_numbers": result.particle_numbers,
        "radii_fm": {key: result.diagnostics[key] for key in (
            "rms_n_no_com", "rms_p_no_com", "rms_matter_no_com", "charge_radius_no_com")},
        "stationarity": {
            "orbital_residual_rms_mev": result.orbital_residual_rms,
            "occupation_residual_rms_mev": result.occupation_residual_rms,
        },
        "no_sea": {
            "minimum_branch_score": float(inverse.minimum_branch_score.detach().cpu()),
            "mean_branch_score": float(inverse.mean_branch_score.detach().cpu()),
            "sign_iterations": backend.sign_iterations,
            "sea_leakage_self_consistent": _sea_leakage(backend, result),
            "projector_iteration": projector_info,
        },
        "pair_sign_flips": backend.pair_sign_flips,
        "counts": {"eigensolve_count": 0, "refresh_count": 0, "training_dirac_solves": 0,
                   "fortran_used": False},
        "canonical_levels": result.canonical,
        "diagnostics": result.diagnostics,
        "config": asdict(config),
        "epochs_completed": len(history) - 1 if history else 0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    densities = {name: value.detach().cpu().numpy() for name, value in rhf.densities.items()
                 if isinstance(value, torch.Tensor)}
    pairing_local = {}
    state = backend.occupation_state(network.logits())
    for block, pair in zip(rhf.blocks, state.pair_amplitudes):
        q = block["factors"].detach()
        split = q.shape[0] // 2
        mu = float(block["degeneracy"])
        diag = (q.square() * pair.detach()[None, :]).sum(dim=1) * mu / backend.functional.h
        name = "n" if float(block["species_sign"]) > 0 else "p"
        key = f"kappa_local_{name}"
        value = (diag[:split] + diag[split:]).cpu().numpy()
        pairing_local[key] = pairing_local.get(key, 0.0) + value
    # Reference-program definitions: Cooper pairs = int sum_blocks
    # (|kappa_GG(r,r)| + |kappa_FF(r,r)|) dr and gap = 2|E_pair| / pairs.
    from dpl_rhf.rhf.quadrature import simpson_weights
    weights = simpson_weights(len(backend.r), float(backend.functional.h),
                              device=torch.device("cpu"), dtype=torch.float64)[1:-1]
    cooper = {"n": 0.0, "p": 0.0}
    for block, pair in zip(rhf.blocks, state.pair_amplitudes):
        q = block["factors"].detach().cpu()
        split = q.shape[0] // 2
        mu = float(block["degeneracy"])
        local = (q.square() * pair.detach().cpu()[None, :]).sum(dim=1) * mu / backend.functional.h
        name = "n" if float(block["species_sign"]) > 0 else "p"
        cooper[name] += float(torch.sum(weights * (local[:split].abs() + local[split:].abs())))
    summary["cooper_pairs_reference_definition"] = cooper
    summary["pairing_gap_reference_definition_mev"] = {
        name: (2.0 * abs(terms[f"E_pair_{name}"]) / cooper[name] if cooper[name] > 1e-12 else 0.0)
        for name in ("n", "p")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(out / "profiles.npz", r=backend.r, **densities, **pairing_local)
    torch.save({"model_state": network.state_dict(), "config": asdict(config),
                "pair_signs": [p.cpu() for p in backend.pair_signs]},
               out / "model_final.pt")
    write_history(out / "history.csv", history)
    return summary


def train_rhfb(backend: SphericalRHFBBackend, config: RHFBTrainingConfig, out_dir):
    torch.manual_seed(config.seed)
    out = ensure_out(out_dir)
    lock_file = (out / ".training.lock").open("w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another RHFB training process is writing {out}") from exc
    network = build_network(backend, config)
    if config.initial_model and not config.resume:
        warm = torch.load(config.initial_model, map_location=backend.device, weights_only=False)
        network.load_state_dict(warm["model_state"])
    logit_ids = {id(p) for p in network.occupation_logits}
    optimizer = torch.optim.Adam([
        {"params": [p for p in network.parameters() if id(p) not in logit_ids],
         "lr": config.learning_rate},
        {"params": list(network.occupation_logits), "lr": config.logit_learning_rate},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50, threshold=1.0e-3)
    history: list[dict] = []
    best_monitor = float("inf")
    best_state = copy.deepcopy(network.state_dict())
    stationarity_reference = energy_reference = None
    weight_s = config.stationarity_weight
    start_epoch = 0
    checkpoint = out / "training_checkpoint.pt"
    # "recompute" regenerates the exact Fock activations during backward
    # (checkpointing), trading time for device memory.
    backend.functional.recompute_fock = config.graph_offload == "recompute"

    # The first no-sea projector comes from the free Dirac blocks, then from the
    # self-consistent field of the initial state; both need matrix products only.
    with torch.no_grad():
        projectors = backend.positive_projectors(backend.kinetic_hamiltonians())
    if config.resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=backend.device, weights_only=False)
        network.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        history = saved["history"]
        best_monitor, best_state = saved["best_monitor"], saved["best_state"]
        stationarity_reference = saved["stationarity_reference"]
        energy_reference = saved["energy_reference"]
        weight_s = saved["stationarity_weight"]
        start_epoch = int(saved["next_epoch"])
        projectors = [p.to(backend.device) for p in saved["projectors"]]
        backend.pair_signs = [p.to(backend.device) for p in saved["pair_signs"]]
        print(f"resuming RHFB training at epoch {start_epoch}", flush=True)
    else:
        initial = _evaluate(backend, network, projectors, create_graph=False)
        projectors = backend.positive_projectors(
            {k: v.detach() for k, v in initial.rhf.hamiltonians.items()})
        backend.update_pair_signs(initial)
        del initial

    start = time.time()
    a = backend.case.a
    monitor = float("inf")
    for epoch in range(start_epoch, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        offload = (torch.autograd.graph.save_on_cpu(pin_memory=False)
                   if config.graph_offload == "cpu" and backend.device.type == "cuda"
                   else nullcontext())
        with offload:
            result = _evaluate(backend, network, projectors,
                               create_graph=epoch < config.epochs and weight_s != 0.0)
            alignment = 0.0
            if config.alignment_epochs > 0:
                alignment = config.alignment_weight * max(
                    0.0, 1.0 - epoch / config.alignment_epochs)
            inverse = _inverse(backend, result, config, alignment)
            energy_per_a = result.energy_total_tensor / a
            stationarity_square = result.stationarity_square_tensor
        if weight_s is None:
            parameters = [p for p in network.parameters() if p.requires_grad]
            g_e = torch.autograd.grad(energy_per_a, parameters, retain_graph=True, allow_unused=True)
            g_s = torch.autograd.grad(stationarity_square, parameters, retain_graph=True,
                                      allow_unused=True)
            norm_e = torch.sqrt(sum(torch.sum(g.square()) for g in g_e if g is not None))
            norm_s = torch.sqrt(sum(torch.sum(g.square()) for g in g_s if g is not None))
            weight_s = float((config.energy_weight * norm_e / norm_s.clamp_min(1e-14)).detach().cpu())
        loss = config.energy_weight * energy_per_a + config.inverse_weight * inverse.loss
        if weight_s != 0.0:
            loss = loss + weight_s * stationarity_square
        stationarity = float(torch.sqrt(stationarity_square).detach().cpu())
        if stationarity_reference is None:
            stationarity_reference = max(stationarity, 1e-14)
        if energy_reference is None:
            energy_reference = result.energy_total / a
        branch_violation = max(0.0, config.branch_floor
                               - float(inverse.minimum_branch_score.detach().cpu()))
        if weight_s == 0.0:
            # Exact constraints: the lowest energy is the best variational state.
            monitor = result.energy_total / a + 1.0e3 * branch_violation ** 2
        else:
            monitor = (stationarity / stationarity_reference
                       + config.energy_weight * (result.energy_total / a - energy_reference)
                       + 1.0e3 * branch_violation ** 2)
        row = {
            "epoch": epoch,
            "loss": float(loss.detach().cpu()),
            "energy_total_no_com_mev": result.energy_total,
            "energy_per_a_no_com_mev": result.energy_total / a,
            "energy_rhf_part_mev": result.rhf.energy_total_no_com,
            "pairing_energy_n_mev": result.diagnostics["pairing_energy_n_mev"],
            "pairing_energy_p_mev": result.diagnostics["pairing_energy_p_mev"],
            "lambda_n_mev": result.chemical_potentials["n"],
            "lambda_p_mev": result.chemical_potentials["p"],
            "gap_n_mev": result.average_gaps["n"],
            "gap_p_mev": result.average_gaps["p"],
            "number_n": result.particle_numbers["n"],
            "number_p": result.particle_numbers["p"],
            "stationarity_rms_mev": stationarity,
            "orbital_residual_rms_mev": result.orbital_residual_rms,
            "occupation_residual_rms_mev": result.occupation_residual_rms,
            "minimum_branch_score": float(inverse.minimum_branch_score.detach().cpu()),
            "alignment_weight": alignment,
            "inverse_tangent_residual": float(inverse.tangent_residual.detach().cpu()),
            "rms_matter_no_com_fm": result.diagnostics["rms_matter_no_com"],
            "charge_radius_no_com_fm": result.diagnostics["charge_radius_no_com"],
            "stationarity_weight": weight_s,
            "selection_objective": monitor,
            "eigensolve_count": 0,
            "refresh_count": 0,
            "sign_iterations": backend.sign_iterations,
            "pair_sign_flips": backend.pair_sign_flips,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.time() - start,
        }
        history.append(row)
        if np.isfinite(monitor) and monitor < best_monitor:
            best_monitor = monitor
            best_state = copy.deepcopy(network.state_dict())
        if epoch % max(config.print_every, 1) == 0 or epoch == config.epochs:
            print(f"epoch={epoch:5d} E/A={row['energy_per_a_no_com_mev']: .8f} "
                  f"Epair=({row['pairing_energy_n_mev']:.4f},{row['pairing_energy_p_mev']:.4f}) "
                  f"lam=({row['lambda_n_mev']:.3f},{row['lambda_p_mev']:.3f}) "
                  f"S={stationarity:.3e} branch={row['minimum_branch_score']:.3f}", flush=True)
            write_history(out / "history.csv", history)
        if epoch == config.epochs:
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), config.grad_clip)
        optimizer.step()
        projectors = backend.positive_projectors(
            {k: v.detach() for k, v in result.rhf.hamiltonians.items()})
        backend.update_pair_signs(result)
        del result, inverse, loss
        scheduler.step(monitor)
        if config.checkpoint_every > 0 and (epoch + 1) % config.checkpoint_every == 0:
            payload = {
                "next_epoch": epoch + 1, "model_state": network.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(), "history": history,
                "best_monitor": best_monitor, "best_state": best_state,
                "stationarity_reference": stationarity_reference,
                "energy_reference": energy_reference, "stationarity_weight": weight_s,
                "projectors": [p.cpu() for p in projectors],
                "pair_signs": [p.cpu() for p in backend.pair_signs],
            }
            temporary = checkpoint.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(checkpoint)

    # Release the last training graph before the two final evaluations.
    result = inverse = loss = None
    if backend.device.type == "cuda":
        torch.cuda.empty_cache()
    network.load_state_dict(best_state)
    final, projectors, info = _self_consistent_evaluation(backend, network, projectors)
    inverse = _inverse(backend, final, config)
    summary = save_result(out, backend, network, final, config, history, inverse, info)
    return network, final, summary


def evaluate_saved(backend: SphericalRHFBBackend, config: RHFBTrainingConfig, model_path, out_dir,
                   *, state: str = "best"):
    """Re-evaluate a saved state with the current code (no optimization)."""
    out = ensure_out(out_dir)
    network = build_network(backend, config)
    saved = torch.load(model_path, map_location=backend.device, weights_only=False)
    # Accept model_final.pt or a training checkpoint (its selected best state).
    if state not in {"best", "last"}:
        raise ValueError("state must be 'best' or 'last'")
    key = "best_state" if state == "best" and "best_state" in saved else "model_state"
    network.load_state_dict(saved[key])
    if "pair_signs" in saved:
        backend.pair_signs = [p.to(backend.device) for p in saved["pair_signs"]]
    if "projectors" in saved:
        projectors = [p.to(backend.device) for p in saved["projectors"]]
    else:
        with torch.no_grad():
            projectors = backend.positive_projectors(backend.kinetic_hamiltonians())
    result, projectors, info = _self_consistent_evaluation(backend, network, projectors)
    info["evaluated_state"] = key
    inverse = _inverse(backend, result, config)
    history = []
    previous = Path(model_path).with_name("history.csv")
    if not previous.exists():
        previous = Path(out_dir) / "history.csv"
    if previous.exists():
        import csv
        history = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(previous.open())]
    return save_result(out, backend, network, result, config, history, inverse, info)
