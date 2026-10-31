"""Fixed-field neural RHFB stationarity prototype for finite validation cases.

This is an intentionally bounded validation stage.  It trains the existing
GeneralizedDensityNet on the primitive generalized-density commutator with
one immutable h/Delta pair.  It does not update or rebuild fields and it never
uses spectral information to initialize the network or define its loss.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .basis import AngularRadialBasis
from .manifold import GeneralizedDensity, blocking_overlap, generalized_density_matrix
from .network import GeneralizedDensityNet
from .parity_diagnostic import number_parity_from_density


@dataclass(frozen=True)
class FrozenFields:
    """Frozen one-body and pairing fields in the primitive basis (MeV)."""

    h: torch.Tensor
    delta: torch.Tensor
    particle_number: int
    species: str
    metadata: dict[str, Any]

    def validate(self, dimension: int) -> None:
        if self.h.shape != (dimension, dimension) or self.delta.shape != (dimension, dimension):
            raise ValueError("frozen h and Delta must both be D by D primitive matrices")
        if self.h.dtype != torch.complex128 or self.delta.dtype != torch.complex128:
            raise TypeError("frozen h and Delta must use complex128")
        if not torch.allclose(self.h.mH, self.h, atol=1e-9, rtol=0):
            raise ValueError("frozen h must be Hermitian")
        if not torch.allclose(self.delta.mT, -self.delta, atol=1e-9, rtol=0):
            raise ValueError("frozen Delta must be antisymmetric")
        if self.particle_number < 0 or self.particle_number > dimension:
            raise ValueError("particle number lies outside the primitive space")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transported_full_reference(
    seed_q: torch.Tensor,
    seed_rho: torch.Tensor,
    seed_chi: torch.Tensor,
    *,
    parity: int,
    blocked_index: int | None,
) -> tuple[torch.Tensor, GeneralizedDensity, torch.Tensor | None]:
    """Complete the seed frame once and transport its pure reference state."""
    d, m = seed_q.shape
    if d != m:
        raise ValueError("the prototype currently requires a complete D by D seed frame")
    if d > 96:
        raise ValueError("dense frozen-field prototype is restricted to D <= 96")
    if parity not in {-1, 1}:
        raise ValueError("seed parity must be even (+1) or odd (-1)")
    q_full, _ = torch.linalg.qr(seed_q, mode="complete")
    t = q_full.mH @ seed_q
    rho_ref = t @ seed_rho @ t.mH
    chi_ref = t @ seed_chi @ t.mT
    matrix = generalized_density_matrix(rho_ref, chi_ref)
    reference = GeneralizedDensity(rho_ref, chi_ref, matrix, parity)
    if not torch.allclose(matrix @ matrix, matrix, atol=2e-7, rtol=0):
        raise ValueError("transported seed reference lost idempotency")
    blocked = None
    if blocked_index is not None:
        if parity != -1 or not 0 <= blocked_index < d:
            raise ValueError("blocked index requires an odd reference and an in-range column")
        # The source mode is represented by a definite fully occupied column.
        if abs(float(seed_rho[blocked_index, blocked_index].real) - 1.0) > 1e-8:
            raise ValueError("blocked source column is not fully occupied")
        expected_row = torch.zeros(d, dtype=seed_rho.dtype, device=seed_rho.device)
        expected_row[blocked_index] = 1.0
        if torch.linalg.vector_norm(seed_rho[blocked_index, :] - expected_row) > 1e-8:
            raise ValueError("blocked source index is not an exact occupied coordinate")
        if torch.linalg.vector_norm(seed_chi[blocked_index, :]) > 1e-8:
            raise ValueError("blocked source row has nonzero anomalous density")
        # QR may only phase-rotate a square orthonormal seed frame.  Check that
        # it did not mix the marked orbital with any other coordinate.
        leakage = torch.abs(t[:, blocked_index]).clone()
        leakage[blocked_index] = 0
        if float(leakage.max()) > 2e-7:
            raise ValueError("QR completion mixed the blocked reference coordinate")
        if abs(float(rho_ref[blocked_index, blocked_index].real) - 1.0) > 1e-8:
            raise ValueError("blocked QR-completed coordinate is not fully occupied")
        # Its particle member is the occupied column of the transported Nambu frame.
        blocked = torch.cat((q_full[:, blocked_index], torch.zeros_like(q_full[:, blocked_index])))
    return q_full, reference, blocked


def _nambu_hamiltonian(fields: FrozenFields, mu: torch.Tensor) -> torch.Tensor:
    d = fields.h.shape[0]
    eye = torch.eye(d, dtype=torch.complex128, device=fields.h.device)
    upper = torch.cat((fields.h - mu * eye, fields.delta), dim=1)
    lower = torch.cat((-fields.delta.conj(), -fields.h.conj() + mu * eye), dim=1)
    return torch.cat((upper, lower), dim=0)


def _lift_density(q: torch.Tensor, gd: GeneralizedDensity) -> GeneralizedDensity:
    """Transport active canonical density to primitive coordinates."""
    rho = q @ gd.rho @ q.mH
    chi = q @ gd.chi @ q.mT
    return GeneralizedDensity(rho, chi, generalized_density_matrix(rho, chi),
                              gd.fermion_parity)


def _diagnostic(q: torch.Tensor, gd: GeneralizedDensity, fields: FrozenFields,
                mu: torch.Tensor) -> dict[str, float]:
    gd_primitive = _lift_density(q, gd)
    hbdg = _nambu_hamiltonian(fields, mu)
    residual = hbdg @ gd_primitive.matrix - gd_primitive.matrix @ hbdg
    number = gd_primitive.rho.diagonal().real.sum()
    return {
        "commutator_frobenius_mev": float(torch.linalg.vector_norm(residual).detach()),
        "commutator_rms_mev": float((torch.linalg.vector_norm(residual) / residual.numel() ** 0.5).detach()),
        "particle_number": float(number.detach()),
        "particle_number_error": float((number - fields.particle_number).detach()),
        "purity_error": float(torch.linalg.matrix_norm(gd_primitive.matrix @ gd_primitive.matrix - gd_primitive.matrix).detach()),
        "pair_antisymmetry_error": float(torch.linalg.matrix_norm(gd_primitive.chi.mT + gd_primitive.chi).detach()),
        "parity": float(gd.fermion_parity),
    }


def _residual_tensors(q: torch.Tensor, gd: GeneralizedDensity, fields: FrozenFields,
                      mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable primitive-space commutator and particle-number error."""
    gd_primitive = _lift_density(q, gd)
    hbdg = _nambu_hamiltonian(fields, mu)
    commutator = hbdg @ gd_primitive.matrix - gd_primitive.matrix @ hbdg
    number_error = gd_primitive.rho.diagonal().real.sum() - fields.particle_number
    return commutator, number_error


def _gradient_diagnostics(model: GeneralizedDensityNet, mu: torch.Tensor) -> dict[str, float]:
    grouped: dict[str, float] = {}
    total_square = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        norm = float(torch.linalg.vector_norm(parameter.grad.detach()))
        group = name.split(".", 1)[0]
        grouped[group] = grouped.get(group, 0.0) + norm * norm
        total_square += norm * norm
    grouped = {name: value ** 0.5 for name, value in grouped.items()}
    grouped["mu"] = float(torch.abs(mu.grad.detach())) if mu.grad is not None else 0.0
    grouped["all_parameters"] = (total_square + grouped["mu"] ** 2) ** 0.5
    return grouped


def _forbid_spectral_torch_calls():
    names = ("eig", "eigh", "eigvals", "eigvalsh")
    originals = {("torch", name): getattr(torch.linalg, name) for name in names}
    originals.update({("numpy", name): getattr(np.linalg, name) for name in names})
    attempts = {f"{module}.linalg.{name}": 0 for module, name in originals}
    def forbidden(module, name):
        def call(*args, **kwargs):
            attempts[f"{module}.linalg.{name}"] += 1
            raise RuntimeError(f"spectral routine {module}.linalg.{name} called during NN training")
        return call
    for name in names:
        setattr(torch.linalg, name, forbidden("torch", name))
        setattr(np.linalg, name, forbidden("numpy", name))
    return originals, attempts


def _restore_spectral_torch_calls(originals):
    for (module, name), function in originals.items():
        setattr(torch.linalg if module == "torch" else np.linalg, name, function)


def train_frozen_fields(
    *,
    basis: AngularRadialBasis,
    fields: FrozenFields,
    seed_q: torch.Tensor,
    seed_rho: torch.Tensor,
    seed_chi: torch.Tensor,
    labels: list[str] | tuple[str, ...],
    blocked_index: int,
    output_dir: str | Path,
    source_files: tuple[str | Path, ...] = (),
    max_iter: int = 120,
    learning_rate: float = 0.2,
    fixed_mu: float | None = None,
) -> dict[str, Any]:
    """Train one fixed h/Delta case and preserve complete diagnostic artifacts."""
    d = basis.dimension
    fields.validate(d)
    if d > 96:
        raise ValueError("dense frozen-field prototype is restricted to D <= 96")
    if len(labels) != d:
        raise ValueError("full primitive frame requires one label per seed column")
    if fields.particle_number % 2 != 1:
        raise ValueError("odd particle number is required for this validation stage")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    start_dir = out / "run_start_sources"
    start_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        start_dir / "training_inputs.npz",
        h=fields.h.detach().cpu().numpy(), delta=fields.delta.detach().cpu().numpy(),
        seed_q=seed_q.detach().cpu().numpy(), seed_rho=seed_rho.detach().cpu().numpy(),
        seed_chi=seed_chi.detach().cpu().numpy(),
    )
    hashes = {str(start_dir / "training_inputs.npz"): _sha256(start_dir / "training_inputs.npz")}
    for source in source_files:
        source = Path(source)
        destination = start_dir / source.name
        shutil.copy2(source, destination)
        hashes[str(source)] = _sha256(destination)

    q0, reference, initial_blocked = transported_full_reference(
        seed_q, seed_rho, seed_chi, parity=-1, blocked_index=blocked_index
    )
    initial_pfaffian = number_parity_from_density(
        reference.rho.detach().cpu().numpy(), reference.chi.detach().cpu().numpy()
    )
    if initial_pfaffian["number_parity"] != -1:
        raise ValueError("transported initial density failed independent odd-Pfaffian check")
    model = GeneralizedDensityNet(
        basis, q0, tuple(labels), reference, blocked_index=blocked_index,
        pairing_seed=0.0,
    ).to(device=fields.h.device)
    with torch.no_grad():
        q_initial, gd_initial = model()
        _, _, mode_initial = model.forward_with_blocked_mode()
    if initial_blocked is None or float(blocking_overlap(initial_blocked, mode_initial)) < 1.0 - 1e-8:
        raise ValueError("network's initial blocked mode does not match the occupied seed coordinate")
    # All model parameters remain trainable.  μ is a scalar Lagrange multiplier.
    mu_parameter = torch.nn.Parameter(torch.tensor(
        0.0 if fixed_mu is None else float(fixed_mu), dtype=torch.float64,
        device=fields.h.device,
    ), requires_grad=fixed_mu is None)
    parameters = list(model.parameters()) + ([mu_parameter] if fixed_mu is None else [])
    optimizer = torch.optim.LBFGS(
        parameters, lr=learning_rate, max_iter=1, history_size=20,
        line_search_fn="strong_wolfe", tolerance_grad=1e-10,
        tolerance_change=1e-12,
    )
    initial_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()
                     if isinstance(tensor, torch.Tensor)}
    history: list[dict[str, Any]] = []
    with torch.no_grad():
        initial_terms = _objective_components(q_initial, gd_initial, model, fields,
                                             mu_parameter, number_penalty=1.0e6)
        initial_row = {"iteration": -1.0, **_diagnostic(q_initial, gd_initial, fields, mu_parameter),
                       "mu_mev": float(mu_parameter.detach()),
                       "loss": float(closure_loss(model, fields, mu_parameter)),
                       "objective_terms": {name: float(value.detach()) for name, value in initial_terms.items()
                                           if name.endswith("_term") or name == "training_objective"},
                       "blocked_mode_overlap": float(blocking_overlap(initial_blocked, mode_initial))}
    history.append(initial_row)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "mu": mu_parameter.detach(), "history": history},
               out / "checkpoint.partial.pt")
    chart_limits = {"raw_coordinates": 2.0, "raw_a": 4.0, "raw_b": 4.0}
    accepted_parameters = [parameter.detach().clone() for parameter in parameters]
    accepted_optimizer = copy.deepcopy(optimizer.state_dict())
    accepted_model_state = copy.deepcopy(model.state_dict())
    accepted_mu = mu_parameter.detach().clone()
    accepted_updates = 0
    stop_reason = "maximum_iterations_reached"
    chart_boundary_hit = False
    optimizer_exception = None
    last_gradient_diagnostics: dict[str, float] = {}
    line_search_chart_bound_crossings = 0
    penalty_number = 1.0e6
    last_objective_terms: dict[str, float] = {}
    spectral_originals, spectral_attempts = _forbid_spectral_torch_calls()
    try:
        for iteration in range(int(max_iter)):
            step_start_parameters = [parameter.detach().clone() for parameter in parameters]
            def closure():
                nonlocal last_gradient_diagnostics, line_search_chart_bound_crossings
                nonlocal last_objective_terms
                optimizer.zero_grad(set_to_none=True)
                chart_norms = {
                    "raw_coordinates": torch.linalg.vector_norm(model.raw_coordinates),
                    "raw_a": torch.linalg.vector_norm(model.raw_a),
                    "raw_b": torch.linalg.vector_norm(model.raw_b),
                }
                if any(float(chart_norms[name].detach()) > chart_limits[name]
                       for name in chart_limits):
                    line_search_chart_bound_crossings += 1
                q, gd = model()
                # The number penalty selects the required stationary sector;
                # μ is also varied as the particle-number Lagrange multiplier.
                terms = _objective_components(q, gd, model, fields, mu_parameter,
                                              number_penalty=penalty_number)
                loss = terms["training_objective"]
                last_objective_terms = {
                    name: float(value.detach()) for name, value in terms.items()
                    if name.endswith("_term") or name == "training_objective"
                }
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite fixed-field residual objective")
                loss.backward()
                for parameter in parameters:
                    if parameter.grad is not None and not parameter.grad.is_contiguous():
                        parameter.grad = parameter.grad.contiguous()
                last_gradient_diagnostics = _gradient_diagnostics(model, mu_parameter)
                return loss
            try:
                optimizer.step(closure)
            except Exception as exc:
                optimizer_exception = f"{type(exc).__name__}: {exc}"
                stop_reason = "optimizer_exception_restored_last_accepted_state"
                model.load_state_dict(accepted_model_state)
                with torch.no_grad():
                    for parameter, value in zip(parameters, accepted_parameters):
                        parameter.copy_(value)
                    mu_parameter.copy_(accepted_mu)
                optimizer.load_state_dict(accepted_optimizer)
                break
            with torch.no_grad():
                update_norm = float(torch.stack([
                    torch.linalg.vector_norm(parameter - before)
                    for parameter, before in zip(parameters, step_start_parameters)
                ]).norm())
                if update_norm <= 1e-14:
                    stop_reason = "optimizer_made_no_parameter_update_last_state_restored"
                    model.load_state_dict(accepted_model_state)
                    for parameter, value in zip(parameters, accepted_parameters):
                        parameter.copy_(value)
                    mu_parameter.copy_(accepted_mu)
                    optimizer.load_state_dict(accepted_optimizer)
                    break
                current_norms = {name: float(torch.linalg.vector_norm(getattr(model, name)))
                                 for name in chart_limits}
                if any(current_norms[name] > chart_limits[name] + 1e-10 for name in chart_limits):
                    chart_boundary_hit = True
                    stop_reason = "accepted_step_rejected_at_chart_boundary_last_state_restored"
                    model.load_state_dict(accepted_model_state)
                    for parameter, value in zip(parameters, accepted_parameters):
                        parameter.copy_(value)
                    mu_parameter.copy_(accepted_mu)
                    optimizer.load_state_dict(accepted_optimizer)
                    break
                accepted_parameters = [parameter.detach().clone() for parameter in parameters]
                accepted_model_state = copy.deepcopy(model.state_dict())
                accepted_optimizer = copy.deepcopy(optimizer.state_dict())
                accepted_mu = mu_parameter.detach().clone()
                accepted_updates += 1
                q, gd = model()
                _, _, mode = model.forward_with_blocked_mode()
                accepted_terms = _objective_components(
                    q, gd, model, fields, mu_parameter, number_penalty=penalty_number
                )
                row = {"iteration": float(iteration), **_diagnostic(q, gd, fields, mu_parameter),
                       "mu_mev": float(mu_parameter.detach()),
                       "loss": float(closure_loss(model, fields, mu_parameter)),
                       "objective_terms": {
                           name: float(value.detach()) for name, value in accepted_terms.items()
                           if name.endswith("_term") or name == "training_objective"
                       },
                       "blocked_mode_overlap": float(blocking_overlap(initial_blocked, mode)),
                       "accepted_parameter_update_norm": update_norm,
                       "chart_norms": current_norms,
                       "last_closure_gradient_norms": dict(last_gradient_diagnostics),
                       "line_search_chart_bound_crossings": line_search_chart_bound_crossings}
                history.append(row)
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "mu": mu_parameter.detach(), "history": history},
                           out / "checkpoint.partial.pt")
    finally:
        _restore_spectral_torch_calls(spectral_originals)

    final_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()
                   if isinstance(tensor, torch.Tensor)}
    with torch.no_grad():
        q_final, gd_final, mode_final = model.forward_with_blocked_mode()
        final_diagnostic = _diagnostic(q_final, gd_final, fields, mu_parameter)
        final_terms = _objective_components(q_final, gd_final, model, fields,
                                            mu_parameter, number_penalty=penalty_number)
        final_loss = float(closure_loss(model, fields, mu_parameter))
        final_chart_norms = {name: float(torch.linalg.vector_norm(getattr(model, name)))
                             for name in chart_limits}
    optimizer.zero_grad(set_to_none=True)
    closure_loss(model, fields, mu_parameter).backward()
    final_gradient_diagnostics = _gradient_diagnostics(model, mu_parameter)
    optimizer.zero_grad(set_to_none=True)
    final_row = {"iteration": float(accepted_updates), **final_diagnostic,
                 "mu_mev": float(mu_parameter.detach()), "loss": final_loss,
                 "objective_terms": {name: float(value.detach()) for name, value in final_terms.items()
                                     if name.endswith("_term") or name == "training_objective"},
                 "blocked_mode_overlap": float(blocking_overlap(initial_blocked, mode_final)),
                 "chart_norms": final_chart_norms,
                 "final_gradient_norms": final_gradient_diagnostics,
                 "line_search_chart_bound_crossings": line_search_chart_bound_crossings}
    history.append(final_row)
    gd_final_primitive = _lift_density(q_final, gd_final)
    gd_initial_primitive = _lift_density(q_initial, gd_initial)
    final_pfaffian = number_parity_from_density(
        gd_final_primitive.rho.detach().cpu().numpy(), gd_final_primitive.chi.detach().cpu().numpy()
    )
    if final_pfaffian["number_parity"] != -1:
        raise RuntimeError("trained state failed independent odd-Pfaffian check")
    # Reference spectra are deliberately evaluated only after training, at the
    # final learned μ, and are never exposed to NN initialization or loss.
    # The independent branch code is currently under a physics audit.  Keep
    # the comparison explicitly pending until its odd-blocked occupation and
    # projected-space particle-hole conventions pass that audit.
    reference_comparison: dict[str, Any] = {
        "status": "pending_independent_reference_physics_audit",
        "note": "No current branch-reference output is accepted or used in the NN objective.",
        "chemical_potential_mev_for_later_comparison": float(mu_parameter.detach()),
    }
    hbdg_final = _nambu_hamiltonian(fields, mu_parameter.detach())
    # This explicit spectrum is post-training validation only.
    spectral_values, spectral_vectors = torch.linalg.eigh(hbdg_final)
    blocked_overlaps = torch.abs(spectral_vectors.mH @ mode_final)
    blocked_match = int(torch.argmax(blocked_overlaps))
    reference_comparison["post_training_spectral_diagnostic"] = {
        "nearest_matched_eigenstate_index": blocked_match,
        "matched_eigenvalue_mev": float(spectral_values[blocked_match]),
        "matched_blocked_mode_overlap": float(blocked_overlaps[blocked_match]),
        "minimum_absolute_quasiparticle_energy_mev": float(spectral_values.abs().min()),
        "particle_hole_spectral_symmetry_error_mev": float((spectral_values + spectral_values.flip(0)).abs().max()),
    }
    parameter_names = {name for name, _ in model.named_parameters()}
    layer_changes = {
        name: float(torch.linalg.vector_norm(final_state[name] - initial_state[name]))
        for name in parameter_names if name in final_state and name in initial_state
    }
    np.savez_compressed(
        out / "states.npz",
        initial_q=q_initial.detach().cpu().numpy(),
        initial_rho_active=gd_initial.rho.detach().cpu().numpy(),
        initial_chi_active=gd_initial.chi.detach().cpu().numpy(),
        initial_rho=gd_initial_primitive.rho.detach().cpu().numpy(),
        initial_chi=gd_initial_primitive.chi.detach().cpu().numpy(),
        final_q=q_final.detach().cpu().numpy(),
        final_rho_active=gd_final.rho.detach().cpu().numpy(),
        final_chi_active=gd_final.chi.detach().cpu().numpy(),
        final_rho=gd_final_primitive.rho.detach().cpu().numpy(),
        final_chi=gd_final_primitive.chi.detach().cpu().numpy(),
        final_generalized_density=gd_final_primitive.matrix.detach().cpu().numpy(),
        final_blocked_mode=mode_final.detach().cpu().numpy() if mode_final is not None else np.empty(0),
    )
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "mu": mu_parameter.detach(), "history": history}, out / "checkpoint.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (out / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    (out / "reference_comparison.json").write_text(json.dumps(reference_comparison, indent=2) + "\n")
    final = final_row
    gate_tolerances = {"commutator_frobenius_mev": 1.0e-4,
                       "particle_number_abs_error": 1.0e-6,
                       "purity_error": 1.0e-10,
                       "pair_antisymmetry_error": 1.0e-10}
    structure_pass = (
        final_diagnostic["purity_error"] <= gate_tolerances["purity_error"]
        and final_diagnostic["pair_antisymmetry_error"] <= gate_tolerances["pair_antisymmetry_error"]
    )
    parity_pass = final_pfaffian["number_parity"] == -1
    reference_pass = reference_comparison.get("status") == "certified_same_branch_pass"
    solver_gate = {
        "tolerances": gate_tolerances,
        "raw_commutator_pass": final_diagnostic["commutator_frobenius_mev"] <= gate_tolerances["commutator_frobenius_mev"],
        "particle_number_pass": abs(final_diagnostic["particle_number_error"]) <= gate_tolerances["particle_number_abs_error"],
        "structure_pass": structure_pass,
        "parity_pass": parity_pass,
        "branch_certification_status": "certified" if reference_pass else "pending",
        "branch_certification_pass": reference_pass,
        "blocked_mode_overlap_initial_to_final": final_row["blocked_mode_overlap"],
        "blocked_mode_overlap_role": "descriptive continuity diagnostic only; excluded from tolerances",
    }
    solver_gate["accepted_as_stationary_solver"] = all(
        solver_gate[key] for key in ("raw_commutator_pass", "particle_number_pass", "structure_pass",
                                     "parity_pass", "branch_certification_pass")
    )
    report = {
        "kind": "fixed-field neural primitive RHFB validation; not self-consistent",
        "fields_frozen": True, "h_delta_functional": fields.metadata,
        "dimension": d, "species": fields.species, "particle_number_target": fields.particle_number,
        "parity_target": -1, "blocked_index": blocked_index,
        "spectral_attempts_during_training": sum(spectral_attempts.values()),
        "spectral_attempt_counters": spectral_attempts,
        "spectral_guard": "torch and NumPy linalg eig/eigh/eigvals/eigvalsh blocked",
        "max_iter": max_iter, "accepted_updates": accepted_updates,
        "stop_reason": stop_reason, "optimizer_exception": optimizer_exception,
        "chart_boundary_hit": chart_boundary_hit, "chart_norm_limits": chart_limits,
        "number_penalty_mev2": penalty_number,
        "chart_penalty": {"type": "smooth softplus squared barrier added to physical objective at every trial",
                          "coefficient": 1.0e8, "softplus_width": 0.05},
        "optimizer": "torch LBFGS strong_wolfe, exact autodiff gradients",
        "mu_mev": float(mu_parameter.detach()),
        "final_diagnostics": final, "final_gradient_norms": final_gradient_diagnostics,
        "solver_acceptance_gate": solver_gate,
        "per_layer_initial_final_norm_changes": layer_changes,
        "initial_independent_pfaffian": initial_pfaffian,
        "final_independent_pfaffian": final_pfaffian,
        "post_training_reference_comparison": reference_comparison,
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "source_sha256": hashes,
        "limitations": ["single frozen PKA1+D1S field evaluation", "prototype only, D<=96",
                         "not a self-consistent solution or ground-state energy"],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _smooth_chart_penalty(value: torch.Tensor, limit: float, *, coefficient: float = 1.0e8,
                          width: float = 0.05) -> torch.Tensor:
    """C-infinity chart regularizer, continuous through the declared norm limit."""
    hinge = width * F.softplus((value - limit) / width)
    return coefficient * hinge.square()


def _objective_components(q, gd, model, fields, mu, *, number_penalty: float):
    commutator, error = _residual_tensors(q, gd, fields, mu)
    raw_residual_term = commutator.abs().square().sum()
    number_penalty_term = number_penalty * error.square()
    limits = {"raw_coordinates": 2.0, "raw_a": 4.0, "raw_b": 4.0}
    chart_penalty_term = torch.stack([
        _smooth_chart_penalty(torch.linalg.vector_norm(getattr(model, name)), limit)
        for name, limit in limits.items()
    ]).sum()
    return {
        "raw_residual_term": raw_residual_term,
        "number_penalty_term": number_penalty_term,
        "chart_penalty_term": chart_penalty_term,
        "training_objective": raw_residual_term + number_penalty_term + chart_penalty_term,
        "particle_number_error": error,
        "commutator": commutator,
    }


def closure_loss(model: GeneralizedDensityNet, fields: FrozenFields, mu: torch.Tensor) -> torch.Tensor:
    q, gd = model()
    return _objective_components(q, gd, model, fields, mu,
                                 number_penalty=1.0e6)["training_objective"]


def run_odd_neutron_probe(
    *,
    seed_state_path: str | Path,
    seed_fields_path: str | Path,
    pairing_fields_path: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    max_iter: int = 120,
) -> dict[str, Any]:
    """Run the bounded N=3 odd-neutron probe from the shared frozen-field cache."""
    state_path, fields_path = Path(seed_state_path), Path(seed_fields_path)
    pairing_path, meta_path = Path(pairing_fields_path), Path(metadata_path)
    metadata = json.loads(meta_path.read_text())
    field_meta_path = fields_path.parent / "metadata.json"
    field_metadata = json.loads(field_meta_path.read_text())
    expected_hash = metadata.get("state_sha256")
    if expected_hash and _sha256(state_path) != expected_hash:
        raise ValueError("seed-state hash does not match its metadata")
    if field_metadata.get("field_sha256") and _sha256(fields_path) != field_metadata["field_sha256"]:
        raise ValueError("normal-field hash does not match its metadata")
    basis = AngularRadialBasis(
        radius_fm=float(metadata["radius_fm"]),
        spacing_fm=float(metadata["spacing_fm"]),
        kappa_max=int(metadata["kappa_max"]),
        n_theta=int(metadata["n_theta"]), n_phi=int(metadata["n_phi"]),
        derivative_order=int(metadata["derivative_order"]),
        mass_n_mev=float(metadata["basis_masses_mev"]["n"]),
        mass_p_mev=float(metadata["basis_masses_mev"]["p"]),
    )
    state = np.load(state_path)
    normal = np.load(fields_path)
    pairing = np.load(pairing_path)
    for key in ("n_Q", "n_rho", "n_chi"):
        if key not in state:
            raise KeyError(f"seed state lacks {key}")
    if "n_h" not in normal or "n_delta" not in pairing:
        raise KeyError("frozen field inputs must contain n_h and n_delta")
    if basis.dimension != int(metadata["primitive_dimension"]):
        raise ValueError("reconstructed angular-radial basis dimension disagrees with metadata")
    d = basis.dimension
    fields = FrozenFields(
        h=torch.as_tensor(normal["n_h"], dtype=torch.complex128),
        delta=torch.as_tensor(pairing["n_delta"], dtype=torch.complex128),
        particle_number=int(metadata["n_branch_target_number"]), species="neutron",
        metadata={
            "functional": "PKA1 particle-hole + Gogny D1S pairing",
            "normal_field_cache": str(fields_path),
            "pairing_field_cache": str(pairing_path),
            "normal_field_kind": "full PH total-energy derivative at saved seed state",
            "pairing_field_kind": "full D1S pairing field at saved seed state",
            "frozen_density_hash": expected_hash,
            "normal_field_hash": field_metadata.get("field_sha256"),
            "normal_field_method": field_metadata.get("field_method"),
            "normal_field_source_hashes": field_metadata.get("source_sha256", {}),
            "pairing_source_hash": json.loads(pairing_path.with_suffix(".json").read_text()).get("pairing_source_sha256"),
        },
    )
    code_dir = Path(__file__).parent
    code_sources = tuple(code_dir / name for name in (
        "frozen_field_neural.py", "network.py", "manifold.py", "basis.py",
        "fixed_field_dirac_branch_reference.py", "branch.py", "parity_diagnostic.py",
        "functional.py", "cached_normal_action.py", "normal_action.py", "gaussian_pairing.py",
    ))
    field_metadata_path = fields_path.parent / "metadata.json"
    pairing_metadata = pairing_path.with_suffix(".json")
    source_files = (state_path, fields_path, pairing_path, meta_path, *code_sources,
                    *((field_metadata_path,) if field_metadata_path.exists() else ()),
                    *((pairing_metadata,) if pairing_metadata.exists() else ()))
    return train_frozen_fields(
        basis=basis, fields=fields,
        seed_q=torch.as_tensor(state["n_Q"], dtype=torch.complex128),
        seed_rho=torch.as_tensor(state["n_rho"], dtype=torch.complex128),
        seed_chi=torch.as_tensor(state["n_chi"], dtype=torch.complex128),
        labels=tuple(f"seed-{i:03d}" for i in range(d)),
        blocked_index=int(metadata["n_blocked_mode_index"]),
        output_dir=output_dir, source_files=source_files, max_iter=max_iter,
    )
