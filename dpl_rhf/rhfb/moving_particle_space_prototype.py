"""Tiny moving-particle-space odd-HFB stationarity benchmark.

The candidate differentiates through a rational-sign positive projector and
QR chart.  A separate dense-eigensolver path solves frozen-projector inner
roots and updates the projector outside.  This is a finite-model algorithm
comparison, not a theorem that a coupled stationary root is a min-max RHFB
state or a production PKA1 result.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager

import torch

from .branch import no_sea_branch_reference, nambu_hamiltonian
from .branch_prototype import make_full_dirac_model
from .no_sea import RationalSignOperator
from .no_sea_particle_space import (
    embed_active_density,
    positive_frame_from_projector,
    positive_projector_from_sign,
)
from .pairing import D1S, D1S_PROVENANCE
from .stationary_solver import StationaryOptions, solve_stationarity


@dataclass(frozen=True)
class MovingToy:
    h0: torch.Tensor
    source: torch.Tensor
    coupling: float
    spinors: torch.Tensor
    positions: torch.Tensor
    weights: torch.Tensor
    seed_frame: torch.Tensor
    shift: float = -5.0
    target_number: float = 3.0

    @property
    def d(self) -> int:
        return self.h0.shape[0]

    @property
    def rank(self) -> int:
        return self.seed_frame.shape[1]

    def normal(self, rho: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.trace(self.source @ rho).real
        energy = torch.trace(self.h0 @ rho).real + 0.5 * self.coupling * x.square()
        field = self.h0 + self.coupling * x * self.source
        return energy, field, x

    def positive_frame(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Weyl enclosure: the six positive bare levels lie in [0,2.8],
        # the two sea levels at -12,-14, and |g x S| is bounded by the
        # trace/rho contraction and the source row-sum norm.
        base = torch.diagonal(self.h0).real
        gap0 = float(torch.min(torch.abs(base - self.shift)))
        bound0 = float(torch.max(torch.abs(base - self.shift)))
        source_norm = float(torch.sum(torch.abs(self.source), dim=1).max())
        # Every candidate H is generated as h0+g*eta*S with |eta|<=16;
        # this enclosure therefore covers trial points as well as roots.
        perturbation = abs(self.coupling) * 16.0 * source_norm
        gap, bound = gap0 - perturbation, bound0 + perturbation
        if gap <= 0:
            raise RuntimeError("declared Weyl enclosure closes")
        sign = RationalSignOperator(
            h, self.d, shift=self.shift, spectral_gap=gap,
            spectral_bound=bound,
            bound_certificate=("diagonal h0 exact; auxiliary eta=16*tanh(s) "
                               "bounds every trial H=h0+g*eta*S; ||S||2<=max row sum; Weyl bound"),
            quadrature_order=121, tolerance=2e-10, solve_tolerance=2e-10,
        ).apply_sign(torch.eye(self.d, dtype=torch.complex128))
        q = positive_projector_from_sign(sign)
        frame = positive_frame_from_projector(
            q, self.seed_frame, projector_tolerance=None,
        )
        return frame, q


def make_moving_toy() -> tuple[MovingToy, object]:
    """Six nucleon states plus two sea states, with live particle/sea mixing."""
    full = make_full_dirac_model(3, pairing_enabled=True)
    # Retain the documented nonzero PKA-like density-dependent source in the
    # full primitive space, including its explicitly nonzero positive/sea
    # matrix elements.  rho itself is embedded in the moving rank-six space.
    source = full.source_a + full.source_a.mH
    source = 0.5 * source
    # Compress the active single-particle spacings so the physical D1S
    # strength-one channel supports a paired stationary branch in this tiny
    # benchmark; the separated negative-energy levels remain untouched.
    h0 = full.h0.clone()
    h0[:6, :6] *= 0.10
    energies = torch.diagonal(h0).real
    frame = torch.zeros((8, 6), dtype=torch.complex128)
    frame[:6] = torch.eye(6, dtype=torch.complex128)
    ref = no_sea_branch_reference(
        energies[:6].contiguous(),
        torch.tensor([1., 1., 0., 0., 0., 0.], dtype=torch.float64),
        dirac_cutoff=-5., chemical_potential=.5, blocked_index=5,
        blocking_operation="particle",
    )
    # Occupied-occupied and empty-empty pairing channels seed a nonzero κ
    # while preserving odd parity; no forbidden occupied-empty seed is used.
    return MovingToy(h0, source, full.coupling_a, full.spinors,
                     full.positions, full.weights, frame), ref


def _chart_basis(device: torch.device) -> tuple[torch.Tensor, ...]:
    """Full 66-real-coordinate canonical chart for six active orbitals."""
    from .branch_prototype import _coordinate_basis
    a, b = _coordinate_basis(6, device)
    return tuple(a.unbind(0)), tuple(b.unbind(0))


def _hermitian_basis(dimension: int, device: torch.device) -> torch.Tensor:
    """Orthonormal real basis for Hermitian matrices under ReTr(A†B)."""
    basis = []
    for i in range(dimension):
        item = torch.zeros((dimension, dimension), dtype=torch.complex128, device=device)
        item[i, i] = 1.
        basis.append(item)
    inv_sqrt2 = 2.0 ** -0.5
    for i in range(dimension):
        for j in range(i + 1, dimension):
            item = torch.zeros((dimension, dimension), dtype=torch.complex128, device=device)
            item[i, j] = item[j, i] = inv_sqrt2
            basis.append(item)
            item = torch.zeros((dimension, dimension), dtype=torch.complex128, device=device)
            item[i, j] = 1j * inv_sqrt2
            item[j, i] = -1j * inv_sqrt2
            basis.append(item)
    return torch.stack(basis)


def _active_state(reference, coordinates: torch.Tensor, a_basis, b_basis):
    from .manifold import transform_reference
    a = torch.einsum("i,ijk->jk", coordinates[:len(a_basis)].to(torch.complex128),
                     torch.stack(a_basis))
    b = torch.einsum("i,ijk->jk", coordinates[len(a_basis):].to(torch.complex128),
                     torch.stack(b_basis))
    return transform_reference(reference.density, a, b)


def _objective_and_fields(toy: MovingToy, frame: torch.Tensor, active):
    from .pairing import pairing_gap
    full = embed_active_density(frame, active.rho, active.chi,
                                fermion_parity=active.fermion_parity)
    normal, h, x = toy.normal(full.rho)
    epair, delta = pairing_gap(toy.spinors, full.chi, toy.positions, toy.weights,
                               species="neutron", chunk_size=8, force=D1S)
    energy = normal + epair
    return full, h, delta, energy, normal, epair, x


def _stationarity_residual(toy, reference, initial_frame, moving, options,
                           initial_parameters: torch.Tensor | None = None):
    a_basis, b_basis = _chart_basis(toy.h0.device)
    ncoord = len(a_basis) + len(b_basis)
    device = toy.h0.device
    x0 = torch.zeros(ncoord + 1, dtype=torch.float64, device=device)
    # Nonzero B(0,1), B(2,3) seeds are explicit canonical tangent coordinates.
    # In the complete real-skew basis, pair 2,3 is pair ordinal 9.
    x0[len(a_basis)] = .32
    x0[len(a_basis) + 18] = -.30
    x0[-1] = .45
    if initial_parameters is not None and not moving:
        if initial_parameters.shape != x0.shape:
            raise ValueError("nested warm-start vector has the wrong shape")
        x0 = initial_parameters.detach().clone()

    if moving:
        # This normal functional has a rank-one field family
        # H(eta)=h0+g*eta*S.  The bounded eta chart keeps the Weyl enclosure
        # valid for every LM trial; F=eta-Tr(S rho)=0 closes self-consistency.
        # KKT variables are c, unconstrained s, μ and scalar Λ.
        initial_state = _active_state(reference, x0[:-1], a_basis, b_basis)
        initial_full = embed_active_density(toy.seed_frame, initial_state.rho,
                                            initial_state.chi,
                                            fermion_parity=initial_state.fermion_parity)
        initial_scalar = torch.trace(toy.source @ initial_full.rho).real
        xinit = torch.cat((x0[:-1], torch.atanh((initial_scalar/16.).clamp(-.99,.99)).reshape(1),
                           x0[-1:], torch.zeros(1, dtype=torch.float64, device=device)))

        def evaluate(params):
            c = params[:ncoord]
            eta_raw = params[ncoord]
            mu, lam = params[ncoord+1], params[ncoord+2]
            eta = 16. * torch.tanh(eta_raw)
            haux = toy.h0 + toy.coupling * eta * toy.source
            active = _active_state(reference, c, a_basis, b_basis)
            frame, _ = toy.positive_frame(haux)
            st, hfield, delta, energy, normal, epair, scalar = _objective_and_fields(toy, frame, active)
            constraint = eta - scalar
            return active, frame, _q_from_h(toy, haux), st, hfield, delta, energy, normal, epair, scalar, mu, lam, constraint, haux, eta

        def residual(params):
            if not params.requires_grad:
                params = params.detach().requires_grad_(True)
            values = evaluate(params)
            st, energy, mu, lam, constraint = values[3], values[6], values[10], values[11], values[12]
            lagrangian = (energy - mu * (torch.trace(st.rho).real - toy.target_number)
                          + lam * constraint)
            gradient = torch.autograd.grad(lagrangian, params, create_graph=True)[0]
            gc = gradient[:ncoord]
            geta = gradient[ncoord]
            number = torch.trace(st.rho).real - toy.target_number
            return torch.cat((gc, geta.reshape(1), number.reshape(1), constraint.reshape(1)))

        solved = solve_stationarity(residual, xinit, options=options)
        with torch.enable_grad():
            values = evaluate(solved.parameters.detach().requires_grad_(True))
        return solved, values, ncoord

    def evaluate(params):
        active = _active_state(reference, params[:ncoord], a_basis, b_basis)
        st, h, delta, energy, normal, epair, scalar = _objective_and_fields(toy, initial_frame, active)
        return active, initial_frame, _q_from_h(toy, h), st, h, delta, energy, normal, epair, scalar

    def residual(params):
        if not params.requires_grad:
            params = params.detach().requires_grad_(True)
        values = evaluate(params)
        st, energy = values[3], values[6]
        lagrangian = energy - params[-1] * (torch.trace(st.rho).real - toy.target_number)
        grad = torch.autograd.grad(lagrangian, params, create_graph=True)[0]
        return torch.cat((grad[:-1], (torch.trace(st.rho).real - toy.target_number).reshape(1)))

    solved = solve_stationarity(residual, x0, options=options)
    with torch.enable_grad():
        values = evaluate(solved.parameters.detach().requires_grad_(True))
    return solved, values, ncoord


@contextmanager
def _candidate_eigensolver_guard():
    """Fail immediately if the candidate root calls any spectral solver."""
    names = ("eigh", "eigvalsh", "eig")
    originals = {name: getattr(torch.linalg, name) for name in names}
    count = {"calls": 0}

    def forbidden(*args, **kwargs):
        count["calls"] += 1
        raise AssertionError("candidate moving-Q solve called a spectral eigensolver")

    try:
        for name in names:
            setattr(torch.linalg, name, forbidden)
        yield count
    finally:
        for name, function in originals.items():
            setattr(torch.linalg, name, function)


def _q_from_h(toy: MovingToy, h: torch.Tensor) -> torch.Tensor:
    base = torch.diagonal(toy.h0).real
    gap0 = float(torch.min(torch.abs(base - toy.shift)))
    bound0 = float(torch.max(torch.abs(base - toy.shift)))
    source_norm = float(torch.sum(torch.abs(toy.source), dim=1).max())
    perturbation = abs(toy.coupling) * 16.0 * source_norm
    sign = RationalSignOperator(
        h, toy.d, shift=toy.shift, spectral_gap=gap0-perturbation,
        spectral_bound=bound0+perturbation,
        bound_certificate="diagonal h0 plus Weyl row-sum bound for all eta=16*tanh(s) trials",
        quadrature_order=121, tolerance=2e-10, solve_tolerance=2e-10,
    ).apply_sign(torch.eye(toy.d, dtype=torch.complex128))
    return positive_projector_from_sign(sign)


def _diagnostics(toy, values) -> dict[str, float]:
    active, frame, qsign, full, h, delta, energy, normal, epair, x = values[:10]
    qframe = frame @ frame.mH
    exact_values, exact_vectors = torch.linalg.eigh(h)
    qexact = exact_vectors[:, exact_values > toy.shift] @ exact_vectors[:, exact_values > toy.shift].mH
    chi_active = frame.mH @ full.chi @ frame.conj()
    g_perp = (torch.eye(toy.d, dtype=h.dtype) - qexact) @ delta @ frame.conj() @ chi_active.mH
    nambu = nambu_hamiltonian(h, delta, chemical_potential=.45)
    return {
        "energy": float(energy.detach()), "normal_energy": float(normal.detach()),
        "D1S_pair_energy": float(epair.detach()), "chi_frobenius": float(torch.linalg.matrix_norm(full.chi).detach()),
        "number": float(torch.trace(full.rho).real.detach()),
        "number_abs_residual": abs(float(torch.trace(full.rho).real.detach())-toy.target_number),
        "R_purity_abs_frobenius": float(torch.linalg.matrix_norm(full.matrix@full.matrix-full.matrix).detach()),
        "field_density_scalar": float(x.detach()),
        "positive_negative_bare_coupling_norm": float(torch.linalg.matrix_norm(toy.source[:6,6:]).detach()),
        "positive_negative_field_coupling_norm": float(torch.linalg.matrix_norm(h[:6,6:]).detach()),
        "projector_sign_exact_frobenius": float(torch.linalg.matrix_norm(qsign-qexact).detach()),
        "projector_frame_fixed_point_residual": float(torch.linalg.matrix_norm(qframe-qexact).detach()),
        "projector_motion_from_bare": float(torch.linalg.matrix_norm(
            qexact - toy.seed_frame @ toy.seed_frame.mH).detach()),
        "moving_response_pair_coupling_norm": float(torch.linalg.matrix_norm(g_perp).detach()),
        "projector_idempotency_exact_sign": float(torch.linalg.matrix_norm(qsign@qsign-qsign).detach()),
        "field_projector_commutator": float(torch.linalg.matrix_norm(h@qexact-qexact@h).detach()),
        "field_hermiticity": float(torch.linalg.matrix_norm(h-h.mH).detach()),
        "delta_antisymmetry": float(torch.linalg.matrix_norm(delta+delta.mT).detach()),
        "nambu_hermiticity": float(torch.linalg.matrix_norm(nambu-nambu.mH).detach()),
        "bare_negative_basis_density_support": float(torch.linalg.matrix_norm(full.rho[6:,:]).detach()),
        "normal_excluded_projector_support": float(torch.linalg.matrix_norm(
            (torch.eye(toy.d, dtype=h.dtype)-qexact) @ full.rho).detach()),
        "normal_excluded_projector_support_fraction": float((torch.linalg.matrix_norm(
            (torch.eye(toy.d, dtype=h.dtype)-qexact) @ full.rho)
            / torch.linalg.matrix_norm(full.rho).clamp_min(1.e-30)).detach()),
        "anomalous_excluded_projector_support": float(torch.linalg.matrix_norm(
            (torch.eye(toy.d, dtype=h.dtype)-qexact) @ full.chi).detach()),
        "anomalous_excluded_projector_support_fraction": float((torch.linalg.matrix_norm(
            (torch.eye(toy.d, dtype=h.dtype)-qexact) @ full.chi)
            / torch.linalg.matrix_norm(full.chi).clamp_min(1.e-30)).detach()),
    }


def run_moving_particle_space_benchmark(*, max_iterations: int = 20) -> dict:
    """Run full-response AD root and independent frozen-Q nested reference."""
    toy, reference = make_moving_toy()
    options = StationaryOptions(max_iterations=max_iterations,
                                residual_tolerance=2e-7,
                                initial_damping=1e-2,
                                maximum_trial_steps=10,
                                maximum_variables=512,
                                maximum_residuals=1024)
    # Candidate: candidate path has rational sign + QR only; no eigensolver.
    with _candidate_eigensolver_guard() as spectral_calls:
        candidate, candidate_values, _ = _stationarity_residual(
            toy, reference, toy.seed_frame, True, options)
    candidate_metrics = _diagnostics(toy, candidate_values)
    candidate_metrics.update({"solver_converged": float(candidate.converged),
                              "solver_residual_max": float(candidate.residual.abs().max()),
                              "solver_stop_reason": candidate.stop_reason,
                              "candidate_eigensolver_calls": spectral_calls["calls"]})
    candidate_metrics["auxiliary_field_constraint_abs"] = abs(float(candidate_values[12].detach()))

    # Independent reference: freeze Q in the inner variational root, then
    # rediagonalize h and update Q between inner solves (nested SCF).
    frame = toy.seed_frame
    nested_metrics = {}
    nested_converged = False
    nested_solver_histories = []
    nested_parameters = None
    for outer in range(8):
        inner, vals, _ = _stationarity_residual(
            toy, reference, frame, False, options, initial_parameters=nested_parameters)
        nested_parameters = inner.parameters
        nested_solver_histories.append(inner.history)
        nested_metrics = _diagnostics(toy, vals)
        h = vals[4]
        ev, vec = torch.linalg.eigh(h)
        new = vec[:, ev > toy.shift]
        old_q = frame @ frame.mH
        pnew = new @ new.mH
        # Keep the active coordinates and blocked reference in one fixed
        # chart: QR(Pnew @ Bseed), not raw eigensolver vectors with arbitrary
        # phases or rotations. The eigensolver only supplies Pnew here.
        new_frame = positive_frame_from_projector(
            pnew, toy.seed_frame, projector_tolerance=None)
        new_q = new_frame @ new_frame.mH
        change = float(torch.linalg.matrix_norm(new_q-old_q).detach())
        frame = new_frame
        nested_converged = inner.converged and change < 2e-6
        if nested_converged:
            break
    nested_metrics.update({"outer_iterations": outer+1,
                           "outer_projector_change": change,
                           "inner_solver_converged": float(inner.converged),
                           "inner_residual_max": float(inner.residual.abs().max()),
                           "nested_fixed_point_converged": float(nested_converged)})
    return {
        "model_scope": "8-state moving-Q odd-HFB toy with D1S strength 1; not production PKA1",
        "source_version": {
            "module": "dpl_rhf.rhfb.moving_particle_space_prototype",
            "snapshot_schema": 1,
            "basis_note": "working-tree source; matrices and final parameters are serialized below",
        },
        "model_specification": {
            "h0": _complex_array_snapshot(toy.h0),
            "normal_source": _complex_array_snapshot(toy.source),
            "normal_coupling": toy.coupling,
            "projector_shift": toy.shift,
            "target_number": toy.target_number,
            "active_rank": toy.rank,
            "primitive_dimension": toy.d,
            "dirac_cutoff": reference.dirac_cutoff,
            "blocking_index_in_active_basis": 5,
            "blocking_operation": "particle added to initially empty orbital 5",
            "reference_occupations_before_blocking": [1, 1, 0, 0, 0, 0],
            "fermion_parity": reference.density.fermion_parity,
            "pairing_force": {
                "name": "Gogny D1S finite-range nn",
                "parameters": {"ranges_fm": list(D1S.ranges_fm), "w_mev": list(D1S.w_mev),
                               "b_mev": list(D1S.b_mev), "h_mev": list(D1S.h_mev),
                               "m_mev": list(D1S.m_mev), "strength": D1S.strength},
                "provenance": D1S_PROVENANCE,
            },
        },
        "solver_settings": {
            "max_iterations": options.max_iterations,
            "residual_tolerance": options.residual_tolerance,
            "initial_damping": options.initial_damping,
            "maximum_trial_steps": options.maximum_trial_steps,
            "maximum_variables": options.maximum_variables,
            "rational_sign_order": 121,
            "rational_sign_solve_tolerance": 2e-10,
        },
        "candidate_full_projector_response": candidate_metrics,
        "independent_frozen_Q_nested_root": nested_metrics,
        "candidate_final_state": _state_snapshot(
            toy, candidate_values, candidate.parameters, candidate.parameters[-2],
            role="moving-Q KKT candidate; exact projector is reference-only eig diagnostic"),
        "nested_final_state": _state_snapshot(
            toy, vals, inner.parameters, inner.parameters[-1],
            role="frozen-Q inner root after outer self-consistency; eig is reference-only"),
        "candidate_solver_history": candidate.history,
        "nested_solver_histories_by_outer_iteration": nested_solver_histories,
        "energy_difference_candidate_minus_nested": candidate_metrics["energy"]-nested_metrics["energy"],
        "rho_difference_frobenius": float(torch.linalg.matrix_norm(
            candidate_values[3].rho - vals[3].rho).detach()),
        "chi_difference_frobenius": float(torch.linalg.matrix_norm(
            candidate_values[3].chi - vals[3].chi).detach()),
        "chi_difference_phase_aligned_frobenius": _phase_aligned_chi_difference(
            candidate_values[3].chi, vals[3].chi),
    }


def _phase_aligned_chi_difference(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    overlap = torch.sum(candidate.conj() * reference)
    if float(torch.abs(overlap).detach()) <= 1.e-14:
        return float(torch.linalg.matrix_norm(candidate-reference).detach())
    phase = overlap.conj() / torch.abs(overlap)
    return float(torch.linalg.matrix_norm(phase*candidate-reference).detach())


def _complex_array_snapshot(value: torch.Tensor) -> dict[str, object]:
    value = value.detach().cpu()
    return {"real": value.real.tolist(), "imag": value.imag.tolist(),
            "shape": list(value.shape)}


def _state_snapshot(toy: MovingToy, values, parameters: torch.Tensor,
                    chemical_potential: torch.Tensor, *, role: str) -> dict:
    active, frame, _qsign, full, h, delta, energy, normal, epair, scalar = values[:10]
    eigvals, eigvecs = torch.linalg.eigh(h)
    selected = eigvecs[:, eigvals > toy.shift]
    p_exact = selected @ selected.mH
    snapshot = {
        "role": role,
        "parameters": parameters.detach().cpu().tolist(),
        "chemical_potential": float(chemical_potential.detach()),
        "energy": float(energy.detach()),
        "normal_energy": float(normal.detach()),
        "D1S_pair_energy": float(epair.detach()),
        "field_density_scalar": float(scalar.detach()),
        "rho": _complex_array_snapshot(full.rho),
        "chi": _complex_array_snapshot(full.chi),
        "projector_frame": _complex_array_snapshot(frame @ frame.mH),
        "projector_reference_eigh": _complex_array_snapshot(p_exact),
        "normal_field": _complex_array_snapshot(h),
        "pairing_field": _complex_array_snapshot(delta),
        "auxiliary_field": _complex_array_snapshot(values[13]) if len(values) > 13 else None,
        "active_fermion_parity": int(active.fermion_parity),
    }
    return snapshot


if __name__ == "__main__":
    import json
    print(json.dumps(run_moving_particle_space_benchmark(), indent=2))
