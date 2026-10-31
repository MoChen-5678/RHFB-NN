"""Residual interface for a moving positive-space RHFB fixed-point toy.

The physical equations are the active canonical commutator, scalar-field
self-consistency, and particle number.  The projector has a differentiable
rational-sign/QR implementation.  No energy-KKT multiplier is added.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from .branch import NoSeaBranchReference, nambu_hamiltonian, no_sea_branch_reference
from .branch_prototype import _coordinate_basis
from .manifold import GeneralizedDensity, transform_reference
from .moving_particle_space_prototype import MovingToy, make_moving_toy
from .no_sea import RationalSignOperator
from .no_sea_particle_space import (
    active_nambu_hamiltonian, embed_active_density,
    positive_frame_from_projector, positive_projector_from_sign,
)
from .pairing import D1S, D1S_PROVENANCE, pairing_gap
from .stationary_solver import StationaryOptions, real_residual, solve_stationarity


@dataclass(frozen=True)
class ProjectedRHFBState:
    active: GeneralizedDensity
    rho: torch.Tensor
    chi: torch.Tensor
    frame: torch.Tensor
    projector: torch.Tensor
    h_aux: torch.Tensor
    h_field: torch.Tensor
    delta: torch.Tensor
    mu: torch.Tensor
    eta: torch.Tensor
    energy: torch.Tensor
    number: torch.Tensor
    source_expectation: torch.Tensor
    active_hamiltonian: torch.Tensor


@dataclass(frozen=True)
class ProjectedResidualBlocks:
    stationarity: torch.Tensor
    field_self_consistency: torch.Tensor
    number: torch.Tensor

    def as_vector(self) -> torch.Tensor:
        return real_residual(self.stationarity,
                             self.field_self_consistency.reshape(()),
                             self.number.reshape(()))


def _basis_tuples(d: int, device: torch.device):
    a, b = _coordinate_basis(d, device)
    return tuple(a.unbind(0)), tuple(b.unbind(0))


def _projector_frame(toy: MovingToy, h: torch.Tensor):
    """Differentiable rational sign and fixed-seed QR, with a trial-wide bound."""
    base = torch.diagonal(toy.h0).real
    gap0 = float(torch.min(torch.abs(base - toy.shift)))
    bound0 = float(torch.max(torch.abs(base - toy.shift)))
    source_norm = float(torch.sum(torch.abs(toy.source), dim=1).max())
    perturb = abs(toy.coupling) * 16.0 * source_norm
    gap, bound = gap0 - perturb, bound0 + perturb
    if gap <= 0:
        raise RuntimeError("Weyl enclosure closes over the declared eta range")
    sign = RationalSignOperator(
        h, toy.d, shift=toy.shift, spectral_gap=gap, spectral_bound=bound,
        bound_certificate="diagonal h0 plus Weyl row-sum bound; eta=16*tanh(s)",
        quadrature_order=121, tolerance=2e-10, solve_tolerance=2e-10,
    ).apply_sign(torch.eye(toy.d, dtype=torch.complex128, device=h.device))
    q = positive_projector_from_sign(sign)
    b = positive_frame_from_projector(q, toy.seed_frame, projector_tolerance=None)
    return b, q


@dataclass
class ProjectedFixedPointProblem:
    toy: MovingToy
    reference: NoSeaBranchReference
    a_basis: tuple[torch.Tensor, ...]
    b_basis: tuple[torch.Tensor, ...]
    target_number: float

    @property
    def n_coordinates(self) -> int:
        return len(self.a_basis) + len(self.b_basis)

    def _active_from_coordinates(self, coordinates: torch.Tensor) -> GeneralizedDensity:
        n_a = len(self.a_basis)
        if coordinates.dtype != torch.float64 or coordinates.shape != (self.n_coordinates,):
            raise TypeError("coordinates must be a float64 vector of the canonical chart size")
        a = torch.einsum("i,ijk->jk", coordinates[:n_a].to(torch.complex128),
                         torch.stack(self.a_basis))
        b = torch.einsum("i,ijk->jk", coordinates[n_a:].to(torch.complex128),
                         torch.stack(self.b_basis))
        return transform_reference(self.reference.density, a, b)

    def state_from_coordinates(self, coordinates: torch.Tensor,
                               eta_raw: torch.Tensor | float,
                               mu: torch.Tensor | float) -> ProjectedRHFBState:
        """Canonical odd state -> density/fields; AD is retained through Q."""
        device = self.toy.h0.device
        c = torch.as_tensor(coordinates, dtype=torch.float64, device=device)
        eta_raw = torch.as_tensor(eta_raw, dtype=torch.float64, device=device).reshape(())
        mu = torch.as_tensor(mu, dtype=torch.float64, device=device).reshape(())
        eta = 16.0 * torch.tanh(eta_raw)
        h_aux = self.toy.h0 + self.toy.coupling * eta * self.toy.source
        frame, q = _projector_frame(self.toy, h_aux)
        active = self._active_from_coordinates(c)
        full = embed_active_density(frame, active.rho, active.chi,
                                    fermion_parity=active.fermion_parity)
        normal, h, source = self.toy.normal(full.rho)
        epair, delta = pairing_gap(
            self.toy.spinors, full.chi, self.toy.positions, self.toy.weights,
            species="neutron", chunk_size=8, force=D1S,
        )
        h_nambu = nambu_hamiltonian(h, delta, chemical_potential=mu)
        h_active = active_nambu_hamiltonian(h_nambu, frame)
        return ProjectedRHFBState(
            active, full.rho, full.chi, frame, q, h_aux, h, delta, mu, eta,
            normal + epair, torch.trace(full.rho).real, source, h_active,
        )

    def evaluate_network_outputs(self, rho_active: torch.Tensor,
                                 chi_active: torch.Tensor,
                                 eta_raw: torch.Tensor | float,
                                 mu: torch.Tensor | float,
                                 *, fermion_parity: int = -1) -> ProjectedRHFBState:
        """Unsafe algebraic adapter, not the canonical physical training path.

        Arbitrary density outputs need separate Hermiticity, skew, purity,
        parity, and particle-number constraints. Prefer canonical coordinates.
        """
        device = self.toy.h0.device
        rho = torch.as_tensor(rho_active, dtype=torch.complex128, device=device)
        chi = torch.as_tensor(chi_active, dtype=torch.complex128, device=device)
        eta_raw = torch.as_tensor(eta_raw, dtype=torch.float64, device=device).reshape(())
        mu = torch.as_tensor(mu, dtype=torch.float64, device=device).reshape(())
        eta = 16.0 * torch.tanh(eta_raw)
        h_aux = self.toy.h0 + self.toy.coupling * eta * self.toy.source
        frame, q = _projector_frame(self.toy, h_aux)
        full = embed_active_density(frame, rho, chi, fermion_parity=fermion_parity)
        normal, h, source = self.toy.normal(full.rho)
        epair, delta = pairing_gap(
            self.toy.spinors, full.chi, self.toy.positions, self.toy.weights,
            species="neutron", chunk_size=8, force=D1S,
        )
        matrix = torch.cat((torch.cat((rho, chi), 1),
                            torch.cat((-chi.conj(),
                                       torch.eye(rho.shape[0], dtype=rho.dtype,
                                                 device=device)-rho.conj()), 1)), 0)
        active = GeneralizedDensity(rho, chi, matrix, fermion_parity)
        hn = nambu_hamiltonian(h, delta, chemical_potential=mu)
        return ProjectedRHFBState(
            active, full.rho, full.chi, frame, q, h_aux, h, delta, mu, eta,
            normal + epair, torch.trace(full.rho).real, source,
            active_nambu_hamiltonian(hn, frame),
        )

    def residual_blocks(self, state: ProjectedRHFBState) -> ProjectedResidualBlocks:
        comm = state.active_hamiltonian @ state.active.matrix - state.active.matrix @ state.active_hamiltonian
        return ProjectedResidualBlocks(
            comm, state.eta - state.source_expectation,
            state.number - self.target_number,
        )

    def residual_vector(self, state: ProjectedRHFBState) -> torch.Tensor:
        return self.residual_blocks(state).as_vector()


def make_projected_fixedpoint_problem(
    toy: MovingToy | None = None,
    *, blocked_index: int = 5,
) -> ProjectedFixedPointProblem:
    """Use the existing MovingToy levels/force and a declared odd block seed."""
    if toy is None:
        toy, _ = make_moving_toy()
    occupations = torch.tensor([1., 1., 0., 0., 0., 0.],
                               dtype=torch.float64, device=toy.h0.device)
    reference = no_sea_branch_reference(
        torch.diagonal(toy.h0).real[:toy.rank].contiguous(), occupations,
        dirac_cutoff=toy.shift, chemical_potential=.5,
        blocked_index=blocked_index, blocking_operation="particle",
    )
    a, b = _basis_tuples(toy.rank, toy.h0.device)
    return ProjectedFixedPointProblem(toy, reference, a, b, toy.target_number)


def initial_pair_coordinates(problem: ProjectedFixedPointProblem) -> torch.Tensor:
    """Same fixed QR chart: occupied pair (0,1), empty pair avoiding block."""
    c = torch.zeros(problem.n_coordinates, dtype=torch.float64, device=problem.toy.h0.device)
    n_a = len(problem.a_basis)
    c[n_a] = .32
    occupied = torch.nonzero(problem.reference.occupied_particles).flatten().tolist()
    blocked = next(i for i in reversed(occupied) if i >= 2)
    pair_id = 14 if blocked in (2, 3) else 9
    c[n_a + 2 * pair_id] = -.30
    return c


def solve_projected_fixedpoint(problem: ProjectedFixedPointProblem,
                               initial_coordinates: torch.Tensor,
                               initial_eta_raw: torch.Tensor | float,
                               initial_mu: torch.Tensor | float,
                               *, options: StationaryOptions | None = None):
    """Solve [Ha,Ra]=0, eta-Tr(S rho)=0, N-N0=0; no energy KKT."""
    device = problem.toy.h0.device
    x0 = torch.cat((
        torch.as_tensor(initial_coordinates, dtype=torch.float64, device=device),
        torch.as_tensor(initial_eta_raw, dtype=torch.float64, device=device).reshape(1),
        torch.as_tensor(initial_mu, dtype=torch.float64, device=device).reshape(1),
    ))

    def state(x):
        return problem.state_from_coordinates(x[:-2], x[-2], x[-1])

    def residual(x):
        return problem.residual_vector(state(x))

    options = options or StationaryOptions(max_iterations=8, residual_tolerance=2e-6,
                                            initial_damping=1e-2, maximum_trial_steps=6,
                                            maximum_variables=512, maximum_residuals=512)
    result = solve_stationarity(residual, x0, options=options)
    return result, state(result.parameters)


@contextmanager
def candidate_spectral_solver_guard():
    """Count and reject eigensolver attempts on the complete candidate path."""
    names = ("eigh", "eigvalsh", "eig")
    original = {n: getattr(torch.linalg, n) for n in names}
    counts = {f"torch.linalg.{n}": 0 for n in names}

    def forbidden(name):
        def call(*_args, **_kwargs):
            counts[f"torch.linalg.{name}"] += 1
            raise AssertionError(f"candidate called torch.linalg.{name}")
        return call

    try:
        for name in names:
            setattr(torch.linalg, name, forbidden(name))
        yield counts
    finally:
        for name, fn in original.items():
            setattr(torch.linalg, name, fn)


@contextmanager
def reference_spectral_solver_counter():
    """Count exact Hermitian eigensolves used only by the independent oracle."""
    original = torch.linalg.eigh
    count = {"torch.linalg.eigh": 0}

    def counted(*args, **kwargs):
        count["torch.linalg.eigh"] += 1
        return original(*args, **kwargs)

    try:
        torch.linalg.eigh = counted
        yield count
    finally:
        torch.linalg.eigh = original


def warmup_fixed_frame(problem: ProjectedFixedPointProblem, *, max_iterations: int = 2,
                       coordinate_bound: float = 1.25):
    """Bounded fixed-frame energy descent with exact number restoration."""
    seed = initial_pair_coordinates(problem)
    n = problem.n_coordinates

    def fixed_state(coords):
        active = problem._active_from_coordinates(coords)
        full = embed_active_density(problem.toy.seed_frame, active.rho, active.chi,
                                    fermion_parity=active.fermion_parity)
        normal, h, source = problem.toy.normal(full.rho)
        epair, delta = pairing_gap(
            problem.toy.spinors, full.chi, problem.toy.positions, problem.toy.weights,
            species="neutron", chunk_size=8, force=D1S,
        )
        return active, full, normal + epair, source

    def number_correct(coords):
        trial = coords.detach().clone()
        for _ in range(20):
            with torch.enable_grad():
                x = trial.detach().requires_grad_(True)
                _active, full, _energy, _source = fixed_state(x)
                err = torch.trace(full.rho).real - problem.target_number
                if abs(float(err.detach())) <= 2e-11:
                    return trial, True
                grad_n = torch.autograd.grad(err, x)[0].detach()
            norm2 = torch.dot(grad_n, grad_n)
            if float(norm2) <= 1e-16:
                return trial, False
            trial = (trial - err.detach() * grad_n / norm2).clamp(
                -coordinate_bound, coordinate_bound)
        return trial, False

    coords, exact_n = number_correct(seed)
    history = []
    reason = "iteration_limit"
    for iteration in range(max_iterations):
        with torch.enable_grad():
            x = coords.detach().requires_grad_(True)
            _active, full, energy, _source = fixed_state(x)
            nerr = torch.trace(full.rho).real - problem.target_number
            grad_e = torch.autograd.grad(energy, x, retain_graph=True)[0].detach()
            grad_n = torch.autograd.grad(nerr, x)[0].detach()
        norm_n = torch.dot(grad_n, grad_n)
        if float(norm_n) <= 1e-16:
            reason = "number_tangent_singular"
            break
        # Orthogonally remove the number-changing component of the descent.
        descent = -grad_e + (torch.dot(grad_e, grad_n) / norm_n) * grad_n
        dmax = float(descent.abs().max())
        energy_before = float(energy.detach())
        if dmax <= 1e-12:
            reason = "projected_gradient_small"
            history.append({"iteration": iteration, "energy": energy_before,
                            "number_abs": abs(float(nerr.detach())),
                            "projected_gradient_max": dmax, "accepted": False})
            break
        descent = descent / dmax
        accepted = False
        alpha = .08
        for trial_id in range(12):
            proposal = (coords + alpha * descent).clamp(-coordinate_bound, coordinate_bound)
            proposal, restored = number_correct(proposal)
            if not restored:
                alpha *= .5
                continue
            _a_new, _f_new, energy_new, _s_new = fixed_state(proposal)
            if float(energy_new.detach()) < energy_before - 1e-12:
                coords = proposal.detach()
                accepted = True
                history.append({"iteration": iteration, "energy": float(energy_new.detach()),
                                "number_abs": abs(float(
                                    (torch.trace(_f_new.rho).real-problem.target_number).detach())),
                                "step": alpha, "line_search_trials": trial_id+1,
                                "accepted": True})
                break
            alpha *= .5
        if not accepted:
            reason = "no_energy_decrease"
            history.append({"iteration": iteration, "energy": energy_before,
                            "number_abs": abs(float(nerr.detach())),
                            "accepted": False})
            break
    active, full, energy, source = fixed_state(coords)
    number_error = torch.trace(full.rho).real - problem.target_number
    eta_raw = torch.atanh((source / 16.).clamp(-.99, .99))
    initial = problem.state_from_coordinates(coords, eta_raw,
                                             torch.tensor(.45, dtype=torch.float64,
                                                          device=coords.device))
    class WarmupResult:
        pass
    result = WarmupResult()
    result.coordinates = coords.detach()
    result.parameters = coords.detach()
    result.energy = energy.detach()
    result.number_error = number_error.detach()
    result.residual = number_error.detach().reshape(1)
    result.exact_number_restored = bool(exact_n and abs(float(number_error.detach())) <= 2e-9)
    result.stop_reason = reason
    result.history = history
    result.bounded = True
    result.energy_decreased = bool(history and any(row.get("accepted") for row in history))
    return result, initial


def reference_positive_bdg_state(problem: ProjectedFixedPointProblem,
                                 h_field: torch.Tensor, delta: torch.Tensor,
                                 frame: torch.Tensor, mu: float | torch.Tensor):
    """Exact reference-only BdG diagonalization with PH-pair/block tracking."""
    from .branch import select_ph_branches_reference_only
    ham = nambu_hamiltonian(h_field, delta, chemical_potential=mu)
    ha = active_nambu_hamiltonian(ham, frame)
    selected_frame, branch_metrics = select_ph_branches_reference_only(
        ha, problem.reference.frame
    )
    projector = selected_frame @ selected_frame.mH
    blocked = max(i for i, occ in enumerate(
        problem.reference.occupied_particles.tolist()) if occ and i >= 2)
    blocked_vector = problem.reference.frame[:, blocked]
    branch_metrics = dict(branch_metrics)
    branch_metrics["blocked_reference_mode_overlap"] = float(
        (blocked_vector.conj() @ projector @ blocked_vector).real
    )
    branch_metrics["blocked_reference_index"] = blocked
    d = frame.shape[1]
    rho, chi = projector[:d, :d], projector[:d, d:]
    active = GeneralizedDensity(rho, chi, projector,
                               problem.reference.density.fermion_parity)
    full = embed_active_density(frame, rho, chi,
                                fermion_parity=active.fermion_parity)
    return active, full, branch_metrics


def _reference_positive_frame(toy: MovingToy, h: torch.Tensor):
    """Reference-only exact positive h subspace and transported seed chart."""
    values, vectors = torch.linalg.eigh(h)
    positive = vectors[:, values > toy.shift]
    projector = positive @ positive.mH
    frame = positive_frame_from_projector(projector, toy.seed_frame,
                                          projector_tolerance=None)
    return frame, projector, values


def run_projected_fixedpoint_benchmark(*, warmup_iterations: int = 2,
                                       root_iterations: int = 2,
                                       reference_iterations: int = 2) -> dict:
    """One bounded model run plus an independent eigensolver reference path."""
    toy, _base_ref = make_moving_toy()
    # Screen odd seeds by their common fixed-frame energy at exact N before
    # spending the single bounded warmup on the lowest seed. This is a branch
    # screen, not a global odd-nucleus energy minimization.
    screened = []
    for block in (2, 3, 4, 5):
        problem_i = make_projected_fixedpoint_problem(toy, blocked_index=block)
        # Compare equal-N Slater blocked seeds before adding the common paired
        # warmup perturbation. Every branch has exactly N0 at this screen.
        seed = torch.zeros(problem_i.n_coordinates, dtype=torch.float64,
                           device=toy.h0.device)
        active = problem_i._active_from_coordinates(seed)
        full = embed_active_density(toy.seed_frame, active.rho, active.chi,
                                    fermion_parity=active.fermion_parity)
        normal, _h, _x = toy.normal(full.rho)
        epair, _delta = pairing_gap(toy.spinors, full.chi, toy.positions, toy.weights,
                                    species="neutron", chunk_size=8, force=D1S)
        screened.append((problem_i, seed, normal + epair, block))
    chosen, chosen_seed, _seed_energy, block = min(
        screened, key=lambda item: float(item[2].detach())
        if bool(torch.isfinite(item[2])) else float("inf")
    )
    with candidate_spectral_solver_guard() as warmup_calls:
        warm, warm_state = warmup_fixed_frame(chosen, max_iterations=warmup_iterations)
    eta_raw = torch.atanh((warm_state.eta / 16.).clamp(-.99, .99))
    warm_coordinates = warm.coordinates
    with candidate_spectral_solver_guard() as calls:
        root, candidate = solve_projected_fixedpoint(
            chosen, warm_coordinates,
            eta_raw, warm_state.mu,
            options=StationaryOptions(max_iterations=root_iterations,
                                      residual_tolerance=2e-6,
                                      initial_damping=1e-2,
                                      maximum_trial_steps=6,
                                      maximum_variables=512,
                                      maximum_residuals=512),
        )
    candidate_calls = {key: warmup_calls[key] + calls[key] for key in calls}
    # Reference diagnostics: exact h projector + conventional BdG branch solve.
    frame = toy.seed_frame
    rho_a, chi_a = warm_state.active.rho.detach(), warm_state.active.chi.detach()
    ref_history = []
    ref_data = None
    mu_ref = float(warm_state.mu.detach())
    with reference_spectral_solver_counter() as reference_eig_calls:
      for it in range(reference_iterations):
        full = embed_active_density(frame, rho_a, chi_a,
                                    fermion_parity=chosen.reference.density.fermion_parity)
        _en, h, _x = toy.normal(full.rho)
        frame, pexact, evals = _reference_positive_frame(toy, h)
        _pair, delta = pairing_gap(toy.spinors, full.chi, toy.positions, toy.weights,
                                   species="neutron", chunk_size=8, force=D1S)
        # Exact-number chemical-potential correction for the reference only.
        low, high = -4.0, 4.0
        best = None
        for _ in range(16):
            trial_mu = .5 * (low + high)
            active_try, full_try, branch_try = reference_positive_bdg_state(
                chosen, h, delta, frame, trial_mu
            )
            n_try = float(torch.trace(full_try.rho).real)
            best = (active_try, full_try, branch_try, trial_mu, n_try)
            if n_try > toy.target_number:
                high = trial_mu
            else:
                low = trial_mu
        active, ref_full, branch, mu_ref, n_ref = best
        new_energy = toy.normal(ref_full.rho)[0] + pairing_gap(
            toy.spinors, ref_full.chi, toy.positions, toy.weights,
            species="neutron", chunk_size=8, force=D1S,
        )[0]
        _e_new, h_new, _x_new = toy.normal(ref_full.rho)
        _pair_new, delta_new = pairing_gap(
            toy.spinors, ref_full.chi, toy.positions, toy.weights,
            species="neutron", chunk_size=8, force=D1S,
        )
        rho_change = float(torch.linalg.matrix_norm(active.rho-rho_a).detach())
        chi_change = float(torch.linalg.matrix_norm(active.chi-chi_a).detach())
        normal_field_change = float(torch.linalg.matrix_norm(h_new-h).detach())
        pairing_field_change = float(torch.linalg.matrix_norm(delta_new-delta).detach())
        ref_data = (active, ref_full, h, delta, branch, pexact, evals)
        ref_history.append({"iteration": it, "number": n_ref,
                            "number_abs": abs(n_ref-toy.target_number),
                            "energy": float(new_energy.detach()),
                            "density_change_rho": rho_change,
                            "density_change_chi": chi_change,
                            "normal_field_selfconsistency": normal_field_change,
                            "pairing_field_selfconsistency": pairing_field_change,
                            "branch": branch})
        rho_a, chi_a = .5*(rho_a+active.rho), .5*(chi_a+active.chi)
    blocks = chosen.residual_blocks(candidate)
    warm_summaries = [{
        "blocked_index": idx,
        "fixed_frame_seed_energy": float(energy.detach()),
        "fixed_frame_seed_number": toy.target_number,
        "selected_for_warmup": idx == block,
    } for _p, _seed, energy, idx in screened]
    if ref_data is not None:
        ar, fr, hr, dr, br, pr, er = ref_data
        ref_report = {
            "history": ref_history,
            "rho": {"real": fr.rho.real.tolist(), "imag": fr.rho.imag.tolist()},
            "chi": {"real": fr.chi.real.tolist(), "imag": fr.chi.imag.tolist()},
            "h_field": {"real": hr.real.tolist(), "imag": hr.imag.tolist()},
            "delta": {"real": dr.real.tolist(), "imag": dr.imag.tolist()},
            "projector": {"real": pr.real.tolist(), "imag": pr.imag.tolist()},
            "branch_metrics": br, "mu": mu_ref,
            "chemical_potential_update": (
                "diagnostic bisection over [-4,4] for the fixed initial reference-frame "
                "overlap-selected PH branch; monotonicity is unproved"
            ),
            "branch_continuation": (
                "not implemented: branch selection reuses problem.reference.frame instead "
                "of transporting the previous quasiframe"
            ),
            "reference_eigenvalues": er.tolist(),
            "number": float(torch.trace(fr.rho).real),
            "number_abs": abs(float(torch.trace(fr.rho).real)-toy.target_number),
            "purity_abs_frobenius": float(torch.linalg.matrix_norm(fr.matrix@fr.matrix-fr.matrix)),
            "converged": bool(ref_history and
                              ref_history[-1]["number_abs"] < 2e-6 and
                              ref_history[-1]["normal_field_selfconsistency"] < 2e-6 and
                              ref_history[-1]["pairing_field_selfconsistency"] < 2e-6 and
                              ref_history[-1]["density_change_rho"] < 2e-6 and
                              ref_history[-1]["density_change_chi"] < 2e-6 and
                              ref_history[-1]["branch"]["minimum_overlap_margin"] > 1e-4 and
                              ref_history[-1]["branch"]["blocked_reference_mode_overlap"] > .5),
        }
    else:
        ref_report = {"history": ref_history, "failure": "no reference step"}
    return {
        "scope": "MovingToy finite model; not production odd PKA1 RHFB",
        "equations": ["[H_a,R_a]=0", "eta-Tr(S rho)=0", "Tr(rho)-N0=0"],
        "model": {
            "normal_level_scale": .10, "normal_coupling": toy.coupling,
            "N0": toy.target_number, "D1S_strength": D1S.strength,
            "D1S_provenance": D1S_PROVENANCE, "selected_block": block,
            "h0": {"real": toy.h0.real.tolist(), "imag": toy.h0.imag.tolist()},
            "source": {"real": toy.source.real.tolist(), "imag": toy.source.imag.tolist()},
            "spinors": {"real": toy.spinors.real.tolist(), "imag": toy.spinors.imag.tolist()},
            "positions": toy.positions.tolist(),
            "weights": toy.weights.tolist(),
            "pairing_parameters": {
                "ranges_fm": list(D1S.ranges_fm), "w_mev": list(D1S.w_mev),
                "b_mev": list(D1S.b_mev), "h_mev": list(D1S.h_mev),
                "m_mev": list(D1S.m_mev),
            },
            "warmup_coordinate_bound": 1.25,
            "candidate_projector": "rational sign + fixed-seed QR; no eigensolver",
            "reference": "exact Python eigensolve of h and BdG; overlap branch tracking",
        },
        "branch_screen": warm_summaries,
        "warmup_exact_number_restored": warm.exact_number_restored,
        "warmup_energy_decreased": warm.energy_decreased,
        "warmup_stop_reason": warm.stop_reason,
        "warmup_residual_abs_max": float(warm.residual.abs().max()),
        "warmup_history": warm.history,
        "candidate": {
            "converged": root.converged, "stop_reason": root.stop_reason,
            "residual_abs_max": float(root.residual.abs().max()),
            "stationarity_frobenius": float(torch.linalg.matrix_norm(blocks.stationarity).detach()),
            "field_abs": abs(float(blocks.field_self_consistency.detach())),
            "number_abs": abs(float(blocks.number.detach())),
            "candidate_eigensolver_calls": candidate_calls,
            "blocked_reference_mode_overlap": float((
                chosen.reference.frame[:, block].conj()
                @ candidate.active.matrix @ chosen.reference.frame[:, block]
            ).real.detach()),
            "active_R_purity_frobenius": float(torch.linalg.matrix_norm(
                candidate.active.matrix @ candidate.active.matrix - candidate.active.matrix
            ).detach()),
            "bare_negative_basis_density_support": float(
                torch.linalg.matrix_norm(candidate.rho[6:, :]).detach()
            ),
            "residual_vector_length": int(root.residual.numel()),
            "stationarity_abs_max": float(blocks.stationarity.abs().max().detach()),
            "history": root.history,
        },
        "candidate_final": {
            "coordinates": root.parameters.detach().cpu().tolist(),
            "rho": {"real": candidate.rho.real.tolist(), "imag": candidate.rho.imag.tolist()},
            "chi": {"real": candidate.chi.real.tolist(), "imag": candidate.chi.imag.tolist()},
            "frame": {"real": candidate.frame.real.tolist(), "imag": candidate.frame.imag.tolist()},
            "projector": {"real": candidate.projector.real.tolist(), "imag": candidate.projector.imag.tolist()},
            "h_aux": {"real": candidate.h_aux.real.tolist(), "imag": candidate.h_aux.imag.tolist()},
            "h_field": {"real": candidate.h_field.real.tolist(), "imag": candidate.h_field.imag.tolist()},
            "delta": {"real": candidate.delta.real.tolist(), "imag": candidate.delta.imag.tolist()},
            "mu": float(candidate.mu.detach()), "eta": float(candidate.eta.detach()),
            "energy": float(candidate.energy.detach()), "number": float(candidate.number.detach()),
        },
        "reference": ref_report,
        "reference_eigensolver_calls": reference_eig_calls,
        "comparison_scope": (
            "candidate and reference target the same projected-nucleon-space fixed-point "
            "equations; reference diagonalizes h/BdG only to construct the branch state. "
            "Agreement validates this finite prescription implementation only, not its "
            "equivalence to full Dirac-quasiparticle no-sea RHFB or the earlier energy-KKT model."
        ),
        "reference_followup_required": (
            "continue the previous quasiframe or impose explicit occupation labels, then "
            "verify N(mu) bracket behavior; do not assume monotonicity for this overlap-selected branch"
        ),
        "execution_attempts": [
            {
                "attempt": 1,
                "outcome": "candidate warmup/root ran, then reference path raised ImportError "
                           "for a helper in the wrong module; no complete artifact was written",
            },
            {
                "attempt": 2,
                "outcome": "same bounded settings completed and wrote the complete artifact",
            },
        ],
        "solver_settings": {
            "warmup_iterations": warmup_iterations,
            "root_iterations": root_iterations,
            "reference_iterations": reference_iterations,
            "warmup_number_tolerance": 2e-11,
            "warmup_coordinate_bound": 1.25,
            "candidate_residual_tolerance": 2e-6,
            "candidate_rational_order": 121,
            "candidate_rational_solve_tolerance": 2e-10,
        },
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_projected_fixedpoint_benchmark(), indent=2))
