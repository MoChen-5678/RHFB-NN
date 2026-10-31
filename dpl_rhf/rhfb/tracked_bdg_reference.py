"""Reference-only projected RHFB solve with tracked blocked BdG branches.

This module is an independent tiny-model oracle.  It is not imported by the
candidate/training path.  Every density update is a rank-D Nambu projector;
only the one-body and pairing fields may be linearly mixed.

The blocking convention follows Li et al., Phys. Lett. B 732 (2014) 266,
arXiv:1203.1363, Eqs. (4)-(5): an odd state is a one-quasiparticle excitation
and blocking exchanges the corresponding creation/annihilation operators.
Geng et al., Phys. Rev. C 105, 014321 (2022), arXiv:2205.01329, Sec. III A,
compare orbital-fixed and lowest-quasiparticle automatic blocking, and note
near-degenerate convergence difficulty. This implementation tracks only the
designated blocked quasiparticle by overlap; overlap of the occupied frame is
used for gauge alignment after the physical projector is chosen. That
continuation detail is an algorithmic choice for this finite reference, not a
unique prescription asserted by either paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from itertools import permutations
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import torch

from .branch import nambu_hamiltonian, no_sea_branch_reference, particle_hole_conjugate
from .moving_particle_space_prototype import MovingToy, make_moving_toy
from .no_sea_particle_space import embed_active_density, positive_frame_from_projector
from .pairing import D1S, pairing_gap
from .parity_diagnostic import number_parity_from_density
from .projected_rhfb_fixedpoint import (
    initial_pair_coordinates, make_projected_fixedpoint_problem,
)


@dataclass(frozen=True)
class TrackedBdGState:
    mu: float
    active_frame: torch.Tensor
    quasiframe: torch.Tensor
    blocked_qp: torch.Tensor
    projector: torch.Tensor
    rho: torch.Tensor
    chi: torch.Tensor
    h_field: torch.Tensor
    delta: torch.Tensor
    energy: torch.Tensor
    number: torch.Tensor
    metrics: dict[str, Any]


class ParityBranchError(RuntimeError):
    """Reference branch rejected after preserving its constructed R for audit."""

    def __init__(self, message: str, partial_state: dict[str, Any]):
        super().__init__(message)
        self.partial_state = partial_state


def _best_assignment(scores: torch.Tensor) -> tuple[list[int], float, float]:
    """Exact maximum-overlap assignment; dimensions are at most six here."""
    d = scores.shape[0]
    best_value, second = -1.0, -1.0
    best: tuple[int, ...] | None = None
    for candidate in permutations(range(d)):
        value = sum(float(scores[i, candidate[i]]) for i in range(d))
        if value > best_value:
            second, best_value, best = best_value, value, candidate
        elif value > second:
            second = value
    assert best is not None
    return list(best), best_value / d, max(0.0, (best_value-second) / d)


def _monotone_orientation(samples: list[tuple[float, float]], *,
                          tolerance: float = 2e-5) -> int:
    """Return +1/-1 for sampled monotone N(mu); reject reversals/flat curves."""
    ordered = sorted(samples, key=lambda item: item[0])
    slopes = [b[1]-a[1] for a, b in zip(ordered, ordered[1:])
              if abs(b[1]-a[1]) > tolerance]
    if not slopes:
        raise RuntimeError("tracked N(mu) is flat over the sampled interval")
    orientation = 1 if slopes[0] > 0 else -1
    if any(orientation*delta_n < -tolerance for delta_n in slopes):
        raise RuntimeError("tracked N(mu) is not monotone on the sampled bracket")
    return orientation


@contextmanager
def reference_spectral_counter():
    """Count exact reference spectral calls and restore the torch API safely."""
    names = ("eigh", "eigvalsh", "eig")
    original = {name: getattr(torch.linalg, name) for name in names}
    counts = {f"torch.linalg.{name}": 0 for name in names}
    try:
        for name in names:
            base = original[name]
            def counted(*args, _name=name, _base=base, **kwargs):
                counts[f"torch.linalg.{_name}"] += 1
                return _base(*args, **kwargs)
            setattr(torch.linalg, name, counted)
        yield counts
    finally:
        for name, function in original.items():
            setattr(torch.linalg, name, function)


def _positive_frame_reference(toy: MovingToy, h: torch.Tensor):
    values, vectors = torch.linalg.eigh(h)
    positive = vectors[:, values > toy.shift]
    projector = positive @ positive.mH
    frame = positive_frame_from_projector(projector, toy.seed_frame,
                                          projector_tolerance=None)
    gap = float(torch.min(torch.abs(values - toy.shift)))
    return frame, projector, values, gap


def _initial_quasiframe(problem, coords: torch.Tensor) -> torch.Tensor:
    """Extract and align the pure chart seed in the initial Nambu basis."""
    active = problem._active_from_coordinates(coords)
    values, vectors = torch.linalg.eigh(active.matrix)
    frame = vectors[:, values > .5]
    if frame.shape[1] != problem.toy.rank:
        raise RuntimeError("canonical seed does not define a rank-D pure projector")
    overlap = torch.abs(problem.reference.frame.mH @ frame).square()
    assignment, _score, _margin = _best_assignment(overlap)
    return frame[:, assignment]


def select_odd_bdg_projector(h_active: torch.Tensor, block_tag: torch.Tensor,
                             *, previous_blocked_qp: torch.Tensor | None = None,
                             previous_quasiframe: torch.Tensor | None = None,
                             minimum_block_overlap: float = .20):
    """Choose the unblocked vacuum plus exactly one designated qp flip.

    The unblocked projected nucleon-space vacuum is the negative BdG spectral
    subspace in this package's generalized-density convention. Its Pfaffian
    parity is measured. If odd, its lowest-energy qp is toggled to define a
    fixed-field even reference; only then is the designated blocked qp toggled.
    Relative to the unconstrained vacuum, the final state can therefore differ
    by zero, one, or two PH-pair toggles. The block tag identifies that qp
    initially; later overlap follows only that qp. ``previous_quasiframe`` is
    used for gauge alignment after the physical projector is fixed, never to
    choose unblocked PH polarizations.
    """
    h_active = torch.as_tensor(h_active)
    d = h_active.shape[0] // 2
    if h_active.dtype != torch.complex128 or h_active.shape != (2*d, 2*d):
        raise TypeError("h_active must be a square complex128 BdG matrix")
    tag = torch.as_tensor(block_tag, dtype=torch.complex128, device=h_active.device)
    if tag.shape != (d,) or float(torch.linalg.vector_norm(tag)) <= 1e-12:
        raise ValueError("block_tag must be a nonzero active particle-space vector")
    tag = tag / torch.linalg.vector_norm(tag)
    values, vectors = torch.linalg.eigh(h_active)
    scale = max(1.0, float(torch.max(torch.abs(values))))
    gap = float(torch.min(torch.abs(values)))
    if gap <= 1e-9*scale:
        raise RuntimeError("quasiparticle gap closed; branch continuation is undefined")
    positive = [int(i) for i in torch.nonzero(values > 0).flatten().tolist()]
    negative_indices = [int(i) for i in torch.nonzero(values < 0).flatten().tolist()]
    negative = set(negative_indices)
    if len(positive) != d or len(negative) != d:
        raise RuntimeError("BdG spectrum is not split into D positive and D negative states")
    pairs: list[tuple[int, int, float]] = []
    for ip in positive:
        ph = particle_hole_conjugate(vectors[:, ip])
        im = max(negative, key=lambda j: float(torch.abs(torch.vdot(vectors[:, j], ph))))
        negative.remove(im)
        if abs(float(values[im] + values[ip])) > 1e-7*scale:
            raise RuntimeError("BdG particle-hole partner mismatch")
        pairs.append((im, ip, float(torch.abs(torch.vdot(vectors[:, im], ph)))))

    if previous_blocked_qp is None:
        scores = [float(torch.abs(torch.vdot(tag, vectors[:d, ip])).square())
                  for _im, ip, _ph_overlap in pairs]
        block_reason = "initial_particle_tag_overlap"
    else:
        prior = torch.as_tensor(previous_blocked_qp, dtype=torch.complex128,
                                device=h_active.device)
        if prior.shape != (2*d,):
            raise ValueError("previous_blocked_qp must be one active Nambu vector")
        scores = [float(torch.abs(torch.vdot(prior, vectors[:, ip])).square())
                  for _im, ip, _ph_overlap in pairs]
        block_reason = "designated_blocked_qp_continuation_overlap"
    order = sorted(range(d), key=lambda j: scores[j], reverse=True)
    if len(order) > 1 and scores[order[0]]-scores[order[1]] < 1e-8:
        raise RuntimeError("blocked quasiparticle identity is ambiguous")
    block_pair = order[0]
    block_score = scores[block_pair]
    if block_score < minimum_block_overlap:
        raise RuntimeError(f"blocked quasiparticle overlap too small ({block_score:.3f})")
    blocked_negative, blocked_positive, _ph_overlap = pairs[block_pair]

    # Compute the parity of the actual all-negative annihilator vacuum. In a
    # time-odd field it need not be even. If it is odd, define the even
    # reference by toggling its lowest positive-energy qp; then perform the
    # designated blocking toggle. This yields the requested odd state without
    # silently assuming the vacuum's parity.
    def parity_for_indices(indices: set[int]):
        occupied = torch.stack([vectors[:, i] for i in sorted(indices)], dim=1)
        r = occupied @ occupied.mH
        return number_parity_from_density(r[:d, :d].detach().cpu().numpy(),
                                          r[:d, d:].detach().cpu().numpy(),
                                          tolerance=1e-8)

    vacuum_indices = set(negative_indices)
    vacuum_parity = parity_for_indices(vacuum_indices)
    even_reference_indices = set(vacuum_indices)
    even_reference_compensation_pair = None
    if vacuum_parity["number_parity"] != 1:
        even_reference_compensation_pair = min(range(d), key=lambda j: float(values[pairs[j][1]]))
        im, ip, _ = pairs[even_reference_compensation_pair]
        even_reference_indices.remove(im)
        even_reference_indices.add(ip)
    even_reference_parity = parity_for_indices(even_reference_indices)
    if even_reference_parity["number_parity"] != 1:
        raise RuntimeError(
            "the chosen fixed-field even reference failed its independent parity check: "
            f"{even_reference_parity['number_parity']}"
        )
    selected_indices = set(even_reference_indices)
    if blocked_negative in selected_indices:
        selected_indices.remove(blocked_negative)
        selected_indices.add(blocked_positive)
    else:
        selected_indices.remove(blocked_positive)
        selected_indices.add(blocked_negative)
    selected_parity = parity_for_indices(selected_indices)
    selected_vectors = [vectors[:, i] for i in sorted(selected_indices)]
    selected = torch.stack(selected_vectors, dim=1)

    # Procrustes alignment is a unitary gauge choice inside the selected range.
    gauge_min_sv = 1.0
    if previous_quasiframe is not None:
        previous = torch.as_tensor(previous_quasiframe, dtype=torch.complex128,
                                   device=h_active.device)
        if previous.shape != selected.shape:
            raise ValueError("previous_quasiframe has wrong shape")
        left, singular, right_h = torch.linalg.svd(selected.mH @ previous)
        selected = selected @ (left @ right_h)
        gauge_min_sv = float(torch.min(singular))
    projector = selected @ selected.mH
    rho, chi = projector[:d, :d], projector[:d, d:]
    pair_diagnostics = [{"E_positive": float(values[ip]),
                         "E_negative": float(values[im]),
                         "designated_block": j == block_pair,
                         "selected_positive_member": ip in selected_indices,
                         "even_reference_compensation": j == even_reference_compensation_pair,
                         "ph_pair_overlap": ph_overlap,
                         "block_tag_or_track_score": scores[j]}
                        for j, (im, ip, ph_overlap) in enumerate(pairs)]
    metrics = {"bdg_gap": gap, "pair_diagnostics": pair_diagnostics,
               "blocked_pair_index": block_pair,
               "blocked_qp_selection_reason": block_reason,
               "blocked_qp_overlap": block_score,
               "basis_gauge_min_principal_overlap": gauge_min_sv,
               "positive_branches_occupied": len(selected_indices.intersection(positive)),
               "vacuum_parity": vacuum_parity,
               "computed_even_reference_parity": even_reference_parity,
               "vacuum_positive_branches_occupied": 0,
               "even_reference_positive_branches_occupied": len(
                   even_reference_indices.intersection(positive)),
               "even_reference_compensation_pair": even_reference_compensation_pair,
               "computed_selected_parity": selected_parity,
               "projector_purity_abs_frobenius": float(
                   torch.linalg.matrix_norm(projector@projector-projector)),
               "rho": _json_matrix(rho), "chi": _json_matrix(chi),
               "projector": _json_matrix(projector),
               "quasiframe": _json_matrix(selected),
               "blocked_qp": _json_matrix(vectors[:, blocked_positive].reshape(-1,1))}
    return selected, projector, rho, chi, vectors[:, blocked_positive], metrics


def _tracked_bdg_state(problem, h: torch.Tensor, delta: torch.Tensor,
                       active_frame: torch.Tensor, mu: float,
                       previous_quasiframe: torch.Tensor,
                       block_tag: torch.Tensor,
                       previous_blocked_qp: torch.Tensor | None = None,
                       *, minimum_block_overlap: float = .20) -> TrackedBdGState:
    """Reference state: vacuum occupation rule plus exactly one blocked qp."""
    d = active_frame.shape[1]
    h_nambu = nambu_hamiltonian(h, delta, chemical_potential=mu)
    q = torch.block_diag(active_frame, active_frame.conj())
    h_active = q.mH @ h_nambu @ q
    selected, projector, rho, chi, blocked_qp, metrics = select_odd_bdg_projector(
        h_active, block_tag, previous_blocked_qp=previous_blocked_qp,
        previous_quasiframe=previous_quasiframe,
        minimum_block_overlap=minimum_block_overlap,
    )
    parity = number_parity_from_density(rho.detach().cpu().numpy(),
                                        chi.detach().cpu().numpy(), tolerance=1e-8)
    metrics["computed_parity"] = parity
    metrics["independent_density_parity"] = parity
    full = embed_active_density(active_frame, rho, chi,
                                fermion_parity=problem.reference.density.fermion_parity)
    normal, h_target, _source = problem.toy.normal(full.rho)
    epair, delta_target = pairing_gap(
        problem.toy.spinors, full.chi, problem.toy.positions, problem.toy.weights,
        species="neutron", chunk_size=8, force=D1S,
    )
    purity = float(torch.linalg.matrix_norm(projector@projector-projector))
    metrics.update({"purity_abs_frobenius": purity,
                    "particle_number": float(torch.trace(full.rho).real),
                    "number_error_abs": abs(float(torch.trace(full.rho).real)
                                             - problem.target_number),
                    "energy": float((normal+epair).detach()),
                    "normal_field_residual_fro": float(torch.linalg.matrix_norm(h_target-h)),
                    "pairing_field_residual_fro": float(torch.linalg.matrix_norm(delta_target-delta)),
                    "field_residual_max": max(float(torch.linalg.matrix_norm(h_target-h)),
                                              float(torch.linalg.matrix_norm(delta_target-delta))),
                    "active_frame": _json_matrix(active_frame),
                    "h_field_used": _json_matrix(h), "delta_used": _json_matrix(delta),
                    "h_field_target": _json_matrix(h_target),
                    "delta_target": _json_matrix(delta_target),
                    "full_rho": _json_matrix(full.rho),
                    "full_chi": _json_matrix(full.chi)})
    if parity["number_parity"] != -1:
        raise ParityBranchError(
            f"constructed blocked RHFB projector has parity {parity['number_parity']}, expected -1",
            metrics,
        )
    return TrackedBdGState(
        mu, active_frame, selected, blocked_qp, projector, rho, chi, h, delta,
        normal+epair, torch.trace(full.rho).real, metrics,
    )


def _number_state(problem, h: torch.Tensor, delta: torch.Tensor,
                  active_frame: torch.Tensor, anchor: TrackedBdGState,
                  block_tag: torch.Tensor,
                  *, target: float, tolerance: float, half_width: float,
                  continuation_steps: int, max_bisections: int,
                  history: list[dict[str, Any]], round_index: int):
    """Find μ only after a branch-preserving, overlap-checked number bracket."""
    center = anchor.mu
    lo_mu, hi_mu = center-half_width, center+half_width

    def continue_to(mu: float, start: TrackedBdGState, side: str):
        distance = abs(mu-start.mu)
        steps = max(1, int(continuation_steps * distance / max(2*half_width, 1e-12)))
        current = start
        for j in range(1, steps+1):
            trial_mu = start.mu + (mu-start.mu)*j/steps
            try:
                state = _tracked_bdg_state(
                    problem, h, delta, active_frame, trial_mu,
                    current.quasiframe, block_tag, current.blocked_qp,
                )
            except ParityBranchError as exc:
                history.append({"round": round_index, "phase": "rejected_parity_state",
                                "side": side, "mu": trial_mu,
                                "partial_state": exc.partial_state})
                raise
            history.append({"round": round_index, "phase": "mu_continue",
                            "side": side, "mu": trial_mu,
                            "number": float(state.number),
                            **state.metrics})
            current = state
        return current

    history_start = len(history)
    left = continue_to(lo_mu, anchor, "left")
    right = continue_to(hi_mu, anchor, "right")
    nlo, nhi = float(left.number), float(right.number)
    bracketed = (nlo-target)*(nhi-target) <= 0
    if not bracketed:
        raise RuntimeError(
            f"tracked μ interval did not bracket N={target}: N(lo)={nlo:.6f}, N(hi)={nhi:.6f}"
        )
    ordered_samples = [(anchor.mu, float(anchor.number))]
    ordered_samples.extend((float(row["mu"]), float(row["number"]))
                           for row in history[history_start:])
    orientation = _monotone_orientation(ordered_samples)
    low, high = (left, right) if left.mu < right.mu else (right, left)
    # Retain samples proving the local monotone trend at endpoints and midpoint.
    mid = continue_to(.5*(low.mu+high.mu), low, "bracket_mid")
    if not (min(float(low.number), float(high.number))-2e-5 <= float(mid.number) <=
            max(float(low.number), float(high.number))+2e-5):
        raise RuntimeError("continued midpoint violates observed N(mu) monotonicity")
    if (float(mid.number)-target)*(float(low.number)-target) <= 0:
        high = mid
    else:
        low = mid
    best = min((left, right, mid), key=lambda s: abs(float(s.number)-target))
    for iteration in range(max_bisections):
        if abs(float(best.number)-target) <= tolerance:
            return best, {"bracketed": True, "monotone_samples": True,
                          "number_orientation_vs_mu": orientation,
                          "number_error": float(best.number)-target,
                          "bisections": iteration,
                          "initial_interval": [lo_mu, hi_mu],
                          "endpoint_numbers": [nlo, nhi]}
        trial_mu = .5*(low.mu+high.mu)
        nearest = low if abs(trial_mu-low.mu) <= abs(high.mu-trial_mu) else high
        trial = continue_to(trial_mu, nearest, "bisection")
        ntrial = float(trial.number)
        if not (min(float(low.number), float(high.number))-2e-5 <= ntrial <=
                max(float(low.number), float(high.number))+2e-5):
            raise RuntimeError("continued bisection sample violates observed N(mu) monotonicity")
        if (float(low.number)-target)*(ntrial-target) <= 0:
            high = trial
        else:
            low = trial
        if abs(ntrial-target) < abs(float(best.number)-target):
            best = trial
    return best, {"bracketed": True, "monotone_samples": True,
                  "number_orientation_vs_mu": orientation,
                  "number_error": float(best.number)-target,
                  "bisections": max_bisections,
                  "initial_interval": [lo_mu, hi_mu],
                  "endpoint_numbers": [nlo, nhi], "bisection_limit": True}


def _json_matrix(matrix: torch.Tensor) -> dict[str, Any]:
    return {"real": matrix.detach().cpu().real.tolist(),
            "imag": matrix.detach().cpu().imag.tolist()}


def run_tracked_bdg_reference(*, max_iterations: int = 8,
                              field_mixing: float = .30,
                              number_tolerance: float = 2e-4,
                              field_tolerance: float = 2e-3,
                              mu_half_width: float = 2.0,
                              continuation_steps: int = 10,
                              max_bisections: int = 12) -> dict[str, Any]:
    """One bounded reference-only solve; no scans or density mixing."""
    if not 1 <= max_iterations <= 50:
        raise ValueError("max_iterations must be in [1, 50]")
    if not 0 < field_mixing <= 1:
        raise ValueError("field_mixing must lie in (0,1]")
    history: list[dict[str, Any]] = []
    round_states: list[dict[str, Any]] = []
    counter = reference_spectral_counter()
    eig_counts = counter.__enter__()

    stop_reason = "iteration_limit"
    final: TrackedBdGState | None = None
    final_number_bracket: dict[str, Any] | None = None
    toy = None
    try:
        toy, _ = make_moving_toy()
        problem = make_projected_fixedpoint_problem(toy, blocked_index=2)
        coords = initial_pair_coordinates(problem)
        anchor_quasiframe = _initial_quasiframe(problem, coords)
        active_seed = problem._active_from_coordinates(coords)
        full_seed = embed_active_density(toy.seed_frame, active_seed.rho,
                                         active_seed.chi, fermion_parity=-1)
        _e, h_seed, _s = toy.normal(full_seed.rho)
        _ep, delta_seed = pairing_gap(
            toy.spinors, full_seed.chi, toy.positions, toy.weights,
            species="neutron", chunk_size=8, force=D1S,
        )
        h_input, delta_input, mu = h_seed, delta_seed, .45
        previous_active_frame = toy.seed_frame
        for iteration in range(max_iterations):
            # Projector h and frame are recomputed from the current normal field.
            active_frame, pexact, h_spectrum, h_gap = _positive_frame_reference(toy, h_input)
            # Transport previous Nambu vectors into the newly charted active frame.
            orbital_transport = active_frame.mH @ previous_active_frame
            transport = torch.block_diag(orbital_transport, orbital_transport.conj())
            anchor_quasiframe = transport @ anchor_quasiframe
            # QR removes only roundoff; the whole occupied subspace stays pure.
            anchor_quasiframe, _ = torch.linalg.qr(anchor_quasiframe, mode="reduced")
            block_tag = active_frame.mH @ toy.seed_frame[:, 2]
            block_tag = block_tag / torch.linalg.vector_norm(block_tag)
            previous_blocked_qp = None
            if final is not None:
                previous_blocked_qp = transport @ final.blocked_qp
            try:
                anchor = _tracked_bdg_state(
                    problem, h_input, delta_input, active_frame, mu,
                    anchor_quasiframe, block_tag, previous_blocked_qp,
                )
            except ParityBranchError as exc:
                history.append({"round": iteration, "phase": "rejected_parity_state",
                                "side": "outer_anchor", "mu": mu,
                                "partial_state": exc.partial_state})
                raise
            number_state, bracket = _number_state(
                problem, h_input, delta_input, active_frame, anchor,
                block_tag,
                target=toy.target_number, tolerance=number_tolerance,
                half_width=mu_half_width, continuation_steps=continuation_steps,
                max_bisections=max_bisections, history=history,
                round_index=iteration,
            )
            nerr = abs(float(number_state.number)-toy.target_number)
            full = embed_active_density(active_frame, number_state.rho,
                                        number_state.chi, fermion_parity=-1)
            normal, h_target, source = toy.normal(full.rho)
            epair, delta_target = pairing_gap(
                toy.spinors, full.chi, toy.positions, toy.weights,
                species="neutron", chunk_size=8, force=D1S,
            )
            h_error = float(torch.linalg.matrix_norm(h_target-h_input))
            delta_error = float(torch.linalg.matrix_norm(delta_target-delta_input))
            field_error = max(h_error, delta_error)
            record = {
                "iteration": iteration, "mu": number_state.mu,
                "number": float(number_state.number), "number_abs": nerr,
                "energy": float((normal+epair).detach()),
                "normal_field_residual_fro": h_error,
                "pairing_field_residual_fro": delta_error,
                "field_residual_max": field_error,
                "source_expectation": float(source),
                "positive_projector_gap": h_gap,
                "positive_projector_spectrum": h_spectrum.detach().cpu().tolist(),
                "positive_projector_rank": int(torch.trace(pexact).real.round()),
                "number_bracket": bracket,
                "branch": number_state.metrics,
                "rho": _json_matrix(full.rho), "chi": _json_matrix(full.chi),
                "projector": _json_matrix(number_state.projector),
                "active_frame": _json_matrix(active_frame),
                "quasiframe": _json_matrix(number_state.quasiframe),
                "h_field_target": _json_matrix(h_target),
                "delta_target": _json_matrix(delta_target),
                "h_field_used": _json_matrix(h_input),
                "delta_used": _json_matrix(delta_input),
            }
            history.append({"round": iteration, "phase": "outer",
                            "mu": number_state.mu, "number": float(number_state.number),
                            "field_residual_max": field_error,
                            "branch": number_state.metrics})
            round_states.append(record)
            final = number_state
            final_number_bracket = bracket
            if nerr <= number_tolerance and field_error <= field_tolerance:
                stop_reason = "converged"
                break
            # Only fields are mixed.  The state remains an exact pure projector.
            h_input = (1-field_mixing)*h_input + field_mixing*h_target
            delta_input = (1-field_mixing)*delta_input + field_mixing*delta_target
            mu = number_state.mu
            anchor_quasiframe = number_state.quasiframe
            previous_active_frame = active_frame
    except Exception as exc:
        stop_reason = f"reference_failed: {type(exc).__name__}: {exc}"
    finally:
        counter.__exit__(None, None, None)
    converged = stop_reason == "converged"
    source_path = Path(inspect.getsourcefile(run_tracked_bdg_reference) or __file__)
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if toy is None:
        raise RuntimeError(f"reference initialization failed: {stop_reason}")
    model = {
        "name": "MovingToy projected nucleon-space RHFB",
        "normal_level_scale": .10, "D1S_strength": D1S.strength,
        "target_number": toy.target_number, "blocked_index": 2,
            "block_operation": "compute all-negative vacuum parity; if odd, toggle lowest-energy qp to define even reference; toggle designated block-2 qp to odd; track only this blocked qp by overlap",
        "chart": "projected fixed-seed QR; candidate convention initial_pair_coordinates(problem)",
        "h0": _json_matrix(toy.h0), "source": _json_matrix(toy.source),
        "seed_coordinates": coords.tolist(),
        "initial_mu": .45,
    }
    return {
        "schema": "tracked-bdg-reference-v1",
        "scope": "reference-only finite MovingToy; not a production PK A1 result",
        "prescription": "P+(h) projected nucleon-space BdG with tracked pure odd R projector",
        "stationarity_target": "self-consistent fields and exact-number μ root on continued blocked PH branch",
        "model": model,
        "run": {"max_iterations": max_iterations, "field_mixing": field_mixing,
                "number_tolerance": number_tolerance, "field_tolerance": field_tolerance,
                "mu_half_width": mu_half_width,
                "continuation_steps_per_half_interval": continuation_steps,
                "max_bisections": max_bisections,
                "density_mixing": False, "field_mixing_only": True},
        "status": {"converged": converged, "stop_reason": stop_reason,
                   "iterations_completed": len(round_states),
                   "final_number_bracket": final_number_bracket},
        "eig_counts": eig_counts,
        "source": {"module": str(source_path), "sha256": source_sha,
                   "run_invocation": "run_tracked_bdg_reference(max_iterations=8)",
                   "snapshot": source_path.read_text()},
        "history": history,
        "round_states": round_states,
        "final": round_states[-1] if round_states else None,
    }


def write_reference_artifact(path: str | Path, result: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
