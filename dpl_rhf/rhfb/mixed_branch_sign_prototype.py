"""Isolated finite-space mixed-sign RHB branch prototype.

This module is not imported by the RHFB trainer and is not a production
no-sea prescription.  It realizes the even-even branch convention stated in
DIRHB, arXiv:1403.4039, Eqs. (46)--(48), using a matrix-sign integral and
shifted linear solves.  The formula and its scope are discussed in
``docs/NOSEA_NEXT_IMPLEMENTATION_20261015.md``.  The correspondence with the
Long/Geng RHFB field-operator convention (arXiv:0812.1103 and 2202.00482),
the complete-band enclosure at arbitrary self-consistent fields, and an odd
blocked extension are not established by those sources.

The caller must supply explicit positive gaps at 0 and +/-Ec, complete-band
claims, and an evidence label.  These are assertions to be audited, not
physical certification.  No Ec is chosen here.  The optional small-space
``eigh`` auditor is isolated from the projector function and is forbidden when
``training_path=True``; it is a validation oracle only.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .execution_limits import require_bounded_cgroup
from .parity_diagnostic import number_parity_from_density


@dataclass(frozen=True)
class BranchAssumptions:
    """Explicit finite-space hypothesis required by the mixed-sign map.

    ``particle_abs_upper < Ec < dirac_abs_lower`` is the complete-band
    condition for the scalar sign formula.  ``gap_at_*`` are caller-provided
    lower bounds for distances of the entire spectrum to the indicated sign
    discontinuity.  Booleans document supplied evidence; they do not verify
    it.  Only the independent small-space auditor can check these claims
    numerically, and it cannot establish physical band identity.
    """

    ec: float
    gap_at_zero: float
    gap_at_plus_ec: float
    gap_at_minus_ec: float
    particle_abs_lower: float
    particle_abs_upper: float
    dirac_abs_lower: float
    complete_particle_band: bool
    complete_dirac_band: bool
    no_unclassified_modes: bool
    evidence: str

    def validate(self, dimension: int) -> None:
        values = (
            self.ec,
            self.gap_at_zero,
            self.gap_at_plus_ec,
            self.gap_at_minus_ec,
            self.particle_abs_lower,
            self.particle_abs_upper,
            self.dirac_abs_lower,
        )
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("Ec, band bounds, and all three spectral gaps must be finite")
        if self.ec <= 0 or min(
            self.gap_at_zero, self.gap_at_plus_ec, self.gap_at_minus_ec
        ) <= 0:
            raise ValueError("explicit positive gaps at 0, +Ec, and -Ec are required")
        if not 0 < self.particle_abs_lower <= self.particle_abs_upper < self.ec < self.dirac_abs_lower:
            raise ValueError("complete-band condition requires particle_abs_upper < Ec < dirac_abs_lower")
        if self.gap_at_zero > self.particle_abs_lower + 1.0e-14:
            raise ValueError("declared zero gap exceeds the lower bound of the complete particle band")
        if min(
            self.ec - self.particle_abs_upper,
            self.dirac_abs_lower - self.ec,
        ) + 1.0e-14 < max(self.gap_at_plus_ec, self.gap_at_minus_ec):
            raise ValueError("declared +/-Ec gap exceeds the declared complete-band margin")
        if not all(
            isinstance(flag, bool) and flag
            for flag in (
                self.complete_particle_band,
                self.complete_dirac_band,
                self.no_unclassified_modes,
            )
        ):
            raise ValueError("complete-band claims must all be explicitly true")
        if not isinstance(self.evidence, str) or not self.evidence.strip():
            raise ValueError("evidence must identify the source of the assumptions")
        if dimension <= 0 or dimension % 2 or dimension > 192:
            raise ValueError("prototype accepts even Nambu dimensions up to 192 only")


@dataclass(frozen=True)
class RationalSignQuadrature:
    """Finite log-trapezoid rule for the integral representation of sign."""

    log_step: float
    half_order: int

    def validate(self) -> None:
        if not math.isfinite(self.log_step) or self.log_step <= 0:
            raise ValueError("log_step must be finite and positive")
        if not isinstance(self.half_order, int) or self.half_order < 8:
            raise ValueError("half_order must be an integer >= 8")


@dataclass(frozen=True)
class BranchDiagnostics:
    status: str
    physical_branch_certified: bool
    assumption_evidence: str
    ec: float
    gap_at_zero: float
    gap_at_plus_ec: float
    gap_at_minus_ec: float
    hermiticity_error: float
    h_particle_hole_error: float
    purity_error: float
    particle_hole_error: float
    chi_antisymmetry_error: float
    max_shifted_solve_residual: float
    sign_evaluations: int


@dataclass(frozen=True)
class BranchResult:
    generalized_density: torch.Tensor
    rho: torch.Tensor
    chi: torch.Tensor
    diagnostics: BranchDiagnostics


@dataclass(frozen=True)
class CompleteBandAudit:
    """Small-space spectral audit; never used inside the sign projector."""

    complete_band_holds: bool
    minimum_gap_zero: float
    minimum_gap_plus_ec: float
    minimum_gap_minus_ec: float
    unclassified_abs_energies: tuple[float, ...]
    oracle_eigh_call_count: int
    is_physical_band_identity_certified: bool = False


def _ph_matrix(matrix: torch.Tensor) -> torch.Tensor:
    n = matrix.shape[0] // 2
    return torch.cat(
        (
            torch.cat((matrix[n:, n:].conj(), matrix[n:, :n].conj()), dim=1),
            torch.cat((matrix[:n, n:].conj(), matrix[:n, :n].conj()), dim=1),
        ),
        dim=0,
    )


def _sign_by_shifted_solves(
    operator: torch.Tensor,
    vectors: torch.Tensor,
    quadrature: RationalSignQuadrature,
) -> tuple[torch.Tensor, float]:
    """Approximate sign(A)@V via (2/pi) int A(A^2+t^2)^-1 V dt.

    The finite log trapezoid is a numerical quadrature with no claimed error
    certificate.  ``torch.linalg.solve`` remains in the autograd graph, so
    field dependence through every shifted solve is retained.
    """
    dim = operator.shape[0]
    identity = torch.eye(dim, dtype=operator.dtype, device=operator.device)
    square = operator @ operator
    result = torch.zeros_like(vectors)
    max_residual = 0.0
    h = quadrature.log_step
    for index in range(-quadrature.half_order, quadrature.half_order + 1):
        t = math.exp(index * h)
        solved = torch.linalg.solve(square + (t * t) * identity, vectors)
        residual = (square + (t * t) * identity) @ solved - vectors
        max_residual = max(max_residual, float(torch.linalg.vector_norm(residual).detach()))
        result = result + (2.0 / math.pi) * h * t * (operator @ solved)
    return result, max_residual


def mixed_branch_projector(
    h_nambu: torch.Tensor,
    *,
    assumptions: BranchAssumptions,
    quadrature: RationalSignQuadrature,
    tolerance: float = 2.0e-8,
) -> BranchResult:
    """Return the conditional DIRHB mixed-sign projector, never a certificate.

    For a full Hermitian BdG matrix ``H``, computes

    ``R = 1/2 [I - sign(H) + sign(H+Ec I) + sign(H-Ec I)]``.

    It is the complement of the selected annihilator branches only if the
    caller's entire finite Fermi band lies within ``|E|<Ec`` and entire Dirac
    band lies outside it, with no spectrum at 0 or +/-Ec.  This requires a
    prior physically justified complete-band assessment.  The function has no
    guessed threshold, eigenvalue solver, basis refresh, field truncation, or
    runtime-training integration.
    """
    require_bounded_cgroup()
    h = torch.as_tensor(h_nambu)
    if h.dtype != torch.complex128 or h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("H must be a square complex128 full Nambu matrix")
    if not bool(torch.isfinite(h).all()):
        raise ValueError("H contains non-finite values")
    assumptions.validate(h.shape[0])
    quadrature.validate()
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    h_norm = float(torch.linalg.matrix_norm(h).detach())
    herm_error = float(torch.linalg.matrix_norm(h - h.mH).detach())
    ph_h_error = float(torch.linalg.matrix_norm(_ph_matrix(h) + h).detach())
    if herm_error > tolerance * max(1.0, h_norm):
        raise ValueError("H must be Hermitian")
    if ph_h_error > tolerance * max(1.0, h_norm):
        raise ValueError("H must satisfy the declared BdG particle-hole symmetry")

    dimension = h.shape[0]
    identity = torch.eye(dimension, dtype=h.dtype, device=h.device)
    sign_h, r0 = _sign_by_shifted_solves(h, identity, quadrature)
    sign_plus, r_plus = _sign_by_shifted_solves(
        h + assumptions.ec * identity, identity, quadrature
    )
    sign_minus, r_minus = _sign_by_shifted_solves(
        h - assumptions.ec * identity, identity, quadrature
    )
    generalized = 0.5 * (identity - sign_h + sign_plus + sign_minus)
    n = dimension // 2
    rho = generalized[:n, :n]
    chi = generalized[:n, n:]
    herm_r = float(torch.linalg.matrix_norm(generalized - generalized.mH).detach())
    purity = float(torch.linalg.matrix_norm(generalized @ generalized - generalized).detach())
    ph_r = float(torch.linalg.matrix_norm(_ph_matrix(generalized) - (identity - generalized)).detach())
    chi_skew = float(torch.linalg.matrix_norm(chi + chi.mT).detach())
    diagnostics = BranchDiagnostics(
        status="conditional_candidate_not_physics_certified",
        physical_branch_certified=False,
        assumption_evidence=assumptions.evidence,
        ec=float(assumptions.ec),
        gap_at_zero=float(assumptions.gap_at_zero),
        gap_at_plus_ec=float(assumptions.gap_at_plus_ec),
        gap_at_minus_ec=float(assumptions.gap_at_minus_ec),
        hermiticity_error=herm_r,
        h_particle_hole_error=ph_h_error,
        purity_error=purity,
        particle_hole_error=ph_r,
        chi_antisymmetry_error=chi_skew,
        max_shifted_solve_residual=max(r0, r_plus, r_minus),
        sign_evaluations=3 * (2 * quadrature.half_order + 1),
    )
    return BranchResult(generalized, rho, chi, diagnostics)


def swap_odd_blocked_occupation(
    even_density: torch.Tensor,
    previously_unoccupied: torch.Tensor,
    *,
    tolerance: float = 2.0e-8,
) -> torch.Tensor:
    """Algebraically swap one PH pair; does not identify a physical odd orbit."""
    require_bounded_cgroup()
    r = torch.as_tensor(even_density)
    w = torch.as_tensor(previously_unoccupied, dtype=r.dtype, device=r.device)
    if r.dtype != torch.complex128 or r.ndim != 2 or r.shape[0] != r.shape[1] or r.shape[0] % 2:
        raise ValueError("R must be square, even-dimensional, and complex128")
    if r.shape[0] > 192 or w.shape != (r.shape[0],):
        raise ValueError("prototype requires one mode vector and dimension <= 192")
    if not bool(torch.isfinite(r).all()) or not bool(torch.isfinite(w).all()):
        raise ValueError("R and blocked mode must be finite")
    identity = torch.eye(r.shape[0], dtype=r.dtype, device=r.device)
    partner = torch.cat((w[r.shape[0] // 2 :].conj(), w[: r.shape[0] // 2].conj()))
    errors = (
        float(torch.linalg.matrix_norm(r - r.mH).detach()),
        float(torch.linalg.matrix_norm(r @ r - r).detach()),
        float(torch.linalg.matrix_norm(_ph_matrix(r) - (identity - r)).detach()),
        abs(float(torch.linalg.vector_norm(w).detach()) - 1.0),
        float(torch.linalg.vector_norm(r @ w).detach()),
        float(torch.linalg.vector_norm(r @ partner - partner).detach()),
        abs(complex(torch.vdot(w, partner).detach().cpu())),
    )
    if max(errors) > tolerance:
        raise ValueError("R, blocked vector, and PH partner do not define a valid projector swap")
    n = r.shape[0] // 2
    even_parity = number_parity_from_density(
        r[:n, :n].detach().cpu().numpy(),
        r[:n, n:].detach().cpu().numpy(),
        tolerance=10.0 * tolerance,
    )
    if even_parity["number_parity"] != 1:
        raise ValueError("odd occupation swap requires an even-parity reference projector")
    odd = r + torch.outer(w, w.conj()) - torch.outer(partner, partner.conj())
    odd_parity = number_parity_from_density(
        odd[:n, :n].detach().cpu().numpy(),
        odd[:n, n:].detach().cpu().numpy(),
        tolerance=20.0 * tolerance,
    )
    if odd_parity["number_parity"] != -1:
        raise RuntimeError("particle-hole occupation swap failed to produce odd parity")
    return odd


def audit_complete_band_with_eigh_oracle(
    h_nambu: torch.Tensor,
    assumptions: BranchAssumptions,
    *,
    training_path: bool = False,
    tolerance: float = 1.0e-8,
) -> CompleteBandAudit:
    """Explicit independent small-space audit, counted and training-forbidden.

    This uses one ``torch.linalg.eigh`` call and reports whether actual
    eigenvalues violate the caller's absolute-energy band enclosure or any
    of the three supplied gap lower bounds.  It cannot infer particle/Dirac
    identity for a spectrum in which bands have crossed; that remains a
    physical labelling obligation.
    """
    require_bounded_cgroup()
    if training_path:
        raise RuntimeError("the independent eigh oracle is forbidden on training paths")
    h = torch.as_tensor(h_nambu)
    if h.dtype != torch.complex128 or h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("H must be a square complex128 matrix")
    assumptions.validate(h.shape[0])
    values_tensor, _ = torch.linalg.eigh(h)
    values = values_tensor.detach().real.cpu().tolist()
    zero_gap = min(abs(value) for value in values)
    plus_gap = min(abs(value + assumptions.ec) for value in values)
    minus_gap = min(abs(value - assumptions.ec) for value in values)
    abs_values = [abs(value) for value in values]
    unclassified = tuple(
        sorted(
            value
            for value in abs_values
            if assumptions.particle_abs_upper + tolerance
            < value
            < assumptions.dirac_abs_lower - tolerance
        )
    )
    gaps_hold = (
        zero_gap + tolerance >= assumptions.gap_at_zero
        and plus_gap + tolerance >= assumptions.gap_at_plus_ec
        and minus_gap + tolerance >= assumptions.gap_at_minus_ec
    )
    return CompleteBandAudit(
        complete_band_holds=not unclassified and gaps_hold,
        minimum_gap_zero=zero_gap,
        minimum_gap_plus_ec=plus_gap,
        minimum_gap_minus_ec=minus_gap,
        unclassified_abs_energies=unclassified,
        oracle_eigh_call_count=1,
    )
