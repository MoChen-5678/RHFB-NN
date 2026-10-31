"""Finite-space DIRHB-style mixed-sign BdG projector (diagnostic only).

This is a bounded small-matrix prototype, not an O17 acceptance path or a
replacement for continuum RHFB.  It applies two rational signs using only
shifted linear solves.  A caller-supplied structured enclosure certificate is
required; the certificate's physical band claims remain the caller's proof
obligation and must come from an analytic toy construction or independent
reference, never a spectral teacher in NN training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import math

import torch

from .execution_limits import require_bounded_cgroup
from .no_sea import RationalSignOperator
from .parity_diagnostic import number_parity_from_density


class BandBoundSource(str, Enum):
    ANALYTIC_TOY = "analytic_toy"
    INDEPENDENT_REFERENCE = "independent_reference"


@dataclass(frozen=True)
class BandEnclosureCertificate:
    """Caller-certified absolute-energy enclosures for two complete BdG bands.

    `particle_abs_upper < lambda < sea_abs_lower` is the finite-window
    separability condition.  All booleans are explicit audit gates rather
    than a free-form string; they do not make a false caller assertion true.
    """

    particle_abs_lower: float
    particle_abs_upper: float
    sea_abs_lower: float
    sea_abs_upper: float
    particle_annihilator_energy_sign: int
    sea_annihilator_energy_sign: int
    zero_gap_lower: float
    max_abs_upper: float
    finite_space_complete: bool
    hermitian_verified: bool
    particle_hole_verified: bool
    particle_band_complete: bool
    sea_band_complete: bool
    no_unclassified_modes: bool
    outside_training_spectral_teacher: bool
    source: BandBoundSource
    source_id: str

    def validate(self, dimension: int, divider: float) -> None:
        if not isinstance(self.source, BandBoundSource):
            raise TypeError("source must be a BandBoundSource enum")
        if not self.source_id.strip():
            raise ValueError("source_id must identify the analytic/reference evidence")
        flags = (
            self.finite_space_complete,
            self.hermitian_verified,
            self.particle_hole_verified,
            self.particle_band_complete,
            self.sea_band_complete,
            self.no_unclassified_modes,
            self.outside_training_spectral_teacher,
        )
        if not all(isinstance(flag, bool) and flag for flag in flags):
            raise ValueError("all finite-space, band, symmetry, and non-teacher gates must be true")
        if (self.particle_annihilator_energy_sign != 1
                or self.sea_annihilator_energy_sign != -1):
            raise ValueError("DIRHB convention requires particle E>0 and sea E<0 annihilator bands")
        bounds = (
            self.particle_abs_lower,
            self.particle_abs_upper,
            self.sea_abs_lower,
            self.sea_abs_upper,
            self.zero_gap_lower,
            self.max_abs_upper,
            divider,
        )
        if not all(math.isfinite(x) for x in bounds):
            raise ValueError("all enclosure bounds and divider must be finite")
        if dimension <= 0 or dimension > 192 or dimension % 2:
            raise ValueError("prototype accepts even Nambu dimensions up to 192 only")
        if not (0.0 < self.particle_abs_lower <= self.particle_abs_upper):
            raise ValueError("invalid physical particle-band enclosure")
        if not (self.particle_abs_upper < divider < self.sea_abs_lower):
            raise ValueError("divider must lie strictly between complete particle and sea bands")
        if not (self.sea_abs_lower <= self.sea_abs_upper <= self.max_abs_upper):
            raise ValueError("invalid sea-band/max-absolute enclosure")
        if not (0.0 < self.zero_gap_lower <= min(
            self.particle_abs_lower, self.sea_abs_lower
        )):
            raise ValueError("zero_gap_lower must be a positive lower bound for the whole spectrum")


@dataclass(frozen=True)
class BranchFilterDiagnostics:
    status: str
    accepted_as_physics: bool
    sign_h_error_bound: float
    sign_cut_error_bound: float
    # Product propagation of the two uniform quadrature errors only.  It
    # excludes linear-solve residual/roundoff, matrix-product roundoff, and
    # certificate uncertainty; those are reported/qualified separately.
    product_projector_quadrature_only_error_bound: float
    max_shifted_solve_residual: float
    hermiticity_error: float
    purity_error: float
    particle_hole_error: float
    chi_skew_error: float
    lambda_gap: float
    cut_gap: float
    certificate_source: str
    certificate_source_id: str


@dataclass(frozen=True)
class BranchFilterResult:
    generalized_density: torch.Tensor
    rho: torch.Tensor
    chi: torch.Tensor
    diagnostics: BranchFilterDiagnostics


def _particle_hole(matrix: torch.Tensor) -> torch.Tensor:
    """Canonical Nambu particle-hole conjugate for the (particle,hole) order."""
    n = matrix.shape[0] // 2
    swap = torch.cat((
        torch.cat((torch.zeros((n, n), dtype=matrix.dtype, device=matrix.device),
                   torch.eye(n, dtype=matrix.dtype, device=matrix.device)), dim=1),
        torch.cat((torch.eye(n, dtype=matrix.dtype, device=matrix.device),
                   torch.zeros((n, n), dtype=matrix.dtype, device=matrix.device)), dim=1),
    ), dim=0)
    return swap @ matrix.conj() @ swap


def particle_hole_partner(vector: torch.Tensor) -> torch.Tensor:
    """Return the convention's particle-hole partner without a spectral solve."""
    vector = torch.as_tensor(vector)
    if vector.ndim != 1 or vector.numel() % 2:
        raise ValueError("particle-hole vector must be a one-dimensional Nambu vector")
    n = vector.numel() // 2
    return torch.cat((vector[n:].conj(), vector[:n].conj()))


def _operator_certificate(cert: BandEnclosureCertificate, label: str) -> str:
    payload = asdict(cert)
    payload["source"] = cert.source.value
    payload["operator"] = label
    return json.dumps(payload, sort_keys=True)


def mixed_sign_bdg_projector(
    h_nambu: torch.Tensor,
    *,
    divider: float,
    certificate: BandEnclosureCertificate,
    quadrature_order: int = 161,
    log_step: float = 0.35,
    tolerance: float = 1.0e-9,
    solve_tolerance: float = 1.0e-11,
) -> BranchFilterResult:
    """Build ``(I-sign(H) sign(Λ²I-H²))/2`` through rational linear solves.

    The defaults use a conservative quadrature order; the existing rational
    operator still checks the actual enclosure-wide error and fails if these
    settings are insufficient for the supplied bounds.  No eig/eigh API is
    called.  This remains a finite-space auxiliary target/reference, not a
    production continuum branch prescription.
    """
    require_bounded_cgroup()
    h = torch.as_tensor(h_nambu)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("H must be a square Nambu matrix")
    if h.dtype != torch.complex128:
        raise TypeError("H must use complex128")
    if not bool(torch.isfinite(h).all()):
        raise ValueError("H contains nonfinite entries")
    certificate.validate(h.shape[0], divider)
    scale_h = max(1.0, float(torch.linalg.matrix_norm(h).detach()))
    hermiticity_error = float(torch.linalg.matrix_norm(h - h.mH).detach())
    ph_error = float(torch.linalg.matrix_norm(_particle_hole(h) + h).detach())
    if hermiticity_error > 1.0e-10 * scale_h:
        raise ValueError("H must be Hermitian")
    if ph_error > 1.0e-10 * scale_h:
        raise ValueError("H violates the declared Nambu particle-hole symmetry")

    dim = h.shape[0]
    identity = torch.eye(dim, dtype=h.dtype, device=h.device)
    sign_h = RationalSignOperator(
        h, dim, shift=0.0,
        spectral_gap=certificate.zero_gap_lower,
        spectral_bound=certificate.max_abs_upper,
        bound_certificate=_operator_certificate(certificate, "H"),
        quadrature_order=quadrature_order, log_step=log_step,
        tolerance=tolerance, solve_tolerance=solve_tolerance,
    )
    h2 = h @ h
    cut_operator = divider * divider * identity - h2
    cut_gap = min(
        divider * divider - certificate.particle_abs_upper**2,
        certificate.sea_abs_lower**2 - divider * divider,
    )
    if cut_gap <= 0.0:
        raise ValueError("divider intersects a certified physical or sea band")
    cut_bound = max(
        abs(divider * divider - certificate.particle_abs_lower**2),
        abs(divider * divider - certificate.particle_abs_upper**2),
        abs(divider * divider - certificate.sea_abs_lower**2),
        abs(divider * divider - certificate.sea_abs_upper**2),
    )
    sign_cut = RationalSignOperator(
        cut_operator, dim, shift=0.0,
        spectral_gap=cut_gap,
        spectral_bound=cut_bound,
        bound_certificate=_operator_certificate(certificate, "LambdaSquaredMinusH2"),
        quadrature_order=quadrature_order, log_step=log_step,
        tolerance=tolerance, solve_tolerance=solve_tolerance,
    )
    sign_h_matrix = sign_h.apply_sign(identity)
    sign_cut_matrix = sign_cut.apply_sign(identity)
    rho_generalized = 0.5 * (identity - sign_h_matrix @ sign_cut_matrix)
    n = dim // 2
    rho, chi = rho_generalized[:n, :n], rho_generalized[:n, n:]
    herm_r = float(torch.linalg.matrix_norm(rho_generalized - rho_generalized.mH).detach())
    purity = float(torch.linalg.matrix_norm(
        rho_generalized @ rho_generalized - rho_generalized
    ).detach())
    chi_skew = float(torch.linalg.matrix_norm(chi + chi.mT).detach())
    ph_r = float(torch.linalg.matrix_norm(
        _particle_hole(rho_generalized) - (identity - rho_generalized)
    ).detach())
    # This is a quadrature-only operator-norm bound: for exact signs S,T and
    # approximants S~,T~, ||S~T~-ST|| <= eS + eT + eS*eT.  The rational
    # operator's measured solve residual is reported separately and is not
    # folded into this bound.
    error = 0.5 * (
        sign_h.quadrature_error_bound
        + sign_cut.quadrature_error_bound
        + sign_h.quadrature_error_bound * sign_cut.quadrature_error_bound
    )
    max_solve = torch.maximum(sign_h.last_max_residual, sign_cut.last_max_residual)
    diagnostics = BranchFilterDiagnostics(
        status="finite_window_candidate_not_physics_accepted",
        accepted_as_physics=False,
        sign_h_error_bound=sign_h.quadrature_error_bound,
        sign_cut_error_bound=sign_cut.quadrature_error_bound,
        product_projector_quadrature_only_error_bound=error,
        max_shifted_solve_residual=float(max_solve.detach()),
        hermiticity_error=herm_r,
        purity_error=purity,
        particle_hole_error=ph_r,
        chi_skew_error=chi_skew,
        lambda_gap=certificate.sea_abs_lower - certificate.particle_abs_upper,
        cut_gap=cut_gap,
        certificate_source=certificate.source.value,
        certificate_source_id=certificate.source_id,
    )
    return BranchFilterResult(rho_generalized, rho, chi, diagnostics)


def swap_odd_blocked_occupation(
    even_density: torch.Tensor,
    previously_unoccupied: torch.Tensor,
    *,
    tolerance: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Toggle one occupied/unoccupied particle-hole pair in a pure R projector.

    `previously_unoccupied` must be normalized and satisfy Rw=0; its PH partner
    must be occupied.  The returned R is an explicit odd blocked candidate.
    It does not choose which orbit is physical; callers track that identity.
    """
    r = torch.as_tensor(even_density)
    w = torch.as_tensor(previously_unoccupied, dtype=r.dtype, device=r.device)
    if r.ndim != 2 or r.shape[0] != r.shape[1] or r.shape[0] % 2:
        raise ValueError("generalized density must be a square even-dimensional Nambu matrix")
    if r.shape[0] > 192 or r.dtype != torch.complex128:
        raise ValueError("blocked prototype requires complex128 Nambu dimension <= 192")
    if w.shape != (r.shape[0],):
        raise ValueError("blocked mode must have one component per Nambu row")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if not bool(torch.isfinite(r).all()) or not bool(torch.isfinite(w).all()):
        raise ValueError("R and blocked vector must contain only finite values")
    identity = torch.eye(r.shape[0], dtype=r.dtype, device=r.device)
    herm = float(torch.linalg.matrix_norm(r - r.mH).detach())
    purity = float(torch.linalg.matrix_norm(r @ r - r).detach())
    ph_error = float(torch.linalg.matrix_norm(
        _particle_hole(r) - (identity - r)
    ).detach())
    if max(herm, purity, ph_error) > tolerance:
        raise ValueError("input R must be Hermitian, idempotent, and particle-hole canonical")
    n = r.shape[0] // 2
    even_parity = number_parity_from_density(
        r[:n, :n].detach().cpu().numpy(), r[:n, n:].detach().cpu().numpy(),
        tolerance=max(tolerance * 10.0, 1.0e-10),
    )
    if even_parity["number_parity"] != 1:
        raise ValueError("odd blocking swap requires an even-parity reference density")
    norm_error = abs(float(torch.linalg.vector_norm(w).detach()) - 1.0)
    partner = particle_hole_partner(w)
    overlap = abs(complex(torch.vdot(w, partner).detach().cpu()))
    if norm_error > tolerance or overlap > tolerance:
        raise ValueError("blocked vector and particle-hole partner must be normalized and orthogonal")
    unoccupied_error = float(torch.linalg.vector_norm(r @ w).detach())
    occupied_error = float(torch.linalg.vector_norm(r @ partner - partner).detach())
    if max(unoccupied_error, occupied_error) > tolerance:
        raise ValueError("R does not contain the requested occupied/unoccupied PH pair")
    odd = r + torch.outer(w, w.conj()) - torch.outer(partner, partner.conj())
    odd_herm = float(torch.linalg.matrix_norm(odd - odd.mH).detach())
    odd_purity = float(torch.linalg.matrix_norm(odd @ odd - odd).detach())
    odd_ph_error = float(torch.linalg.matrix_norm(
        _particle_hole(odd) - (identity - odd)
    ).detach())
    rho, chi = odd[:n, :n], odd[:n, n:]
    parity = number_parity_from_density(
        rho.detach().cpu().numpy(), chi.detach().cpu().numpy(),
        tolerance=max(tolerance * 10.0, 1.0e-10),
    )
    if max(odd_herm, odd_purity, odd_ph_error) > tolerance * 4.0:
        raise RuntimeError("blocked occupation swap failed generalized-density invariants")
    if parity["number_parity"] != -1:
        raise RuntimeError("explicit occupation swap did not produce odd number parity")
    return odd, {
        "reference_number_parity": int(even_parity["number_parity"]),
        "number_parity": int(parity["number_parity"]),
        "hermiticity_error": odd_herm,
        "purity_error": odd_purity,
        "particle_hole_error": odd_ph_error,
        "unoccupied_mode_error": unoccupied_error,
        "occupied_partner_error": occupied_error,
    }
