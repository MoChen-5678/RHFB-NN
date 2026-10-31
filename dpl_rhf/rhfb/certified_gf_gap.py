"""Certified finite-matrix G/F gap policy for reference-only RHFB branches.

This module does not identify physical quasiparticle labels by itself. It
certifies a real shift W at which the finite normal matrix h-W is
quasi-definite in the primitive upper/lower Dirac-component block order.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

import torch


class CertifiedGFGapError(RuntimeError):
    """The finite h matrix has no certified G/F quasi-definite shift."""

    def __init__(self, message: str, evidence: dict[str, Any]):
        super().__init__(message)
        self.evidence = evidence


def gf_indices_from_basis(basis) -> tuple[torch.Tensor, torch.Tensor]:
    """Return primitive upper-G and lower-F rows in basis storage order."""
    g_rows: list[int] = []
    f_rows: list[int] = []
    nrad = int(basis.n_radial)
    for channel in basis.channels:
        start = int(channel.offset)
        g_rows.extend(range(start, start + nrad))
        f_rows.extend(range(start + nrad, start + 2 * nrad))
    if len(g_rows) + len(f_rows) != int(basis.dimension):
        raise ValueError("basis channel offsets do not cover the primitive dimension")
    if len(g_rows) != len(f_rows):
        raise ValueError("G/F primitive subspaces must have equal dimensions")
    return torch.tensor(g_rows, dtype=torch.long), torch.tensor(f_rows, dtype=torch.long)


def certify_gf_interval_from_indices(
    h: torch.Tensor,
    g_rows: torch.Tensor,
    f_rows: torch.Tensor,
    *,
    rounding_relative_margin: float = 1.0e-10,
) -> dict[str, Any]:
    """Certify W from Hermitian Gershgorin bounds on h_GG and h_FF.

    The returned interval is (upper_bound(lambda_max(h_FF)),
    lower_bound(lambda_min(h_GG))). For each W in it,
    A=h_GG-W>0 and C=W-h_FF>0, so Veselić & Veselić Eq. (2) applies to
    [[A,h_GF],[h_FG,-C]]. No eigensystem is used to choose W.
    """
    h = torch.as_tensor(h)
    if h.dtype != torch.complex128 or h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise TypeError("h must be square complex128")
    if not bool(torch.isfinite(h).all()):
        raise ValueError("h contains NaN or infinity")
    if rounding_relative_margin < 0 or not math.isfinite(rounding_relative_margin):
        raise ValueError("rounding_relative_margin must be finite and nonnegative")
    d = h.shape[0]
    g_rows = torch.as_tensor(g_rows, dtype=torch.long, device=h.device)
    f_rows = torch.as_tensor(f_rows, dtype=torch.long, device=h.device)
    if (g_rows.ndim != 1 or f_rows.ndim != 1 or g_rows.numel() + f_rows.numel() != d
            or g_rows.numel() != f_rows.numel()):
        raise ValueError("G/F index vectors must partition h into equal-size blocks")
    all_rows = torch.cat((g_rows, f_rows))
    if not torch.equal(torch.sort(all_rows).values, torch.arange(d, device=h.device)):
        raise ValueError("G/F rows must be a disjoint complete partition of primitive indices")
    hermitian_max = float((h - h.mH).abs().max())
    if hermitian_max > 1.0e-8:
        raise ValueError(f"h is not Hermitian within reference tolerance: {hermitian_max:.6g}")
    h_raw = h
    h = 0.5 * (h_raw + h_raw.mH)
    sym_max = float((h-h_raw).abs().max())
    sym_frob = float(torch.linalg.matrix_norm(h-h_raw))

    gg = h[g_rows[:, None], g_rows[None, :]]
    ff = h[f_rows[:, None], f_rows[None, :]]
    # The off-diagonal B block is unrestricted by the theorem.
    g_diag = gg.diagonal().real
    f_diag = ff.diagonal().real
    g_radius = gg.abs().sum(dim=1) - gg.diagonal().abs()
    f_radius = ff.abs().sum(dim=1) - ff.diagonal().abs()
    g_raw = float((g_diag - g_radius).min())
    f_raw = float((f_diag + f_radius).max())
    row_sum = float(h.abs().sum(dim=1).max())
    if not all(math.isfinite(x) for x in (g_raw, f_raw, row_sum)):
        raise ValueError("non-finite Gershgorin bound from h")
    pad = float(rounding_relative_margin) * max(1.0, row_sum)
    g_lower = g_raw - pad
    f_upper = f_raw + pad
    width = g_lower - f_upper
    evidence: dict[str, Any] = {
        "policy": "certified_gf_interval",
        "source_of_W": "midpoint of outward-padded Hermitian Gershgorin bounds of actual h_GG and h_FF; no normal eigenvalues used to select W",
        "g_dimension": int(g_rows.numel()),
        "f_dimension": int(f_rows.numel()),
        "g_gershgorin_lower_raw_mev": g_raw,
        "f_gershgorin_upper_raw_mev": f_raw,
        "outward_rounding_pad_mev": pad,
        "g_gershgorin_lower_certified_mev": g_lower,
        "f_gershgorin_upper_certified_mev": f_upper,
        "certified_W_interval_open_mev": [f_upper, g_lower],
        "interval_width_mev": width,
        "midpoint_W_mev": 0.5 * (f_upper + g_lower),
        "A_lower_bound_at_midpoint_mev": 0.5 * width,
        "C_lower_bound_at_midpoint_mev": 0.5 * width,
        "whole_h_hermiticity_max_abs_mev": hermitian_max,
        "h_symmetrization_correction_max_abs_mev": sym_max,
        "h_symmetrization_correction_frobenius_mev": sym_frob,
        "arithmetic_scope": "float64/complex128 Gershgorin computation with explicit outward relative pad; conservative floating-point safety margin, not interval-arithmetic proof",
        "whole_h_max_row_sum_mev": row_sum,
        "actual_h_matrix_sha256": hashlib.sha256(
            h.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest(),
        "raw_h_matrix_sha256": hashlib.sha256(
            h_raw.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest(),
        "G_F_row_partition_sha256": hashlib.sha256(
            torch.cat((g_rows.cpu(), f_rows.cpu())).numpy().tobytes()
        ).hexdigest(),
        "theorem": "Veselic & Veselic, arXiv:1404.4991, PDF p.1 Eq.(2): for Hermitian [[A,B],[B†,-C]] with A,C>0, spectrum avoids (-lambda_min(C),lambda_min(A)); arbitrary B.",
        "physical_label_limit": "A certified mathematical gap and inertia are not by themselves a proof that the two subspaces are the nuclear physical Dirac sea and positive-particle branches.",
        "safe": width > 0.0,
    }
    if width <= 0.0:
        raise CertifiedGFGapError("G/F Gershgorin bounds do not certify a nonempty W interval", evidence)
    return evidence


def certify_basis_gf_interval(h: torch.Tensor, basis, *, mass_mev: float,
                              species: str) -> dict[str, Any]:
    """Apply the G/F certificate with actual basis rows and PKA1 mass check."""
    if species not in {"n", "p"}:
        raise ValueError("species must be 'n' or 'p'")
    expected = float(basis.mass_n_mev if species == "n" else basis.mass_p_mev)
    if not math.isclose(float(mass_mev), expected, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError("caller mass does not match the supplied PKA1 basis species mass")
    g_rows, f_rows = gf_indices_from_basis(basis)
    report = certify_gf_interval_from_indices(h, g_rows, f_rows)
    report.update({
        "species": species,
        "mass_mev": float(mass_mev),
        "basis": {"radius_fm": float(basis.radius_fm),
                  "spacing_fm": float(basis.spacing_fm),
                  "kappa_max": int(basis.kappa_max),
                  "n_radial": int(basis.n_radial),
                  "primitive_dimension": int(basis.dimension)},
        "mass_role": "recorded and checked against the PKA1 AngularRadialBasis; W is derived from actual h_GG/h_FF and does not use a substituted mass cutoff",
    })
    return report


def validate_spectrum_for_gf_interval(eigenvalues: torch.Tensor,
                                      certificate: dict[str, Any], *,
                                      tolerance_mev: float = 1.0e-7) -> dict[str, Any]:
    """Post-check the h spectrum against the independent G/F W certificate."""
    values = torch.as_tensor(eigenvalues, dtype=torch.float64).detach().cpu()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("normal spectrum contains NaN or infinity")
    if values.ndim != 1 or values.numel() != certificate["g_dimension"] + certificate["f_dimension"]:
        raise ValueError("normal spectrum size disagrees with certified G/F dimensions")
    if bool((values[1:] < values[:-1]).any()):
        raise ValueError("normal eigenvalues must be sorted ascending")
    lo, hi = map(float, certificate["certified_W_interval_open_mev"])
    w = float(certificate["midpoint_W_mev"])
    lower_gap = float(certificate["A_lower_bound_at_midpoint_mev"])
    lower_gap = min(lower_gap, float(certificate["C_lower_bound_at_midpoint_mev"]))
    distance = float((values - w).abs().min())
    below = int((values < w).sum())
    above = int(values.numel() - below)
    expected = int(certificate["g_dimension"])
    checked = {
        "midpoint_W_mev": w,
        "certified_interval_mev": [lo, hi],
        "certified_resolvent_gap_about_W_lower_bound_mev": lower_gap,
        "observed_nearest_eigenvalue_distance_from_W_mev": distance,
        "gap_bound_check_passed": distance + tolerance_mev >= lower_gap,
        "eigenvalues_below_W": below,
        "eigenvalues_above_W": above,
        "equal_G_F_inertia_check_passed": below == above == expected,
        "ordered_216_217_values_if_432_dimensional": (
            [float(values[215]), float(values[216])] if values.numel() == 432 else None),
        "normal_spectrum_confirms_certificate": (
            distance + tolerance_mev >= lower_gap and below == above == expected),
    }
    if not checked["normal_spectrum_confirms_certificate"]:
        raise CertifiedGFGapError("post-solve normal spectrum did not confirm G/F gap certificate", checked)
    return checked
