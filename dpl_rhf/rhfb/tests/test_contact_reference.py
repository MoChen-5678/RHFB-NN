"""Bounded contact/Hessian check against the RHF-e spherical reduction."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from dpl_rhf.rhfb.contact_reference import rhfe_contact_reference


def test_derivative_hessian_plus_compensation_partial_spherical_diagnostic(record_property):
    """One same-state radial refinement; partial consistency, never readiness."""
    # Match the O17 production metadata: five-point, fourth-order radial ADF.
    coarse = rhfe_contact_reference(spacing_fm=0.125, radius_fm=6.5,
                                    derivative_order=5)
    fine = rhfe_contact_reference(spacing_fm=0.0625, radius_fm=6.5,
                                  derivative_order=5)
    for label, result in (("coarse", coarse), ("fine", fine)):
        assert result["case"]["derivative_order"] == 5
        assert result["case"]["adf_points"] == 5
        assert result["case"]["adf_formal_accuracy"] == "fourth order"
        record_property(f"{label}_derivative_order", result["case"]["derivative_order"])
        record_property(f"{label}_adf_points", result["case"]["adf_points"])
        record_property(f"{label}_adf_formal_accuracy",
                        result["case"]["adf_formal_accuracy"])
    for label, result in (("coarse", coarse), ("fine", fine)):
        for channel in ("rho_tensor", "pion_pv"):
            # Explicit H-delta expectation is checked against independent
            # RHF-e local trace identities, not contact.py's construction.
            assert result[channel]["h_delta_abs_error_mev"] < 2e-12
            assert math.isfinite(result[channel]["relative_error"])
            for key in ("general_m_mev", "spherical_mev", "abs_error_mev",
                        "relative_error", "h_delta_contact_mev",
                        "h_delta_reference_mev"):
                record_property(f"{label}_{channel}_{key}", result[channel][key])
        # A time-reversal-invariant closed doublet has no spatial axial source.
        assert abs(result["pion_pv"]["finite_range_direct_mev"]) < 1e-12

    # The single refinement reduces both discrepancies; this records a trend,
    # not an asymptotic convergence law or a complete physical closure.
    for channel in ("rho_tensor", "pion_pv"):
        assert fine[channel]["abs_error_mev"] < coarse[channel]["abs_error_mev"]
        record_property(f"{channel}_error_reduction_factor",
                        coarse[channel]["abs_error_mev"] /
                        fine[channel]["abs_error_mev"])

    result = {
        "status": "small-space partial consistency only; not a physical closure certificate",
        "coarse_h_0p125": coarse,
        "fine_h_0p0625": fine,
        "diagnostic_limits": {
            "h_delta_contact_abs_error_mev": 2e-12,
            "pion_direct_abs_mev": 1e-12,
            "total_channel_error_threshold": None,
            "interpretation": (
                "finite-range plus H-delta differences are reported, not gated; "
                "one h refinement cannot establish continuum convergence"
            ),
        },
    }
    artifact = Path(__file__).resolve().parents[3] / "outputs" / "validation" / (
        "rhf_e_contact_reference_20261006.json"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    # Keep the reported reference values in pytest's XML as well as JSON.
    assert torch.isfinite(torch.tensor([
        coarse["rho_tensor"]["general_m_mev"],
        coarse["pion_pv"]["general_m_mev"],
        fine["rho_tensor"]["general_m_mev"],
        fine["pion_pv"]["general_m_mev"],
    ])).all()
