import pytest
import torch

from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.certified_gf_gap import (
    CertifiedGFGapError,
    certify_basis_gf_interval,
    certify_gf_interval_from_indices,
    gf_indices_from_basis,
    validate_spectrum_for_gf_interval,
)
from dpl_rhf.rhfb.fixed_field_dirac_branch_reference import compare_fixed_fields


def test_gf_gap_certificate_is_independent_of_scalar_shift():
    dtype = torch.complex128
    h = torch.tensor([[4.0, 0.7j], [-0.7j, -3.0]], dtype=dtype)
    g = torch.tensor([0])
    f = torch.tensor([1])
    cert = certify_gf_interval_from_indices(h, g, f)
    shift = 17.25
    shifted = certify_gf_interval_from_indices(
        h + shift * torch.eye(2, dtype=dtype), g, f)
    assert cert["safe"] and shifted["safe"]
    assert shifted["certified_W_interval_open_mev"] == pytest.approx(
        [cert["certified_W_interval_open_mev"][0] + shift,
         cert["certified_W_interval_open_mev"][1] + shift])
    assert shifted["midpoint_W_mev"] == pytest.approx(cert["midpoint_W_mev"] + shift)
    assert shifted["A_lower_bound_at_midpoint_mev"] == pytest.approx(
        cert["A_lower_bound_at_midpoint_mev"])
    spectrum = torch.linalg.eigvalsh(h)
    check = validate_spectrum_for_gf_interval(spectrum, cert)
    assert check["normal_spectrum_confirms_certificate"]
    assert check["eigenvalues_below_W"] == check["eigenvalues_above_W"] == 1
    # Different W values inside one certified empty interval must define the
    # same spectral projector, independent of the scalar mass origin.
    values, vectors = torch.linalg.eigh(h)
    lo, hi = cert["certified_W_interval_open_mev"]
    w1, w2 = lo + .2 * (hi-lo), lo + .8 * (hi-lo)
    p1 = vectors[:, values > w1] @ vectors[:, values > w1].mH
    p2 = vectors[:, values > w2] @ vectors[:, values > w2].mH
    assert int((values > w1).sum()) == int((values > w2).sum()) == 1
    assert torch.allclose(p1, p2, atol=1e-14, rtol=0.)


def test_actual_basis_gf_indices_partition_primitive_rows_and_check_mass():
    basis = AngularRadialBasis(radius_fm=8.0, spacing_fm=1.0, kappa_max=1,
                               derivative_order=5)
    g, f = gf_indices_from_basis(basis)
    assert g.numel() == f.numel() == basis.dimension // 2
    assert torch.equal(torch.sort(torch.cat((g, f))).values,
                       torch.arange(basis.dimension))
    with pytest.raises(ValueError, match="mass"):
        certify_basis_gf_interval(torch.eye(basis.dimension, dtype=torch.complex128),
                                  basis, mass_mev=939.0, species="n")


def test_nonempty_certificate_required():
    h = torch.diag(torch.tensor([-5.0, 5.0], dtype=torch.complex128))
    # G lies below F, so no W makes G-W positive and F-W negative.
    with pytest.raises(CertifiedGFGapError):
        certify_gf_interval_from_indices(h, torch.tensor([0]), torch.tensor([1]))


def test_certificate_rejects_nonfinite_h_and_records_small_symmetrization():
    h = torch.tensor([[4.0, 0.7j], [-0.7j + 1e-12, -3.0]], dtype=torch.complex128)
    cert = certify_gf_interval_from_indices(h, torch.tensor([0]), torch.tensor([1]))
    assert cert["h_symmetrization_correction_max_abs_mev"] > 0.
    bad = h.clone()
    bad[0, 0] = complex(float("nan"), 0.)
    with pytest.raises(ValueError, match="NaN"):
        certify_gf_interval_from_indices(bad, torch.tensor([0]), torch.tensor([1]))


def test_explicit_certified_policy_runs_tiny_actual_basis_reference():
    basis = AngularRadialBasis(radius_fm=8.0, spacing_fm=1.0, kappa_max=1,
                               derivative_order=5)
    g, f = gf_indices_from_basis(basis)
    h = torch.zeros((basis.dimension, basis.dimension), dtype=torch.complex128)
    h[g, g] = torch.linspace(.5, 2.5, g.numel(), dtype=torch.float64).to(torch.complex128)
    h[f, f] = torch.linspace(-6., -3., f.numel(), dtype=torch.float64).to(torch.complex128)
    result = compare_fixed_fields(h, torch.zeros_like(h), mass_mev=938.9,
        target_number=2, continuation_steps=1, sector_policy="certified_gf_gap",
        basis=basis, species="n")
    report = result.report
    assert report["sector_policy"] == "certified_gf_gap"
    assert report["normal_dirac_split"]["normal_spectrum_confirms_certificate"]
    assert report["normal_dirac_split"]["eigenvalues_below_W"] == basis.dimension // 2
    assert report["normal_dirac_split"]["eigenvalues_above_W"] == basis.dimension // 2
    assert report["conditional_reference_hard_gates"]["all_passed"]
    assert report["conditional_reference_hard_gates"]["physical_solution_accepted"] is False
    assert report["chemical_potential_source"] == "normal-spectrum midpoint"
    assert report["strict_normal_positive_projected_bdg"][
        "same_full_field_commutator_frobenius_mev"] >= 0.
