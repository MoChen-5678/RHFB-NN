"""Finite-window branch-filter algebra tests; never run as a training teacher."""
from dataclasses import replace

import numpy as np
import pytest
import torch

import dpl_rhf.rhfb.bdg_branch_filter as branch
from dpl_rhf.rhfb.bdg_branch_filter import (
    BandBoundSource,
    BandEnclosureCertificate,
    mixed_sign_bdg_projector,
    swap_odd_blocked_occupation,
)


def _analytic_weyl_certificate(**changes):
    """Prove the toy bands without diagonalizing H.

    At zero pairing, H0 has magnitudes 1, 20, and an uncoupled spectator
    magnitude 0.4 (each with its particle-hole partner). Pairing couples only
    the p/sea four-dimensional block. For |d| <= .21, the BdG perturbation
    norm is ||delta||2=|d|<=.21; Weyl therefore encloses the coupled physical
    and sea magnitudes by [1-.21,1+.21] and [20-.21,20+.21]. The spectator
    remains exactly at .4 and supplies the physical lower bound. Thus the
    complete absolute bands are [.4,1.21] and [19.79,20.21], with zero gap
    .4 and max magnitude 20.21. This analytic argument uses no eigensolver or
    training-time spectral target.
    """
    values = dict(
        particle_abs_lower=0.40,
        particle_abs_upper=1.21,
        sea_abs_lower=19.79,
        sea_abs_upper=20.21,
        particle_annihilator_energy_sign=1,
        sea_annihilator_energy_sign=-1,
        zero_gap_lower=0.40,
        max_abs_upper=20.21,
        finite_space_complete=True,
        hermitian_verified=True,
        particle_hole_verified=True,
        particle_band_complete=True,
        sea_band_complete=True,
        no_unclassified_modes=True,
        outside_training_spectral_teacher=True,
        source=BandBoundSource.ANALYTIC_TOY,
        source_id="Weyl bound: H0 diag(1,-20,.4), ||pairing perturbation||<=.21",
    )
    values.update(changes)
    return BandEnclosureCertificate(**values)


def _hamiltonian(pairing=0.2, *, requires_grad=False):
    dtype = torch.complex128
    h = torch.diag(torch.tensor([1.0, -20.0, 0.4], dtype=torch.float64)).to(dtype)
    delta0 = torch.tensor(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=dtype,
    )
    d = torch.as_tensor(pairing, dtype=torch.float64)
    delta = d * delta0
    return torch.cat((
        torch.cat((h, delta), dim=1),
        torch.cat((-delta.conj(), -h.conj()), dim=1),
    ), dim=0)


def _spectral_reference(h, divider):
    """Independent NumPy reference, intentionally outside the no-eig path."""
    values, vectors = np.linalg.eigh(h.detach().cpu().numpy())
    sign_product = np.sign(values) * np.sign(divider**2 - values**2)
    return (vectors * ((1.0 - sign_product) / 2.0)[None, :]) @ vectors.conj().T


def _forbid_spectra(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("branch-filter training path must not diagonalize")
    for name in ("eig", "eigh", "eigvals", "eigvalsh"):
        monkeypatch.setattr(torch.linalg, name, forbidden)
        monkeypatch.setattr(np.linalg, name, forbidden)


def _run(h, cert=None):
    return mixed_sign_bdg_projector(
        h, divider=5.0, certificate=cert or _analytic_weyl_certificate(),
        quadrature_order=161, log_step=0.35, tolerance=1.0e-9,
        solve_tolerance=1.0e-11,
    )


def test_full_bdg_filter_matches_independent_reference_and_keeps_cross_chi(monkeypatch):
    h = _hamiltonian(0.2)
    reference = _spectral_reference(h, 5.0)
    monkeypatch.setattr(branch, "require_bounded_cgroup", lambda: {"bounded": True})
    _forbid_spectra(monkeypatch)
    result = _run(h)

    actual = result.generalized_density.detach().numpy()
    np.testing.assert_allclose(actual, reference, atol=1e-10, rtol=0)
    assert result.diagnostics.accepted_as_physics is False
    assert result.diagnostics.status == "finite_window_candidate_not_physics_accepted"
    assert result.diagnostics.product_projector_quadrature_only_error_bound < 1.0e-9
    assert result.diagnostics.max_shifted_solve_residual < 1e-11
    assert result.diagnostics.purity_error < 1e-10
    assert result.diagnostics.particle_hole_error < 1e-10
    assert result.diagnostics.chi_skew_error < 1e-10
    commutator = h @ result.generalized_density - result.generalized_density @ h
    assert torch.linalg.matrix_norm(commutator).item() < 1e-10
    assert abs(result.chi[0, 1]) > 1e-3

    # The mixed-sign BdG branch retains the p-sea anomalous element; a strict
    # normal-positive double projection removes it, so these are distinct.
    p = torch.diag(torch.tensor([1.0, 0.0, 1.0], dtype=result.chi.dtype))
    strict_chi = p @ result.chi @ p.mT
    assert torch.linalg.matrix_norm(result.chi - strict_chi).item() > 1e-3


def test_zero_pairing_limit_has_empty_dirac_sea_density(monkeypatch):
    monkeypatch.setattr(branch, "require_bounded_cgroup", lambda: {"bounded": True})
    _forbid_spectra(monkeypatch)
    result = _run(_hamiltonian(0.0))
    assert torch.linalg.matrix_norm(result.rho).item() < 2e-8
    assert torch.linalg.matrix_norm(result.chi).item() < 2e-8


def test_explicit_particle_hole_swap_produces_odd_parity(monkeypatch):
    monkeypatch.setattr(branch, "require_bounded_cgroup", lambda: {"bounded": True})
    _forbid_spectra(monkeypatch)
    even = _run(_hamiltonian(0.2)).generalized_density
    blocked_unoccupied = torch.zeros(6, dtype=torch.complex128)
    blocked_unoccupied[2] = 1.0  # positive-energy spectator particle vector
    odd, report = swap_odd_blocked_occupation(even, blocked_unoccupied)
    assert report["number_parity"] == -1
    assert report["reference_number_parity"] == 1
    assert torch.linalg.matrix_norm(odd @ odd - odd).item() < 1e-8
    assert torch.real(odd[2, 2]).item() == pytest.approx(1.0, abs=1e-8)


def test_odd_swap_rejects_nonfinite_noncanonical_and_odd_reference(monkeypatch):
    monkeypatch.setattr(branch, "require_bounded_cgroup", lambda: {"bounded": True})
    _forbid_spectra(monkeypatch)
    even = _run(_hamiltonian(0.2)).generalized_density
    w = torch.zeros(6, dtype=torch.complex128)
    w[2] = 1.0

    nonfinite = even.clone()
    nonfinite[0, 0] = complex(float("nan"), 0.0)
    with pytest.raises(ValueError, match="finite"):
        swap_odd_blocked_occupation(nonfinite, w)

    noncanonical = torch.zeros_like(even)
    noncanonical[0, 0] = 1.0  # pure projector, but not a canonical Nambu R
    with pytest.raises(ValueError, match="particle-hole canonical"):
        swap_odd_blocked_occupation(noncanonical, w)

    odd, _ = swap_odd_blocked_occupation(even, w)
    with pytest.raises(ValueError, match="even-parity"):
        swap_odd_blocked_occupation(odd, w)


def test_rational_solve_autograd_matches_centered_finite_difference(monkeypatch):
    monkeypatch.setattr(branch, "require_bounded_cgroup", lambda: {"bounded": True})
    _forbid_spectra(monkeypatch)
    probe = torch.tensor(
        [[0.3, 0.1j, 0, 0, 0.04, 0], [-0.1j, -0.2, 0, 0, 0, 0],
         [0, 0, 0.1, 0, 0, 0], [0, 0, 0, -0.1, 0, 0],
         [0.04, 0, 0, 0, 0.2, 0], [0, 0, 0, 0, 0, -0.3]],
        dtype=torch.complex128,
    )
    scale = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    value = torch.real(torch.sum(probe.conj() * _run(_hamiltonian(scale)).generalized_density))
    (gradient,) = torch.autograd.grad(value, scale)

    def objective(x):
        return float(torch.real(torch.sum(
            probe.conj() * _run(_hamiltonian(x)).generalized_density
        )).detach())

    step = 2.0e-5
    finite_difference = (objective(0.2 + step) - objective(0.2 - step)) / (2 * step)
    assert float(gradient) == pytest.approx(finite_difference, rel=2e-5, abs=2e-7)


def test_overlapping_or_unverified_bands_fail_closed(monkeypatch):
    monkeypatch.setattr(branch, "require_bounded_cgroup", lambda: {"bounded": True})
    h = _hamiltonian(0.2)
    overlap = _analytic_weyl_certificate(particle_abs_upper=5.1)
    with pytest.raises(ValueError, match="strictly between"):
        _run(h, overlap)

    missing_gate = _analytic_weyl_certificate(particle_band_complete=False)
    with pytest.raises(ValueError, match="gates"):
        _run(h, missing_gate)

    forged_source = replace(_analytic_weyl_certificate(), source="looks like a citation")
    with pytest.raises(TypeError, match="BandBoundSource"):
        _run(h, forged_source)
