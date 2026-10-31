"""Isolated mixed-sign matrix-sign prototype tests; never a training path."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dpl_rhf.rhfb.mixed_branch_sign_prototype import (
    BranchAssumptions,
    RationalSignQuadrature,
    audit_complete_band_with_eigh_oracle,
    mixed_branch_projector,
    swap_odd_blocked_occupation,
)
from dpl_rhf.rhfb.parity_diagnostic import number_parity_from_density


def _assumptions(**changes):
    args = dict(
        ec=5.0,
        gap_at_zero=0.35,
        gap_at_plus_ec=3.79,
        gap_at_minus_ec=3.79,
        particle_abs_lower=0.4,
        particle_abs_upper=1.21,
        dirac_abs_lower=19.79,
        complete_particle_band=True,
        complete_dirac_band=True,
        no_unclassified_modes=True,
        evidence="analytic Weyl enclosure for this 6x6 toy only",
    )
    args.update(changes)
    return BranchAssumptions(**args)


def _quadrature():
    return RationalSignQuadrature(log_step=0.18, half_order=180)


def _hamiltonian(pairing=0.0, continuum_energy=0.4):
    dtype = torch.complex128
    h = torch.diag(torch.tensor([1.0, -20.0, continuum_energy], dtype=torch.float64)).to(dtype)
    delta0 = torch.tensor(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=dtype
    )
    d = torch.as_tensor(pairing, dtype=torch.float64)
    delta = d * delta0
    return torch.cat(
        (torch.cat((h, delta), dim=1), torch.cat((-delta.conj(), -h.conj()), dim=1)),
        dim=0,
    )


def _oracle(h, ec):
    values, vectors = np.linalg.eigh(h.detach().cpu().numpy())
    weights = 0.5 * (
        1.0 - np.sign(values) + np.sign(values + ec) + np.sign(values - ec)
    )
    return (vectors * weights[None, :]) @ vectors.conj().T


def test_full_pairing_zero_and_nonzero_match_small_space_oracle_without_training_eigh(monkeypatch):
    h0 = _hamiltonian(0.0)
    h_pair = _hamiltonian(0.2)
    refs = (_oracle(h0, 5.0), _oracle(h_pair, 5.0))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("production prototype must use shifted solves, never eigensolvers")

    monkeypatch.setattr(torch.linalg, "eigh", forbidden)
    monkeypatch.setattr(torch.linalg, "eigvalsh", forbidden)
    for h, reference in zip((h0, h_pair), refs):
        result = mixed_branch_projector(
            h, assumptions=_assumptions(), quadrature=_quadrature()
        )
        np.testing.assert_allclose(result.generalized_density.detach().numpy(), reference, atol=2e-8)
        assert result.diagnostics.physical_branch_certified is False
        assert result.diagnostics.hermiticity_error < 1e-8
        assert result.diagnostics.purity_error < 2e-8
        assert result.diagnostics.particle_hole_error < 2e-8
        assert result.diagnostics.chi_antisymmetry_error < 2e-8
    assert torch.linalg.matrix_norm(
        mixed_branch_projector(h_pair, assumptions=_assumptions(), quadrature=_quadrature()).chi
    ).item() > 1e-3


def test_particle_hole_transformed_full_matrix_matches_its_independent_oracle():
    h = _hamiltonian(0.2)
    transformed = _ph_matrix(h)
    result = mixed_branch_projector(
        transformed, assumptions=_assumptions(), quadrature=_quadrature()
    )
    np.testing.assert_allclose(
        result.generalized_density.detach().numpy(), _oracle(transformed, 5.0), atol=2e-8
    )
    original = mixed_branch_projector(
        h, assumptions=_assumptions(), quadrature=_quadrature()
    ).generalized_density
    identity = torch.eye(h.shape[0], dtype=h.dtype)
    assert torch.linalg.matrix_norm(result.generalized_density - (identity - original)).item() < 2e-8


def test_complex_unitary_nambu_rotation_matches_rotated_density_and_oracle(monkeypatch):
    h = _hamiltonian(0.2)
    # A fixed genuinely complex matrix; QR constructs a unitary without any
    # spectral decomposition. The Nambu transform is diag(U, U*).
    seed = torch.tensor(
        [[1.0 + 0.2j, 0.3 - 0.4j, -0.2 + 0.7j],
         [0.1 + 0.8j, -0.6 + 0.1j, 0.5 + 0.3j],
         [0.4 - 0.5j, 0.2 + 0.9j, 1.1 - 0.2j]],
        dtype=torch.complex128,
    )
    unitary, _ = torch.linalg.qr(seed)
    zero = torch.zeros((3, 3), dtype=torch.complex128)
    transform = torch.cat(
        (torch.cat((unitary, zero), dim=1),
         torch.cat((zero, unitary.conj()), dim=1)),
        dim=0,
    )
    rotated_h = transform @ h @ transform.mH
    rotated_reference = _oracle(rotated_h, 5.0)
    base_reference = _oracle(h, 5.0)
    covariance_reference = transform @ torch.as_tensor(base_reference) @ transform.mH
    np.testing.assert_allclose(rotated_reference, covariance_reference, atol=1e-12)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("mixed-sign algorithm call must not use an eigensolver")

    for name in ("eig", "eigh", "eigvals", "eigvalsh"):
        monkeypatch.setattr(torch.linalg, name, forbidden)
        monkeypatch.setattr(np.linalg, name, forbidden)
    result = mixed_branch_projector(
        rotated_h, assumptions=_assumptions(), quadrature=_quadrature()
    )
    np.testing.assert_allclose(
        result.generalized_density.detach().numpy(), rotated_reference, atol=2e-8
    )
    assert torch.linalg.matrix_norm(
        result.generalized_density - covariance_reference
    ).item() < 2e-8


def test_positive_discrete_continuum_states_are_included_by_the_declared_window():
    result = mixed_branch_projector(
        _hamiltonian(0.0), assumptions=_assumptions(), quadrature=_quadrature()
    )
    # The positive-energy discrete spectator is a Fermi-sector state inside Ec;
    # its Nambu particle component is empty and its PH hole partner is filled.
    assert result.generalized_density[2, 2].real.item() == pytest.approx(0.0, abs=2e-8)
    assert result.generalized_density[5, 5].real.item() == pytest.approx(1.0, abs=2e-8)


def test_odd_occupation_swap_preserves_purity_and_ph_canonical_structure():
    even = mixed_branch_projector(
        _hamiltonian(0.0), assumptions=_assumptions(), quadrature=_quadrature()
    ).generalized_density
    w = torch.zeros(6, dtype=torch.complex128)
    w[2] = 1.0
    odd = swap_odd_blocked_occupation(even, w)
    identity = torch.eye(6, dtype=odd.dtype)
    assert torch.linalg.matrix_norm(odd @ odd - odd).item() < 2e-8
    assert torch.linalg.matrix_norm(_ph_matrix(odd) - (identity - odd)).item() < 2e-8
    assert odd[2, 2].real.item() == pytest.approx(1.0)
    assert odd[5, 5].real.item() == pytest.approx(0.0)
    assert number_parity_from_density(
        odd[:3, :3].detach().numpy(), odd[:3, 3:].detach().numpy(), tolerance=1e-6
    )["number_parity"] == -1


def _ph_matrix(matrix):
    n = matrix.shape[0] // 2
    return torch.cat(
        (
            torch.cat((matrix[n:, n:].conj(), matrix[n:, :n].conj()), dim=1),
            torch.cat((matrix[:n, n:].conj(), matrix[:n, :n].conj()), dim=1),
        ),
        dim=0,
    )


def test_eigh_oracle_marks_fermi_state_outside_window_as_failed_complete_band():
    # The API cannot infer band labels from signs. The independent audit is
    # deliberately separate and explicitly reports the violated enclosure.
    h = _hamiltonian(0.0, continuum_energy=6.0)
    audit = audit_complete_band_with_eigh_oracle(h, _assumptions())
    assert audit.oracle_eigh_call_count == 1
    assert audit.complete_band_holds is False
    assert audit.unclassified_abs_energies == pytest.approx((6.0, 6.0))
    failed_assumptions = _assumptions(complete_particle_band=False)
    with pytest.raises(ValueError, match="complete-band"):
        mixed_branch_projector(h, assumptions=failed_assumptions, quadrature=_quadrature())


def test_invalid_or_missing_spectral_assumptions_fail_closed():
    h = _hamiltonian()
    with pytest.raises(ValueError, match="complete"):
        mixed_branch_projector(
            h, assumptions=_assumptions(complete_particle_band=False), quadrature=_quadrature()
        )
    with pytest.raises(ValueError, match="gap"):
        mixed_branch_projector(
            h, assumptions=_assumptions(gap_at_plus_ec=0.0), quadrature=_quadrature()
        )


def test_gradient_flows_through_shifted_linear_solves():
    pairing = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    h = _hamiltonian(pairing)
    result = mixed_branch_projector(
        h, assumptions=_assumptions(), quadrature=_quadrature()
    )
    probe = torch.zeros_like(result.generalized_density)
    probe[0, 4] = 1.0
    probe[4, 0] = 1.0
    loss = torch.real(torch.sum(probe.conj() * result.generalized_density))
    loss.backward()
    assert pairing.grad is not None
    assert torch.isfinite(pairing.grad)
    expected = 2.0 * 19.0**2 / (19.0**2 + 4.0 * 0.2**2) ** 1.5
    assert pairing.grad.item() == pytest.approx(expected, rel=2e-6, abs=2e-8)


def test_independent_eigh_oracle_rejects_training_path():
    with pytest.raises(RuntimeError, match="forbidden on training paths"):
        audit_complete_band_with_eigh_oracle(
            _hamiltonian(), _assumptions(), training_path=True
        )
