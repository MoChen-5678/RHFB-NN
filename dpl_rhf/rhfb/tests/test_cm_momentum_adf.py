from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dpl_rhf.rhfb import cm_momentum_adf as adf_cm
from dpl_rhf.rhfb.basis import AngularRadialBasis


def _operators():
    p = np.zeros((3, 2, 2), dtype=np.complex128)
    p[0] = np.array([[0.0, -1j], [1j, 0.0]])
    p[1] = np.array([[0.2, 0.1], [0.1, -0.2]])
    p2 = np.array([[1.4, 0.1], [0.1, 1.7]], dtype=np.complex128)
    return adf_cm.BasisMomentumOperators(
        momentum_fm_inverse=p,
        momentum_squared_fm_inverse2=p2,
        raw_momentum_fm_inverse=p.copy(),
        integration_by_parts_defect_fm_inverse=np.zeros_like(p),
        basis_signature="basis-hash",
        operator_sha256="operators-hash",
        hbar_c_mev_fm=197.3269804,
        diagnostics={"physical_convergence_certified": False},
    )


def _states():
    q = torch.eye(2, dtype=torch.complex128)
    rho = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.complex128))
    chi = torch.zeros((2, 2), dtype=torch.complex128)
    state = SimpleNamespace(q=q, rho_active=rho, chi_active=chi)
    return {"n": state, "p": state}


def test_adf_com_evaluation_is_only_accepted_with_matching_full_certificate():
    ops = _operators()
    common = dict(states=_states(), operators=ops, mass_number=16,
                  nucleon_mass_mev=938.9)
    unvalidated = adf_cm.evaluate_basis_center_of_mass_post_variation(**common)
    assert unvalidated["included_in_variation"] is False
    assert unvalidated["physical_result_accepted"] is False
    assert unvalidated["external_validation_asserted"] is False
    assert unvalidated["external_validation_machine_verified"] is False
    assert unvalidated["energy_correction_mev"] != 0.0

    certificate = {
        "certificate_id": "independent-validation-record",
        "basis_signature": ops.basis_signature,
        "operator_sha256": ops.operator_sha256,
        "checks": {name: True for name in adf_cm._REQUIRED_VALIDATION},
    }
    declared = adf_cm.evaluate_basis_center_of_mass_post_variation(
        **common, validation_certificate=certificate
    )
    assert declared["external_validation_asserted"] is True
    assert declared["external_validation_machine_verified"] is False
    assert declared["physical_result_accepted"] is False
    assert "normal_field" not in declared and "pairing_field" not in declared

    wrong_basis = dict(certificate, basis_signature="different")
    mismatch = adf_cm.evaluate_basis_center_of_mass_post_variation(
        **common, validation_certificate=wrong_basis
    )
    assert mismatch["physical_result_accepted"] is False


def test_actual_adf_basis_builder_matches_spherical_gaussian_p2_and_observer(monkeypatch):
    import dpl_rhf.rhfb.execution_limits as limits
    monkeypatch.setattr(limits, "require_bounded_cgroup", lambda: {"bounded": True})

    basis = AngularRadialBasis(
        radius_fm=4.0, spacing_fm=0.2, kappa_max=1,
        derivative_order=5, hbar_c_mev_fm=197.328284,
    )
    result = adf_cm.basis_momentum_operators(
        basis, column_chunk_size=8, device="cpu")
    assert result.status == "candidate_unvalidated"
    assert result.hbar_c_mev_fm == pytest.approx(197.328284)
    assert max(result.diagnostics["p_hermiticity_relative_by_axis"]) < 1e-10
    assert result.diagnostics["p2_hermiticity_relative"] < 1e-10
    assert "reduced G,F" in result.diagnostics["radial_representation"]

    # Independent Cartesian reference: psi=(exp(-r^2/2a^2)Y_00, 0)
    # has <p^2>=3/(2a^2).  Its exact outer-surface density envelope is
    # R^2 exp(-R^2/a^2); the origin flux vanishes with the r^2 surface area.
    width = 1.0
    channel = next(c for c in basis.channels if c.kappa == -1 and c.m2 == 1)
    reduced_g = basis.x * np.exp(-basis.x**2 / (2.0 * width**2))
    vector = np.zeros(basis.dimension, dtype=np.complex128)
    vector[channel.offset:channel.offset + basis.n_radial] = (
        np.sqrt(basis.spacing_fm) * reduced_g)
    norm2 = float(np.vdot(vector, vector).real)
    p2_observed = float(np.vdot(
        vector, result.momentum_squared_fm_inverse2 @ vector).real / norm2)
    p2_reference = 3.0 / (2.0 * width**2)
    assert p2_observed == pytest.approx(p2_reference, rel=0.08)
    outer_flux_envelope = basis.radius_fm**2 * np.exp(-basis.radius_fm**2 / width**2)
    assert outer_flux_envelope < 1e-5
    assert result.diagnostics["surface_flux_zero_assumed_not_certified"] is True

    state = SimpleNamespace(
        q=torch.eye(basis.dimension, dtype=torch.complex128)[:, :1],
        rho_active=torch.ones((1, 1), dtype=torch.complex128),
        chi_active=torch.zeros((1, 1), dtype=torch.complex128),
    )
    states = {"n": state, "p": state}
    observer = adf_cm.PostVariationCoMObserver(
        basis, result, mass_number=16, nucleon_mass_mev=938.9)
    metadata = observer.checkpoint_metadata()
    assert metadata["hbar_c_mev_fm"] == pytest.approx(197.328284)
    assert metadata["basis_signature"] == result.basis_signature
    assert metadata["observer_sha256"]
    value = observer.evaluate(states)
    assert value["physical_result_accepted"] is False
    assert value["included_in_variation"] is False
    assert value["hbar_c_mev_fm"] == pytest.approx(197.328284)

    certificate = {
        "certificate_id": "external-only",
        "basis_signature": result.basis_signature,
        "operator_sha256": result.operator_sha256,
        "checks": {name: True for name in adf_cm._REQUIRED_VALIDATION},
    }
    certified_observer = adf_cm.PostVariationCoMObserver(
        basis, result, mass_number=16, nucleon_mass_mev=938.9,
        validation_certificate=certificate)
    certified_metadata = certified_observer.checkpoint_metadata()
    assert certified_metadata["validation_certificate"] == certificate
    assert certified_metadata["observer_sha256"] != metadata["observer_sha256"]
    declared_value = certified_observer.evaluate(states)
    assert declared_value["external_validation_asserted"] is True
    assert declared_value["external_validation_machine_verified"] is False
    assert declared_value["physical_result_accepted"] is False


def test_weak_pz_s_to_p12_matches_cartesian_oscillator_reference_and_converges(monkeypatch):
    import dpl_rhf.rhfb.execution_limits as limits
    monkeypatch.setattr(limits, "require_bounded_cgroup", lambda: {"bounded": True})

    width = 1.1
    reference = -1j / (np.sqrt(6.0) * width)
    errors = []
    ibp_defects = []
    for spacing in (0.2, 0.1):
        basis = AngularRadialBasis(
            radius_fm=5.0, spacing_fm=spacing, kappa_max=1,
            derivative_order=5, hbar_c_mev_fm=197.328284,
        )
        operators = adf_cm.basis_momentum_operators(
            basis, column_chunk_size=8, device="cpu")

        s_channel = next(c for c in basis.channels if c.kappa == -1 and c.m2 == 1)
        p_channel = next(c for c in basis.channels if c.kappa == 1 and c.m2 == 1)
        gs = basis.x * np.exp(-basis.x**2 / (2.0 * width**2))
        gp = basis.x**2 * np.exp(-basis.x**2 / (2.0 * width**2))
        s_state = np.zeros(basis.dimension, dtype=np.complex128)
        p_state = np.zeros_like(s_state)
        s_state[s_channel.offset:s_channel.offset + basis.n_radial] = np.sqrt(spacing) * gs
        p_state[p_channel.offset:p_channel.offset + basis.n_radial] = np.sqrt(spacing) * gp
        s_state /= np.linalg.norm(s_state)
        p_state /= np.linalg.norm(p_state)
        observed = np.vdot(p_state, operators.momentum_fm_inverse[2] @ s_state)
        raw_observed = np.vdot(p_state, operators.raw_momentum_fm_inverse[2] @ s_state)
        ibp_observed = np.vdot(
            p_state, operators.integration_by_parts_defect_fm_inverse[2] @ s_state)
        assert observed - raw_observed == pytest.approx(0.5j * ibp_observed, abs=1e-10)
        errors.append(abs(observed - reference) / abs(reference))
        ibp_defects.append(abs(ibp_observed) / abs(reference))
        assert np.max(np.abs(
            operators.momentum_fm_inverse[2]
            - operators.momentum_fm_inverse[2].conj().T)) < 1e-10

        # This normalized HO pair has a Gaussian tail at the outer box; the
        # surface-density envelope bounds the omitted surface-flux term.
        surface_envelope = basis.radius_fm**2 * np.exp(
            -basis.radius_fm**2 / width**2)
        assert surface_envelope < 1e-5

    assert errors[1] < errors[0]
    assert errors[1] < 0.08
    assert ibp_defects[1] < ibp_defects[0]
