from __future__ import annotations

import math

import pytest
import torch

from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.bogoliubov_blocking import (
    bogoliubov_frame,
    slater_bogoliubov_frame,
)
from dpl_rhf.rhfb.manifold import slater_reference
from dpl_rhf.rhfb.network import GeneralizedDensityNet
from dpl_rhf.rhfb.network_number_transport import (
    apply_mean_number_correction,
    restore_number_transport,
    snapshot_number_transport,
)
from dpl_rhf.rhfb.network import upgrade_legacy_seed_state
from dpl_rhf.rhfb.o17_selfconsistent_neural import UnrestrictedO17DensityNet
from dpl_rhf.rhfb.spatial_parity import ParityConservingDensityNet


def _model(network_class, *, pairing_seed=0.0):
    basis = AngularRadialBasis(
        radius_fm=8.0, spacing_fm=0.5, kappa_max=2, derivative_order=5
    )
    q0, labels = basis.build_initial_q(
        2, oscillator_length_fm=1.8, minimum=6, empty_states=2
    )
    m = q0.shape[1]
    occupations = torch.zeros(m, dtype=torch.float64)
    blocked = m - 1
    reference = slater_reference(occupations, blocked_index=blocked)
    model = network_class(
        basis, q0, labels, reference, blocked_index=blocked, hidden=8,
        depth=1, radial_basis_size=4, channel_chunk_size=3,
        pairing_seed=pairing_seed,
    )
    with torch.no_grad():
        model.raw_a.zero_()
        model.raw_b.zero_()
        model.raw_coordinates.zero_()
    return model, blocked


def _paired_odd_frame(model, blocked, theta=0.4):
    m = model.active_dimension
    assert m >= 6
    u = torch.eye(m, dtype=torch.complex128)
    v = torch.zeros_like(u)
    c, s = math.cos(theta), math.sin(theta)
    for start in (0, 2):
        u[start, start] = c
        u[start + 1, start + 1] = c
        v[start, start + 1] = s
        v[start + 1, start] = -s
    # Keep the selected odd reference orbital occupied; remaining states are
    # particle-empty. This gives an odd pure HFB state with nonzero N20.
    u[blocked, blocked] = 0.0
    v[blocked, blocked] = 1.0
    return bogoliubov_frame(u, v)


@pytest.mark.parametrize("network_class", [GeneralizedDensityNet, UnrestrictedO17DensityNet])
def test_number_retraction_is_the_next_forward_state_without_q_or_parameter_refresh(network_class):
    model, blocked = _model(network_class)
    reference_frame = model._slater_reference_frame()
    desired_frame = _paired_odd_frame(model, blocked)
    # Encode a known paired odd state through the persistent right-reference
    # transport, leaving the neural transform and Q basis unchanged.
    with torch.no_grad():
        model.canonical_number_transport.copy_(reference_frame.mH @ desired_frame)
        model._number_transport_active = True

    q_before, _density_before, state_before = model.forward_with_bogoliubov_frame()
    assert float(torch.trace(state_before.density.rho).real) != pytest.approx(2.0)
    trainable_ids = {name: id(value) for name, value in model.named_parameters()}
    optimizer_parameters = [id(parameter) for parameter in model.parameters()]
    transport_snapshot = snapshot_number_transport(model)

    report = apply_mean_number_correction(model, 2)

    assert report.converged, report
    assert report.transport_updated
    assert report.q_unchanged
    assert report.transport_before_sha256 != report.transport_after_sha256
    assert abs(report.mean_number_after - 2.0) < 1.0e-10
    q_after, density_after, state_after = model.forward_with_bogoliubov_frame()
    torch.testing.assert_close(q_after, q_before, atol=2.0e-12, rtol=0.0)
    torch.testing.assert_close(state_after.frame, report.retraction.state.frame,
                               atol=2.0e-9, rtol=0.0)
    torch.testing.assert_close(density_after.matrix, report.retraction.state.density.matrix,
                               atol=2.0e-9, rtol=0.0)
    assert state_after.density.fermion_parity == -1
    assert state_after.physical_identity_certified is False
    q_mode, gd_mode, primitive_mode = model.forward_with_blocked_mode()
    torch.testing.assert_close(q_mode, q_after, atol=2.0e-12, rtol=0.0)
    torch.testing.assert_close(gd_mode.matrix, density_after.matrix, atol=2.0e-12, rtol=0.0)
    active_mode = state_after.blocked_mode
    expected_mode = torch.cat(
        (q_after @ active_mode[: model.active_dimension],
         q_after.conj() @ active_mode[model.active_dimension :])
    )
    torch.testing.assert_close(primitive_mode, expected_mode, atol=2.0e-12, rtol=0.0)
    assert not state_after.physical_identity_certified
    assert {name: id(value) for name, value in model.named_parameters()} == trainable_ids
    assert [id(parameter) for parameter in model.parameters()] == optimizer_parameters

    restore_number_transport(model, transport_snapshot)
    torch.testing.assert_close(model.canonical_number_transport, transport_snapshot,
                               atol=0.0, rtol=0.0)


def test_zero_number_response_fails_closed_without_inserting_pairing():
    model, _blocked = _model(GeneralizedDensityNet, pairing_seed=0.0)
    before = snapshot_number_transport(model)
    raw_b = model.raw_b.detach().clone()

    report = apply_mean_number_correction(model, 2)

    assert not report.converged
    assert report.reason == "singular_number_response"
    assert not report.transport_updated
    torch.testing.assert_close(model.canonical_number_transport, before, atol=0.0, rtol=0.0)
    torch.testing.assert_close(model.raw_b, raw_b, atol=0.0, rtol=0.0)
    assert torch.count_nonzero(model.raw_b) == 0


def test_transport_buffer_is_checkpointed_and_legacy_state_is_rejected_even_non_strictly():
    model, blocked = _model(GeneralizedDensityNet)
    state = model.state_dict()
    assert "canonical_number_transport" in state
    assert model.checkpoint_metadata()["format"] == "dpl-rhfb-generalized-density-net-v2"
    assert model.checkpoint_metadata()["canonical_number_transport"] == "right-thouless-reference-frame-v1"

    restored, _ = _model(GeneralizedDensityNet)
    restored.load_state_dict(state, strict=True)
    torch.testing.assert_close(restored.canonical_number_transport,
                               model.canonical_number_transport, atol=0.0, rtol=0.0)

    legacy_state = state.copy()
    del legacy_state["canonical_number_transport"]
    legacy_state["_extra_state"] = dict(legacy_state["_extra_state"])
    legacy_state["_extra_state"]["format"] = "dpl-rhfb-generalized-density-net-v1"
    with pytest.raises(RuntimeError, match="missing canonical_number_transport"):
        restored.load_state_dict(legacy_state, strict=False)


def test_transport_rejects_canonical_particle_hole_swap_that_flips_parity():
    model, _blocked = _model(GeneralizedDensityNet)
    occupations = torch.zeros(model.active_dimension, dtype=torch.float64)
    occupations[0] = 1.0
    parity_flipping_swap = slater_bogoliubov_frame(occupations)

    with pytest.raises(ValueError, match="parity-preserving canonical component"):
        model._validate_number_transport(parity_flipping_swap)


def test_parity_network_number_transport_preserves_spatial_parity_and_blocked_frame():
    model, blocked = _model(ParityConservingDensityNet)
    parity = model.orbital_parities
    candidates = [i for i in range(model.active_dimension) if i != blocked]
    pair = next(
        (i, j)
        for offset, i in enumerate(candidates)
        for j in candidates[offset + 1 :]
        if int(parity[i]) == int(parity[j])
    )
    i, j = pair
    m = model.active_dimension
    u = torch.eye(m, dtype=torch.complex128)
    v = torch.zeros_like(u)
    theta = 0.35
    c, s = math.cos(theta), math.sin(theta)
    u[i, i] = u[j, j] = c
    v[i, j] = s
    v[j, i] = -s
    u[blocked, blocked] = 0.0
    v[blocked, blocked] = 1.0
    desired_frame = bogoliubov_frame(u, v)
    reference_frame = model._slater_reference_frame()
    with torch.no_grad():
        model.canonical_number_transport.copy_(reference_frame.mH @ desired_frame)
        model._number_transport_active = True

    report = apply_mean_number_correction(model, 1)
    assert report.converged, report
    q, gd, mode = model.forward_with_blocked_mode()
    parity_sign = model.orbital_parities[:, None] * model.orbital_parities[None, :]
    assert float((gd.rho - gd.rho * parity_sign).abs().max()) < 2.0e-9
    assert float((gd.chi - gd.chi * parity_sign).abs().max()) < 2.0e-9
    assert torch.linalg.vector_norm(mode).item() == pytest.approx(1.0, abs=2.0e-9)
    assert model.checkpoint_metadata()["format"] == "dpl-rhfb-parity-conserving-density-net-v2"


def test_legacy_seed_conversion_is_explicit_and_requires_exact_metadata():
    model, _blocked = _model(GeneralizedDensityNet)
    legacy = model.state_dict()
    del legacy["canonical_number_transport"]
    legacy["_extra_state"] = dict(legacy["_extra_state"])
    legacy["_extra_state"]["format"] = "dpl-rhfb-generalized-density-net-v1"

    upgraded = upgrade_legacy_seed_state(model, legacy)
    torch.testing.assert_close(
        upgraded["canonical_number_transport"],
        torch.eye(2 * model.active_dimension, dtype=torch.complex128),
        atol=0.0,
        rtol=0.0,
    )
    assert upgraded["_extra_state"] == model.checkpoint_metadata()
    restored, _blocked = _model(GeneralizedDensityNet)
    restored.load_state_dict(upgraded, strict=True)

    altered = dict(legacy)
    altered["_extra_state"] = dict(legacy["_extra_state"])
    altered["_extra_state"]["labels"] = ["not-the-same-seed"]
    with pytest.raises(ValueError, match="metadata differs"):
        upgrade_legacy_seed_state(model, altered)

