import math

import pytest
import torch

from dpl_rhf.rhfb.blocking_tracking import transport_occupied_tag
from dpl_rhf.rhfb.manifold import GeneralizedDensity, generalized_density_matrix


def _density(rho, chi=None):
    if chi is None:
        chi = torch.zeros_like(rho)
    return GeneralizedDensity(
        rho=rho,
        chi=chi,
        matrix=generalized_density_matrix(rho, chi),
        fermion_parity=-1,
    )


def _slater_density(angle):
    c, s = math.cos(angle), math.sin(angle)
    occupied = torch.tensor([c, s], dtype=torch.complex128)
    rho = occupied[:, None] @ occupied.conj()[None, :]
    return _density(rho)


def test_transport_is_invariant_under_active_occupied_frame_gauge_rotation():
    q = torch.eye(2, dtype=torch.complex128)
    initial_rho = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.complex128))
    original = _density(initial_rho)
    theta = 0.41
    gauge = torch.tensor(
        [[math.cos(theta), -math.sin(theta)],
         [math.sin(theta), math.cos(theta)]],
        dtype=torch.complex128,
    )
    rotated_q = q @ gauge
    rotated_rho = gauge.mH @ initial_rho @ gauge
    rotated = _density(rotated_rho)
    tag = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex128)

    first = transport_occupied_tag(q, original, tag, tag)
    second = transport_occupied_tag(rotated_q, rotated, tag, tag)

    assert torch.allclose(first.mode, second.mode, atol=1e-12, rtol=0.0)
    assert first.tag_resolved is True
    assert first.local_overlap == pytest.approx(1.0)
    assert second.local_overlap == pytest.approx(1.0)
    assert first.physical_identity_certified is False


def test_local_continuation_allows_cumulative_rotation_from_initial_tag():
    q = torch.eye(2, dtype=torch.complex128)
    initial = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex128)
    previous = initial
    last = None
    for step in range(1, 11):
        last = transport_occupied_tag(
            q,
            _slater_density(0.06 * step),
            previous,
            initial_mode=initial,
            minimum_overlap=0.90,
        )
        assert last.local_overlap > 0.99
        previous = last.mode

    assert last is not None
    assert last.tag_resolved is True
    assert last.initial_overlap < 0.90
    assert last.projection_norm == pytest.approx(last.local_overlap)
    assert last.particle_hole_error < 1e-10


def test_overlap_is_diagnostic_and_zero_projection_leaves_trial_unrejected():
    q = torch.eye(2, dtype=torch.complex128)
    density = _density(torch.diag(torch.tensor([1.0, 0.0], dtype=torch.complex128)))
    tag = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex128)

    low_but_resolved = transport_occupied_tag(
        q, _slater_density(1.2), tag, minimum_overlap=0.90
    )
    assert low_but_resolved.tag_resolved is True
    assert low_but_resolved.local_overlap < 0.90
    assert low_but_resolved.above_minimum_overlap is False

    zero_projection = transport_occupied_tag(
        q, _slater_density(math.pi / 2), tag, minimum_overlap=0.90
    )
    assert zero_projection.tag_resolved is False
    assert zero_projection.mode is None
    assert zero_projection.local_overlap < 1e-14
    assert zero_projection.particle_hole_error is None
    assert zero_projection.above_minimum_overlap is False

    with pytest.raises(ValueError, match="normalized"):
        transport_occupied_tag(q, density, 2.0 * tag)
    nan_tag = tag.clone()
    nan_tag[0] = complex(float("nan"), 0.0)
    with pytest.raises(ValueError, match="non-finite"):
        transport_occupied_tag(q, density, nan_tag)


def test_transport_rejects_wrong_particle_hole_structure():
    q = torch.eye(2, dtype=torch.complex128)
    rho = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.complex128))
    wrong_chi = torch.zeros_like(rho)
    wrong_chi[0, 1] = 0.1
    wrong_chi[1, 0] = 0.1  # Pairing tensor must be antisymmetric.
    wrong_density = _density(rho, wrong_chi)
    tag = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex128)

    with pytest.raises(ValueError, match="antisymmetric"):
        transport_occupied_tag(q, wrong_density, tag)
