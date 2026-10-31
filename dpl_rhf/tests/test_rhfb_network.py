from __future__ import annotations

import torch
import pytest

from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.manifold import GeneralizedDensity, generalized_density_matrix, slater_reference
from dpl_rhf.rhfb.network import GeneralizedDensityNet


def make_network(*, odd: bool = True, blocked_index: int = 2) -> GeneralizedDensityNet:
    basis = AngularRadialBasis(
        radius_fm=8.0,
        spacing_fm=0.5,
        kappa_max=2,
        derivative_order=5,
    )
    initial_q, labels = basis.build_initial_q(
        2, oscillator_length_fm=1.8, minimum=6, empty_states=2
    )
    occupations = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    reference = slater_reference(
        occupations,
        blocked_index=blocked_index if odd else None,
        blocking_operation="particle",
    )
    return GeneralizedDensityNet(
        basis,
        initial_q,
        labels,
        reference,
        blocked_index=blocked_index if odd else None,
        hidden=8,
        depth=1,
        radial_basis_size=4,
        channel_chunk_size=3,
    )


def test_forward_returns_orthonormal_q_and_pure_even_or_blocked_density() -> None:
    for odd in (False, True):
        model = make_network(odd=odd)
        q, density = model()
        m = q.shape[1]
        torch.testing.assert_close(q.mH @ q, torch.eye(m, dtype=q.dtype), atol=2e-12, rtol=2e-12)
        assert density.rho.shape == (m, m)
        assert density.chi.shape == (m, m)
        assert density.matrix.shape == (2 * m, 2 * m)
        torch.testing.assert_close(density.rho, density.rho.mH, atol=2e-12, rtol=0.0)
        torch.testing.assert_close(density.chi, -density.chi.mT, atol=2e-12, rtol=0.0)
        torch.testing.assert_close(
            density.matrix @ density.matrix,
            density.matrix,
            atol=2e-11,
            rtol=2e-11,
        )
        assert density.fermion_parity == (-1 if odd else 1)
        assert torch.linalg.matrix_norm(density.chi) > 0.0
        if odd:
            assert torch.count_nonzero(model.raw_b[model.blocked_index]) == 0
            assert torch.count_nonzero(density.chi[model.blocked_index]) == 0


def test_blocked_nambu_mode_tracks_canonical_mixing_when_q_column_is_fixed():
    model = make_network()
    q0, _, mode0 = model.forward_with_blocked_mode()
    with torch.no_grad():
        model.raw_b[2, 4] = .8
        model.raw_b[4, 2] = -.8
        model.raw_a[2, 5] = .6
        model.raw_a[5, 2] = -.6
    q, gd, mode = model.forward_with_blocked_mode()
    torch.testing.assert_close(q, q0)
    # A Q-only overlap would incorrectly be exactly one here.
    assert torch.abs(torch.vdot(mode0, mode)) < .99
    torch.testing.assert_close(torch.vdot(mode, mode).real,
                               torch.tensor(1., dtype=torch.float64), atol=2e-12, rtol=0)
    d = q.shape[0]
    active = torch.cat((q.mH @ mode[:d], q.mT @ mode[d:]))
    torch.testing.assert_close(gd.matrix @ active, active, atol=2e-12, rtol=0)
    assert torch.linalg.vector_norm(mode[d:]) > .1
    objective = mode[d:].abs().square().sum()
    objective.backward()
    assert model.raw_b.grad is not None and model.raw_b.grad.abs().max() > 1e-6
    assert gd.fermion_parity == -1


def test_even_network_has_no_blocked_nambu_vector():
    _, _, mode = make_network(odd=False).forward_with_blocked_mode()
    assert mode is None


def test_direct_complex_coordinates_have_trainable_directions_outside_initial_span() -> None:
    model = make_network()
    q, _ = model()
    generator = torch.Generator().manual_seed(112)
    target = torch.randn(q.shape, generator=generator, dtype=torch.float64).to(q.dtype)
    target = target + 1j * torch.randn(q.shape, generator=generator, dtype=torch.float64)
    q0 = model.initial_q
    external_target = target - q0 @ (q0.mH @ target)
    assert torch.linalg.matrix_norm(external_target) > 1.0

    loss = torch.sum(q.conj() * external_target).real
    loss.backward()
    gradient = model.raw_coordinates.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    external_gradient = gradient - q0 @ (q0.mH @ gradient)
    assert torch.linalg.matrix_norm(external_gradient) > 1.0e-10


def test_component_envelopes_enforce_origin_and_box_regularity() -> None:
    model = make_network()
    basis = model.basis
    scaled = torch.as_tensor(basis.x / basis.radius_fm, dtype=torch.float64)
    for channel in basis.channels:
        if channel.kappa < 0:
            upper_l, lower_l = -channel.kappa - 1, -channel.kappa
        else:
            upper_l, lower_l = channel.kappa, channel.kappa - 1
        upper = scaled.pow(upper_l + 1) * (1.0 - scaled)
        lower = scaled.pow(lower_l + 1) * (1.0 - scaled)
        upper = upper / upper.max()
        lower = lower / lower.max()
        actual_upper = model.envelope[channel.offset : channel.offset + basis.n_radial]
        actual_lower = model.envelope[
            channel.offset + basis.n_radial : channel.offset + 2 * basis.n_radial
        ]
        torch.testing.assert_close(actual_upper, upper, atol=1e-14, rtol=1e-13)
        torch.testing.assert_close(actual_lower, lower, atol=1e-14, rtol=1e-13)
        assert actual_upper[0] > 0.0 and actual_lower[0] > 0.0
        assert actual_upper[-1] < actual_upper.max()
        assert actual_lower[-1] < actual_lower.max()


def test_complex_gradients_are_finite_and_no_eigensolver_is_used(monkeypatch) -> None:
    model = make_network()

    def forbidden(*args, **kwargs):
        raise AssertionError("the generalized-density network must not diagonalize")

    monkeypatch.setattr(torch.linalg, "eigh", forbidden)
    monkeypatch.setattr(torch.linalg, "eigvalsh", forbidden)
    monkeypatch.setattr(torch.linalg, "eig", forbidden)
    q, density = model()
    loss = (
        q[0, 0].real
        + density.rho[0, 0].real
        + density.chi[0, 1].real
        + torch.linalg.matrix_norm(density.chi).square()
    )
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    assert model.raw_coordinates.grad is not None
    assert model.raw_b.grad is not None
    assert model.output_head.weight.grad is not None
    assert torch.linalg.vector_norm(model.output_head.weight.grad) > 0.0


def test_constructor_rejects_a_non_idempotent_reference() -> None:
    basis = AngularRadialBasis(
        radius_fm=8.0,
        spacing_fm=0.5,
        kappa_max=2,
        derivative_order=5,
    )
    q, labels = basis.build_initial_q(2, oscillator_length_fm=1.8, minimum=6, empty_states=2)
    rho = torch.diag(torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.complex128))
    rho[1, 1] = 0.5
    chi = torch.zeros((6, 6), dtype=torch.complex128)
    malformed = GeneralizedDensity(rho, chi, generalized_density_matrix(rho, chi), 1)
    with pytest.raises(ValueError, match="pure and idempotent"):
        GeneralizedDensityNet(basis, q, labels, malformed, hidden=8, radial_basis_size=4)


def test_checkpoint_metadata_rejects_another_blocking_chart() -> None:
    model = make_network()
    state = model.state_dict()
    restored = make_network()
    restored.load_state_dict(state)
    assert restored.checkpoint_metadata() == model.checkpoint_metadata()

    other_block = make_network(blocked_index=3)
    try:
        other_block.load_state_dict(state)
    except RuntimeError as error:
        assert "metadata mismatch" in str(error)
    else:
        raise AssertionError("loading a checkpoint with another block must fail")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_complex_cuda_construction_and_module_to_keep_all_tensors_colocated() -> None:
    cpu_model = make_network()
    model = cpu_model.to("cuda")
    q, density = model()
    assert q.device.type == "cuda"
    assert density.chi.device.type == "cuda"
    loss = q[0, 0].real + density.chi.abs().square().sum()
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert parameter.grad.device.type == "cuda"
            assert torch.isfinite(parameter.grad).all()
