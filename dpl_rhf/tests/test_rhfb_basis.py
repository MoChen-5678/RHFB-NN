from __future__ import annotations

import numpy as np
import torch

from dpl_rhf.rhfb.basis import AngularRadialBasis, active_size


def small_basis(**overrides) -> AngularRadialBasis:
    options = {
        "radius_fm": 8.0,
        "spacing_fm": 0.5,
        "kappa_max": 4,
        "derivative_order": 5,
    }
    options.update(overrides)
    return AngularRadialBasis(**options)


def test_active_size_uses_occupied_plus_empty_and_complete_multiplets() -> None:
    basis = small_basis()
    assert active_size(8) == 40
    assert basis.active_size(8) == 40
    assert small_basis(kappa_max=6).active_size(50) == 82


def test_angular_channels_and_spinor_harmonics_are_orthonormal() -> None:
    basis = small_basis(kappa_max=3)
    assert basis.angular_state_count == 2 * 3 * 4
    # Sum over both spin components while retaining the angular channel axes.
    weighted_gram = np.einsum(
        "cas,a,das->cd",
        basis._harmonics.conj(),
        basis.angular_weights,
        basis._harmonics,
    )
    np.testing.assert_allclose(weighted_gram, np.eye(basis.angular_state_count), atol=2e-13)

    cos_theta = np.cos(basis.theta)
    sin_theta = np.sin(basis.theta)
    sigma_r = np.zeros((basis.n_angular_points, 2, 2), dtype=np.complex128)
    sigma_r[:, 0, 0] = cos_theta
    sigma_r[:, 1, 1] = -cos_theta
    sigma_r[:, 0, 1] = sin_theta * np.exp(-1j * basis.phi)
    sigma_r[:, 1, 0] = sin_theta * np.exp(1j * basis.phi)
    for index, channel in enumerate(basis.channels):
        partner = basis._harmonics[
            basis.channel_index[(-channel.kappa, channel.m2)]
        ]
        applied = np.einsum("aij,aj->ai", sigma_r, basis._harmonics[index])
        np.testing.assert_allclose(applied, -partner, atol=2e-13)


def test_initial_q_is_complex_orthonormal_and_contains_empty_states() -> None:
    basis = small_basis()
    assert basis.mass_n_mev == 938.9
    assert basis.mass_p_mev == 938.9
    assert basis.hbar_c_mev_fm == 197.328284
    q, labels = basis.build_initial_q(8, oscillator_length_fm=1.8)
    assert q.shape == (basis.dimension, 40)
    assert q.dtype == torch.complex128
    assert len(labels) == 40
    torch.testing.assert_close(q.mH @ q, torch.eye(40, dtype=q.dtype), atol=2e-12, rtol=2e-12)
    assert sum(":m2=" in label for label in labels) == 40

    q_ws, _ = basis.build_initial_q(
        8,
        mass_mev=938.272,
        oscillator_length_fm=1.8,
        initialization="woods-saxon",
        species="p",
        nucleus_a=16,
        proton_number=8,
    )
    torch.testing.assert_close(
        q_ws.mH @ q_ws, torch.eye(40, dtype=q_ws.dtype), atol=2e-12, rtol=2e-12
    )


def test_sampled_spinors_reproduce_radial_metric_and_chunked_sampling() -> None:
    basis = small_basis(kappa_max=2)
    q, _ = basis.build_initial_q(2, oscillator_length_fm=1.8, minimum=6, empty_states=2)
    spinors, weights, positions = basis.sample(q)
    gram = torch.einsum("p,pam,pan->mn", weights, spinors.conj(), spinors)
    torch.testing.assert_close(gram, q.mH @ q, atol=3e-12, rtol=3e-12)
    assert spinors.shape == (basis.point_count, 4, q.shape[1])
    assert weights.shape == (basis.point_count,)
    assert positions.shape == (basis.point_count, 3)

    # The coordinate spinor is G/r and iF/r. Check a pointwise density and
    # the matching r^2 dr dOmega quadrature weight so norm cancellation cannot
    # hide a missing radial Jacobian.
    channel = basis.channels[basis.channel_index[(-1, -1)]]
    radial_index = 3
    angular_index = 2
    point = radial_index * basis.n_angular_points + angular_index
    radius = basis.x[radial_index]
    upper_radial = q[channel.offset + radial_index, 0] / basis.spacing_fm**0.5
    lower_radial = q[
        channel.offset + basis.n_radial + radial_index, 0
    ] / basis.spacing_fm**0.5
    upper_omega = basis._harmonics[basis.channel_index[(-1, -1)]][angular_index]
    lower_omega = basis._harmonics[basis.channel_index[(1, -1)]][angular_index]
    expected_density = (
        upper_radial.abs().square() * np.vdot(upper_omega, upper_omega).real
        + lower_radial.abs().square() * np.vdot(lower_omega, lower_omega).real
    ) / radius**2
    actual_density = spinors[point, :, 0].abs().square().sum()
    torch.testing.assert_close(actual_density, expected_density, atol=2e-13, rtol=2e-12)
    expected_weight = (
        basis.spacing_fm * radius**2 * basis.angular_weights[angular_index]
    )
    assert weights[point].item() == expected_weight

    pieces = list(basis.iter_samples(q, chunk_size=37))
    stitched = torch.cat([piece[1] for piece in pieces], dim=0)
    torch.testing.assert_close(stitched, spinors, atol=0.0, rtol=0.0)
    assert sum(piece[1].shape[0] for piece in pieces) == basis.point_count


def test_complex_active_factors_allow_unrestricted_m_and_kappa_mixing() -> None:
    basis = small_basis()
    q, _ = basis.build_initial_q(8, oscillator_length_fm=1.8)
    seed = torch.Generator().manual_seed(91)
    trial = torch.randn(q.shape[1], q.shape[1], generator=seed, dtype=torch.float64)
    trial = trial.to(torch.complex128) + 1j * torch.randn(
        q.shape[1], q.shape[1], generator=seed, dtype=torch.float64
    )
    unitary, _ = torch.linalg.qr(trial)
    mixed = q @ unitary
    spinors, weights, _ = basis.sample(mixed)
    sampled_gram = torch.einsum("p,pam,pan->mn", weights, spinors.conj(), spinors)
    torch.testing.assert_close(sampled_gram, torch.eye(40, dtype=q.dtype), atol=4e-12, rtol=4e-12)

    # Each mixed state has nonzero amplitude in several angular channels,
    # confirming that the representation does not freeze m or kappa blocks.
    channel_weights = []
    for channel in basis.channels:
        block = mixed[channel.offset : channel.offset + 2 * basis.n_radial]
        channel_weights.append(torch.linalg.vector_norm(block, dim=0))
    assert torch.stack(channel_weights).gt(1e-8).sum(dim=0).min() > 1


def test_free_kinetic_operator_is_hermitian_and_preserves_multiplets() -> None:
    basis = small_basis(kappa_max=2)
    generator = torch.Generator().manual_seed(27)
    left = torch.randn(basis.dimension, 2, generator=generator, dtype=torch.float64).to(torch.complex128)
    right = torch.randn(basis.dimension, 2, generator=generator, dtype=torch.float64).to(torch.complex128)
    h_left = basis.apply_kinetic(left, species="n")
    h_right = basis.apply_kinetic(right, species="n")
    lhs = torch.sum(left.conj() * h_right)
    rhs = torch.sum(h_left.conj() * right)
    torch.testing.assert_close(lhs, rhs, atol=2e-9, rtol=2e-12)


def test_memory_estimate_separates_factor_and_full_sample_storage() -> None:
    basis = small_basis(kappa_max=2)
    estimate = basis.memory_estimate(12)
    assert estimate["factor_bytes"] == basis.dimension * 12 * 16
    assert estimate["sampled_full_grid_bytes"] == basis.point_count * 4 * 12 * 16
    assert estimate["sampled_chunk_bytes"] == min(1024, basis.point_count) * 4 * 12 * 16
    assert estimate["sampled_full_grid_bytes"] > estimate["factor_bytes"]
