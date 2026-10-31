from __future__ import annotations

import pytest
import torch

from dpl_rhf.rhfb.currents import invariant_baryon_density, local_bilinears
from dpl_rhf.rhfb.pairing import (
    D1S,
    GognyD1S,
    pairing_gap,
    pairing_gap_from_sampler,
    pairing_gap_oracle,
)


def _random_case(seed: int = 17):
    torch.manual_seed(seed)
    points, states = 4, 3
    q = torch.randn(points, 4, states, dtype=torch.complex128)
    z = torch.randn(states, states, dtype=torch.complex128)
    chi = (z - z.T) / 3.0
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [0.45, 0.0, 0.0], [1.0, 0.2, 0.0], [1.7, 0.0, 0.0]],
        dtype=torch.float64,
    )
    weights = torch.tensor([0.2, 0.3, 0.25, 0.15], dtype=torch.float64)
    return q, chi, xyz, weights


@pytest.mark.parametrize("species", ["neutron", "proton"])
def test_streamed_pairing_matches_explicit_antisymmetrized_oracle(species: str) -> None:
    q, chi, xyz, weights = _random_case()
    e, gap = pairing_gap(q, chi, xyz, weights, species=species, chunk_size=2)
    e_ref, gap_ref = pairing_gap_oracle(q, chi, xyz, weights, species=species)
    torch.testing.assert_close(e, e_ref, rtol=2e-12, atol=2e-11)
    torch.testing.assert_close(gap, gap_ref, rtol=2e-12, atol=2e-11)
    torch.testing.assert_close(gap, -gap.T, rtol=0, atol=2e-12)
    torch.testing.assert_close(e, 0.5 * torch.sum(chi.conj() * gap).real)


def test_complex_directional_derivative_matches_returned_gap() -> None:
    q, chi, xyz, weights = _random_case(seed=31)
    chi.requires_grad_(True)
    z = torch.randn_like(chi)
    direction = z - z.T
    energy, gap = pairing_gap(q, chi, xyz, weights, species="proton", chunk_size=3)
    grad, = torch.autograd.grad(energy, chi)
    torch.testing.assert_close(grad, gap, rtol=2e-12, atol=2e-11)

    eps = 1e-6
    with torch.no_grad():
        e_plus = pairing_gap(q, chi + eps * direction, xyz, weights, species="proton")[0]
        e_minus = pairing_gap(q, chi - eps * direction, xyz, weights, species="proton")[0]
    finite_difference = (e_plus - e_minus) / (2 * eps)
    analytic = torch.sum(direction.conj() * gap).real
    torch.testing.assert_close(finite_difference, analytic, rtol=3e-8, atol=3e-7)


def test_global_pair_gauge_and_coordinate_permutation_invariance() -> None:
    q, chi, xyz, weights = _random_case(seed=43)
    energy, gap = pairing_gap(q, chi, xyz, weights, species="neutron")
    phase = torch.exp(torch.tensor(0.731j, dtype=torch.complex128))
    energy_phase, gap_phase = pairing_gap(phase * q, chi, xyz, weights, species="neutron")
    torch.testing.assert_close(energy_phase, energy, rtol=2e-12, atol=2e-11)
    torch.testing.assert_close(gap_phase, gap, rtol=2e-12, atol=2e-11)

    perm = torch.tensor([2, 0, 3, 1])
    energy_perm, gap_perm = pairing_gap(
        q[perm], chi, xyz[perm], weights[perm], species="neutron", chunk_size=1
    )
    torch.testing.assert_close(energy_perm, energy, rtol=2e-12, atol=2e-11)
    torch.testing.assert_close(gap_perm, gap, rtol=2e-12, atol=2e-11)


def test_two_spin_singlet_pair_is_attractive_and_uses_unscaled_d1s() -> None:
    q = torch.zeros((2, 4, 2), dtype=torch.complex128)
    q[0, 0, 0] = 1.0  # upper spin-up orbital at point 0
    q[1, 1, 1] = 1.0  # upper spin-down orbital at point 1
    chi = torch.tensor([[0, 1], [-1, 0]], dtype=torch.complex128)
    xyz = torch.tensor([[0.0, 0, 0], [2.0, 0, 0]], dtype=torch.float64)
    weights = torch.ones(2, dtype=torch.float64)
    energy, gap = pairing_gap(q, chi, xyz, weights, species="neutron")
    assert energy.item() < 0.0
    assert gap[0, 1].real.item() < 0.0
    assert D1S.strength == 1.0
    assert D1S.m_mev[1] == -223.934
    with pytest.raises(ValueError, match="strength to 1"):
        pairing_gap(q, chi, xyz, weights, species="neutron", force=GognyD1S(strength=0.9))


def test_lower_dirac_components_contribute_to_pairing() -> None:
    q, chi, xyz, weights = _random_case(seed=57)
    upper_only = q.clone()
    upper_only[:, 2:, :] = 0
    e_upper, d_upper = pairing_gap(upper_only, chi, xyz, weights, species="proton")
    e_full, d_full = pairing_gap(q, chi, xyz, weights, species="proton")
    assert not torch.allclose(e_upper, e_full, rtol=1e-9, atol=1e-9)
    assert not torch.allclose(d_upper, d_full, rtol=1e-9, atol=1e-9)
    e_ref, d_ref = pairing_gap_oracle(q, chi, xyz, weights, species="proton")
    torch.testing.assert_close(e_full, e_ref, rtol=2e-12, atol=2e-11)
    torch.testing.assert_close(d_full, d_ref, rtol=2e-12, atol=2e-11)


def test_checkpointed_two_endpoint_sampler_bounds_chunks_and_preserves_q_gradients() -> None:
    q, chi, xyz, weights = _random_case(seed=63)
    q_plain = q.detach().clone().requires_grad_(True)
    chi_plain = chi.detach().clone().requires_grad_(True)
    e_plain, d_plain = pairing_gap(
        q_plain, chi_plain, xyz, weights, species="neutron", chunk_size=2
    )
    gq_plain, gc_plain = torch.autograd.grad(e_plain, (q_plain, chi_plain))

    q_checkpointed = q.detach().clone().requires_grad_(True)
    chi_checkpointed = chi.detach().clone().requires_grad_(True)
    calls: list[tuple[int, int]] = []

    def sampler(start: int, stop: int):
        calls.append((start, stop))
        return q_checkpointed[start:stop], weights[start:stop], xyz[start:stop]

    e_checkpointed, d_checkpointed = pairing_gap_from_sampler(
        chi_checkpointed,
        q.shape[0],
        sampler,
        species="neutron",
        chunk_size=2,
        checkpoint_blocks=True,
    )
    forward_sample_calls = len(calls)
    gq_checkpointed, gc_checkpointed = torch.autograd.grad(
        e_checkpointed, (q_checkpointed, chi_checkpointed)
    )
    torch.testing.assert_close(e_checkpointed, e_plain, rtol=2e-12, atol=2e-11)
    torch.testing.assert_close(d_checkpointed, d_plain, rtol=2e-12, atol=2e-11)
    torch.testing.assert_close(gq_checkpointed, gq_plain, rtol=2e-11, atol=2e-10)
    torch.testing.assert_close(gc_checkpointed, gc_plain, rtol=2e-11, atol=2e-10)
    assert calls and max(stop - start for start, stop in calls) <= 2
    assert (0, q.shape[0]) not in calls
    assert len(calls) > forward_sample_calls  # backward regenerated sampler blocks

    dq = torch.randn_like(q_checkpointed)
    eps = 1.0e-6
    with torch.no_grad():
        q_plus = q_checkpointed + eps * dq
        q_minus = q_checkpointed - eps * dq
        e_plus = pairing_gap_from_sampler(
            chi_checkpointed.detach(), q.shape[0],
            lambda start, stop: (q_plus[start:stop], weights[start:stop], xyz[start:stop]),
            species="neutron", chunk_size=2, checkpoint_blocks=False,
        )[0]
        e_minus = pairing_gap_from_sampler(
            chi_checkpointed.detach(), q.shape[0],
            lambda start, stop: (q_minus[start:stop], weights[start:stop], xyz[start:stop]),
            species="neutron", chunk_size=2, checkpoint_blocks=False,
        )[0]
    finite_difference = (e_plus - e_minus) / (2 * eps)
    analytic = torch.sum(gq_checkpointed.conj() * dq).real
    torch.testing.assert_close(finite_difference, analytic, rtol=4e-8, atol=4e-7)


def test_pairing_rejects_invalid_species_and_non_skew_chi() -> None:
    q, chi, xyz, weights = _random_case()
    with pytest.raises(ValueError, match="pn pairing"):
        pairing_gap(q, chi, xyz, weights, species="pn")
    with pytest.raises(ValueError, match="skew-symmetric"):
        pairing_gap(q, chi + torch.eye(chi.shape[0], dtype=chi.dtype), xyz, weights, species="neutron")


def test_local_bilinears_retain_currents_axial_tensor_and_lower_signs() -> None:
    q = torch.zeros((1, 4, 1), dtype=torch.complex128)
    q[0, 0, 0] = (0.96) ** 0.5
    q[0, 2, 0] = 0.2
    rho = torch.ones((1, 1), dtype=torch.complex128)
    out = local_bilinears(q, rho)
    torch.testing.assert_close(out["scalar"], torch.tensor([0.92], dtype=torch.float64))
    # A real upper/lower interference gives a nonzero spatial baryon current.
    assert torch.linalg.vector_norm(out["baryon_current"][0, 1:]).item() > 0.3
    assert out["axial_current"].shape == (1, 4)
    assert out["tensor"].shape == (1, 4, 4)
    torch.testing.assert_close(out["tensor"], -out["tensor"].transpose(-1, -2), atol=1e-14, rtol=0)
    rho_b = invariant_baryon_density(out["baryon_current"])
    torch.testing.assert_close(rho_b, torch.tensor([0.92], dtype=torch.float64))

    lower_only = torch.zeros_like(q)
    lower_only[0, 2, 0] = 1
    lower = local_bilinears(lower_only, rho)
    torch.testing.assert_close(lower["scalar"], torch.tensor([-1.0], dtype=torch.float64))
    torch.testing.assert_close(lower["baryon_current"][..., 0], torch.tensor([1.0], dtype=torch.float64))


def test_invariant_baryon_density_rejects_spacelike_except_vacuum() -> None:
    current = torch.tensor(
        [[2.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    torch.testing.assert_close(
        invariant_baryon_density(current),
        torch.tensor([3.75**0.5, 0.0], dtype=torch.float64),
    )
    rho_b = invariant_baryon_density(current)
    (gradient,) = torch.autograd.grad(rho_b[0], current)
    expected_gradient = torch.tensor(
        [[2.0 / 3.75**0.5, -0.5 / 3.75**0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(gradient, expected_gradient)
    with pytest.raises(ValueError, match="future-timelike"):
        invariant_baryon_density(torch.tensor([[1.0, 1.1, 0.0, 0.0]], dtype=torch.float64))
    with pytest.raises(ValueError, match="future-timelike"):
        invariant_baryon_density(torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float64))
