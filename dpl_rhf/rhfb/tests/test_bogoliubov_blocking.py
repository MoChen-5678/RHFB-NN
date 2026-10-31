from __future__ import annotations

import torch
import pytest

from dpl_rhf.rhfb.bogoliubov_blocking import (
    bogoliubov_density,
    bogoliubov_frame,
    slater_bogoliubov_frame,
    state_from_frame,
    swap_quasiparticle_column,
)
from dpl_rhf.rhfb.manifold import slater_reference
from dpl_rhf.rhfb.network import GeneralizedDensityNet
from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.o17_selfconsistent_neural import UnrestrictedO17DensityNet


def _fock_annihilators() -> tuple[torch.Tensor, torch.Tensor]:
    """Two-mode Fock matrices in basis |00>, |10>, |01>, |11>."""
    c1 = torch.zeros((4, 4), dtype=torch.complex128)
    c2 = torch.zeros_like(c1)
    c1[0, 1] = 1.0
    c1[2, 3] = 1.0
    c2[0, 2] = 1.0
    c2[1, 3] = -1.0
    return c1, c2


def _fock_densities(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    annihilators = _fock_annihilators()
    rho = torch.empty((2, 2), dtype=torch.complex128)
    chi = torch.empty_like(rho)
    for a in range(2):
        for b in range(2):
            rho[a, b] = torch.vdot(state, annihilators[b].mH @ annihilators[a] @ state)
            chi[a, b] = torch.vdot(state, annihilators[b] @ annihilators[a] @ state)
    return rho, chi


def test_uv_density_matches_full_occupied_frame_projector() -> None:
    u0, v0 = 0.8, 0.6
    u = u0 * torch.eye(2, dtype=torch.complex128)
    v = torch.tensor([[0.0, v0], [-v0, 0.0]], dtype=torch.complex128)
    frame = bogoliubov_frame(u, v)
    state = state_from_frame(frame, fermion_parity=1)
    occupied = torch.zeros((4, 4), dtype=torch.complex128)
    occupied[2:, 2:] = torch.eye(2, dtype=torch.complex128)
    direct_projector = frame @ occupied @ frame.mH
    density = bogoliubov_density(u, v, fermion_parity=1)
    torch.testing.assert_close(density.matrix, direct_projector, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(state.density.matrix, direct_projector, atol=2e-14, rtol=2e-14)

    # Independent Fock-space contractions fix the pair-tensor orientation.
    vacuum_pair = torch.tensor([u0, 0.0, 0.0, v0], dtype=torch.complex128)
    rho_fock, chi_fock = _fock_densities(vacuum_pair)
    torch.testing.assert_close(density.rho, rho_fock, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(density.chi, chi_fock, atol=2e-14, rtol=2e-14)


def test_explicit_ph_column_swap_matches_one_qp_fock_state() -> None:
    u0, v0 = 0.8, 0.6
    u = u0 * torch.eye(2, dtype=torch.complex128)
    v = torch.tensor([[0.0, v0], [-v0, 0.0]], dtype=torch.complex128)
    even_frame = bogoliubov_frame(u, v)
    blocked = swap_quasiparticle_column(even_frame, 0, fermion_parity=1)

    # Independently perform the stated PH-conjugate column exchange.
    direct_frame = even_frame.clone()
    direct_frame[:, [0, 2]] = even_frame[:, [2, 0]]
    direct_state = state_from_frame(
        direct_frame,
        fermion_parity=-1,
        blocked_index=0,
        blocked_mode_column=2,
    )
    torch.testing.assert_close(blocked.frame, direct_frame, atol=0.0, rtol=0.0)
    torch.testing.assert_close(blocked.density.matrix, direct_state.density.matrix,
                               atol=2e-14, rtol=2e-14)
    assert blocked.density.fermion_parity == -1
    assert blocked.physical_identity_certified is False

    c1, c2 = _fock_annihilators()
    even_fock = torch.tensor([u0, 0.0, 0.0, v0], dtype=torch.complex128)
    beta1_dagger = u0 * c1.mH - v0 * c2
    one_qp_fock = beta1_dagger @ even_fock
    rho_fock, chi_fock = _fock_densities(one_qp_fock)
    torch.testing.assert_close(blocked.density.rho, rho_fock, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(blocked.density.chi, chi_fock, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(blocked.blocked_mode, direct_frame[:, 2], atol=0.0, rtol=0.0)
    assert torch.linalg.vector_norm(blocked.blocked_mode) == 1.0


@pytest.mark.parametrize("network_class", [GeneralizedDensityNet, UnrestrictedO17DensityNet])
def test_network_exposes_transformed_full_active_frame_without_double_blocking(network_class) -> None:
    basis = AngularRadialBasis(
        radius_fm=8.0,
        spacing_fm=0.5,
        kappa_max=2,
        derivative_order=5,
    )
    q0, labels = basis.build_initial_q(
        2, oscillator_length_fm=1.8, minimum=6, empty_states=2
    )
    for occupations, blocked_index, operation in (
        ([1, 1, 0, 0, 0, 0], 2, "particle"),
        ([1, 1, 0, 0, 0, 0], 1, "hole"),
    ):
        reference = slater_reference(
            torch.tensor(occupations, dtype=torch.float64),
            blocked_index=blocked_index,
            blocking_operation=operation,
        )
        model = network_class(
            basis,
            q0,
            labels,
            reference,
            blocked_index=blocked_index,
            hidden=8,
            depth=1,
            radial_basis_size=4,
            channel_chunk_size=3,
        )
        q_default, density_default = model()
        q, density, state = model.forward_with_bogoliubov_frame()
        q_mode, _gd_mode, old_api_mode = model.forward_with_blocked_mode()
        torch.testing.assert_close(q, q_default, atol=0.0, rtol=0.0)
        torch.testing.assert_close(q_mode, q, atol=0.0, rtol=0.0)
        torch.testing.assert_close(density.matrix, density_default.matrix, atol=2e-12, rtol=0.0)
        torch.testing.assert_close(state.density.matrix, density.matrix, atol=2e-12, rtol=0.0)
        assert state.frame.shape == (2 * q.shape[1], 2 * q.shape[1])
        assert state.blocked_mode is not None
        active_mode = state.blocked_mode
        torch.testing.assert_close(density.matrix @ active_mode, active_mode, atol=2e-12, rtol=0.0)
        primitive_mode = torch.cat(
            (q @ active_mode[: q.shape[1]], q.conj() @ active_mode[q.shape[1] :])
        )
        torch.testing.assert_close(old_api_mode, primitive_mode, atol=2e-12, rtol=2e-12)
        assert state.physical_identity_certified is False


def test_slater_frame_reconstructs_reference_density() -> None:
    occupations = torch.tensor([1, 0, 1], dtype=torch.float64)
    frame = slater_bogoliubov_frame(occupations)
    state = state_from_frame(frame, fermion_parity=-1)
    reference = slater_reference(occupations)
    torch.testing.assert_close(state.density.matrix, reference.matrix, atol=0.0, rtol=0.0)
