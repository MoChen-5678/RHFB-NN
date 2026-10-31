"""Numerical equivalence of streamed PKA1 fields under x-tile repartitioning.

Run only through scripts/run_rhfb_limited.py (kernel cgroup <=4 GiB, no swap).
This compares the same fixed pure Slater densities; it does not train a model.
"""
import gc
import os

import pytest
import torch

from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.cuda_resources import configure_cuda
from dpl_rhf.rhfb.execution_limits import require_bounded_cgroup
from dpl_rhf.rhfb.manifold import generalized_density_matrix
from dpl_rhf.rhfb.operators import DensityFactors
from dpl_rhf.rhfb.streamed_normal_functional import StreamedPrimitiveNormalFunctional


@pytest.fixture(scope="module")
def fixed_o17_like_slater_states():
    if os.environ.get("RHFB_CUDA_UUID"):
        budget_text = os.environ.get("RHFB_CUDA_BUDGET_MIB")
        if budget_text is None:
            raise RuntimeError("protected CUDA test job omitted RHFB_CUDA_BUDGET_MIB")
        cuda_evidence = configure_cuda("cuda:0", budget_mib=int(budget_text))
        device = torch.device(cuda_evidence["device"])
    else:
        require_bounded_cgroup()
        device = torch.device("cpu")
    # K=1 gives four angular channels. With six radial points, D=48;
    # 6 radial x (3 theta x 6 phi) points gives P=108.
    basis = AngularRadialBasis(
        radius_fm=1.4, spacing_fm=0.2, kappa_max=1,
        n_theta=3, n_phi=6, derivative_order=5,
    )
    assert basis.dimension == 48
    assert basis.point_count == 108

    generator = torch.Generator(device="cpu").manual_seed(20261015)
    states = {}
    for species, occupied_count in (("n", 3), ("p", 2)):
        raw = torch.randn((basis.dimension, 6), dtype=torch.float64,
                          generator=generator).to(torch.complex128)
        raw = raw + 1j * torch.randn((basis.dimension, 6), dtype=torch.float64,
                                     generator=generator)
        q, _ = torch.linalg.qr(raw, mode="reduced")
        q = q.to(device=device)
        occupations = torch.tensor(
            [1.0] * occupied_count + [0.0] * (6 - occupied_count),
            dtype=torch.float64, device=device,
        )
        rho = torch.diag(occupations).to(torch.complex128)
        chi = torch.zeros((6, 6), dtype=torch.complex128, device=device)
        assert q.device == rho.device == chi.device == device
        assert torch.allclose(q.mH @ q, torch.eye(6, dtype=torch.complex128,
                                                   device=device),
                              atol=1e-13, rtol=0.0)
        active_r = generalized_density_matrix(rho, chi)
        assert torch.allclose(active_r @ active_r, active_r,
                              atol=1e-13, rtol=0.0)
        states[species] = DensityFactors(q, rho, chi)
        torch.testing.assert_close(
            states[species].particle_number(),
            torch.tensor(float(occupied_count), dtype=torch.float64, device=device),
            atol=1e-13, rtol=0.0,
        )
    assert states["n"].rho_active.diagonal().real.sum().item() == 3.0
    assert states["p"].rho_active.diagonal().real.sum().item() == 2.0
    return basis, states


def test_full_pka1_energy_channels_and_h_are_tile_size_equivalent(
        fixed_o17_like_slater_states):
    # This is a real streamed PKA1 evaluation. The production provider itself
    # also calls this guard before allocating numerical workspaces.
    require_bounded_cgroup()
    basis, states = fixed_o17_like_slater_states
    tile_sizes = (8, 16, 32)
    def evaluate_one_tile(tile):
        backend = StreamedPrimitiveNormalFunctional(
            basis, tile_size=tile, chunk_size=32, lmax=2,
        )
        energy, fields = backend.differentiable_energy_and_fields(states)
        # Keep only tiny detached outputs; release the custom-AD result, fields,
        # backend, and any tape before evaluating the next partition.
        total = energy.detach().clone()
        channels = {name: value.detach().clone()
                    for name, value in backend.last_energy_terms.items()}
        primitive_fields = {species: value.detach().clone()
                            for species, value in fields.items()}
        del energy, fields, backend
        gc.collect()
        return total, channels, primitive_fields

    expected_channels = {
        "kinetic", "sigma_direct", "sigma_fock",
        "omega_direct", "omega_fock", "rho_vector_direct", "rho_vector_fock",
        "rho_tensor_direct", "rho_tensor_fock", "rho_vt_direct", "rho_vt_fock",
        "rho_tv_direct", "rho_tv_fock", "pion_pv_direct", "pion_pv_fock",
        "coulomb_direct", "coulomb_fock", "rho_tensor_contact",
        "pion_pv_contact",
    }
    reference = None
    for tile in tile_sizes:
        candidate = evaluate_one_tile(tile)
        energy, channels, primitive_fields = candidate
        assert set(channels) == expected_channels
        if reference is None:
            reference = candidate
            continue
        reference_energy, reference_channels, reference_fields = reference
        torch.testing.assert_close(
            energy, reference_energy,
            atol=1e-9, rtol=1e-11, msg=f"total energy, tile={tile}",
        )
        for channel in sorted(expected_channels):
            torch.testing.assert_close(
                channels[channel], reference_channels[channel],
                atol=1e-9, rtol=1e-11,
                msg=f"energy channel {channel}, tile={tile}",
            )
        assert set(primitive_fields) == {"n", "p"}
        for species in ("n", "p"):
            torch.testing.assert_close(
                primitive_fields[species], reference_fields[species],
                atol=1e-9, rtol=1e-11,
                msg=f"full primitive h[{species}], tile={tile}",
            )
        del candidate, energy, channels, primitive_fields
        gc.collect()

