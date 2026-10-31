from __future__ import annotations

import torch

from dpl_rhf.backends.base import NucleusCase
from dpl_rhf.backends.density_matrix_rmf import DensityMatrixRMFBackend
from dpl_rhf.models.density_matrix_net import DensityMatrixNet
from dpl_rhf.rhf.backend import DensityMatrixRHFBackend
from dpl_rhf.training.inverse_hamiltonian import inverse_hamiltonian_constraint


def test_inverse_hamiltonian_score_separates_positive_and_sea_branches():
    hamiltonian = torch.diag(torch.tensor([-2.0, 1.0], dtype=torch.float64))
    common = {
        "occupations": torch.ones(1, dtype=torch.float64),
        "species_sign": torch.tensor(1.0, dtype=torch.float64),
        "kappa": torch.tensor(-1),
    }
    positive = inverse_hamiltonian_constraint(
        [{**common, "factors": torch.tensor([[0.0], [1.0]], dtype=torch.float64)}],
        {"n_kappa_-1": hamiltonian},
        {"n": -1.0, "p": -1.0},
    )
    sea = inverse_hamiltonian_constraint(
        [{**common, "factors": torch.tensor([[1.0], [0.0]], dtype=torch.float64)}],
        {"n_kappa_-1": hamiltonian},
        {"n": -1.0, "p": -1.0},
    )
    assert positive.minimum_branch_score > 0.0
    assert sea.minimum_branch_score < 0.0
    assert positive.linear_solve_residual < 1.0e-14


def test_inverse_hamiltonian_branch_test_is_occupied_gauge_invariant():
    hamiltonian = torch.diag(
        torch.tensor([-2.0, 0.5, 1.0, 2.0], dtype=torch.float64)
    )
    q = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    angle = torch.tensor(0.37, dtype=torch.float64)
    rotation = torch.stack(
        [
            torch.stack([torch.cos(angle), -torch.sin(angle)]),
            torch.stack([torch.sin(angle), torch.cos(angle)]),
        ]
    )
    common = {
        "occupations": torch.ones(2, dtype=torch.float64),
        "species_sign": torch.tensor(1.0, dtype=torch.float64),
        "kappa": torch.tensor(-1),
    }
    original = inverse_hamiltonian_constraint(
        [{**common, "factors": q}],
        {"n_kappa_-1": hamiltonian},
        {"n": -1.0, "p": -1.0},
    )
    rotated = inverse_hamiltonian_constraint(
        [{**common, "factors": q @ rotation}],
        {"n_kappa_-1": hamiltonian},
        {"n": -1.0, "p": -1.0},
    )
    assert torch.allclose(original.minimum_branch_score, rotated.minimum_branch_score)
    assert torch.allclose(original.mean_branch_score, rotated.mean_branch_score)


def test_inverse_hamiltonian_detects_a_sea_direction_hidden_by_diagonal_scores():
    hamiltonian = torch.diag(
        torch.tensor([-2.0, 1.0, 1.0], dtype=torch.float64)
    )
    q = torch.tensor(
        [
            [2.0**-0.5, 2.0**-0.5],
            [0.5, -0.5],
            [0.5, -0.5],
        ],
        dtype=torch.float64,
    )
    result = inverse_hamiltonian_constraint(
        [{
            "occupations": torch.ones(2, dtype=torch.float64),
            "species_sign": torch.tensor(1.0, dtype=torch.float64),
            "kappa": torch.tensor(-1),
            "factors": q,
        }],
        {"n_kappa_-1": hamiltonian},
        {"n": -1.0, "p": -1.0},
    )
    # Both diagonal quotients vanish, yet the occupied plane contains a pure
    # sea direction. The invariant matrix test must expose it.
    assert result.minimum_branch_score < 0.0


def test_inverse_hamiltonian_auxiliary_step_holds_current_hamiltonian_fixed():
    diagonal = torch.tensor([-2.0, 1.0], dtype=torch.float64, requires_grad=True)
    hamiltonian = torch.diag(diagonal)
    factors = torch.tensor([[0.1], [0.9]], dtype=torch.float64, requires_grad=True)
    factors = factors / torch.linalg.norm(factors)
    result = inverse_hamiltonian_constraint(
        [{
            "occupations": torch.ones(1, dtype=torch.float64),
            "species_sign": torch.tensor(1.0, dtype=torch.float64),
            "kappa": torch.tensor(-1),
            "factors": factors,
        }],
        {"n_kappa_-1": hamiltonian},
        {"n": -1.0, "p": -1.0},
    )
    factor_gradient, hamiltonian_gradient = torch.autograd.grad(
        result.loss, (factors, diagonal), allow_unused=True
    )
    assert factor_gradient is not None
    assert torch.linalg.vector_norm(factor_gradient) > 0.0
    assert hamiltonian_gradient is None


def test_rmf_and_rhf_initialization_use_zero_eigensolves():
    original = torch.linalg.eigh

    def forbidden(*args, **kwargs):
        raise AssertionError("eigh was called during no-refresh initialization")

    torch.linalg.eigh = forbidden
    try:
        rmf = DensityMatrixRMFBackend(
            NucleusCase("PKDD", 8, 8, 16), device=torch.device("cpu")
        )
        rhf = DensityMatrixRHFBackend(
            NucleusCase("PKO2", 8, 8, 16), device=torch.device("cpu")
        )
        for backend in (rmf, rhf):
            network = DensityMatrixNet(
                torch.as_tensor(backend.r), backend.block_specs,
                hidden=8, depth=1, radial_basis_size=4,
            )
            blocks = network()
            assert all("positive_basis" not in block for block in blocks)
            assert backend.initialization_block_diagonalizations == 0
    finally:
        torch.linalg.eigh = original


def test_no_sea_projector_matches_spectral_projector_without_eigensolves():
    from dpl_rhf.training.inverse_hamiltonian import positive_energy_projector

    generator = torch.Generator().manual_seed(7)
    basis, _ = torch.linalg.qr(
        torch.randn(12, 12, dtype=torch.float64, generator=generator)
    )
    spectrum = torch.tensor(
        [-2900.0, -2100.0, -1900.0, -1500.0, -1200.0, -1050.0,
         -60.0, -35.0, -8.0, 4.0, 900.0, 4000.0],
        dtype=torch.float64,
    )
    hamiltonian = basis @ torch.diag(spectrum) @ basis.T
    reference = basis[:, spectrum > -938.9] @ basis[:, spectrum > -938.9].T
    original = torch.linalg.eigh

    def forbidden(*args, **kwargs):
        raise AssertionError("eigh was called by the no-sea projector")

    torch.linalg.eigh = forbidden
    try:
        projector, iterations = positive_energy_projector(hamiltonian, -938.9)
    finally:
        torch.linalg.eigh = original
    assert iterations < 50
    assert torch.allclose(projector, reference, atol=1.0e-12)


def test_no_sea_projection_removes_dirac_sea_from_network_factors():
    from dpl_rhf.training.inverse_hamiltonian import (
        dirac_sea_weight,
        positive_energy_projectors,
    )

    backend = DensityMatrixRHFBackend(
        NucleusCase("PKO2", 8, 8, 16), device=torch.device("cpu")
    )
    network = DensityMatrixNet(
        torch.as_tensor(backend.r), backend.block_specs,
        hidden=8, depth=1, radial_basis_size=4,
    )
    blocks = network()
    # A Hamiltonian whose sea is the free lower-component branch: the free
    # Dirac operator itself, shifted by the rest mass convention.
    hamiltonians = {
        f"{'n' if float(b['species_sign']) > 0 else 'p'}_kappa_{int(b['kappa']):+d}":
        b["free_hamiltonian"] * backend.params.hbar_c
        for b in blocks
    }
    shifts = {"n": -backend.params.mass_n, "p": -backend.params.mass_p}
    no_sea = positive_energy_projectors(hamiltonians, shifts)
    assert no_sea.max_trace_error < 1.0e-8
    # Deliberately mix a lower-component (sea-like) direction into the raw
    # factors; projection must remove it exactly before QR.
    with torch.no_grad():
        head = network.output_heads[0]
        head.bias.copy_(torch.tensor([0.0, 50.0], dtype=torch.float64))
    unprojected = dirac_sea_weight(network(), no_sea.projectors)
    network.set_positive_projectors(no_sea.projectors)
    projected = dirac_sea_weight(network(), no_sea.projectors)
    assert unprojected > 1.0e-6
    assert projected < 1.0e-12


def test_batched_fock_matches_reference_pair_loops():
    torch.set_default_dtype(torch.float64)
    backend = DensityMatrixRHFBackend(
        NucleusCase("PKA1", 8, 12, 20), device=torch.device("cpu")
    )
    network = DensityMatrixNet(
        torch.as_tensor(backend.r), backend.block_specs,
        hidden=8, depth=1, radial_basis_size=4,
    )
    torch.manual_seed(5)
    with torch.no_grad():
        for head in network.output_heads:
            head.weight.normal_(std=0.3)
            head.bias.normal_(std=0.3)
    results = {}
    for batched in (False, True):
        backend.functional.batched_fock = batched
        network.zero_grad()
        result = backend.evaluate(network(), create_graph=True)
        loss = result.energy_total_tensor + sum(
            result.stationarity_quadratic_terms.values()
        )
        loss.backward()
        gradient = torch.cat(
            [p.grad.flatten() for p in network.parameters() if p.grad is not None]
        )
        results[batched] = (result, gradient)
    (reference, reference_gradient), (batched, batched_gradient) = (
        results[False], results[True]
    )
    for name, value in reference.action_terms.items():
        if value.numel() == 1:
            assert torch.allclose(batched.action_terms[name], value, atol=1.0e-9, rtol=0.0), name
    for name, value in reference.hamiltonians.items():
        assert torch.allclose(batched.hamiltonians[name], value, atol=1.0e-9, rtol=0.0), name
    assert torch.allclose(batched_gradient, reference_gradient, atol=1.0e-10, rtol=1.0e-10)
