from __future__ import annotations

import math

import torch

from dpl_rhf.backends.base import NucleusCase
from dpl_rhf.models.density_matrix_net import DensityMatrixNet
from dpl_rhf.rhf.backend import DensityMatrixRHFBackend
from dpl_rhf.rhf.asymptotics import (
    bound_density_tail_parameters,
    continue_bound_radial_density,
    physical_local_density_robin_coefficient,
)
from dpl_rhf.rhf.density import (
    kappa_quantum_numbers,
    local_source_densities,
    nonlocal_density_from_block,
)
from dpl_rhf.rhf.dirac import RHFRadialDiracOperator
from dpl_rhf.rhf.kernels import (
    PKO2AngularKernels,
    SphericalPropagators,
    orbital_cg_squared,
    spin_cg_squared,
)
from dpl_rhf.rhf.functional import (
    PKA1_DIRECT_CHANNEL_TERMS,
    PKA1_FOCK_CHANNEL_TERMS,
    PKO2RHFFunctional,
)
from dpl_rhf.rhf.parameters import pka1_parameters, pko2_parameters


def test_pko2_parameters_match_core1204_define():
    p = pko2_parameters()
    assert p.rho_sat == 0.151021
    assert p.mass_n == p.mass_p == 938.9
    assert p.m_sigma == 534.461766
    assert p.g_sigma == 8.920597
    assert p.g_omega == 10.550553
    assert p.g_rho == 4.068299
    assert p.f_pi == p.f_rho == 0.0


def test_pka1_parameters_match_core1204_define_and_tensor_conversion():
    p = pka1_parameters()
    assert p.model == "PKA1"
    assert p.rho_sat == 0.159996
    assert p.m_sigma == 488.227904
    assert p.g_sigma == 8.372672
    assert p.g_omega == 11.270457
    assert p.g_rho == 3.649857
    assert p.f_pi == 1.030722
    assert p.a_pi == 1.2
    assert p.f_rho == 3.199491
    assert p.a_rho_tensor == 0.820583
    expected = p.g_rho * p.f_rho / (2.0 * p.mass_n / p.hbar_c)
    assert p.rho_tensor_vertex == expected


def test_kappa_convention_matches_spherical_dirac_blocks():
    assert kappa_quantum_numbers(-1) == (0, 1, 0.5)
    assert kappa_quantum_numbers(-2) == (1, 2, 1.5)
    assert kappa_quantum_numbers(1) == (1, 0, 0.5)


def test_rhf_kinetic_operator_is_book_equation_3393_not_an_rmf_field_matrix():
    r = torch.arange(21, dtype=torch.float64) * 0.1
    r[0] = 0.001
    operator = RHFRadialDiracOperator(
        r,
        mass_n_mev=939.0,
        mass_p_mev=938.0,
        hbar_c_mev_fm=197.328284,
        derivative_order=7,
    )
    matrix = operator.matrix_fm_inverse(1.0, -1)
    size = r.numel() - 2
    assert torch.count_nonzero(matrix[:size, :size]) == 0
    assert torch.allclose(
        torch.diagonal(matrix[size:, size:]),
        torch.full((size,), -2.0 * 939.0 / 197.328284, dtype=torch.float64),
    )
    assert torch.allclose(matrix[:size, size:], matrix[size:, :size].T)


def test_bound_density_tail_uses_shifted_single_particle_energy():
    mass = 938.9
    epsilon = torch.tensor(-8.0, dtype=torch.float64)
    tail = bound_density_tail_parameters(epsilon, mass, 197.328284)
    momentum = math.sqrt(8.0 * (2.0 * mass - 8.0))
    assert torch.allclose(
        tail["total_energy_mev"], torch.tensor(930.9, dtype=torch.float64)
    )
    assert abs(float(tail["decay_fm_inverse"]) - momentum / 197.328284) < 1.0e-14
    assert abs(float(tail["scalar_to_vector_ratio"]) - 930.9 / mass) < 1.0e-14
    assert abs(float(tail["tensor_to_vector_ratio"]) + momentum / mass) < 1.0e-14

    radius = torch.tensor(20.0, dtype=torch.float64)
    coefficient = physical_local_density_robin_coefficient(
        radius, tail["decay_fm_inverse"]
    )
    assert torch.allclose(
        coefficient, 2.0 * tail["decay_fm_inverse"] + 2.0 / radius
    )


def test_density_tail_continuation_satisfies_dirac_ratios_and_exact_moments():
    upper_match = torch.tensor([0.3, 0.2], dtype=torch.float64)
    epsilon = torch.tensor([-8.0, -4.0], dtype=torch.float64)
    tail = continue_bound_radial_density(
        upper_match, epsilon, 938.9, 197.328284,
        torch.tensor(19.9), torch.tensor(20.0),
    )
    beta = -tail["lower_to_upper_ratio"]
    assert torch.allclose(tail["lower"], beta.square() * tail["upper"])
    assert torch.allclose(tail["mixed"], -beta * tail["upper"])
    rate = 2.0 * tail["decay_fm_inverse"]
    assert torch.allclose(tail["exterior_number"], tail["vector"] / rate)


def test_yukawa_and_coulomb_multipoles_are_symmetric_and_finite():
    r = torch.arange(41, dtype=torch.float64) * 0.1
    r[0] = 0.001
    propagators = SphericalPropagators(r, 197.328284)
    for multipole in range(4):
        for kernel in (
            propagators.yukawa(multipole, 534.461766),
            propagators.coulomb(multipole),
        ):
            assert torch.isfinite(kernel).all()
            assert torch.max(torch.abs(kernel - kernel.T)).item() < 1.0e-13
    expected = 1.0 / torch.maximum(r[:, None], r[None, :])
    assert torch.max(torch.abs(propagators.coulomb(0) - expected)).item() < 1.0e-13


def test_six_point_half_mesh_folding_integrates_constant_exactly():
    raw = torch.ones((3, 401), dtype=torch.float64)
    folded = SphericalPropagators._fold_half_mesh(raw, 0.1)
    assert folded.shape == (3, 201)
    assert torch.max(torch.abs(folded.sum(dim=1) - 20.0)).item() < 1.0e-13


def test_nonlocal_density_has_required_hermitian_component_relations():
    generator = torch.Generator().manual_seed(4)
    factors = torch.randn((12, 2), dtype=torch.float64, generator=generator)
    factors, _ = torch.linalg.qr(factors)
    block = {
        "factors": factors,
        "occupations": torch.ones(2, dtype=torch.float64),
        "degeneracy": torch.tensor(4.0, dtype=torch.float64),
        "species_sign": torch.tensor(1.0, dtype=torch.float64),
        "kappa": torch.tensor(-2),
        "principals": torch.tensor([1, 2]),
    }
    density = nonlocal_density_from_block(block, 0.1)
    assert torch.max(torch.abs(density.matrix[0, 0] - density.matrix[0, 0].T)).item() < 1.0e-14
    assert torch.max(torch.abs(density.matrix[1, 1] - density.matrix[1, 1].T)).item() < 1.0e-14
    assert torch.max(torch.abs(density.matrix[0, 1] - density.matrix[1, 0].T)).item() < 1.0e-14
    assert density.principals == (1, 2)
    reconstructed = density.state_matrix(0) + density.state_matrix(1)
    assert torch.max(torch.abs(reconstructed - density.matrix)).item() < 1.0e-14


def test_nonlocal_density_is_invariant_under_equal_occupation_n_rotation():
    generator = torch.Generator().manual_seed(20260819)
    factors = torch.randn((20, 3), dtype=torch.float64, generator=generator)
    factors, _ = torch.linalg.qr(factors, mode="reduced")
    rotation = torch.randn((3, 3), dtype=torch.float64, generator=generator)
    rotation, _ = torch.linalg.qr(rotation)
    common = {
        "occupations": torch.ones(3, dtype=torch.float64),
        "degeneracy": torch.tensor(2.0, dtype=torch.float64),
        "species_sign": torch.tensor(1.0, dtype=torch.float64),
        "kappa": torch.tensor(-1),
        "principals": torch.tensor([1, 2, 3]),
    }
    original = nonlocal_density_from_block(
        {**common, "factors": factors}, 0.1
    )
    rotated = nonlocal_density_from_block(
        {**common, "factors": factors @ rotation}, 0.1
    )
    assert torch.allclose(original.projector, rotated.projector, atol=2.0e-15)
    assert torch.allclose(original.matrix, rotated.matrix, atol=3.0e-14)


def test_closed_shell_nonlocal_projectors_are_idempotent_and_count_particles():
    for z, n, a in ((8, 8, 16), (20, 20, 40), (20, 28, 48),
                    (50, 82, 132), (82, 126, 208)):
        backend = DensityMatrixRHFBackend(
            NucleusCase("PKA1", z, n, a), device=torch.device("cpu")
        )
        network = DensityMatrixNet(
            torch.as_tensor(backend.r, dtype=torch.float64),
            backend.block_specs, hidden=8, depth=1, radial_basis_size=4,
        )
        blocks = network()
        neutron_number = 0.0
        proton_number = 0.0
        for block in blocks:
            density = nonlocal_density_from_block(block, backend.functional.h)
            error = torch.max(torch.abs(density.projector @ density.projector - density.projector))
            assert error < 2.0e-12
            number = density.degeneracy * torch.trace(density.projector)
            if density.species_sign > 0.0:
                neutron_number += float(number)
            else:
                proton_number += float(number)
        assert abs(neutron_number - n) < 1.0e-10
        assert abs(proton_number - z) < 1.0e-10


def test_pka1_tensor_density_is_diagonal_gf_plus_fg_of_fock_density():
    factors = torch.zeros((12, 1), dtype=torch.float64)
    factors[1, 0] = 0.6
    factors[7, 0] = 0.8
    factors = factors / torch.linalg.norm(factors)
    density = nonlocal_density_from_block(
        {
            "factors": factors,
            "occupations": torch.ones(1, dtype=torch.float64),
            "degeneracy": torch.tensor(2.0),
            "species_sign": torch.tensor(1.0),
            "kappa": torch.tensor(-1),
            "principals": torch.tensor([1]),
        },
        0.1,
    )
    radius = torch.arange(8, dtype=torch.float64) * 0.1
    radius[0] = 0.001
    local = local_source_densities([density], radius, include_tensor=True)
    denominator = 4.0 * torch.pi * radius[2].square()
    expected = (
        density.matrix[0, 1, 1, 1] + density.matrix[1, 0, 1, 1]
    ) / denominator
    assert torch.allclose(local["rho_t_n"][2], expected)
    assert torch.count_nonzero(local["rho_t_p"]) == 0


def test_pko2_o16_forward_channels_have_rhf_signs_and_exact_assembly():
    backend = DensityMatrixRHFBackend(
        NucleusCase("PKO2", 8, 8, 16), device=torch.device("cpu")
    )
    network = DensityMatrixNet(
        torch.as_tensor(backend.r, dtype=torch.float64),
        backend.block_specs,
        hidden=8,
        depth=1,
        radial_basis_size=4,
    )
    result = backend.evaluate(network())
    terms = result.action_terms
    assert terms["E_sigma_fock"] > 0.0
    assert terms["E_omega_fock"] < 0.0
    assert terms["E_rho_fock"] < 0.0
    assert terms["E_coul_fock"] < 0.0
    # Absolute closure against converged occupied states is enforced by
    # scripts/audit_rhf_hamiltonian.py. A randomly initialized network is not
    # a physical Core state and therefore must not carry hard-coded SCF values.
    for channel in ("omega", "rho", "coul"):
        assert torch.allclose(
            terms[f"E_{channel}_fock"],
            terms[f"E_{channel}_fock_time"] + terms[f"E_{channel}_fock_space"],
        )
    assert torch.isfinite(result.fields["sigma_rearr_fock"]).all()
    assert torch.allclose(
        result.fields["sigma_rearr_total"],
        result.fields["sigma_rearr_direct"] + result.fields["sigma_rearr_fock"],
    )
    assert result.diagnostics["n_number_error"] < 1.0e-12
    assert result.diagnostics["z_number_error"] < 1.0e-12
    assert result.diagnostics["max_hamiltonian_hermiticity_error_mev"] < 1.0e-12
    assert result.diagnostics["training_dirac_solves"] == 0
    assert math.isfinite(result.energy_per_a_no_com)
    for name, total in result.hamiltonians.items():
        assert torch.allclose(
            total,
            result.kinetic_hamiltonians[name]
            + result.direct_hamiltonians[name]
            + result.fock_hamiltonians[name],
            rtol=0.0,
            atol=2.0e-12,
        )
    assert result.diagnostics["component_residual_sum_error_mev"] < 1.0e-12
    assert result.diagnostics["stationarity_quadratic_closure_mev2"] < 1.0e-10
    expanded = sum(result.stationarity_quadratic_terms.values())
    assert torch.allclose(
        expanded,
        result.stationarity_residual_tensor.square(),
        rtol=2.0e-13,
        atol=2.0e-12,
    )


def test_only_total_rhf_hamiltonian_is_required_to_be_stationary():
    q = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    kinetic = torch.tensor([[0.0, 2.0], [2.0, 0.0]], dtype=torch.float64)
    direct = torch.tensor([[0.0, -3.0], [-3.0, 0.0]], dtype=torch.float64)
    fock = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)

    def projected(operator):
        operator_q = operator @ q
        return operator_q - q @ (q.T @ operator_q)

    component_norms = [
        torch.linalg.vector_norm(projected(operator))
        for operator in (kinetic, direct, fock)
    ]
    total = projected(kinetic + direct + fock)
    assert all(value > 0.0 for value in component_norms)
    assert torch.linalg.vector_norm(total) == 0.0


def test_pko2_rhf_network_energy_gradient_matches_finite_difference():
    backend = DensityMatrixRHFBackend(
        NucleusCase("PKO2", 8, 8, 16), device=torch.device("cpu")
    )
    network = DensityMatrixNet(
        torch.as_tensor(backend.r, dtype=torch.float64),
        backend.block_specs,
        hidden=8,
        depth=1,
        radial_basis_size=4,
    )
    result = backend.evaluate(network())
    result.energy_total_tensor.backward()
    parameter = network.output_heads[0].weight
    flat_index = int(torch.argmax(parameter.grad.abs()))
    row = flat_index // parameter.shape[1]
    column = flat_index % parameter.shape[1]
    automatic = float(parameter.grad[row, column])
    epsilon = 1.0e-5
    original = float(parameter[row, column].detach())
    with torch.no_grad():
        parameter[row, column] = original + epsilon
    plus = backend.evaluate(network()).energy_total_no_com
    with torch.no_grad():
        parameter[row, column] = original - epsilon
    minus = backend.evaluate(network()).energy_total_no_com
    with torch.no_grad():
        parameter[row, column] = original
    finite_difference = (plus - minus) / (2.0 * epsilon)
    relative = abs(automatic - finite_difference) / max(abs(finite_difference), 1.0e-10)
    assert relative < 1.0e-5


def test_pka1_complete_channels_are_finite_and_network_gradient_is_autodiff_consistent():
    radius = torch.arange(41, dtype=torch.float64) * 0.1
    radius[0] = 0.001
    functional = PKO2RHFFunctional(radius, 1, 1, pka1_parameters())
    blocks = []
    generator = torch.Generator().manual_seed(17)
    for species in (1.0, -1.0):
        factor = torch.randn((78, 1), dtype=torch.float64, generator=generator)
        factor = (factor / torch.linalg.norm(factor)).detach().requires_grad_()
        blocks.append(
            {
                "factors": factor,
                "occupations": torch.tensor([0.5], dtype=torch.float64),
                "degeneracy": torch.tensor(2.0, dtype=torch.float64),
                "species_sign": torch.tensor(species, dtype=torch.float64),
                "kappa": torch.tensor(-1),
                "principals": torch.tensor([1]),
                "free_hamiltonian": torch.zeros((78, 78), dtype=torch.float64),
            }
        )

    def evaluate():
        densities = [nonlocal_density_from_block(block, functional.h) for block in blocks]
        return functional.evaluate(blocks, densities)

    result = evaluate()
    required = (
        "E_pion_fock",
        "E_rho_tensor_fock_time",
        "E_rho_tensor_fock_space",
        "E_rho_vt_fock",
        "E_rho_tensor_direct",
        "E_rho_vt_direct",
    )
    terms = {**result.direct_terms, **result.exchange_terms}
    assert all(torch.isfinite(terms[name]) for name in required)
    required_by_lagrangian = {
        name
        for names in (
            *PKA1_DIRECT_CHANNEL_TERMS.values(),
            *PKA1_FOCK_CHANNEL_TERMS.values(),
        )
        for name in names
    }
    assert required_by_lagrangian.issubset(terms)
    direct_sum = sum(
        (terms[name] for names in PKA1_DIRECT_CHANNEL_TERMS.values() for name in names),
        torch.zeros_like(result.total),
    )
    fock_sum = sum(
        (terms[name] for names in PKA1_FOCK_CHANNEL_TERMS.values() for name in names),
        torch.zeros_like(result.total),
    )
    assert torch.allclose(direct_sum, result.direct_terms["E_direct"])
    assert torch.allclose(fock_sum, result.exchange_terms["E_fock"])
    result.total.backward()
    parameter = blocks[0]["factors"]
    row = int(torch.argmax(parameter.grad.abs()))
    automatic = float(parameter.grad[row, 0])
    epsilon = 1.0e-5
    original = float(parameter[row, 0].detach())
    with torch.no_grad():
        parameter[row, 0] = original + epsilon
    plus = float(evaluate().total.detach())
    with torch.no_grad():
        parameter[row, 0] = original - epsilon
    minus = float(evaluate().total.detach())
    with torch.no_grad():
        parameter[row, 0] = original
    finite_difference = (plus - minus) / (2.0 * epsilon)
    relative = abs(automatic - finite_difference) / max(abs(finite_difference), 1.0e-10)
    assert relative < 1.0e-5


def test_pka1_recompute_preserves_energy_and_second_derivative():
    radius = torch.arange(31, dtype=torch.float64) * 0.1
    radius[0] = 0.001
    generator = torch.Generator().manual_seed(20260816)
    initial = [
        torch.randn((58, 1), dtype=torch.float64, generator=generator)
        for _ in range(2)
    ]

    def evaluate(recompute: bool):
        functional = PKO2RHFFunctional(
            radius, 1, 1, pka1_parameters(), recompute_fock=recompute
        )
        factors = [
            (value / torch.linalg.norm(value)).detach().requires_grad_()
            for value in initial
        ]
        blocks = [
            {
                "factors": factor,
                "occupations": torch.tensor([0.5], dtype=torch.float64),
                "degeneracy": torch.tensor(2.0, dtype=torch.float64),
                "species_sign": torch.tensor(species, dtype=torch.float64),
                "kappa": torch.tensor(-1),
                "principals": torch.tensor([1]),
                "free_hamiltonian": torch.zeros((58, 58), dtype=torch.float64),
            }
            for factor, species in zip(factors, (1.0, -1.0))
        ]
        densities = [
            nonlocal_density_from_block(block, functional.h) for block in blocks
        ]
        energy = functional.evaluate(blocks, densities).total
        first = torch.autograd.grad(energy, factors, create_graph=True)
        gradient_square = sum(value.square().sum() for value in first)
        second = torch.autograd.grad(gradient_square, factors)
        return (
            energy.detach(),
            torch.cat([value.detach().reshape(-1) for value in first]),
            torch.cat([value.detach().reshape(-1) for value in second]),
        )

    ordinary = evaluate(False)
    recomputed = evaluate(True)
    for expected, actual in zip(ordinary, recomputed):
        assert torch.allclose(expected, actual, rtol=2.0e-13, atol=2.0e-12)


def test_pk02_angular_kernel_respects_parity_selection_and_is_finite():
    r = torch.linspace(0.1, 2.0, 20, dtype=torch.float64)
    kernels = PKO2AngularKernels(SphericalPropagators(r, 197.328284))
    scalar = kernels.scalar_time("sigma", -1, -1, 534.461766)
    assert torch.max(torch.abs(scalar)).item() > 0.0
    # This is the folded integration operator R-hat of Eq. (3.404), which
    # already contains source-side quadrature weights and is therefore not a
    # bare symmetric propagator matrix.
    assert torch.isfinite(scalar).all()


def test_vector_space_keeps_highest_cross_component_multipole():
    r = torch.linspace(0.1, 2.0, 20, dtype=torch.float64)
    propagators = SphericalPropagators(r, 197.328284)
    kernels = PKO2AngularKernels(propagators)
    assert kernels.maximum_multipole(-2, 1) == 2
    assert kernels.maximum_space_multipole(-2, 1) == 3
    full = kernels.vector_space("omega", -2, 1, 1, 1, 783.0)
    c3j = spin_cg_squared(-2, 1, 3)
    coefficient = 2.0 * orbital_cg_squared(2, 1, 3) - c3j
    expected_l3 = coefficient * propagators.folded_yukawa(3, 783.0) / (4.0 * math.pi)
    # Rebuild L=1 separately; L=2 is excluded by parity for this component.
    coefficient_l1 = (
        2.0 * orbital_cg_squared(2, 1, 1) - spin_cg_squared(-2, 1, 1)
    )
    expected_l1 = coefficient_l1 * propagators.folded_yukawa(1, 783.0) / (4.0 * math.pi)
    assert torch.max(torch.abs(full - expected_l1 - expected_l3)).item() < 1.0e-13


def test_fock_angular_reduction_matches_core_c3_normalization():
    r = torch.linspace(0.1, 2.0, 20, dtype=torch.float64)
    propagators = SphericalPropagators(r, 197.328284)
    kernels = PKO2AngularKernels(propagators)
    kappa_a, kappa_b, multipole = -2, -2, 2
    coefficient = spin_cg_squared(kappa_a, kappa_b, multipole)
    # Core's C3J is |CG|^2/(2L+1), while hLS contains (2L+1)/(4*pi).
    core_c3j = coefficient / (2 * multipole + 1)
    expected = (
        core_c3j
        * (2 * multipole + 1)
        * propagators.folded_yukawa(multipole, 534.461766)
        / (4.0 * math.pi)
    )
    # For this pair the scalar parity rule admits L=0 and L=2.  Subtract L=0
    # and verify the exact cancellation of the two normalization factors.
    actual = kernels.scalar_time("sigma", kappa_a, kappa_b, 534.461766)
    l0 = (
        spin_cg_squared(kappa_a, kappa_b, 0)
        * propagators.folded_yukawa(0, 534.461766)
        / (4.0 * math.pi)
    )
    assert torch.max(torch.abs(actual - l0 - expected)).item() < 1.0e-13
