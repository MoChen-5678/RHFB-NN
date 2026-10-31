from __future__ import annotations

import numpy as np
import torch
from torch import nn

from dpl_rhf.backends.base import NucleusCase
from dpl_rhf.backends.finite_difference import zhang_7point_adf_coefficients
from dpl_rhf.backends.fortran_rhf_stub import FortranRHFBackend
from dpl_rhf.backends.torch_rmf import (
    SHELLS,
    TorchRMFBackend,
    noncentral_derivative_matrix,
    spherical_shells,
)
from dpl_rhf.backends.density_matrix_rmf import DensityMatrixRMFBackend
from dpl_rhf.cli.prl_train import build_parser
from dpl_rhf.functionals.pkdd_rmf import PKDDRMFFunctionalSpec
from dpl_rhf.functionals.pkdd_action import (
    PKDDParameters,
    PKDDRMFFunctional,
    RMF_PARAMETER_DATA,
    rmf_parameters,
)
from dpl_rhf.models.hamiltonian_net import (
    DirectHamiltonianParameterization,
    LocalHamiltonianNet,
    compose_hamiltonian,
    decompose_hamiltonian,
    physical_component_gradient,
)
from dpl_rhf.models.density_matrix_net import DensityMatrixNet
from dpl_rhf.training.surrogate_gradient import (
    generalized_prl_gradient,
    surrogate_loss,
)
from dpl_rhf.training.prl_hamiltonian_trainer import _compose_backend_hamiltonian
from dpl_rhf.training.inverse_hamiltonian import (
    inverse_hamiltonian_constraint,
    rmf_block_hamiltonians,
)


def test_surrogate_gradient_matches_backend_gradient():
    torch.set_default_dtype(torch.float64)
    hamiltonian = torch.randn(4, 16, requires_grad=True)
    reconstructed = torch.zeros_like(hamiltonian)
    grad_energy = torch.randn_like(hamiltonian)
    grad_h = generalized_prl_gradient(hamiltonian, reconstructed, grad_energy, 0.7)
    loss = surrogate_loss(hamiltonian, grad_h)
    (actual,) = torch.autograd.grad(loss, hamiltonian)
    expected = grad_h.detach()
    assert torch.max(torch.abs(actual - expected)).item() < 1.0e-14


def test_prl_reconstruction_gradient_is_exact_hamiltonian_difference():
    hamiltonian = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    reconstructed = torch.flip(hamiltonian, dims=(1,))
    gradient = generalized_prl_gradient(
        hamiltonian, reconstructed, torch.zeros_like(hamiltonian), 0.7
    )
    expected = 0.7 * (hamiltonian - reconstructed)
    assert torch.max(torch.abs(gradient - expected)).item() < 1.0e-14


def test_physical_component_gradient_matches_channel_assembly():
    components = torch.randn((4, 11), dtype=torch.float64, requires_grad=True)
    channel_gradient = torch.randn((4, 11), dtype=torch.float64)
    value = torch.sum(compose_hamiltonian(components) * channel_gradient)
    (actual,) = torch.autograd.grad(value, components)
    assert torch.max(torch.abs(actual - physical_component_gradient(channel_gradient))).item() < 1.0e-14


def test_7point_adf_moment_conditions():
    coeff = zhang_7point_adf_coefficients()
    offsets = np.arange(7, dtype=np.float64)
    for power in range(7):
        moment = float(np.sum(coeff * offsets**power))
        expected = 1.0 if power == 1 else 0.0
        assert abs(moment - expected) < 1.0e-10


def test_adf_uses_zero_exterior_forward_operator_and_adjoint_pair():
    r = torch.linspace(0.0, 2.0, 21, dtype=torch.float64)
    forward = noncentral_derivative_matrix(r, order=7)
    backend = TorchRMFBackend(NucleusCase("PKDD", 8, 8, 16), device=torch.device("cpu"), derivative_order=7)
    assert torch.count_nonzero(forward[-1, :-1]).item() == 0
    solver = backend.matrix_solver
    adjoint = solver.derivative_minus + solver.derivative_plus.T
    assert torch.max(torch.abs(adjoint)).item() < 1.0e-14


def test_dirac_adf_constructs_b1_b2_explicitly_with_parity_exchange():
    backend = TorchRMFBackend(NucleusCase("PKDD", 8, 8, 16), device=torch.device("cpu"), derivative_order=7)
    solver = backend.matrix_solver
    stack = torch.as_tensor(backend.initial_hamiltonian_np, dtype=torch.float64)
    size = solver.x.numel()
    # G behaves as r^(l+1): kappa=-1 (s1/2) has odd radial G, while
    # kappa=+1 (p1/2) has even radial G.
    for kappa, radial_g_is_odd in ((-1, True), (1, False)):
        matrix = solver.matrix(stack, 1.0, kappa)
        angular = torch.diag(torch.full_like(solver.x, float(kappa)) / solver.x)
        forward = solver.derivative_plus if radial_g_is_odd else solver.derivative_minus
        backward = solver.derivative_minus if radial_g_is_odd else solver.derivative_plus
        physical_b1 = -backward + angular
        physical_b2 = forward + angular
        assert torch.max(torch.abs(matrix[:size, size:] - physical_b1)).item() < 1.0e-14
        assert torch.max(torch.abs(matrix[size:, :size] - physical_b2)).item() < 1.0e-14
        assert torch.max(torch.abs(physical_b1 - physical_b2.T)).item() < 1.0e-14


def test_rhf_backend_is_reserved():
    try:
        FortranRHFBackend()
    except NotImplementedError as exc:
        assert "reserved" in str(exc)
    else:
        raise AssertionError("RHF backend should not be implemented in RMF PRL v1")


def test_prl_default_backend_is_differentiable_torch_rmf():
    args = build_parser().parse_args([])
    assert args.backend == "torch-rmf"
    assert args.functional == "rmf-dd"
    assert args.energy_gradient_weight == 1.0
    assert args.lambda_reconstruct == 1.0e-3
    assert args.lambda_mode == "auto"
    assert args.direct_order == 64
    assert args.derivative_order == 7
    assert args.mode == "density-matrix"
    assert args.network_depth == 3
    assert args.hidden == 64
    assert args.radial_basis_size == 16
    assert args.message_neighbors == 3
    assert args.lr is None
    assert args.scheduler_patience == 120
    assert args.scheduler_cooldown == 40
    activation_action = next(
        action for action in build_parser()._actions if action.dest == "activation"
    )
    assert activation_action.choices == ["silu"]


def test_five_rmf_parameter_sets_match_core_define_values():
    assert tuple(RMF_PARAMETER_DATA) == ("DD-ME1", "DD-ME2", "PKDD", "TW99", "DD-LZ1")
    expected = {
        "DD-ME1": (938.5, 549.5255, 0.151962),
        "DD-ME2": (938.9, 550.1238, 0.152),
        "PKDD": (939.5731, 555.511236, 0.149552),
        "TW99": (939.0, 550.0, 0.153004),
        "DD-LZ1": (938.9, 538.619216, 0.158100),
    }
    for model, values in expected.items():
        params = rmf_parameters(model)
        assert (params.mass_n, params.m_sigma, params.rho_sat) == values


def test_torch_backend_uses_selected_rmf_parameters_everywhere():
    for model in RMF_PARAMETER_DATA:
        backend = TorchRMFBackend(
            NucleusCase(model, 8, 8, 16),
            device=torch.device("cpu"),
            derivative_order=7,
        )
        expected = rmf_parameters(model)
        assert backend.params == expected
        assert backend.functional.params == expected
        assert backend.functional_spec.params == expected
        assert backend.matrix_solver.params == expected
        assert np.isclose(
            backend.initial_hamiltonian_np[1, -1],
            -2.0 * expected.mass_n / expected.hbar_c,
        )


def test_trainer_assembles_channels_with_selected_parameter_set():
    components = torch.zeros((4, 201), dtype=torch.float64)
    for model in RMF_PARAMETER_DATA:
        backend = TorchRMFBackend(
            NucleusCase(model, 8, 8, 16),
            device=torch.device("cpu"),
            derivative_order=7,
        )
        actual = _compose_backend_hamiltonian(components, backend)
        expected = rmf_parameters(model)
        assert torch.max(torch.abs(actual[1] + 2.0 * expected.mass_n / expected.hbar_c)).item() < 1.0e-14
        assert torch.max(torch.abs(actual[3] + 2.0 * expected.mass_p / expected.hbar_c)).item() < 1.0e-14


def test_physical_representation_decomposes_with_selected_parameter_set():
    r = np.linspace(0.001, 20.0, 201)
    for model in RMF_PARAMETER_DATA:
        params = rmf_parameters(model)
        initial = PKDDRMFFunctionalSpec(8, 8, r, params=params).initial_hamiltonian()
        direct = DirectHamiltonianParameterization(
            r, 8, 8, order=8, params=params
        )
        expected = decompose_hamiltonian(
            torch.as_tensor(initial, dtype=torch.float64),
            params=params,
            coulomb=direct.coulomb_base,
        )
        expected = expected.clone()
        expected[:3] *= direct.massive_envelope
        expected[3] = direct.coulomb_base
        assert torch.max(torch.abs(direct.initial_components - expected)).item() < 1.0e-14


def test_torch_rmf_backend_gradient_is_autodiff_consistent():
    torch.set_default_dtype(torch.float64)
    backend = TorchRMFBackend(NucleusCase("PKDD", 8, 8, 16), device=torch.device("cpu"), derivative_order=7)
    hamiltonian = torch.as_tensor(backend.initial_hamiltonian_np, dtype=torch.float64).requires_grad_(True)
    result = backend.evaluate_tensor(hamiltonian)
    assert result.diagnostics["hermiticity_error"] < 2.0e-14
    assert result.diagnostics["gradient_norm"] > 0.0
    check = backend.gradient_check(hamiltonian.detach(), epsilon=1.0e-4)
    assert check["rel_error"] < 2.0e-7


def test_torch_rmf_orbitals_obey_regular_origin_boundary():
    torch.set_default_dtype(torch.float64)
    backend = TorchRMFBackend(NucleusCase("PKDD", 8, 8, 16), device=torch.device("cpu"), derivative_order=7)
    hamiltonian = torch.as_tensor(backend.initial_hamiltonian_np, dtype=torch.float64).requires_grad_(True)
    orbitals = backend.evaluate_tensor(hamiltonian).orbitals
    r = torch.as_tensor(backend.r, dtype=torch.float64)
    for i, kappa in enumerate(orbitals["kappa"]):
        l_upper, l_lower = backend.matrix_solver._component_angular_momenta(int(kappa))
        for values, angular in ((orbitals["G"][i], l_upper), (orbitals["F"][i], l_lower)):
            scaled = values[:3] / r[:3].pow(angular + 1)
            assert torch.isfinite(scaled).all()
            assert abs(float(values[-1])) < 1.0e-14


def test_kinetic_energy_uses_physically_normalized_orbitals():
    backend = TorchRMFBackend(NucleusCase("PKDD", 8, 8, 16), device=torch.device("cpu"), derivative_order=7)
    hamiltonian = torch.as_tensor(backend.initial_hamiltonian_np, dtype=torch.float64).requires_grad_(True)
    orbitals = backend.evaluate_tensor(hamiltonian).orbitals
    p = backend.functional.params
    for index in range(orbitals["G"].shape[0]):
        species = float(orbitals["species_sign"][index])
        kappa = int(orbitals["kappa"][index])
        zero_stack = torch.zeros_like(hamiltonian)
        zero_stack[1] = -2.0 * p.mass_n / p.hbar_c
        zero_stack[3] = -2.0 * p.mass_p / p.hbar_c
        free = backend.matrix_solver.matrix(zero_stack, species, kappa)
        interacting = backend.matrix_solver.matrix(hamiltonian, species, kappa)
        _, eigenvectors = torch.linalg.eigh(interacting)
        vector = eigenvectors[:, orbitals["eigen_index"][index]]
        expected = p.hbar_c * (vector @ free @ vector)
        assert abs(float(expected - orbitals["kinetic_expectation_mev"][index])) < 1.0e-10


def test_uniform_adf_metric_makes_matrix_and_density_bilinears_identical():
    backend = TorchRMFBackend(NucleusCase("PKDD", 8, 8, 16), device=torch.device("cpu"), derivative_order=7)
    hamiltonian = torch.as_tensor(backend.initial_hamiltonian_np, dtype=torch.float64).requires_grad_(True)
    result = backend.evaluate_tensor(hamiltonian)
    h = backend.functional.h
    assert torch.max(torch.abs(backend.functional.orbital_w[1:] - h)).item() < 1.0e-14
    for index in range(result.orbitals["G"].shape[0]):
        species = float(result.orbitals["species_sign"][index])
        kappa = int(result.orbitals["kappa"][index])
        offset = 0 if species > 0 else 2
        g = result.orbitals["G"][index, 1:-1]
        f = result.orbitals["F"][index, 1:-1]
        coefficients = torch.sqrt(torch.as_tensor(h)) * torch.cat([g, f])
        matrix = backend.matrix_solver.matrix(hamiltonian, species, kappa)
        matrix_expectation = coefficients @ matrix @ coefficients
        assert abs(float(torch.linalg.norm(coefficients) - 1.0)) < 1.0e-12
        assert abs(float(matrix_expectation * backend.functional.params.hbar_c - result.orbitals["epsilon"][index])) < 1.0e-9
        local_integral = h * torch.sum(
            g.square() * hamiltonian[offset, 1:-1]
            + f.square() * hamiltonian[offset + 1, 1:-1]
        )
        block_size = g.numel()
        local_matrix = (
            coefficients[:block_size].square() @ hamiltonian[offset, 1:-1]
            + coefficients[block_size:].square() @ hamiltonian[offset + 1, 1:-1]
        )
        assert abs(float(local_integral - local_matrix)) < 1.0e-12


def test_global_occupation_preserves_neutron_and_proton_numbers():
    backend = TorchRMFBackend(NucleusCase("PKDD", 8, 8, 16), device=torch.device("cpu"), derivative_order=7)
    hamiltonian = torch.as_tensor(backend.initial_hamiltonian_np, dtype=torch.float64).requires_grad_(True)
    orbitals = backend.evaluate_tensor(hamiltonian).orbitals
    particles = orbitals["occupancy"] * orbitals["degeneracy"]
    assert abs(float(particles[orbitals["species_sign"] > 0].sum()) - 8.0) < 1.0e-12
    assert abs(float(particles[orbitals["species_sign"] < 0].sum()) - 8.0) < 1.0e-12


def test_hamiltonian_network_hard_physical_boundaries():
    r = np.linspace(0.001, 20.0, 201)
    initial = np.zeros((4, r.size))
    network = LocalHamiltonianNet(r, 8, 8, initial)
    components = network.components()
    diagnostics = network.constraint_diagnostics(components)
    assert diagnostics["analytic_center_derivative_fm_minus_2"] == 0.0
    assert diagnostics["analytic_massive_yukawa_robin_fm_minus_2"] == 0.0
    assert diagnostics["analytic_coulomb_robin_fm_minus_2"] == 0.0
    assert diagnostics["max_scalar_identity_mev"] < 1.0e-10
    direct = DirectHamiltonianParameterization(r, 8, 8, order=8)
    with torch.no_grad():
        direct.coefficients[0, 0] = 1.0
    scalar = direct.components()[0]
    assert float(scalar[-1]) > 0.0
    assert abs(float(scalar[-2] - scalar[-1])) > 1.0e-6


def test_massive_fields_include_exact_yukawa_exterior_action():
    backend = TorchRMFBackend(
        NucleusCase("PKDD", 8, 8, 16),
        device=torch.device("cpu"),
        derivative_order=7,
    )
    functional = backend.functional
    base_gradient = functional.d1.T @ torch.diag(functional.measure) @ functional.d1
    identity = torch.diag(functional.measure)
    for name, mass in (
        ("sigma", functional.params.m_sigma),
        ("omega", functional.params.m_omega),
        ("rho", functional.params.m_rho),
    ):
        mu = mass / functional.params.hbar_c
        interior = base_gradient + mu**2 * identity
        difference = functional.field_full_kernels[name] - interior
        expected = 4.0 * torch.pi * functional.r[-1].square() * (
            mu + 1.0 / functional.r[-1]
        )
        assert abs(float(difference[-1, -1] - expected)) < 1.0e-10
        difference = difference.clone()
        difference[-1, -1] = 0.0
        assert torch.max(torch.abs(difference)).item() < 1.0e-14


def test_n_equal_z_does_not_remove_isovector_variational_channel():
    r = np.linspace(0.001, 20.0, 201)
    direct = DirectHamiltonianParameterization(r, 8, 8, order=12)
    with torch.no_grad():
        direct.coefficients[2, 0] = 1.0
    isovector = direct.components()[2]
    assert torch.max(torch.abs(isovector)).item() > 0.01
    (gradient,) = torch.autograd.grad(isovector.square().sum(), direct.coefficients)
    assert torch.linalg.norm(gradient[2]).item() > 0.0


def test_representation_has_no_woods_saxon_hard_multiplier():
    r = np.linspace(0.001, 20.0, 201)
    direct = DirectHamiltonianParameterization(r, 8, 8, order=12)
    assert not hasattr(direct, "nuclear_shape")


def test_direct_parameterization_uses_same_physical_map():
    r = np.linspace(0.001, 20.0, 201)
    direct = DirectHamiltonianParameterization(r, 8, 8, order=12)
    assert direct().shape == (4, r.size)
    assert direct.components().requires_grad
    assert torch.max(torch.abs(direct.components()[:3, -1])).item() < 1.0e-14


def test_all_hamiltonian_hidden_activations_are_silu():
    r = np.linspace(0.001, 20.0, 201)
    network = LocalHamiltonianNet(r, 8, 8, np.zeros((4, r.size)))
    activations = [
        module for module in network.modules() if isinstance(module, nn.SiLU)
    ]
    assert activations
    assert all(isinstance(module, nn.SiLU) for module in activations)


def test_network_mode_has_no_chebyshev_variational_bypass():
    r = np.linspace(0.001, 20.0, 201)
    network = LocalHamiltonianNet(r, 8, 8, np.zeros((4, r.size)))
    assert not hasattr(network, "variational_coefficients")
    assert not hasattr(network, "variational_basis")
    assert hasattr(network, "message_blocks")
    assert len(network.message_blocks) == 3
    assert len(network.output_heads) == 4
    assert network.node_features.shape == (2 * r.size, 18)
    assert network.edge_index.shape[0] == 2
    full_latent = network.full_latent()
    negative = full_latent[:r.size]
    positive = full_latent[r.size:]
    assert torch.max(torch.abs(negative - torch.flip(positive, dims=(0,)))).item() < 1.0e-12
    assert torch.max(torch.abs(network.components() - network.initial_components)).item() < 1.0e-14
    with torch.no_grad():
        for head in network.output_heads:
            head.weight.normal_(mean=0.0, std=1.0e-3)
    components = network.components()
    weights = torch.arange(
        1, components.numel() + 1, dtype=components.dtype
    ).reshape_as(components)
    gradients = torch.autograd.grad(
        torch.sum(components * weights), tuple(network.parameters()),
        allow_unused=True,
    )
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.linalg.norm(gradient).item() > 0.0 for gradient in gradients)


def test_network_vjp_is_exact_prl_parameter_gradient_without_grid_scaling():
    r = np.linspace(0.001, 20.0, 41)
    network = LocalHamiltonianNet(
        r, 8, 8, np.zeros((4, r.size)), hidden=12, depth=2, radial_basis_size=6
    )
    with torch.no_grad():
        for head in network.output_heads:
            head.weight.normal_(mean=0.0, std=1.0e-3)
    hamiltonian = network()
    prescribed = torch.randn_like(hamiltonian)
    parameters = tuple(network.parameters())
    from_surrogate = torch.autograd.grad(
        surrogate_loss(hamiltonian, prescribed), parameters, retain_graph=True
    )
    from_vjp = torch.autograd.grad(
        hamiltonian, parameters, grad_outputs=prescribed
    )
    for actual, expected in zip(from_surrogate, from_vjp):
        assert torch.max(torch.abs(actual - expected)).item() < 1.0e-12


def test_complete_action_reduces_to_on_shell_energy():
    backend = TorchRMFBackend(NucleusCase("PKDD", 8, 8, 16), device=torch.device("cpu"), derivative_order=7)
    hamiltonian = torch.as_tensor(backend.initial_hamiltonian_np, dtype=torch.float64).requires_grad_(True)
    result = backend.evaluate_tensor(hamiltonian)
    assert abs(result.diagnostics["action_reduction_error_mev"]) < 1.0e-8
    assert result.diagnostics["max_orbital_norm_residual"] < 1.0e-12
    assert result.diagnostics["max_spectral_residual"] < 1.0e-10
    assert max(result.diagnostics[f"weak_el_{name}_rms"] for name in ("sigma", "omega", "rho", "coul")) < 1.0e-9
    assert abs(float(result.action_terms["action_orbital_constraints"])) < 1.0e-10


def test_pkdd_uses_distinct_neutron_and_proton_masses():
    backend = TorchRMFBackend(NucleusCase("PKDD", 8, 8, 16), device=torch.device("cpu"), derivative_order=7)
    p = backend.functional.params
    assert p.mass_n == 939.5731
    assert p.mass_p == 938.2796
    zeros = torch.zeros_like(torch.as_tensor(backend.r, dtype=torch.float64))
    potentials = backend.functional.potentials(
        rho_s_n=zeros, rho_s_p=zeros, rho_v_n=zeros, rho_v_p=zeros,
        sigma=zeros, omega=zeros, rho=zeros, coul=zeros,
    )
    assert torch.max(torch.abs(potentials["vms_n"] + 2.0 * p.mass_n / p.hbar_c)).item() < 1.0e-14
    assert torch.max(torch.abs(potentials["vms_p"] + 2.0 * p.mass_p / p.hbar_c)).item() < 1.0e-14
    expected_split = -2.0 * (p.mass_n - p.mass_p) / p.hbar_c
    actual_split = float((potentials["vms_n"] - potentials["vms_p"])[0])
    assert abs(actual_split - expected_split) < 1.0e-14


def test_pkdd_coupling_derivatives_generate_exact_rearrangement_inputs():
    r = torch.as_tensor(np.arange(201) * 0.1, dtype=torch.float64)
    r[0] = 0.001
    functional = PKDDRMFFunctional(r, 8, 8)
    rho_b = torch.linspace(1.0e-6, 0.16, r.numel(), dtype=torch.float64, requires_grad=True)
    couplings = functional.couplings(rho_b)
    for coupling, derivative in (("gsig", "dgsig"), ("gome", "dgome"), ("grho", "dgrho")):
        (actual,) = torch.autograd.grad(couplings[coupling].sum(), rho_b, retain_graph=True)
        expected = couplings[derivative] / functional.params.rho_sat
        assert torch.max(torch.abs(actual - expected)).item() < 1.0e-11


def test_local_isovector_density_sources_rho_field_even_for_n_equal_z():
    r = torch.as_tensor(np.arange(201) * 0.1, dtype=torch.float64)
    r[0] = 0.001
    functional = PKDDRMFFunctional(r, 8, 8)
    profile = 0.08 * torch.exp(-(r / 2.5).square())
    state = {
        "rho_s_n": profile,
        "rho_s_p": 0.9 * profile,
        "rho_v_n": profile,
        "rho_v_p": 0.9 * profile,
    }
    couplings = functional.couplings(state["rho_v_n"] + state["rho_v_p"])
    fields = functional.reconstruct_fields(state, couplings)
    assert torch.max(torch.abs(fields["rho"])).item() > 1.0e-6


def test_on_shell_field_action_derivative_reproduces_rearranged_potentials():
    r = torch.as_tensor(np.arange(201) * 0.1, dtype=torch.float64)
    r[0] = 0.001
    functional = PKDDRMFFunctional(r, 8, 8)
    base = 0.08 * torch.exp(-(r / 2.6).square())
    rho_s_n = (0.92 * base).clone().requires_grad_(True)
    rho_s_p = (0.86 * base).clone().requires_grad_(True)
    rho_v_n = base.clone().requires_grad_(True)
    rho_v_p = (0.9 * base).clone().requires_grad_(True)
    density = {
        "rho_s_n": rho_s_n, "rho_s_p": rho_s_p,
        "rho_v_n": rho_v_n, "rho_v_p": rho_v_p,
    }
    couplings = functional.couplings(rho_v_n + rho_v_p)
    fields = functional.reconstruct_fields(density, couplings)
    state = {**density, **fields}
    p = functional.params
    rho_s = rho_s_n + rho_s_p
    rho_b = rho_v_n + rho_v_p
    rho_3 = rho_v_n - rho_v_p
    energy = p.hbar_c * (
        0.5 * fields["sigma"] @ functional.field_full_kernels["sigma"] @ fields["sigma"]
        + torch.sum(functional.measure * couplings["gsig"] * rho_s * fields["sigma"])
        - 0.5 * fields["omega"] @ functional.field_full_kernels["omega"] @ fields["omega"]
        + torch.sum(functional.measure * couplings["gome"] * rho_b * fields["omega"])
        - 0.5 * fields["rho"] @ functional.field_full_kernels["rho"] @ fields["rho"]
        + torch.sum(functional.measure * couplings["grho"] * rho_3 * fields["rho"])
        - 0.5 * fields["coul"] @ functional.field_full_kernels["coul"] @ fields["coul"]
        + torch.sum(functional.measure * rho_v_p * fields["coul"])
    )
    gradients = torch.autograd.grad(energy, (rho_s_n, rho_s_p, rho_v_n, rho_v_p))
    potentials = functional.potentials(**state)
    interior = slice(2, -2)
    scale = p.hbar_c * functional.measure[interior]
    expected = (
        potentials["scalar_self_energy"],
        potentials["scalar_self_energy"],
        potentials["vector_isoscalar"] + potentials["vector_isovector"],
        potentials["vector_isoscalar"] - potentials["vector_isovector"] + potentials["vector_coulomb"],
    )
    for actual_gradient, expected_potential in zip(gradients, expected):
        actual = actual_gradient[interior] / scale
        assert torch.max(torch.abs(actual - expected_potential[interior])).item() < 2.0e-10


def test_torch_backend_constructs_core_mesh_without_fortran_runtime():
    backend = TorchRMFBackend(NucleusCase("PKDD", 8, 8, 16), device=torch.device("cpu"))
    assert backend.r.shape == (201,)
    assert backend.r[0] == 0.001
    assert abs(backend.r[-1] - 20.0) < 1.0e-14


def test_network_hamiltonian_obeys_book_channel_decomposition():
    p = PKDDParameters()
    r = np.linspace(0.001, 20.0, 201)
    initial = PKDDRMFFunctionalSpec(8, 8, r).initial_hamiltonian()
    network = LocalHamiltonianNet(r, 8, 8, initial)
    stack = network()
    scalar_n = 0.5 * (stack[0] - stack[1] - 2.0 * p.mass_n / p.hbar_c)
    scalar_p = 0.5 * (stack[2] - stack[3] - 2.0 * p.mass_p / p.hbar_c)
    assert torch.max(torch.abs(scalar_n - scalar_p)).item() < 1.0e-14
    vector_n = 0.5 * (stack[0] + stack[1] + 2.0 * p.mass_n / p.hbar_c)
    vector_p = 0.5 * (stack[2] + stack[3] + 2.0 * p.mass_p / p.hbar_c)
    assert torch.isfinite(vector_n).all() and torch.isfinite(vector_p).all()


def test_density_matrix_mode_has_hard_projector_and_no_training_eigensolve():
    torch.set_default_dtype(torch.float64)
    backend = DensityMatrixRMFBackend(
        NucleusCase("PKDD", 8, 8, 16), device=torch.device("cpu")
    )
    network = DensityMatrixNet(
        torch.as_tensor(backend.r),
        backend.block_specs,
        hidden=16,
        depth=1,
        radial_basis_size=4,
    )
    original_eigh = torch.linalg.eigh

    def forbidden_eigh(*args, **kwargs):
        raise AssertionError("density-matrix training forward called torch.linalg.eigh")

    torch.linalg.eigh = forbidden_eigh
    try:
        blocks = network.density_blocks()
        result = backend.evaluate(blocks)
        result.energy_total_tensor.backward()
    finally:
        torch.linalg.eigh = original_eigh

    assert backend.training_dirac_solves == 0
    assert backend.initialization_block_diagonalizations == 0
    assert result.diagnostics["n_number_error"] < 1.0e-10
    assert result.diagnostics["z_number_error"] < 1.0e-10
    assert result.diagnostics["max_orthogonality_error"] < 1.0e-12
    assert result.diagnostics["max_idempotency_rms"] < 1.0e-12

    parameter = network.output_heads[0].weight
    flat_index = int(torch.argmax(parameter.grad.abs()))
    row = flat_index // parameter.shape[1]
    column = flat_index % parameter.shape[1]
    automatic = float(parameter.grad[row, column])
    epsilon = 1.0e-5
    with torch.no_grad():
        original = float(parameter[row, column])
        parameter[row, column] = original + epsilon
        plus = float(backend.evaluate(network.density_blocks()).energy_total_tensor)
        parameter[row, column] = original - epsilon
        minus = float(backend.evaluate(network.density_blocks()).energy_total_tensor)
        parameter[row, column] = original
    finite_difference = (plus - minus) / (2.0 * epsilon)
    assert abs(automatic - finite_difference) / max(abs(finite_difference), 1.0e-10) < 1.0e-5

    orbital_rows = {key: [] for key in ("G", "F", "occupancy", "degeneracy", "kinetic_expectation_mev")}
    for block in blocks:
        split = block["factors"].shape[0] // 2
        for column in range(block["factors"].shape[1]):
            factor = block["factors"][:, column]
            g = torch.zeros(len(backend.r), dtype=factor.dtype)
            f = torch.zeros_like(g)
            g[1:-1] = factor[:split] / np.sqrt(backend.functional.h)
            f[1:-1] = factor[split:] / np.sqrt(backend.functional.h)
            orbital_rows["G"].append(g)
            orbital_rows["F"].append(f)
            orbital_rows["occupancy"].append(block["occupations"][column])
            orbital_rows["degeneracy"].append(block["degeneracy"])
            orbital_rows["kinetic_expectation_mev"].append(
                factor @ block["free_hamiltonian"] @ factor * backend.functional.params.hbar_c
            )
    orbitals = {name: torch.stack(values) for name, values in orbital_rows.items()}
    orbitals["species_sign"] = torch.cat([
        block["species_sign"].expand(block["factors"].shape[1]) for block in blocks
    ])
    orbital_densities = backend.functional.densities_from_orbitals(orbitals)
    for name, density in result.densities.items():
        # The direct-density backend replaces the two artificial box endpoints
        # by exact kappa-origin and bound-tail values.  The legacy orbital
        # helper intentionally retains its zero-box convention.
        assert torch.max(torch.abs(density[1:-1] - orbital_densities[name][1:-1])).item() < 1.0e-12
    assert abs(
        float(backend.kinetic_energy(blocks) - backend.functional.exact_kinetic_energy(orbitals))
    ) < 1.0e-10

    inverse = inverse_hamiltonian_constraint(
        blocks,
        rmf_block_hamiltonians(backend, result),
        {
            "n": -backend.functional.params.mass_n / backend.functional.params.hbar_c,
            "p": -backend.functional.params.mass_p / backend.functional.params.hbar_c,
        },
    )
    assert inverse.minimum_branch_score > 0.0
    assert inverse.linear_solve_residual < 1.0e-10
    assert inverse.solve_count == len(blocks)
    try:
        backend.refresh_positive_energy_spaces(result.hamiltonian.detach())
    except RuntimeError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("no-refresh backend accepted a subspace refresh")


def test_density_matrix_cli_and_activations_are_explicit():
    args = build_parser().parse_args(["--mode", "density-matrix"])
    assert args.mode == "density-matrix"
    assert args.density_refresh_interval == 0
    assert args.density_inverse_weight == 1.0
    assert args.density_stationarity_weight == 1.0
    assert args.density_energy_weight == 1.0e-2
    assert args.density_spectral_gate == 1.0e-2
    assert args.density_stationarity_gate == 1.0e-2
    backend = DensityMatrixRMFBackend(
        NucleusCase("PKDD", 8, 8, 16), device=torch.device("cpu")
    )
    network = DensityMatrixNet(
        torch.as_tensor(backend.r), backend.block_specs, hidden=16, depth=2, radial_basis_size=4
    )
    activations = [module for module in network.modules() if isinstance(module, nn.SiLU)]
    assert len(activations) == 3
    assert not any(isinstance(module, (nn.ReLU, nn.Tanh, nn.GELU)) for module in network.modules())


def test_spherical_shell_space_extends_without_changing_core_prefix():
    shells = spherical_shells(400)
    assert shells[: len(SHELLS)] == SHELLS
    assert sum(shell[4] for shell in shells) >= 400
    assert len({(shell[1], shell[3]) for shell in shells}) == len(shells)


if __name__ == "__main__":
    test_surrogate_gradient_matches_backend_gradient()
    print("PASS PRL surrogate gradient")
    test_7point_adf_moment_conditions()
    print("PASS 7-point ADF moment conditions")
    test_rhf_backend_is_reserved()
    print("PASS RHF reserved interface")
    test_prl_default_backend_is_differentiable_torch_rmf()
    print("PASS PRL trainer defaults to differentiable torch RMF")
    test_torch_rmf_backend_gradient_is_autodiff_consistent()
    print("PASS torch RMF autograd gradient check")
