from __future__ import annotations

import numpy as np
import torch

from fieldspace_model import FieldSpaceRMFNet
from fieldspace_rmf import (
    DifferentiableDiracMatrix,
    build_reconstructed_state,
    noncentral_derivative_matrix,
    woods_saxon_local_stack,
)
from rmf_functional import PKDDRMFFunctional
from variational_rmf import independent_orbital_data


def make_problem(device: str = "cpu", hidden: int = 16, derivative_order: int = 1):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(20240623)
    r_np = np.concatenate(([0.001], np.arange(0.1, 20.0 + 0.05, 0.1)))
    r = torch.as_tensor(r_np, device=device)
    initial_np = woods_saxon_local_stack(8, 8, r_np)
    initial = torch.as_tensor(initial_np, device=device)
    data, _ = independent_orbital_data(8, 8, r_np)
    functional = PKDDRMFFunctional(r, 8, 8)
    matrix = DifferentiableDiracMatrix(r, data, initial, derivative_order)
    network = FieldSpaceRMFNet(r_np, 8, 8, initial_np, hidden).to(device)
    return r, functional, matrix, network


def objective(functional, matrix, network):
    trial = network()
    orbitals = matrix.solve(trial)
    orbitals, state, _, rebuilt = build_reconstructed_state(functional, orbitals)
    energy = (
        functional.exact_kinetic_energy(orbitals)
        + functional.direct_energy_terms(state)["E_direct"]
    ) / 16.0
    scales = trial.new_tensor([0.15, 0.30, 0.15, 0.30])[:, None]
    mismatch = ((trial - rebuilt) / scales).square().mean()
    return energy / 50.0 + 0.1 * mismatch, orbitals, state


def test_matrix_is_hermitian_and_states_are_normalized():
    for derivative_order in (1, 2, 4, 7):
        _, functional, matrix, network = make_problem(derivative_order=derivative_order)
        trial = network()
        for species, kappa in ((1.0, -1), (1.0, -2), (-1.0, -1), (-1.0, -2)):
            hamiltonian = matrix.matrix(trial, species, kappa)
            assert torch.max(torch.abs(hamiltonian - hamiltonian.T)).item() < 1.0e-13
        _, orbitals, state = objective(functional, matrix, network)
        norms = torch.sum(
            functional.orbital_w[None, :] * (orbitals["G"].square() + orbitals["F"].square()), dim=1
        )
        assert torch.max(torch.abs(norms - 1.0)).item() < 1.0e-12
        neutron_number = torch.sum(functional.measure * state["rho_v_n"])
        proton_number = torch.sum(functional.measure * state["rho_v_p"])
        assert abs(float(neutron_number) - 8.0) < 1.0e-10
        assert abs(float(proton_number) - 8.0) < 1.0e-10


def test_second_order_derivative_is_consistent_inside_domain():
    torch.set_default_dtype(torch.float64)
    r_np = np.concatenate(([0.001], np.arange(0.1, 20.0 + 0.05, 0.1)))
    r = torch.as_tensor(r_np)
    x = r[1:-1]
    derivative = noncentral_derivative_matrix(r, order=2)
    values = x.square()
    numerical = derivative @ values
    exact = 2.0 * x
    assert torch.max(torch.abs(numerical[:-2] - exact[:-2])).item() < 1.0e-10


def test_zhang_adf_stencils():
    torch.set_default_dtype(torch.float64)
    r_np = np.concatenate(([0.001], np.arange(0.1, 20.0 + 0.05, 0.1)))
    r = torch.as_tensor(r_np)
    h = 0.1
    five_point = noncentral_derivative_matrix(r, order=4)[10, 10:15] * h
    expected_five = torch.tensor([-25.0, 48.0, -36.0, 16.0, -3.0]) / 12.0
    assert torch.max(torch.abs(five_point - expected_five)).item() < 1.0e-13

    seven_point = noncentral_derivative_matrix(r, order=7)
    seven_coefficients = seven_point[10, 10:17] * h
    offsets = torch.arange(7, dtype=torch.float64)
    for power in range(7):
        moment = torch.sum(seven_coefficients * offsets**power)
        expected = 1.0 if power == 1 else 0.0
        assert abs(float(moment - expected)) < 1.0e-10


def test_autograd_directional_derivative():
    _, functional, matrix, network = make_problem(hidden=8)
    parameters = tuple(network.parameters())
    value, _, _ = objective(functional, matrix, network)
    gradients = torch.autograd.grad(value, parameters)
    torch.manual_seed(7)
    direction = tuple(torch.randn_like(parameter) for parameter in parameters)
    norm = torch.sqrt(sum(vector.square().sum() for vector in direction))
    direction = tuple(vector / norm for vector in direction)
    analytic = sum((gradient * vector).sum() for gradient, vector in zip(gradients, direction))

    epsilon = 1.0e-5
    with torch.no_grad():
        for parameter, vector in zip(parameters, direction):
            parameter.add_(vector, alpha=epsilon)
        plus = objective(functional, matrix, network)[0]
        for parameter, vector in zip(parameters, direction):
            parameter.add_(vector, alpha=-2.0 * epsilon)
        minus = objective(functional, matrix, network)[0]
        for parameter, vector in zip(parameters, direction):
            parameter.add_(vector, alpha=epsilon)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    relative_error = abs(float(analytic - finite_difference)) / max(
        abs(float(analytic)), abs(float(finite_difference)), 1.0e-10
    )
    assert relative_error < 2.0e-4, (float(analytic), float(finite_difference), relative_error)


if __name__ == "__main__":
    test_matrix_is_hermitian_and_states_are_normalized()
    print("PASS hermiticity, orbital normalization, and particle numbers")
    test_second_order_derivative_is_consistent_inside_domain()
    print("PASS second-order noncentral derivative consistency")
    test_zhang_adf_stencils()
    print("PASS Zhang-style asymmetric finite-difference stencils")
    test_autograd_directional_derivative()
    print("PASS autograd directional derivative")
