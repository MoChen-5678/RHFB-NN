"""Stationary roots, redundant neural charts, and a blocked Nambu benchmark."""
import torch
from torch import nn
from torch.func import functional_call

from dpl_rhf.rhfb.stationary_solver import (
    StationaryOptions, real_residual, solve_stationarity,
)
from dpl_rhf.rhfb.branch import (
    no_sea_branch_reference, nambu_hamiltonian,
    select_ph_branches_reference_only, transform_branch_reference,
)
from dpl_rhf.rhfb.branch_prototype import _coordinate_basis


def test_saddle_is_found_without_energy_minimization():
    # E=x²-y² is unbounded below but its physical equation has a stationary
    # saddle. A redundant third coordinate checks gauge regularization.
    def equations(x):
        return torch.stack((2 * (x[0] + x[2]), -2 * x[1]))
    result = solve_stationarity(
        equations, torch.tensor([.8, -.7, .2], dtype=torch.float64)
    )
    assert result.converged
    assert result.residual.abs().max() < 1e-9
    assert abs(float(result.parameters[0] + result.parameters[2])) < 1e-9


def test_nonzero_equation_is_not_called_converged():
    def equations(x):
        return torch.stack((x[0], x[0] * 0 + 1))
    result = solve_stationarity(equations, torch.tensor([.1], dtype=torch.float64))
    assert not result.converged
    assert result.residual.abs().max() == 1


def test_neural_odd_paired_nambu_root_matches_independent_branch(monkeypatch):
    d = 8
    energies = torch.tensor([-2., -1., 0., .8, 1.8, 3., -12., -14.], dtype=torch.float64)
    reference = no_sea_branch_reference(
        energies, torch.tensor([1., 1., 0., 0., 0., 0., 0., 0.], dtype=torch.float64),
        dirac_cutoff=-5., chemical_potential=-.1, blocked_index=2,
        blocking_operation="particle",
    )
    h = torch.diag(energies).to(torch.complex128)
    delta = torch.zeros_like(h)
    for i, j in ((0, 1), (3, 4)):
        delta[i, j] = .6 + .18j
        delta[j, i] = -delta[i, j]
    ham = nambu_hamiltonian(h, delta, chemical_potential=-.1)
    # Independent oracle runs before the training guard and never supplies
    # candidate coordinates/initial weights.
    oracle_frame, _ = select_ph_branches_reference_only(ham, reference.frame)
    oracle = oracle_frame @ oracle_frame.mH
    a_basis, b_basis = _coordinate_basis(d, torch.device("cpu"))
    na = a_basis.shape[0]
    count = na + b_basis.shape[0]
    torch.manual_seed(20261006)
    net = nn.Linear(3, 1, dtype=torch.float64)
    features = torch.stack((torch.ones(count, dtype=torch.float64),
                            torch.linspace(-1, 1, count, dtype=torch.float64),
                            torch.linspace(-1, 1, count, dtype=torch.float64).square()), dim=1)
    with torch.no_grad():
        net.weight.mul_(.01)
        net.bias.zero_()
    # Full real chart plus a small neural correction; all neural parameters
    # participate in the solve. This tests algorithm, not physical PKA1/D1S.
    initial = torch.cat((torch.zeros(count, dtype=torch.float64),
                         net.weight.detach().reshape(-1), net.bias.detach()))

    def state(x):
        coordinates = x[:count] + functional_call(
            net, {"weight": x[count:count+3].reshape(1, 3),
                  "bias": x[count+3:count+4]}, (features,)
        ).reshape(-1)
        a = torch.einsum("i,ijk->jk", coordinates[:na].to(torch.complex128), a_basis)
        b = torch.einsum("i,ijk->jk", coordinates[na:].to(torch.complex128), b_basis)
        return transform_branch_reference(reference, a, b)

    def equations(x):
        gd = state(x)
        comm = ham @ gd.matrix - gd.matrix @ ham
        return real_residual(comm, torch.trace(gd.rho).real - 3.)

    def forbidden(*args, **kwargs):
        raise AssertionError("candidate stationary solve called an eigensolver")
    monkeypatch.setattr(torch.linalg, "eigh", forbidden)
    monkeypatch.setattr(torch.linalg, "eig", forbidden)
    monkeypatch.setattr(torch.linalg, "eigvalsh", forbidden)
    result = solve_stationarity(equations, initial, options=StationaryOptions(max_iterations=35))
    assert result.converged, (result.stop_reason, result.history[-1])
    gd = state(result.parameters)
    assert gd.fermion_parity == -1
    assert torch.linalg.matrix_norm(gd.chi) > .1
    assert torch.linalg.matrix_norm(gd.matrix - oracle) < 1e-8
    assert gd.rho.diagonal().real[-2:].abs().max() < 1e-10
    assert torch.linalg.vector_norm(result.parameters[count:] - initial[count:]) > 1e-5
