import torch
from torch.func import functional_call

from dpl_rhf.rhfb.bounded_neural_fixedpoint import (
    BoundedFixedPointSiLUNet,
    _RawOutputView,
    solve_bounded_neural_fixedpoint,
)
from dpl_rhf.rhfb.neural_rhfb_fixedpoint import _parameter_views
from dpl_rhf.rhfb.stationary_solver import StationaryOptions


class AnalyticResidualProblem:
    def __init__(self, target):
        self.target = target

    def state_from_coordinates(self, c, eta, mu):
        return c, eta, mu

    def residual_blocks(self, state):
        c, eta, mu = state
        return {
            "stationarity": torch.complex(c - self.target, .2 * (c - self.target)),
            "field_self_consistency": eta - .15,
            "number": mu + .2,
        }


def test_bounded_all_trainable_network_reduces_residual_with_step_and_output_limits():
    center = torch.zeros(3, dtype=torch.float64)
    problem = AnalyticResidualProblem(torch.tensor([.18, -.24, .12], dtype=torch.float64))
    result = solve_bounded_neural_fixedpoint(
        problem, center, 0., 0., hidden=3, seed=14,
        maximum_step_l2=.2,
        options=StationaryOptions(max_iterations=30, residual_tolerance=2e-7,
                                  initial_damping=1e-3, maximum_trial_steps=12,
                                  maximum_variables=128, maximum_residuals=32),
    )
    initial_max = float(torch.tensor([.24, .15, .2, .048, .04]).max())
    assert float(result.residual.abs().max()) < initial_max
    assert torch.all(result.output_coordinates <= result.initial_output + result.output_radii)
    assert torch.all(result.output_coordinates >= result.initial_output - result.output_radii)
    assert result.parameter_change_l2 > 1e-5
    assert all(result.per_layer_change_l2[name] > 0 for name in
               ("hidden.weight", "hidden.bias", "output.weight", "output.bias"))
    accepted = [row["step_l2"] for row in result.solver.history if row.get("step_accepted")]
    assert accepted and max(accepted) <= .2 + 1e-12
    assert all(value == 0 for value in result.eigensolver_counts.values())


def test_fixed_center_and_tanh_bound_reconstruct_initial_output_and_limit_head():
    center = torch.tensor([.32, -.3, 0., .02, .45], dtype=torch.float64)
    radii = torch.tensor([1.25, 1.25, 1.25, .5, 4.], dtype=torch.float64)
    net = BoundedFixedPointSiLUNet(center, radii, hidden=3, seed=5)
    initial = torch.cat([p.detach().reshape(-1) for p in net.parameters()])
    initial_views = _parameter_views(net, initial)
    raw0 = functional_call(_RawOutputView(net), initial_views, (net.features,))
    y0 = net.center + net.radii * torch.tanh(raw0)
    assert torch.allclose(y0, center, atol=1e-15, rtol=0.)

    with torch.no_grad():
        net.output.bias.fill_(100.)
    y = net()
    assert torch.all(y <= center + radii)
    assert torch.all(y >= center - radii)
    assert torch.allclose(y, center + radii, atol=1e-12, rtol=0.)
