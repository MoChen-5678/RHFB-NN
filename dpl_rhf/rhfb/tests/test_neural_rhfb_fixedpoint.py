from dataclasses import dataclass

import torch

from dpl_rhf.rhfb.neural_rhfb_fixedpoint import solve_neural_fixedpoint
from dpl_rhf.rhfb.stationary_solver import StationaryOptions


@dataclass
class AnalyticPhysicalProblem:
    """Interface fixture; physical-equation smoke test, not an RHFB model."""

    target: torch.Tensor

    def state_from_coordinates(self, coordinates, eta_raw, mu):
        return coordinates, eta_raw, mu

    def residual_blocks(self, state):
        coordinates, eta_raw, mu = state
        # The complex block exercises exact real/imaginary equation packing.
        stationarity = torch.complex(
            coordinates - self.target,
            0.25 * (coordinates - self.target),
        )
        return {
            "stationarity": stationarity,
            "field_self_consistency": eta_raw - 0.2,
            "number": mu + 0.35,
        }


def test_silu_weights_are_functional_call_solved_against_physical_residual():
    torch.manual_seed(19)
    target = torch.tensor([0.15, -0.3, 0.1], dtype=torch.float64)
    problem = AnalyticPhysicalProblem(target)
    result = solve_neural_fixedpoint(
        problem,
        torch.zeros(3, dtype=torch.float64),
        0.0,
        0.0,
        options=StationaryOptions(
            max_iterations=50,
            residual_tolerance=2e-8,
            initial_damping=1e-3,
            maximum_trial_steps=12,
            maximum_variables=512,
            maximum_residuals=64,
        ),
    )
    assert result.solver.converged, (result.solver.stop_reason, result.solver.history[-1])
    assert float(result.residual.abs().max()) < 2e-8
    assert torch.allclose(result.output_coordinates[:3], target, atol=2e-8, rtol=0.0)
    assert abs(float(result.output_coordinates[-2]) - 0.2) < 2e-8
    assert abs(float(result.output_coordinates[-1]) + 0.35) < 2e-8
    assert result.parameter_change_l2 > 1e-5
    assert result.per_layer_change_l2["output.bias"] > 1e-5
    assert result.per_layer_change_l2["hidden.weight"] > 0.0
    assert result.per_layer_change_l2["hidden.bias"] > 0.0
    assert result.per_layer_change_l2["output.weight"] > 0.0
