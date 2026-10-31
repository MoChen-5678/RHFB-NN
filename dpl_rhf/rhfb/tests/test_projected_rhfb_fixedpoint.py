import torch

from dpl_rhf.rhfb.moving_particle_space_prototype import make_moving_toy
from dpl_rhf.rhfb.no_sea_particle_space import (
    embed_active_density, positive_frame_from_projector,
)
from dpl_rhf.rhfb.pairing import D1S, pairing_gap
from dpl_rhf.rhfb.projected_rhfb_fixedpoint import (
    candidate_spectral_solver_guard,
    initial_pair_coordinates,
    make_projected_fixedpoint_problem,
    reference_positive_bdg_state,
)


def test_projected_fixedpoint_state_and_residual_interface_are_canonical():
    toy, _ = make_moving_toy()
    problem = make_projected_fixedpoint_problem(toy, blocked_index=5)
    coords = initial_pair_coordinates(problem)
    eta_raw = torch.tensor(.02, dtype=torch.float64, requires_grad=True)
    mu = torch.tensor(.45, dtype=torch.float64)
    with candidate_spectral_solver_guard() as counts:
        state = problem.state_from_coordinates(coords, eta_raw, mu)
        blocks = problem.residual_blocks(state)
        residual = blocks.as_vector()
        # A projector's Frobenius norm is fixed by rank; inspect its
        # positive/negative off-diagonal block to test moving-Q response.
        observable = state.projector[:6, 6:].real.square().sum()
        grad = torch.autograd.grad(observable, eta_raw)[0]
    assert counts == {"torch.linalg.eigh": 0, "torch.linalg.eigvalsh": 0,
                      "torch.linalg.eig": 0}
    assert state.active.fermion_parity == -1
    assert torch.linalg.matrix_norm(state.active.matrix @ state.active.matrix
                                    - state.active.matrix) < 1e-10
    assert torch.allclose(blocks.number, state.number - toy.target_number)
    assert abs(float(blocks.number.detach())) > 1e-8
    assert torch.linalg.matrix_norm(state.chi) > 1e-3
    assert blocks.stationarity.shape == (12, 12)
    assert blocks.field_self_consistency.numel() == 1
    assert blocks.number.numel() == 1
    assert residual.dtype == torch.float64 and torch.isfinite(residual).all()
    assert torch.isfinite(grad)


def test_density_output_adapter_exposes_all_physical_residual_blocks():
    toy, _ = make_moving_toy()
    problem = make_projected_fixedpoint_problem(toy, blocked_index=5)
    active = problem._active_from_coordinates(initial_pair_coordinates(problem))
    state = problem.evaluate_network_outputs(
        active.rho, active.chi, torch.tensor(.01, dtype=torch.float64),
        torch.tensor(.45, dtype=torch.float64), fermion_parity=-1,
    )
    blocks = problem.residual_blocks(state)
    assert blocks.stationarity.shape == (12, 12)
    assert blocks.field_self_consistency.dtype == torch.float64
    assert blocks.number.dtype == torch.float64


def test_reference_path_uses_explicit_bdg_branch_tracking():
    toy, _ = make_moving_toy()
    problem = make_projected_fixedpoint_problem(toy, blocked_index=5)
    active_seed = problem._active_from_coordinates(initial_pair_coordinates(problem))
    full = embed_active_density(toy.seed_frame, active_seed.rho, active_seed.chi,
                                fermion_parity=-1)
    _normal, h, _x = toy.normal(full.rho)
    _epair, delta = pairing_gap(toy.spinors, full.chi, toy.positions, toy.weights,
                                species="neutron", chunk_size=8, force=D1S)
    values, vectors = torch.linalg.eigh(h)
    selected = vectors[:, values > toy.shift]
    p = selected @ selected.mH
    frame = positive_frame_from_projector(p, toy.seed_frame,
                                          projector_tolerance=None)
    active, reference, branch = reference_positive_bdg_state(problem, h, delta, frame, .45)
    assert active.fermion_parity == -1
    assert torch.linalg.matrix_norm(reference.matrix @ reference.matrix
                                    - reference.matrix) < 1e-8
    assert branch["reference_only_eigensolves"] == 1.0
