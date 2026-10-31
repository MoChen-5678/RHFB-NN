import torch

from dpl_rhf.rhfb.frozen_field_neural import (
    FrozenFields,
    _residual_tensors,
    _smooth_chart_penalty,
)
from dpl_rhf.rhfb.manifold import GeneralizedDensity, generalized_density_matrix


def test_fixed_active_density_changes_primitive_residual_when_q_rotates():
    dtype = torch.complex128
    h = torch.diag(torch.tensor([0.0, 1.0], dtype=torch.float64)).to(dtype)
    delta = torch.zeros((2, 2), dtype=dtype)
    fields = FrozenFields(h, delta, particle_number=1, species="neutron", metadata={})
    rho_active = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64)).to(dtype)
    chi_active = torch.zeros((2, 2), dtype=dtype)
    gd = GeneralizedDensity(rho_active, chi_active,
                            generalized_density_matrix(rho_active, chi_active), -1)
    angle = torch.tensor(torch.pi / 4, dtype=torch.float64)
    q_identity = torch.eye(2, dtype=dtype)
    q_rotated = torch.tensor([[torch.cos(angle), -torch.sin(angle)],
                              [torch.sin(angle), torch.cos(angle)]], dtype=torch.float64).to(dtype)
    mu = torch.tensor(0.0, dtype=torch.float64)

    residual_identity, _ = _residual_tensors(q_identity, gd, fields, mu)
    residual_rotated, _ = _residual_tensors(q_rotated, gd, fields, mu)

    # Independent construction of the physical normal density and BdG commutator.
    rho_physical = q_rotated @ rho_active @ q_rotated.mH
    assert torch.linalg.matrix_norm(rho_physical - rho_active) > 0.5
    eye = torch.eye(2, dtype=dtype)
    r_physical = torch.cat((
        torch.cat((rho_physical, torch.zeros_like(rho_physical)), dim=1),
        torch.cat((torch.zeros_like(rho_physical), eye - rho_physical.conj()), dim=1),
    ), dim=0)
    h_bdg = torch.cat((
        torch.cat((h, delta), dim=1),
        torch.cat((-delta.conj(), -h.conj()), dim=1),
    ), dim=0)
    independently_built = h_bdg @ r_physical - r_physical @ h_bdg

    assert torch.linalg.matrix_norm(residual_identity) < 1e-12
    assert torch.linalg.matrix_norm(residual_rotated) > 0.5
    assert torch.allclose(residual_rotated, independently_built, atol=1e-12, rtol=0)


def test_chart_penalty_autodiff_matches_finite_difference_across_limit():
    limit = 2.0
    points = (limit - 0.01, limit, limit + 0.01)
    step = 1.0e-6
    derivatives = []
    for point in points:
        x = torch.tensor(point, dtype=torch.float64, requires_grad=True)
        value = _smooth_chart_penalty(x, limit)
        autodiff = torch.autograd.grad(value, x)[0].item()
        with torch.no_grad():
            plus = _smooth_chart_penalty(torch.tensor(point + step, dtype=torch.float64), limit).item()
            minus = _smooth_chart_penalty(torch.tensor(point - step, dtype=torch.float64), limit).item()
        finite_difference = (plus - minus) / (2.0 * step)
        assert abs(autodiff - finite_difference) <= 2e-4 * max(1.0, abs(autodiff))
        derivatives.append(autodiff)

    # Values on both sides must vary continuously through the nominal norm limit.
    assert derivatives[0] < derivatives[1] < derivatives[2]
    assert derivatives[2] - derivatives[0] < 1.0e8
