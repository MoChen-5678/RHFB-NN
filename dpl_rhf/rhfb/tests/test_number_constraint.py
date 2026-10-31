"""Independent finite-space checks of the particle-number Gram equation."""
from types import SimpleNamespace
import os

import pytest
import torch

from dpl_rhf.rhfb.manifold import generalized_density_matrix
from dpl_rhf.rhfb.number_constraint import number_tangent_projection
from dpl_rhf.rhfb.operators import DensityFactors
from dpl_rhf.rhfb.selfconsistent_stationarity import selfconsistent_stationarity_loss


def _state(angle=0.47):
    angle = torch.as_tensor(angle, dtype=torch.float64)
    rho = torch.diag(torch.stack((torch.ones_like(angle),
                                  torch.sin(angle)**2, torch.sin(angle)**2))).to(torch.complex128)
    skew = torch.tensor([[0, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.complex128)
    chi = torch.sin(angle) * torch.cos(angle) * skew
    raw = torch.tensor([[1, .2j, 0], [.1j, 1, .3], [.2, 0, 1],
                        [0, .3j, .2], [.1, .2, .3j]], dtype=torch.complex128)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return DensityFactors(q, rho, chi)


def _fields(d):
    raw = torch.arange(d*d, dtype=torch.float64).reshape(d, d)
    h = torch.complex(raw+raw.T, .13*(raw-raw.T)) / 17
    delta_raw = torch.complex(.03*raw, .07*raw.T)
    return h, delta_raw-delta_raw.T


def _dense_commutator(state, h, delta, mu):
    d = state.dimension
    eye = torch.eye(d, dtype=torch.complex128)
    rh = state.q @ state.rho_active @ state.q.mH
    ch = state.q @ state.chi_active @ state.q.T
    r = generalized_density_matrix(rh, ch)
    ham = torch.cat((torch.cat((h-mu*eye, delta), 1),
                     torch.cat((-delta.conj(), -h.conj()+mu*eye), 1)), 0)
    return ham @ r-r @ ham


def test_multiplier_is_dense_full_nambu_gram_solution_with_complement_pairing():
    state = _state()
    h, delta = _fields(state.dimension)
    result = number_tangent_projection(state, h@state.q, delta@state.q.conj())
    c0 = _dense_commutator(state, h, delta, 0.)
    direction = c0-_dense_commutator(state, h, delta, 1.)
    expected = (c0.conj()*direction).sum().real / direction.abs().square().sum()
    torch.testing.assert_close(result.chemical_potential, expected, atol=2e-14, rtol=2e-14)
    optimum = _dense_commutator(state, h, delta, result.chemical_potential)
    for shift in (-.3, .3):
        assert (optimum.abs().square().sum()
                < _dense_commutator(state,h,delta,expected+shift).abs().square().sum())
    torch.testing.assert_close((optimum.conj()*direction).sum().real,
                               torch.zeros((), dtype=torch.float64), atol=1e-13, rtol=0)
    # Delta does have a part outside Q. The scalar reduction must not be
    # interpreted as permission to drop that part from the full residual.
    projector = state.q @ state.q.mH
    assert torch.linalg.vector_norm(delta-projector@delta@projector.conj()) > .1
    assert result.identifiable


def test_variance_is_independent_odd_fock_number_distribution():
    angle = .47
    result = number_tangent_projection(_state(angle), *_field_actions(_state(angle)))
    # Independent odd state: cos(theta)|100> + sin(theta)|111>.
    probs = torch.tensor([torch.cos(torch.tensor(angle, dtype=torch.float64))**2,
                          torch.sin(torch.tensor(angle, dtype=torch.float64))**2])
    numbers = torch.tensor([1., 3.], dtype=torch.float64)
    variance = (probs*numbers.square()).sum()-(probs*numbers).sum().square()
    torch.testing.assert_close(result.number_variance, variance, atol=2e-14, rtol=0)
    torch.testing.assert_close(result.number_variance_from_rho, variance, atol=2e-14, rtol=0)
    torch.testing.assert_close(result.constraint_gram, 4*variance, atol=2e-14, rtol=0)


def _field_actions(state):
    h, delta = _fields(state.dimension)
    return h@state.q, delta@state.q.conj()


def test_pairing_collapse_preserves_supplied_mu_and_reports_singular_constraint():
    state = _state(0.)
    fallback = torch.tensor(-7.2, dtype=torch.float64, requires_grad=True)
    result = number_tangent_projection(state, *_field_actions(state), fallback_mu=fallback)
    assert not result.identifiable
    assert result.number_variance == 0
    torch.testing.assert_close(result.chemical_potential, fallback)
    torch.testing.assert_close(torch.autograd.grad(result.chemical_potential, fallback)[0],
                               torch.ones_like(fallback))


def test_multiplier_retains_density_and_field_response_gradient():
    def evaluate(angle):
        state = _state(angle)
        h, delta = _fields(state.dimension)
        rho = state.q@state.rho_active@state.q.mH
        chi = state.q@state.chi_active@state.q.T
        return number_tangent_projection(state, (h+.3*rho)@state.q,
                                         (delta+.4*chi)@state.q.conj()).chemical_potential
    angle = torch.tensor(.47, dtype=torch.float64, requires_grad=True)
    grad, = torch.autograd.grad(evaluate(angle), angle)
    step = 1e-5
    finite_difference = (evaluate(.47+step)-evaluate(.47-step))/(2*step)
    torch.testing.assert_close(grad, finite_difference, atol=2e-9, rtol=2e-8)
    assert grad.abs() > 1e-3


class _Normal:
    def differentiable_energy_and_fields(self, states):
        h, _ = _fields(5)
        return sum(torch.trace(h@s.q@s.rho_active@s.q.mH).real for s in states.values()), {s:h for s in states}


class _Pair:
    def __init__(self, basis, q, chi, **kwargs):
        _, self.delta = _fields(5)
    def apply(self, vector):
        return self.delta@vector


def test_joint_evaluation_uses_solved_mu_but_keeps_actual_number_error():
    basis = SimpleNamespace(dimension=5, kappa_max=0)
    states = {"n":_state(.47), "p":_state(.62)}
    result = selfconsistent_stationarity_loss(
        basis,states,{"n":-9.,"p":-8.},{"n":3,"p":2},
        normal_functional=_Normal(), pairing_action_factory=_Pair,
        number_weight=100., chemical_potential_mode="number_tangent", residual_chunk_size=2)
    h, delta = _fields(5)
    for species in states:
        diag = result.species_diagnostics[species]
        dense = _dense_commutator(states[species], h, delta, diag["chemical_potential_mev"])
        torch.testing.assert_close(diag["full_commutator_frobenius_squared_mev2"],
                                   dense.abs().square().sum(), atol=2e-12, rtol=2e-12)
        assert diag["number_error"].abs() > .1
        assert diag["number_tangent_identifiable"] == 1.
    assert not result.physical_result_accepted
    assert not result.no_sea_branch_certified


def test_invalid_numerical_gram_tolerance_is_rejected():
    state = _state()
    with pytest.raises(ValueError, match="Gram"):
        number_tangent_projection(state, *_field_actions(state), numerical_gram_tolerance=-1.)


@pytest.mark.skipif(not os.environ.get("RHFB_CUDA_UUID"), reason="requires authorized guarded CUDA launcher")
def test_gpu_one_multiplier_and_field_response_match_cpu():
    from dpl_rhf.rhfb.cuda_resources import configure_cuda
    evidence = configure_cuda("cuda:0", budget_mib=int(os.environ["RHFB_CUDA_BUDGET_MIB"]))
    assert evidence["physical_index"] == 1
    cpu_state = _state()
    cpu_h, cpu_delta = _fields(5)
    outputs = []
    gradients = []
    for device in ("cpu", "cuda:0"):
        state = DensityFactors(*(v.to(device) for v in (
            cpu_state.q, cpu_state.rho_active, cpu_state.chi_active)))
        scale = torch.tensor(.37, dtype=torch.float64, device=device, requires_grad=True)
        h = cpu_h.to(device) * (1+scale)
        delta = cpu_delta.to(device) * (1+.2*scale)
        result = number_tangent_projection(state, h@state.q, delta@state.q.conj())
        gradient, = torch.autograd.grad(result.chemical_potential, scale)
        outputs.append(result.chemical_potential.detach().cpu())
        gradients.append(gradient.detach().cpu())
    torch.testing.assert_close(outputs[0],outputs[1],atol=2e-12,rtol=2e-12)
    torch.testing.assert_close(gradients[0],gradients[1],atol=2e-12,rtol=2e-12)
