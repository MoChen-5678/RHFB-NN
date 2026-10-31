"""Independent number response and canonical correction checks."""
import math

import pytest
import torch

from dpl_rhf.rhfb.bogoliubov_blocking import (
    bogoliubov_frame, slater_bogoliubov_frame, state_from_frame,
    swap_quasiparticle_column,
)
from dpl_rhf.rhfb.hfb_gradient import n20_matrix, thouless_density_derivative
from dpl_rhf.rhfb.number_retraction import correct_mean_particle_number


def paired_state(theta, *, odd=False):
    m = 5 if odd else 4
    u = torch.eye(m, dtype=torch.complex128)
    v = torch.zeros_like(u)
    for start in (0, 2):
        u[start:start+2, start:start+2] *= math.cos(theta)
        v[start, start+1] = math.sin(theta)
        v[start+1, start] = -math.sin(theta)
    frame = bogoliubov_frame(u, v)
    return (swap_quasiparticle_column(frame, 4, fermion_parity=1) if odd
            else state_from_frame(frame, fermion_parity=1))


def test_number_response_has_independent_bcs_normalization():
    theta = 0.37
    state = paired_state(theta)
    n20 = n20_matrix(state.u, state.v)
    drho, _ = thouless_density_derivative(state.u, state.v, n20)
    # Two independent identical BCS pairs: ||N20||²=16 u² v².
    expected = 16 * math.cos(theta)**2 * math.sin(theta)**2
    assert float(torch.trace(drho).real) == pytest.approx(expected, abs=1.e-13)
    assert float(n20.abs().square().sum()) == pytest.approx(expected, abs=1.e-13)


@pytest.mark.parametrize("odd,target", [(False, 2), (True, 3)])
@pytest.mark.parametrize("theta", [0.25, 1.12])
def test_corrects_number_and_preserves_canonical_parity_without_eigensolve(
        odd, target, theta, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("number correction must not solve a spectrum")
    monkeypatch.setattr(torch.linalg, "eigh", forbidden)
    monkeypatch.setattr(torch.linalg, "eigvalsh", forbidden)
    result = correct_mean_particle_number(paired_state(theta, odd=odd), target)
    assert result.converged, result
    assert abs(result.errors[-1]) < 1.e-10
    assert all(abs(b) < abs(a) for a, b in zip(result.errors, result.errors[1:]))
    state = result.state
    r = state.density.matrix
    assert torch.max(torch.abs(r @ r-r)) < 1.e-12
    assert torch.max(torch.abs(state.density.chi.mT+state.density.chi)) < 1.e-12
    parity = -1 if odd else 1
    # A determinant accumulates the errors of all columns. Use the declared
    # 1e-10 structural target, while retaining stricter per-entry checks above.
    assert torch.linalg.det(state.frame).real == pytest.approx(parity, abs=1.e-10)
    assert state.density.fermion_parity == parity
    assert torch.diag(state.density.rho).real[:4].tolist() == pytest.approx([0.5]*4, abs=1.e-10)
    if odd:
        assert state.blocked_index == 4
        assert torch.allclose(r @ state.blocked_mode, state.blocked_mode)
        assert float(state.density.rho[-1,-1].real) == pytest.approx(1.)


def test_unpaired_wrong_number_reports_singular_without_injecting_pairing():
    frame = slater_bogoliubov_frame(torch.zeros(4, dtype=torch.float64))
    state = state_from_frame(frame, fermion_parity=1)
    result = correct_mean_particle_number(state, 2)
    assert not result.converged
    assert result.reason == "singular_number_response"
    assert result.accepted_corrections == 0
    assert torch.equal(result.state.frame, frame)
    assert torch.count_nonzero(result.state.density.chi) == 0


def test_existing_correct_unpaired_number_needs_no_response():
    frame = slater_bogoliubov_frame(torch.tensor([1.,1.,0.,0.],dtype=torch.float64))
    result = correct_mean_particle_number(state_from_frame(frame, fermion_parity=1), 2)
    assert result.converged
    assert result.accepted_corrections == 0


def test_number_iteration_limit_is_not_reported_as_success():
    result = correct_mean_particle_number(paired_state(0.25), 2, max_corrections=0)
    assert not result.converged
    assert result.reason == "number_iteration_limit"
