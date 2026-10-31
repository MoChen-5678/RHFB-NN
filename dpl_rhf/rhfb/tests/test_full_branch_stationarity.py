import numpy as np
import pytest
import torch

from dpl_rhf.rhfb.full_branch_stationarity import (
    canonical_tangent,
    frame_h20_residual,
    full_stationarity_residual,
    nambu_hamiltonian,
    tangent_density,
    tangent_energy_derivative,
)
from dpl_rhf.rhfb.manifold import generalized_density_matrix, slater_reference
from dpl_rhf.rhfb.parity_diagnostic import number_parity_from_density


def _forbid_eig(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("stationarity prototype must not diagonalize")
    for name in ("eig", "eigh", "eigvals", "eigvalsh"):
        monkeypatch.setattr(torch.linalg, name, forbidden)
        monkeypatch.setattr(np.linalg, name, forbidden)


def _sample_odd_state():
    dtype = torch.complex128
    a = torch.tensor([
        [0.0, 0.11j, -0.07],
        [0.11j, 0.0, 0.04j],
        [0.07, 0.04j, 0.0],
    ], dtype=dtype)
    # Make A anti-Hermitian explicitly from a generic matrix.
    a = 0.5 * (a - a.mH)
    b = torch.tensor([
        [0.0, 0.08 + 0.03j, -0.02j],
        [-0.08 - 0.03j, 0.0, 0.05],
        [0.02j, -0.05, 0.0],
    ], dtype=dtype)
    seed = slater_reference(torch.tensor([1., 0., 0.], dtype=torch.float64))
    k = canonical_tangent(a, b)
    w = torch.matrix_exp(k)
    r = w @ seed.matrix @ w.mH
    rho, chi = r[:3, :3], r[:3, 3:]
    return k, r, rho, chi


def test_full_h20_is_exactly_full_projector_commutator_without_spectrum(monkeypatch):
    _forbid_eig(monkeypatch)
    dtype = torch.complex128
    k, r, _, _ = _sample_odd_state()
    h = torch.diag(torch.tensor([-0.4, 0.3, 1.2], dtype=torch.float64)).to(dtype)
    delta = torch.tensor([
        [0.0, 0.04 + 0.01j, -0.03j],
        [-0.04 - 0.01j, 0.0, 0.02],
        [0.03j, -0.02, 0.0],
    ], dtype=dtype)
    h_mu = nambu_hamiltonian(h, delta, chemical_potential=0.17)
    comm = full_stationarity_residual(h_mu, r)

    # Build an occupied frame from the known Slater reference columns and its
    # canonical transform, without spectral decomposition of H or R.
    seed = slater_reference(torch.tensor([1., 0., 0.], dtype=torch.float64))
    occ = torch.diag(seed.matrix).real > 0.5
    frame0 = torch.eye(2 * h.shape[0], dtype=dtype)[:, occ]
    frame = torch.matrix_exp(k) @ frame0
    h20 = frame_h20_residual(h_mu, frame)
    assert torch.allclose(frame @ frame.mH, r, atol=3e-15, rtol=0)
    assert torch.linalg.matrix_norm(comm).square().item() == pytest.approx(
        2.0 * torch.linalg.matrix_norm(h20).square().item(), abs=2e-13
    )


def test_thouless_tangent_derivative_matches_centered_difference_and_parity(monkeypatch):
    _forbid_eig(monkeypatch)
    dtype = torch.complex128
    k, r, rho, chi = _sample_odd_state()
    parity = number_parity_from_density(rho.detach().numpy(), chi.detach().numpy())
    assert parity["number_parity"] == -1

    h = torch.tensor([
        [0.2, 0.03j, 0.0],
        [-0.03j, 0.8, 0.04],
        [0.0, 0.04, 1.4],
    ], dtype=dtype)
    delta = torch.tensor([
        [0.0, 0.06 + 0.01j, -0.02j],
        [-0.06 - 0.01j, 0.0, 0.03],
        [0.02j, -0.03, 0.0],
    ], dtype=dtype)
    mu = 0.11
    h_mu = nambu_hamiltonian(h, delta, mu)
    dr = tangent_density(r, k)
    analytic = float(tangent_energy_derivative(h_mu, dr))

    def objective(step):
        w = torch.matrix_exp(step * k)
        rt = w @ r @ w.mH
        return float((0.5 * torch.trace(h_mu @ rt)).real)

    step = 1.0e-6
    finite_difference = (objective(step) - objective(-step)) / (2.0 * step)
    assert abs(analytic - finite_difference) < 2.0e-9

    # A connected canonical path preserves the density's independently
    # computed number parity while allowing general full-space pairing.
    r1 = torch.matrix_exp(0.31 * k) @ r @ torch.matrix_exp(0.31 * k).mH
    rho1, chi1 = r1[:3, :3], r1[:3, 3:]
    assert torch.linalg.matrix_norm(chi1) > 1.0e-3
    assert torch.allclose(r1 @ r1, r1, atol=3e-14, rtol=0)
    assert number_parity_from_density(rho1.numpy(), chi1.numpy())["number_parity"] == -1
