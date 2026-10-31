import torch
import pytest

from dpl_rhf.rhfb.operators import DensityFactors, NambuAction, stationarity_metrics


def test_density_factors_reject_nonprojective_and_mixed_input_states():
    q = torch.eye(4, dtype=torch.complex128)[:, :2]
    rho = torch.eye(2, dtype=torch.complex128)
    chi = torch.zeros_like(rho)
    with pytest.raises(ValueError, match="orthonormal"):
        DensityFactors(2 * q, rho, chi)
    with pytest.raises(ValueError, match="pure"):
        DensityFactors(q, .5 * rho, chi)
    with pytest.raises(ValueError, match="Hermitian"):
        DensityFactors(q, rho + 1j * rho, chi)
    with pytest.raises(ValueError, match="antisymmetric"):
        DensityFactors(q, rho, rho)


def test_factorized_density_and_hamiltonian_match_independent_dense_actions():
    torch.manual_seed(781)
    q, _ = torch.linalg.qr(torch.randn(5, 3, dtype=torch.complex128))
    rhoa = torch.diag(torch.tensor([1., 0.3, 0.3], dtype=torch.complex128))
    chia = torch.zeros_like(rhoa)
    chia[1, 2] = (0.21 ** 0.5)
    chia[2, 1] = -chia[1, 2]
    h = torch.randn(5, 5, dtype=torch.complex128)
    h = (h + h.mH) / 2
    gap = torch.randn_like(h)
    gap = (gap - gap.mT) / 2
    factors = DensityFactors(q, rhoa, chia)
    action = NambuAction(5, lambda x: h @ x, lambda x: gap @ x, 0.7)
    rho, chi = q @ rhoa @ q.mH, q @ chia @ q.mT
    r = torch.cat((torch.cat((rho, chi), dim=1),
                   torch.cat((-chi.conj(), torch.eye(5) - rho.conj()), dim=1)), dim=0)
    identity = torch.eye(5, dtype=torch.complex128)
    hb = torch.cat((torch.cat((h - .7 * identity, gap), dim=1),
                    torch.cat((-gap.conj(), -h.conj() + .7 * identity), dim=1)), dim=0)
    vectors = torch.randn(10, 4, dtype=torch.complex128)
    assert torch.allclose(factors.apply(vectors), r @ vectors, atol=1e-13)
    assert torch.allclose(action.apply(vectors), hb @ vectors, atol=1e-13)
    expected = torch.linalg.vector_norm(hb @ r - r @ hb) / (10 ** .5)
    for chunk in (1, 3, 20):
        assert torch.allclose(stationarity_metrics(action, factors, chunk_size=chunk)["full_rms_mev"], expected)


def test_full_residual_detects_pairing_wholly_outside_active_space():
    q = torch.eye(4, dtype=torch.complex128)[:, :2]
    rhoa = torch.eye(2, dtype=torch.complex128)
    chi = torch.zeros_like(rhoa)
    h = torch.diag(torch.tensor([1., 2., 3., 4.], dtype=torch.complex128))
    delta = torch.zeros_like(h)
    delta[2, 3], delta[3, 2] = .2, -.2
    result = stationarity_metrics(NambuAction(4, lambda x: h @ x, lambda x: delta @ x),
                                  DensityFactors(q, rhoa, chi), chunk_size=2)
    assert result["active_frobenius_mev"] == 0
    assert result["complement_frobenius_mev"] == 0
    assert result["full_rms_mev"] > .1
