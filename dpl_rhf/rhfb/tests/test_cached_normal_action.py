import pytest
import torch

from dpl_rhf.rhfb.cached_normal_action import CachedNormalAction
from dpl_rhf.rhfb.cached_normal_action import evaluate_two_species_normal_columns


class _HermitianOracle:
    def __init__(self, matrix):
        self.matrix = matrix
        self.vector_chunk = 8

    def apply(self, vector):
        return self.matrix @ vector


def test_cached_action_is_exact_on_declared_span_and_rejects_outside():
    torch.manual_seed(713)
    d, k = 8, 3
    raw = torch.randn((d, d), dtype=torch.float64).to(torch.complex128)
    h = (raw + raw.mH) / 2
    raw_u = torch.randn((d, k), dtype=torch.float64).to(torch.complex128)
    u, _ = torch.linalg.qr(raw_u)
    cache = CachedNormalAction.from_oracle(_HermitianOracle(h), u)
    coefficients = torch.randn((k, 2), dtype=torch.float64).to(torch.complex128)
    inside = u @ coefficients
    torch.testing.assert_close(cache.apply(inside), h @ inside, atol=2e-14, rtol=2e-14)
    probe = torch.linalg.qr(torch.randn((d, d), dtype=torch.float64).to(torch.complex128)).Q[:, -1]
    with pytest.raises(ValueError, match="outside cached span"):
        cache.apply(probe)


def test_full_orthonormal_cache_preserves_hermitian_action():
    torch.manual_seed(714)
    d = 6
    raw = torch.randn((d, d), dtype=torch.float64).to(torch.complex128)
    h = (raw + raw.mH) / 2
    identity = torch.eye(d, dtype=torch.complex128)
    cache = CachedNormalAction.from_oracle(_HermitianOracle(h), identity)
    materialized = cache.apply(identity)
    torch.testing.assert_close(materialized, h, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(materialized, materialized.mH, atol=2e-14, rtol=0)


def test_joint_species_tangent_returns_both_complete_linear_actions():
    torch.manual_seed(715)
    d, m = 5, 2
    eye = torch.eye(d, dtype=torch.complex128)
    states = {}
    operators = {}
    for species in ("n", "p"):
        raw = torch.randn((d, d), dtype=torch.float64).to(torch.complex128)
        operators[species] = (raw + raw.mH) / 2
        rho = torch.diag(torch.tensor([0.3, 0.7], dtype=torch.complex128))
        chi = torch.zeros((m, m), dtype=torch.complex128)
        states[species] = type("State", (), {"q": eye[:, :m], "rho_active": rho,
                                              "chi_active": chi})()

    class _Basis:
        dimension = d

    def energy(_basis, tangent):
        value = torch.zeros((), dtype=torch.float64)
        for species in ("n", "p"):
            q, rho = tangent[species].q, tangent[species].rho_active
            density = q @ rho @ q.mH
            value = value + torch.sum(operators[species] * density.T).real
        return value

    v = torch.randn((d, 3), dtype=torch.float64).to(torch.complex128)
    actual = evaluate_two_species_normal_columns(_Basis(), states, energy, v,
                                                  vector_chunk=2)
    for species in ("n", "p"):
        torch.testing.assert_close(actual[species], operators[species] @ v,
                                   atol=2e-13, rtol=2e-13)
