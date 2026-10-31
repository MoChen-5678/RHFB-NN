import torch

from nuclear_matter_rmf import PKDDNuclearMatter


def test_gauss_legendre_polynomial_exactness():
    solver = PKDDNuclearMatter(quadrature_order=16)
    kf = torch.tensor(1.7, dtype=torch.float64)
    for power in range(12):
        value = solver.integrate_fermi_sphere(kf, lambda p, power=power: p**power)
        expected = kf ** (power + 1) / ((power + 1) * torch.pi**2)
        assert torch.allclose(value, expected, atol=1.0e-13, rtol=1.0e-13)


def test_pkdd_symmetric_matter_variational_and_thermodynamic_consistency():
    result = PKDDNuclearMatter(quadrature_order=96).solve(0.149552, 0.0)
    assert result["sigma_field_residual"] < 1.0e-7
    assert result["thermodynamic_residual_mev"] < 1.0e-7
    assert -20.0 < result["binding_mev"] < -10.0


if __name__ == "__main__":
    test_gauss_legendre_polynomial_exactness()
    print("PASS Gauss-Legendre polynomial exactness")
    test_pkdd_symmetric_matter_variational_and_thermodynamic_consistency()
    print("PASS PKDD field equation and thermodynamic consistency")
