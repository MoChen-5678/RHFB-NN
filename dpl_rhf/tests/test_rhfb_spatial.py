import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad

from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.spatial import GreenOperator


def make_operator(step=1., radius=8., lmax=2):
    basis = AngularRadialBasis(radius_fm=radius, spacing_fm=step, kappa_max=1,
                              n_theta=lmax+1, n_phi=2*lmax+2, derivative_order=5)
    return basis, GreenOperator(basis, lmax=lmax)


@pytest.mark.parametrize('mass', [0., 138.])
def test_green_block_equals_action_and_has_finite_diagonal(mass):
    basis, green = make_operator()
    torch.manual_seed(712)
    source = torch.randn(basis.point_count, 2, dtype=torch.complex128)
    block = green.kernel_block(0, basis.point_count, 0, basis.point_count, mass)
    assert torch.isfinite(block).all()
    torch.testing.assert_close(block, block.T, atol=1e-14, rtol=1e-14)
    weights = torch.as_tensor(basis.weights, dtype=torch.float64)
    oracle = block.to(torch.complex128) @ (weights[:, None]*source)
    torch.testing.assert_close(green.apply(source, mass), oracle, atol=2e-13, rtol=2e-13)
    energy = (source.conj()*green.apply(source, mass)*weights[:, None]).sum().real
    assert energy > 0


def test_green_coulomb_uniform_ball_converges_to_analytic_potential():
    errors = []
    for step in (.25, .125):
        basis, green = make_operator(step=step)
        source = torch.ones(basis.point_count, dtype=torch.float64)
        potential = green.apply(source, 0.).reshape(basis.n_radial, -1)[:, 0]
        r = torch.as_tensor(basis.x, dtype=torch.float64)
        exact = basis.radius_fm**2/2-r.square()/6
        errors.append(float((potential-exact).abs().max()))
    # Interior quadrature excludes r=R and r=0, giving first-order box error;
    # this explicitly measures convergence, not a continuum accuracy claim.
    assert errors[1] < .53*errors[0]


def test_green_yukawa_spherical_source_matches_independent_angular_integral():
    basis, green = make_operator(step=.5)
    source = np.exp(-basis.x**2/4)
    field = green.apply(torch.as_tensor(np.repeat(source, basis.n_angular_points)), 138.)
    radius = 2.
    index = round(radius/basis.spacing_fm)-1
    mu = 138./basis.hbar_c_mev_fm
    # Direct numerical angular integral of exp(-mu*s)/(4pi*s), independent
    # of the implementation's Bessel functions and spherical normalization.
    def shell(rp):
        return .5*quad(lambda u: math.exp(-mu*math.sqrt(radius**2+rp**2-2*radius*rp*u)) /
                       math.sqrt(radius**2+rp**2-2*radius*rp*u), -1, 1, epsabs=2e-10)[0]
    expected = sum(basis.spacing_fm*rp**2*f*shell(rp) for rp, f in zip(basis.x, source))
    assert abs(float(field.reshape(basis.n_radial, -1)[index, 0])-expected) < 2e-9


def test_cartesian_adf_gradient_polynomial_and_weighted_adjoint():
    basis, green = make_operator(step=.25, radius=8., lmax=2)
    positions = torch.as_tensor(basis.positions, dtype=torch.float64)
    xyz = positions[:, :]
    gradient = green.gradient(xyz)
    interior = torch.as_tensor(np.repeat((basis.x > 1.5) & (basis.x < 6.), basis.n_angular_points))
    expected = torch.eye(3, dtype=torch.float64).expand(int(interior.sum()), 3, 3)
    torch.testing.assert_close(gradient[interior], expected, atol=2e-12, rtol=2e-12)
    torch.manual_seed(19)
    f = torch.randn(basis.point_count, 2, dtype=torch.complex128)
    v = torch.randn(basis.point_count, 3, 2, dtype=torch.complex128)
    weights = torch.as_tensor(basis.weights, dtype=torch.float64)
    left = (green.gradient(f).conj()*v*weights[:, None, None]).sum()
    right = (f.conj()*green.gradient_adjoint(v)*weights[:, None]).sum()
    torch.testing.assert_close(left, right, atol=5e-12, rtol=3e-13)


def test_derivative_blocks_equal_gradient_of_green_action_and_double_gradient():
    basis, green = make_operator()
    torch.manual_seed(6)
    source = torch.randn(basis.point_count, dtype=torch.complex128)
    weights = torch.as_tensor(basis.weights, dtype=torch.float64)
    dx = green.kernel_block(0, basis.point_count, 0, basis.point_count, 138., derivative_x=True)
    direct = torch.einsum('xyi,y->xi', dx.to(source.dtype), source*weights)
    torch.testing.assert_close(direct, green.gradient(green.apply(source, 138.)), atol=1e-12, rtol=3e-12)
    xy = green.kernel_block(0, basis.point_count, 0, basis.point_count, 138., derivative_x=True, derivative_y=True)
    y = green.kernel_block(0, basis.point_count, 0, basis.point_count, 138., derivative_y=True)
    expected = green.gradient(y.to(torch.complex128)).real.permute(0, 2, 1, 3)
    torch.testing.assert_close(xy, expected, atol=1e-12, rtol=3e-12)


def test_green_source_derivative_has_density_autograd_path():
    basis, green = make_operator()
    torch.manual_seed(3)
    source = torch.randn(basis.point_count, dtype=torch.float64, requires_grad=True)
    weights = torch.as_tensor(basis.weights, dtype=torch.float64)
    energy = .5*(source*green.apply(source, 138.)*weights).sum()
    derivative, = torch.autograd.grad(energy, source)
    torch.testing.assert_close(derivative/weights, green.apply(source, 138.), atol=1e-12, rtol=1e-12)


def test_green_rejects_unresolved_angular_cutoff():
    basis = AngularRadialBasis(radius_fm=8., spacing_fm=1., kappa_max=1)
    with pytest.raises(ValueError, match='angular quadrature'):
        GreenOperator(basis, lmax=4)
