"""Independent checks of the radial ADF stencil used by the Dirac operator."""

from __future__ import annotations

import math

import torch

from dpl_rhf.rhf.dirac import RHFRadialDiracOperator, forward_adf_derivative


def _radial_grid(step: float, radius: float = 4.0) -> torch.Tensor:
    intervals = round(radius / step)
    r = torch.arange(intervals + 1, dtype=torch.float64) * step
    r[0] = 0.001
    return r


def test_order5_forward_coefficients_match_zhang_eq13() -> None:
    # Eq. (13), written here as a source-literal oracle rather than generated
    # by the production Vandermonde coefficient helper.
    expected = torch.tensor([-25.0, 48.0, -36.0, 16.0, -3.0], dtype=torch.float64) / 12.0
    step = 0.1
    r = _radial_grid(step)
    derivative = forward_adf_derivative(r, order=5)
    row = round(1.0 / step) - 1  # x=1 fm, away from either box boundary
    torch.testing.assert_close(
        derivative[row, row : row + 5], expected / step, rtol=0, atol=2e-13
    )
    assert torch.count_nonzero(derivative[row, row + 5 :]).item() == 0


def test_negative_transpose_is_eq14_backward_stencil_away_from_boundary() -> None:
    step = 0.1
    derivative_forward = forward_adf_derivative(_radial_grid(step), order=5)
    derivative_backward = -derivative_forward.T
    expected = torch.tensor([25.0, -48.0, 36.0, -16.0, 3.0], dtype=torch.float64) / (12.0 * step)
    row = round(2.0 / step) - 1  # all four preceding points remain interior
    torch.testing.assert_close(
        derivative_backward[row, row - 4 : row + 1],
        expected.flip(0),
        rtol=0,
        atol=2e-13,
    )
    torch.testing.assert_close(
        derivative_backward,
        -derivative_forward.T,
        rtol=0,
        atol=0,
    )


def _smooth_derivative_error(order: int, step: float) -> float:
    r = _radial_grid(step)
    x = r[1:-1]
    derivative = forward_adf_derivative(r, order=order)
    row = round(1.0 / step) - 1
    approximate = derivative[row] @ torch.exp(x)
    return abs(float(approximate) - math.e)


def test_adf5_and_adf7_recover_fourth_and_sixth_order_interior_accuracy() -> None:
    # f=exp(x) has a known derivative; x=1 fm is held fixed as the grid is
    # refined, and both forward stencils fit fully inside the radial box.
    for order, expected_order in ((5, 4), (7, 6)):
        coarse = _smooth_derivative_error(order, 0.1)
        fine = _smooth_derivative_error(order, 0.05)
        ratio = coarse / fine
        assert ratio > 0.75 * (2**expected_order)
        assert ratio < 1.35 * (2**expected_order)


def test_radial_dirac_parity_switches_g_f_adf_and_keeps_hermiticity() -> None:
    r = _radial_grid(0.1, radius=3.0)
    operator = RHFRadialDiracOperator(
        r,
        mass_n_mev=939.0,
        mass_p_mev=938.0,
        hbar_c_mev_fm=197.328284,
        derivative_order=5,
    )
    forward = operator.derivative_forward
    backward = operator.derivative_backward
    angular_s = torch.diag(torch.full_like(operator.x, -1.0) / operator.x)
    angular_p = torch.diag(torch.full_like(operator.x, 1.0) / operator.x)

    # kappa=-1: l_G=0, hence reduced G ~ r^(l+1) is odd (forward); F uses
    # the adjoint backward operator. kappa=+1: l_G=1 gives even reduced G,
    # so the assignments exchange.
    expected_s = operator.matrix_fm_inverse(1.0, -1)
    expected_p = operator.matrix_fm_inverse(1.0, 1)
    n = operator.x.numel()
    b1_s, b2_s = expected_s[:n, n:], expected_s[n:, :n]
    b1_p, b2_p = expected_p[:n, n:], expected_p[n:, :n]
    torch.testing.assert_close(b1_s, -backward + angular_s)
    torch.testing.assert_close(b2_s, forward + angular_s)
    torch.testing.assert_close(b1_p, -forward + angular_p)
    torch.testing.assert_close(b2_p, backward + angular_p)
    torch.testing.assert_close(b1_s, b2_s.T)
    torch.testing.assert_close(b1_p, b2_p.T)
    torch.testing.assert_close(expected_s, expected_s.T, rtol=0, atol=2e-13)
    torch.testing.assert_close(expected_p, expected_p.T, rtol=0, atol=2e-13)
