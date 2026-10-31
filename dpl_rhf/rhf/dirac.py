from __future__ import annotations

import numpy as np
import torch


def _first_derivative_coefficients(offsets: list[int]) -> np.ndarray:
    matrix = np.asarray(
        [[float(offset) ** power for offset in offsets] for power in range(len(offsets))]
    )
    right_hand_side = np.zeros(len(offsets))
    right_hand_side[1] = 1.0
    return np.linalg.solve(matrix, right_hand_side)


def forward_adf_derivative(r: torch.Tensor, order: int = 7) -> torch.Tensor:
    """Forward asymmetric derivative used by the radial Dirac blocks."""

    size = r.numel() - 2
    h = float(r[2] - r[1])
    if order not in {4, 5, 6, 7}:
        raise ValueError("RHF uses a 5- or 7-point asymmetric derivative")
    points = 5 if order in {4, 5} else 7
    if size < points:
        raise ValueError(f"{points}-point ADF requires at least {points} interior points")
    coefficients = torch.as_tensor(
        _first_derivative_coefficients(list(range(points))) / h,
        dtype=r.dtype,
        device=r.device,
    )
    derivative = torch.zeros((size, size), dtype=r.dtype, device=r.device)
    for row in range(size):
        stop = min(size, row + points)
        derivative[row, row:stop] = coefficients[: stop - row]
    return derivative


class RHFRadialDiracOperator:
    """RHF kinetic operator from RHF-e Eqs. (3.390) and (3.393).

    The program subtracts the species rest mass from the eigenvalue. Therefore
    the upper/lower diagonal entries are 0 and -2M/(hbar*c), respectively.
    Interaction fields are not accepted here: Hartree and Fock operators are
    built independently from the RHF functional derivative.
    """

    def __init__(
        self,
        r: torch.Tensor,
        *,
        mass_n_mev: float,
        mass_p_mev: float,
        hbar_c_mev_fm: float,
        derivative_order: int = 7,
    ) -> None:
        self.r = r
        self.x = r[1:-1]
        self.mass_n_mev = float(mass_n_mev)
        self.mass_p_mev = float(mass_p_mev)
        self.hbar_c_mev_fm = float(hbar_c_mev_fm)
        self.derivative_forward = forward_adf_derivative(r, derivative_order)
        self.derivative_backward = -self.derivative_forward.T

    @staticmethod
    def component_angular_momenta(kappa: int) -> tuple[int, int]:
        if kappa == 0:
            raise ValueError("Dirac kappa cannot be zero")
        if kappa < 0:
            return -kappa - 1, -kappa
        return kappa, kappa - 1

    def matrix_fm_inverse(self, species_sign: float, kappa: int) -> torch.Tensor:
        l_upper, _ = self.component_angular_momenta(kappa)
        radial_g_is_odd = (l_upper + 1) % 2 == 1
        if radial_g_is_odd:
            d_forward = self.derivative_forward
            d_backward = self.derivative_backward
        else:
            d_forward = self.derivative_backward
            d_backward = self.derivative_forward
        angular = torch.diag(torch.full_like(self.x, float(kappa)) / self.x)
        # RHF-e (3.390, 3.393): B1=-D_backward+kappa/r,
        # B2=D_forward+kappa/r.
        b1 = -d_backward + angular
        b2 = d_forward + angular
        mass = self.mass_n_mev if species_sign > 0.0 else self.mass_p_mev
        zero = torch.zeros_like(b1)
        lower_mass = torch.diag(
            torch.full_like(self.x, -2.0 * mass / self.hbar_c_mev_fm)
        )
        return torch.cat(
            [torch.cat([zero, b1], dim=1), torch.cat([b2, lower_mass], dim=1)],
            dim=0,
        )
