from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
import torch
from scipy.special import ive, kve
from sympy import Rational
from sympy.physics.wigner import clebsch_gordan, wigner_3j, wigner_6j

from dpl_rhf.rhf.density import kappa_quantum_numbers


@lru_cache(maxsize=None)
def _cg_squared(j1_twice: int, j2_twice: int, j_twice: int, m1_twice: int, m2_twice: int) -> float:
    value = clebsch_gordan(
        Rational(j1_twice, 2),
        Rational(j2_twice, 2),
        Rational(j_twice, 2),
        Rational(m1_twice, 2),
        Rational(m2_twice, 2),
        Rational(m1_twice + m2_twice, 2),
    )
    return float(value) ** 2


def spin_cg_squared(kappa_a: int, kappa_b: int, multipole: int) -> float:
    ja2 = 2 * abs(kappa_a) - 1
    jb2 = 2 * abs(kappa_b) - 1
    return _cg_squared(ja2, jb2, 2 * multipole, 1, -1)


def orbital_cg_squared(l_a: int, l_b: int, multipole: int) -> float:
    return _cg_squared(2 * l_a, 2 * l_b, 2 * multipole, 0, 0)


class SphericalPropagators:
    """Eq. (3.15)-(3.16) radial propagators on the physical mesh."""

    def __init__(self, r: torch.Tensor, hbar_c: float):
        self.r = r
        self.hbar_c = float(hbar_c)
        self._cache: dict[tuple[str, int, float], torch.Tensor] = {}

    def _yukawa_between(
        self, field_r: np.ndarray, source_r: np.ndarray, multipole: int, mass_mev: float
    ) -> np.ndarray:
        inverse_range = float(mass_mev) / self.hbar_c
        z_field = inverse_range * field_r
        z_source = inverse_range * source_r
        z_left = np.minimum(z_field[:, None], z_source[None, :])
        z_right = np.maximum(z_field[:, None], z_source[None, :])
        order = multipole + 0.5
        product = ive(order, z_left) * kve(order, z_right) * np.exp(z_left - z_right)
        return inverse_range * product / np.sqrt(z_field[:, None] * z_source[None, :])

    @staticmethod
    def _coulomb_between(
        field_r: torch.Tensor, source_r: torch.Tensor, multipole: int
    ) -> torch.Tensor:
        small = torch.minimum(field_r[:, None], source_r[None, :])
        large = torch.maximum(field_r[:, None], source_r[None, :])
        return small.pow(multipole) / (
            (2 * multipole + 1) * large.pow(multipole + 1)
        )

    def yukawa(self, multipole: int, mass_mev: float) -> torch.Tensor:
        key = ("yukawa", int(multipole), float(mass_mev))
        if key in self._cache:
            return self._cache[key]
        radius = self.r.detach().cpu().numpy()
        kernel = self._yukawa_between(radius, radius, multipole, mass_mev)
        result = torch.as_tensor(kernel, dtype=self.r.dtype, device=self.r.device)
        self._cache[key] = result
        return result

    def coulomb(self, multipole: int) -> torch.Tensor:
        key = ("coulomb", int(multipole), 0.0)
        if key in self._cache:
            return self._cache[key]
        result = self._coulomb_between(self.r, self.r, multipole)
        self._cache[key] = result
        return result

    @staticmethod
    def _fold_half_mesh(raw: torch.Tensor, h: float) -> torch.Tensor:
        """RHF-e Eqs. (3.402)-(3.409), identical to Combination:Folding."""

        npt = (raw.shape[1] + 1) // 2
        if raw.shape[1] != 2 * npt - 1 or npt < 13:
            raise ValueError("six-point RHF folding requires a 2*N-1 half mesh")
        ai = raw.new_tensor(
            [
                [0.24609375, 1.23046875, -0.8203125, 0.4921875, -0.17578125, 0.02734375],
                [-0.02734375, 0.41015625, 0.8203125, -0.2734375, 0.08203125, -0.01171875],
                [0.01171875, -0.09765625, 0.5859375, 0.5859375, -0.09765625, 0.01171875],
                [-0.01171875, 0.08203125, -0.2734375, 0.8203125, 0.41015625, -0.02734375],
                [0.02734375, -0.17578125, 0.4921875, -0.8203125, 1.23046875, 0.24609375],
            ]
        )
        folded = raw.new_zeros((raw.shape[0], npt))
        # The following indices are the exact zero-based translation of the
        # five loops in the document's reference implementation.
        for j in range(1, 7):
            for k in range(1, 4):
                folded[:, j - 1] += raw[:, 2 * k - 1] * ai[k - 1, j - 1]
        for j in range(2, 7):
            for k in range(1, j):
                folded[:, j - 1] += raw[:, 2 * k + 5] * ai[2, j - k - 1]
        for j in range(7, npt - 5):
            for k in range(1, 7):
                folded[:, j - 1] += raw[:, 2 * (j - 4 + k) - 1] * ai[2, k - 1]
        for j in range(npt - 5, npt + 1):
            for k in range(3, 6):
                index = j - (npt - 6)
                folded[:, j - 1] += raw[:, 2 * (npt - 6 + k) - 1] * ai[k - 1, index - 1]
        for j in range(npt - 5, npt):
            for k in range(1, npt - j + 1):
                index = j - (npt - 6) + k
                folded[:, j - 1] += raw[:, 2 * (npt - 3 - k) - 1] * ai[2, index - 1]
        folded[:, 0] = 4.0 * folded[:, 0] + raw[:, 0]
        folded[:, 1:-1] = 4.0 * folded[:, 1:-1] + 2.0 * raw[:, 2:-1:2]
        folded[:, -1] = 4.0 * folded[:, -1] + raw[:, -1]
        return folded * (h / 6.0)

    def folded_yukawa(self, multipole: int, mass_mev: float) -> torch.Tensor:
        key = ("folded-yukawa", int(multipole), float(mass_mev))
        if key in self._cache:
            return self._cache[key]
        h = float((self.r[2] - self.r[1]).detach().cpu())
        source = np.arange(2 * self.r.numel() - 1, dtype=np.float64) * (h / 2.0)
        source[0] = float(self.r[0].detach().cpu())
        raw = self._yukawa_between(
            self.r.detach().cpu().numpy(), source, multipole, mass_mev
        )
        raw_tensor = torch.as_tensor(raw, dtype=self.r.dtype, device=self.r.device)
        result = self._fold_half_mesh(raw_tensor, h)
        self._cache[key] = result
        return result

    def folded_coulomb(self, multipole: int) -> torch.Tensor:
        key = ("folded-coulomb", int(multipole), 0.0)
        if key in self._cache:
            return self._cache[key]
        h = float((self.r[2] - self.r[1]).detach().cpu())
        source = torch.arange(
            2 * self.r.numel() - 1, dtype=self.r.dtype, device=self.r.device
        ) * (h / 2.0)
        source[0] = self.r[0]
        raw = self._coulomb_between(self.r, source, multipole)
        result = self._fold_half_mesh(raw, h)
        self._cache[key] = result
        return result

    def _mixed_yukawa_between(
        self,
        field_r: np.ndarray,
        source_r: np.ndarray,
        field_order: int,
        source_order: int,
        mass_mev: float,
        *,
        signed: bool,
    ) -> np.ndarray:
        """Mixed-order Bessel products used by XBIT/XBIV in RHF-e."""

        inverse_range = float(mass_mev) / self.hbar_c
        zf = inverse_range * field_r[:, None]
        zs = inverse_range * source_r[None, :]
        denominator = np.sqrt(field_r[:, None] * source_r[None, :])

        def product(i_order: int, i_z: np.ndarray, k_order: int, k_z: np.ndarray) -> np.ndarray:
            return (
                ive(i_order + 0.5, i_z)
                * kve(k_order + 0.5, k_z)
                * np.exp(i_z - k_z)
            )

        source_inside = source_r[None, :] < field_r[:, None]
        source_outside = source_r[None, :] > field_r[:, None]
        inside = product(source_order, zs, field_order, zf) / denominator
        outside = product(field_order, zf, source_order, zs) / denominator
        if signed:
            inside = -inside
        equal = 0.5 * (inside + outside)
        return np.where(source_inside, inside, np.where(source_outside, outside, equal))

    def folded_mixed_yukawa(
        self,
        field_order: int,
        source_order: int,
        mass_mev: float,
        *,
        signed: bool = False,
    ) -> torch.Tensor:
        key = (
            "folded-mixed-signed" if signed else "folded-mixed",
            int(field_order),
            int(source_order),
            float(mass_mev),
        )
        if key in self._cache:
            return self._cache[key]
        h = float((self.r[2] - self.r[1]).detach().cpu())
        source = np.arange(2 * self.r.numel() - 1, dtype=np.float64) * (h / 2.0)
        source[0] = float(self.r[0].detach().cpu())
        raw = self._mixed_yukawa_between(
            self.r.detach().cpu().numpy(),
            source,
            field_order,
            source_order,
            mass_mev,
            signed=signed,
        )
        result = self._fold_half_mesh(
            torch.as_tensor(raw, dtype=self.r.dtype, device=self.r.device), h
        )
        self._cache[key] = result
        return result


class PKO2AngularKernels:
    """Angular reductions of RHF-e Eqs. (3.324)-(3.335)."""

    def __init__(self, propagators: SphericalPropagators):
        self.propagators = propagators
        self._cache: dict[tuple[str, int, int, int, int], torch.Tensor] = {}

    @staticmethod
    def maximum_multipole(kappa_a: int, kappa_b: int) -> int:
        return abs(kappa_a) + abs(kappa_b) - 1

    @staticmethod
    def maximum_space_multipole(kappa_a: int, kappa_b: int) -> int:
        """Upper L bound for vector spatial components, Eq. (3.329b-e)."""

        la_u, la_l, _ = kappa_quantum_numbers(kappa_a)
        lb_u, lb_l, _ = kappa_quantum_numbers(kappa_b)
        return max(la_u + lb_l, la_l + lb_u)

    def scalar_time(
        self, channel: str, kappa_a: int, kappa_b: int, mass_mev: float
    ) -> torch.Tensor:
        key = (channel + "-time", kappa_a, kappa_b, 0, 0)
        if key in self._cache:
            return self._cache[key]
        l_a, _, _ = kappa_quantum_numbers(kappa_a)
        l_b, _, _ = kappa_quantum_numbers(kappa_b)
        result = torch.zeros_like(self.propagators.folded_coulomb(0))
        for multipole in range(self.maximum_multipole(kappa_a, kappa_b) + 1):
            if (multipole + l_a + l_b) % 2:
                continue
            # Core stores C3J = |CG|^2/(2L+1) and multiplies it by
            # (2L+1)/(4*pi) in PrePotels.  Using the standard CG coefficient
            # directly therefore leaves |CG|^2/(4*pi) here.
            coefficient = spin_cg_squared(kappa_a, kappa_b, multipole) / (4.0 * math.pi)
            radial = (
                self.propagators.folded_coulomb(multipole)
                if channel == "coulomb"
                else self.propagators.folded_yukawa(multipole, mass_mev)
            )
            result = result + coefficient * radial
        self._cache[key] = result
        return result

    def vector_space(
        self,
        channel: str,
        kappa_a: int,
        kappa_b: int,
        mu: int,
        mu_prime: int,
        mass_mev: float,
    ) -> torch.Tensor:
        key = (channel + "-space", kappa_a, kappa_b, mu, mu_prime)
        if key in self._cache:
            return self._cache[key]
        la_u, la_l, _ = kappa_quantum_numbers(kappa_a)
        lb_u, lb_l, _ = kappa_quantum_numbers(kappa_b)
        result = torch.zeros_like(self.propagators.folded_coulomb(0))
        for multipole in range(self.maximum_space_multipole(kappa_a, kappa_b) + 1):
            c3j = spin_cg_squared(kappa_a, kappa_b, multipole)
            if mu == 0 and mu_prime == 0:
                parity_l = la_u + lb_l
                coefficient = 2.0 * orbital_cg_squared(la_u, lb_l, multipole) - c3j
            elif mu == 1 and mu_prime == 1:
                parity_l = la_l + lb_u
                coefficient = 2.0 * orbital_cg_squared(la_l, lb_u, multipole) - c3j
            else:
                parity_l = la_u + lb_l
                coefficient = c3j
            if (multipole + parity_l) % 2:
                continue
            radial = (
                self.propagators.folded_coulomb(multipole)
                if channel == "coulomb"
                else self.propagators.folded_yukawa(multipole, mass_mev)
            )
            result = result + coefficient * radial / (4.0 * math.pi)
        self._cache[key] = result
        return result


@lru_cache(maxsize=None)
def _bj_symbol(kappa_a: int, kappa_b: int, j: int, multipole: int) -> float:
    """Core-1204 Combination:BJLab, including sqrt(2J+1)."""

    if j == 0 and multipole == 0:
        return 0.0
    k1, k2 = abs(kappa_a), abs(kappa_b)
    l_a = k1 - 1 if kappa_a < 0 else k1
    if j == multipole:
        denominator = math.sqrt(j * (j + 1.0))
        if denominator == 0.0:
            return 0.0
        return math.sqrt(2 * j + 1.0) * (
            k1 + k2 * (-1.0) ** (k1 + k2 + j - 1)
        ) / denominator
    tmp = 0.5 * (multipole - j) * (multipole + j + 1.0)
    if tmp == 0.0:
        return 0.0
    return (-1.0) ** (k1 + l_a) * (kappa_a + kappa_b + tmp) / math.sqrt(abs(tmp))


@lru_cache(maxsize=None)
def _f_symbol(multipole: int, j: int, shifted: int) -> float:
    """Core-1204 Combination:FLJLp."""

    if shifted < 0 or (multipole + shifted + 1) % 2:
        return 0.0
    value = (
        math.sqrt((2 * multipole + 1) * (2 * shifted + 1))
        * float(wigner_3j(multipole, 1, shifted, 0, 0, 0))
        * float(wigner_6j(shifted, multipole, 1, 1, 1, j))
    )
    return value


class PKA1AngularKernels(PKO2AngularKernels):
    """PKA1 pion-PV and rho tensor kernels from RHF-e/Core PrePotels."""

    _pairs = ((-1, -1), (-1, 1), (1, -1), (1, 1))

    @staticmethod
    def _core_c3j(kappa_a: int, kappa_b: int, multipole: int) -> float:
        return spin_cg_squared(kappa_a, kappa_b, multipole) / (2 * multipole + 1)

    def _empty_components(self) -> torch.Tensor:
        size = self.propagators.r.numel()
        return torch.zeros((2, 2, size, size), dtype=self.propagators.r.dtype, device=self.propagators.r.device)

    def pion(self, kappa_a: int, kappa_b: int, mass_mev: float) -> torch.Tensor:
        key = ("pion-pv", kappa_a, kappa_b, 0, 0)
        if key in self._cache:
            return self._cache[key]
        lu1, ld1, _ = kappa_quantum_numbers(kappa_a)
        lu2, ld2, _ = kappa_quantum_numbers(kappa_b)
        lower = min(abs(lu1 - lu2), abs(ld1 - ld2))
        upper = max(lu1 + lu2, ld1 + ld2)
        result = self._empty_components()
        first_multipole = lower - 1
        if first_multipole < 0:
            first_multipole += 2
        for multipole in range(first_multipole, upper + 2, 2):
            c3j = self._core_c3j(kappa_a, kappa_b, multipole)
            base = c3j / ((2 * multipole + 1) * 4.0 * math.pi)
            alpha = {-1: -multipole, 1: multipole + 1.0}
            for i1, i2 in self._pairs:
                l1, l2 = multipole + i1, multipole + i2
                if not (lower <= l1 <= upper and lower <= l2 <= upper):
                    continue
                common = i1 * i2 * base
                coefficients = (
                    -(kappa_a + kappa_b + alpha[i1]) * (kappa_a + kappa_b + alpha[i2]) * common,
                    +(kappa_a + kappa_b + alpha[i1]) * (kappa_a + kappa_b - alpha[i2]) * common,
                    +(kappa_a + kappa_b - alpha[i1]) * (kappa_a + kappa_b + alpha[i2]) * common,
                    -(kappa_a + kappa_b - alpha[i1]) * (kappa_a + kappa_b - alpha[i2]) * common,
                )
                radial = self.propagators.folded_mixed_yukawa(l1, l2, mass_mev)
                for index, coefficient in enumerate(coefficients):
                    result[index // 2, index % 2] += coefficient * radial
        radius = self.propagators.r
        contact = 2.0 / (
            3.0 * (mass_mev / self.propagators.hbar_c) ** 2
            * radius.square() * 4.0 * math.pi
        )
        diagonal = torch.arange(radius.numel(), device=radius.device)
        result[0, 1, diagonal, diagonal] += contact
        result[1, 0, diagonal, diagonal] += contact
        self._cache[key] = result
        return result

    def rho_tensor_time(self, kappa_a: int, kappa_b: int, mass_mev: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = ("rho-tensor-time", kappa_a, kappa_b, 0, 0)
        if key in self._cache:
            cached = self._cache[key]
            return cached[0], cached[1], cached[2]
        lu1, ld1, _ = kappa_quantum_numbers(kappa_a)
        lu2, ld2, _ = kappa_quantum_numbers(kappa_b)
        lower = abs(lu1 - lu2)
        upper = lu1 + lu2
        cross_lower = max(abs(ld1 - lu2), abs(lu1 - ld2))
        cross_upper = min(ld1 + lu2, lu1 + ld2)
        cv = self._empty_components()
        ct = self._empty_components()
        tt = self._empty_components()
        inverse_range = mass_mev / self.propagators.hbar_c
        for multipole in range(lower, upper + 1, 2):
            alpha = {-1: -multipole, 1: multipole + 1.0}
            c3j = self._core_c3j(kappa_a, kappa_b, multipole)
            for i1 in (-1, 1):
                l1 = multipole + i1
                if not cross_lower <= l1 <= cross_upper:
                    continue
                app = (kappa_a - kappa_b + alpha[i1]) * i1 * c3j * inverse_range / (4.0 * math.pi)
                amm = (kappa_a - kappa_b - alpha[i1]) * i1 * c3j * inverse_range / (4.0 * math.pi)
                wrvt = self.propagators.folded_mixed_yukawa(multipole, l1, mass_mev, signed=True)
                wrtv = self.propagators.folded_mixed_yukawa(l1, multipole, mass_mev, signed=True)
                cv[0, 0] -= app * wrvt; cv[0, 1] += amm * wrvt
                cv[1, 0] -= app * wrvt; cv[1, 1] += amm * wrvt
                ct[0, 0] += app * wrtv; ct[0, 1] += app * wrtv
                ct[1, 0] -= amm * wrtv; ct[1, 1] -= amm * wrtv
            for i1, i2 in self._pairs:
                l1, l2 = multipole + i1, multipole + i2
                if not (cross_lower <= l1 <= cross_upper and cross_lower <= l2 <= cross_upper):
                    continue
                common = i1 * i2 * c3j * inverse_range**2 / ((2 * multipole + 1) * 4.0 * math.pi)
                coeffs = (
                    (kappa_a-kappa_b+alpha[i1])*(kappa_a-kappa_b+alpha[i2])*common,
                    -(kappa_a-kappa_b+alpha[i1])*(kappa_a-kappa_b-alpha[i2])*common,
                    -(kappa_a-kappa_b-alpha[i1])*(kappa_a-kappa_b+alpha[i2])*common,
                    (kappa_a-kappa_b-alpha[i1])*(kappa_a-kappa_b-alpha[i2])*common,
                )
                radial = self.propagators.folded_mixed_yukawa(l1, l2, mass_mev)
                for index, coefficient in enumerate(coeffs):
                    tt[index // 2, index % 2] += coefficient * radial
        packed = torch.stack([cv, ct, tt])
        self._cache[key] = packed
        return cv, ct, tt

    def rho_tensor_space(self, kappa_a: int, kappa_b: int, mass_mev: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = ("rho-tensor-space", kappa_a, kappa_b, 0, 0)
        if key in self._cache:
            cached = self._cache[key]
            return cached[0], cached[1], cached[2]
        lu1, ld1, _ = kappa_quantum_numbers(kappa_a)
        lu2, ld2, _ = kappa_quantum_numbers(kappa_b)
        lower = min(abs(lu1 - ld2), abs(ld1 - lu2))
        upper = max(lu1 + ld2, ld1 + lu2)
        same_lower = min(abs(lu1-lu2), abs(ld1-ld2))
        same_upper = max(lu1+lu2, ld1+ld2)
        j_min, j_max = abs(abs(kappa_a)-abs(kappa_b)), abs(kappa_a)+abs(kappa_b)-1
        cv = self._empty_components(); ct = self._empty_components(); tt = self._empty_components()
        drt = torch.zeros(4, dtype=self.propagators.r.dtype, device=self.propagators.r.device)
        inverse_range = mass_mev / self.propagators.hbar_c
        for multipole in range(lower, upper + 1, 2):
            for i1 in (-1, 1):
                l1 = multipole + i1
                if not same_lower <= l1 <= same_upper:
                    continue
                coeffs = [0.0, 0.0, 0.0, 0.0]
                for j in range(abs(multipole-1), multipole+2):
                    if not (j_min <= j <= j_max and abs(l1-1) <= j <= l1+1):
                        continue
                    tmp = _f_symbol(multipole, j, l1) * self._core_c3j(kappa_a, kappa_b, j) * math.sqrt(6.0) * inverse_range * (-1.0)**j / (4.0*math.pi)
                    coeffs[0] += _bj_symbol(kappa_a,-kappa_b,j,multipole) * _bj_symbol(kappa_a,kappa_b,j,l1) * tmp
                    coeffs[1] += _bj_symbol(kappa_a,-kappa_b,j,multipole) * _bj_symbol(-kappa_a,-kappa_b,j,l1) * tmp
                    coeffs[2] += _bj_symbol(-kappa_a,kappa_b,j,multipole) * _bj_symbol(kappa_a,kappa_b,j,l1) * tmp
                    coeffs[3] += _bj_symbol(-kappa_a,kappa_b,j,multipole) * _bj_symbol(-kappa_a,-kappa_b,j,l1) * tmp
                wrvt = self.propagators.folded_mixed_yukawa(multipole, l1, mass_mev, signed=True)
                wrtv = self.propagators.folded_mixed_yukawa(l1, multipole, mass_mev, signed=True)
                for index, coefficient in enumerate(coeffs):
                    cv[index//2,index%2] -= coefficient * wrvt
                ct[0,0] += coeffs[0]*wrtv; ct[0,1] += coeffs[2]*wrtv
                ct[1,0] += coeffs[1]*wrtv; ct[1,1] += coeffs[3]*wrtv
            for i1, i2 in self._pairs:
                l1, l2 = multipole+i1, multipole+i2
                if not (same_lower <= l1 <= same_upper and same_lower <= l2 <= same_upper):
                    continue
                id1, id2 = (0 if i1 == -1 else 1), (0 if i2 == -1 else 1)
                coeffs = [0.0, 0.0, 0.0, 0.0]
                for j in range(abs(multipole-1), multipole+2):
                    if not (j_min <= j <= j_max and abs(l1-1)<=j<=l1+1 and abs(l2-1)<=j<=l2+1):
                        continue
                    tmp = _f_symbol(multipole,j,l1) * _f_symbol(multipole,j,l2) * self._core_c3j(kappa_a,kappa_b,j) * 6.0 * inverse_range**2 / (4.0*math.pi)
                    coeffs[0] += _bj_symbol(kappa_a,kappa_b,j,l1)*_bj_symbol(kappa_a,kappa_b,j,l2)*tmp
                    coeffs[2] += _bj_symbol(kappa_a,kappa_b,j,l1)*_bj_symbol(-kappa_a,-kappa_b,j,l2)*tmp
                    coeffs[1] += _bj_symbol(-kappa_a,-kappa_b,j,l1)*_bj_symbol(kappa_a,kappa_b,j,l2)*tmp
                    coeffs[3] += _bj_symbol(-kappa_a,-kappa_b,j,l1)*_bj_symbol(-kappa_a,-kappa_b,j,l2)*tmp
                radial = self.propagators.folded_mixed_yukawa(l1,l2,mass_mev)
                tt[0,0] -= coeffs[0]*radial; tt[0,1] -= coeffs[2]*radial
                tt[1,0] -= coeffs[1]*radial; tt[1,1] -= coeffs[3]*radial
                drt += torch.as_tensor(coeffs, dtype=drt.dtype, device=drt.device)
        radius = self.propagators.r
        diagonal = torch.arange(radius.numel(), device=radius.device)
        contact = 1.0 / (radius.square()*4.0*math.pi)
        compensation = drt[:,None] / (radius.square()[None,:]*inverse_range**2)
        for index in range(4):
            component = tt[index//2,index%2]
            component[diagonal,diagonal] += -contact + compensation[index]
        packed = torch.stack([cv,ct,tt])
        self._cache[key] = packed
        return cv,ct,tt

    def vector_space(
        self,
        channel: str,
        kappa_a: int,
        kappa_b: int,
        mu: int,
        mu_prime: int,
        mass_mev: float,
    ) -> torch.Tensor:
        key = (channel + "-space", kappa_a, kappa_b, mu, mu_prime)
        if key in self._cache:
            return self._cache[key]
        la_u, la_l, _ = kappa_quantum_numbers(kappa_a)
        lb_u, lb_l, _ = kappa_quantum_numbers(kappa_b)
        result = torch.zeros_like(self.propagators.folded_coulomb(0))
        for multipole in range(self.maximum_space_multipole(kappa_a, kappa_b) + 1):
            c3j = spin_cg_squared(kappa_a, kappa_b, multipole)
            if mu == 0 and mu_prime == 0:
                parity_l = la_u + lb_l
                coefficient = 2.0 * orbital_cg_squared(la_u, lb_l, multipole) - c3j
            elif mu == 1 and mu_prime == 1:
                parity_l = la_l + lb_u
                coefficient = 2.0 * orbital_cg_squared(la_l, lb_u, multipole) - c3j
            else:
                parity_l = la_u + lb_l
                coefficient = c3j
            if (multipole + parity_l) % 2:
                continue
            radial = (
                self.propagators.folded_coulomb(multipole)
                if channel == "coulomb"
                else self.propagators.folded_yukawa(multipole, mass_mev)
            )
            result = result + coefficient * radial / (4.0 * math.pi)
        self._cache[key] = result
        return result
