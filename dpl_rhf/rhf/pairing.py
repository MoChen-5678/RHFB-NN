"""Spherical J=0 Gogny-D1S pairing for the relativistic Hartree-Fock-Bogoliubov solver.

The reduction follows the reference RHFB program (S-RHFB-C ``Gogny.f90:gofac``,
``KappaG``, ``DeltaG`` and ``Expect.f90`` pairing energy):

* only J=0 time-reversed pairs enter (j1=j2, j3=j4);
* all four Dirac components (GG, GF, FG, FF) of the pairing tensor enter, with
  gamma-zero vertices at both ends;
* the radial multipole of exp(-|r1-r2|^2/mu^2) is
  V_L(r1,r2) = exp(-x1^2-x2^2) sqrt(pi/(x1 x2)) I_{L+1/2}(2 x1 x2), x=r/mu;
* E_pair = 1/2 sum_{kappa1,kappa2} sum_XY int dr dr' K^XY(kappa1,kappa2;r,r')
  kappa1^XY(r,r') kappa2^XY(r,r'), with the angular coefficients

      gf = (2L+1) c3j (GA+GD)/2,  gg = gf - (2L+1) c3l GD/2,
      ff = gf - (2L+1) c3s GD/2,  GA = W-H,  GD = B-M  (MeV),

  where c3j = [(la lb L;000){L la lb;1/2 jb ja}]^2 (2la+1)(2lb+1),
  c3l = (la lb L;000)^2 and c3s = (la' lb' L;000)^2 (``Self.f90`` Media).

Phase convention.  The reference program's J=0 pair states differ from the
time-reversed pairs |m, T m> of the physical time-reversal operator by the
block phase sigma_kappa = (-1)^(j-1/2) (verified against the all-J m-scheme
oracle: diagonal terms agree, cross terms between j-blocks of opposite sigma
change sign).  ``convention="time_reversal"`` (default) multiplies the kernels
by sigma_kappa1 sigma_kappa2, so that u_n v_n refer to time-reversed pairs;
``convention="reference"`` keeps the reference program's sign.

Radial functions use the u=rR convention of the RHF solver, and the pairing
tensor of one kappa block is ``kappa^XY(r,r') = sum_n w_n X_n(r) Y_n(r')`` with
``w_n = (2j+1) u_n v_n`` (``(2j-1)`` for an equal-filling blocked level).
Quadrature uses the solver's Simpson weights at both endpoints; the reference
program instead folds the r' integral on a half mesh, which differs only at the
quadrature level for the 0.7/1.2 fm Gaussian ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from scipy.special import ive

from dpl_rhf.rhf.density import kappa_quantum_numbers
from dpl_rhf.rhf.quadrature import simpson_weights


@dataclass(frozen=True)
class GognyD1SPairingForce:
    """Central finite-range D1S parameters (MeV, fm), Berger-Girod-Gogny 1991."""

    ranges_fm: tuple[float, float] = (0.7, 1.2)
    w_mev: tuple[float, float] = (-1720.30, 103.639)
    b_mev: tuple[float, float] = (1300.00, -163.483)
    h_mev: tuple[float, float] = (-1813.53, 162.812)
    m_mev: tuple[float, float] = (1397.60, -223.934)
    strength: float = 1.0


D1S_PAIRING = GognyD1SPairingForce()


@lru_cache(maxsize=None)
def _three_j_zero(l1: int, l2: int, l3: int) -> float:
    from sympy import N
    from sympy.physics.wigner import wigner_3j

    return float(N(wigner_3j(l1, l2, l3, 0, 0, 0), 30))


@lru_cache(maxsize=None)
def _six_j_half(big_l: int, la: int, lb: int, twice_jb: int, twice_ja: int) -> float:
    from sympy import N, Rational
    from sympy.physics.wigner import wigner_6j

    return float(N(wigner_6j(big_l, la, lb, Rational(1, 2),
                             Rational(twice_jb, 2), Rational(twice_ja, 2)), 30))


@lru_cache(maxsize=None)
def angular_coefficients(kappa_a: int, kappa_b: int, big_l: int) -> tuple[float, float, float]:
    """Return ``(c3j, c3l, c3s)`` of the reference program for one multipole."""

    la, lap, ja = kappa_quantum_numbers(kappa_a)
    lb, lbp, jb = kappa_quantum_numbers(kappa_b)
    twice_ja, twice_jb = int(round(2 * ja)), int(round(2 * jb))
    if (la + lb + big_l) % 2 == 0:
        x = _three_j_zero(la, lb, big_l) * _six_j_half(big_l, la, lb, twice_jb, twice_ja)
        return (x * x * (2 * la + 1) * (2 * lb + 1),
                _three_j_zero(la, lb, big_l) ** 2,
                _three_j_zero(lap, lbp, big_l) ** 2)
    z = _three_j_zero(lap, lb, big_l) * _six_j_half(big_l, lap, lb, twice_jb, twice_ja)
    return (z * z * (2 * lap + 1) * (2 * lb + 1),
            _three_j_zero(lap, lb, big_l) ** 2,
            _three_j_zero(la, lbp, big_l) ** 2)


def gaussian_radial_multipole(r: np.ndarray, range_fm: float, big_l: int) -> np.ndarray:
    """``exp(-x1^2-x2^2) sqrt(pi/(x1 x2)) I_{L+1/2}(2 x1 x2)`` on ``r x r``."""

    x = np.asarray(r, dtype=np.float64) / float(range_fm)
    x1, x2 = x[:, None], x[None, :]
    z = 2.0 * x1 * x2
    # exp(-(x1-x2)^2) ive(nu,z) == exp(-x1^2-x2^2) I_nu(z), stable for large z.
    return np.exp(-(x1 - x2) ** 2) * ive(big_l + 0.5, z) * np.sqrt(np.pi / (x1 * x2))


class SphericalD1SPairing:
    """J=0 D1S pairing energy for spherical kappa blocks of one or two species.

    ``kernels[(kappa1, kappa2)]`` has shape ``(3, nr, nr)`` holding the
    weight-premultiplied GG, GF(=FG) and FF kernels on the interior mesh.
    """

    def __init__(self, r: torch.Tensor, kappas, *, force: GognyD1SPairingForce = D1S_PAIRING,
                 radial_weights: torch.Tensor | None = None,
                 convention: str = "time_reversal"):
        if convention not in {"time_reversal", "reference"}:
            raise ValueError("convention must be 'time_reversal' or 'reference'")
        self.convention = convention
        r = torch.as_tensor(r, dtype=torch.float64)
        if r.ndim != 1 or r.numel() < 5:
            raise ValueError("pairing mesh must be a one-dimensional radial grid")
        self.device = r.device
        self.force = force
        self.h = float(r[2] - r[1])
        if radial_weights is None:
            full_weights = simpson_weights(r.numel(), self.h, device=r.device, dtype=r.dtype)
            weights = full_weights[1:-1]
        else:
            # Interior-point weights; used to match another quadrature exactly.
            weights = torch.as_tensor(radial_weights, dtype=torch.float64, device=r.device)
            if weights.shape != (r.numel() - 2,):
                raise ValueError("radial_weights must cover the interior mesh points")
        r_interior = r[1:-1].detach().cpu().numpy()
        self.nr = r_interior.size
        kappas = sorted({int(k) for k in kappas})
        self.kappas = tuple(kappas)
        ga = [force.strength * (w - hh) for w, hh in zip(force.w_mev, force.h_mev)]
        gd = [force.strength * (b - m) for b, m in zip(force.b_mev, force.m_mev)]
        max_l = 0
        for kappa in kappas:
            lu, ld, _ = kappa_quantum_numbers(kappa)
            max_l = max(max_l, 2 * max(lu, ld))
        multipoles = {
            (i, big_l): gaussian_radial_multipole(r_interior, mu, big_l)
            for i, mu in enumerate(force.ranges_fm)
            for big_l in range(max_l + 1)
        }
        outer = (weights[:, None] * weights[None, :]).cpu().numpy()
        self.kernels: dict[tuple[int, int], torch.Tensor] = {}
        self.coefficients: dict[tuple[int, int], list[dict[str, float]]] = {}
        for k1 in kappas:
            lu1, ld1, _ = kappa_quantum_numbers(k1)
            for k2 in kappas:
                lu2, ld2, _ = kappa_quantum_numbers(k2)
                low = min(abs(lu1 - lu2), abs(ld1 - ld2))
                high = max(lu1 + lu2, ld1 + ld2)
                gg = np.zeros((self.nr, self.nr))
                gf = np.zeros_like(gg)
                ff = np.zeros_like(gg)
                records = []
                for big_l in range(low, high + 1, 2):
                    c3j, c3l, c3s = angular_coefficients(k1, k2, big_l)
                    for i in range(len(force.ranges_fm)):
                        coefficient_gf = (2 * big_l + 1) * c3j * (ga[i] + gd[i]) * 0.5
                        coefficient_gg = coefficient_gf - (2 * big_l + 1) * c3l * gd[i] * 0.5
                        coefficient_ff = coefficient_gf - (2 * big_l + 1) * c3s * gd[i] * 0.5
                        kernel = multipoles[(i, big_l)]
                        gg += coefficient_gg * kernel
                        gf += coefficient_gf * kernel
                        ff += coefficient_ff * kernel
                        records.append({"L": big_l, "range": i, "gg": coefficient_gg,
                                        "gf": coefficient_gf, "ff": coefficient_ff})
                self.coefficients[(k1, k2)] = records
                phase = 1.0
                if convention == "time_reversal":
                    phase = (-1.0) ** ((abs(k1) - 1) + (abs(k2) - 1))
                self.kernels[(k1, k2)] = torch.as_tensor(
                    phase * np.stack([gg, gf, ff]) * outer[None], dtype=torch.float64,
                    device=self.device,
                )

    @staticmethod
    def pairing_tensor(factors: torch.Tensor, weights: torch.Tensor, h: float) -> torch.Tensor:
        """Return ``(2,2,nr,nr)`` kappa^XY from Q columns (sqrt(h)-normalized)."""

        split = factors.shape[0] // 2
        matrix = (factors * weights[None, :]) @ factors.T / h
        return torch.stack([
            torch.stack([matrix[:split, :split], matrix[:split, split:]]),
            torch.stack([matrix[split:, :split], matrix[split:, split:]]),
        ])

    def _stacked_kernels(self) -> torch.Tensor:
        """K[u, v, c] over all kappa pairs with components (GG, GF, FG, FF)."""

        stacked = getattr(self, "_kernel_stack", None)
        if stacked is None:
            stacked = torch.stack([
                torch.stack([
                    self.kernels[(k1, k2)][[0, 1, 1, 2]] for k2 in self.kappas
                ])
                for k1 in self.kappas
            ]).contiguous()
            self._kernel_stack = stacked
        return stacked

    def energy(self, tensors: list[tuple[int, torch.Tensor]]) -> torch.Tensor:
        """Pairing energy (MeV) of one species from ``[(kappa, kappa^XY), ...]``.

        Batched form of :meth:`energy_reference`: the J=0 kernel couples equal
        Dirac components, so E = 1/2 sum_{u,v,c} T_u^c K_uv^c T_v^c pointwise.
        """

        if not tensors:
            return torch.zeros((), dtype=torch.float64, device=self.device)
        index = {kappa: i for i, kappa in enumerate(self.kappas)}
        reference = tensors[0][1]
        stacked = reference.new_zeros(
            (len(self.kappas), 4, *reference.shape[-2:])
        )
        for kappa, tensor in tensors:
            stacked = stacked.index_add(
                0,
                torch.tensor([index[kappa]], device=reference.device),
                tensor.reshape(1, 4, *tensor.shape[-2:]),
            )
        pairs = stacked[:, None] * stacked[None, :]
        return 0.5 * torch.sum(self._stacked_kernels() * pairs)

    def energy_reference(self, tensors: list[tuple[int, torch.Tensor]]) -> torch.Tensor:
        """Reference pair loop for :meth:`energy`."""

        if not tensors:
            return torch.zeros((), dtype=torch.float64, device=self.device)
        total = None
        for k1, t1 in tensors:
            # Sum the J=0 field of all blocks first: Delta_1 = -sum_2 K_12 kappa_2.
            gg = gf = fg = ff = None
            for k2, t2 in tensors:
                kernel = self.kernels[(k1, k2)]
                parts = (kernel[0] * t2[0, 0], kernel[1] * t2[0, 1],
                         kernel[1] * t2[1, 0], kernel[2] * t2[1, 1])
                if gg is None:
                    gg, gf, fg, ff = parts
                else:
                    gg, gf, fg, ff = gg + parts[0], gf + parts[1], fg + parts[2], ff + parts[3]
            value = (torch.sum(gg * t1[0, 0]) + torch.sum(gf * t1[0, 1])
                     + torch.sum(fg * t1[1, 0]) + torch.sum(ff * t1[1, 1]))
            total = value if total is None else total + value
        return 0.5 * total
