"""Batched evaluation of the spherical RHF Fock channels.

The channel formulas are exactly those of ``SphericalRHFFunctional``'s
reference loops (RHF-e Eqs. (3.324)-(3.337), Core PrePotels): for every
source pair (a, b) and Dirac components,

    profile(r) = sum_r' V(r,r') sum_ab F_ab A_a(r,r') K_ab(r,r') B_b(r,r').

The angular kernel K_ab depends only on (kappa_a, kappa_b) and the isospin
factor F_ab only on the two species.  Densities are therefore scattered onto
a (species, unique kappa) grid and each channel becomes a handful of batched
contractions instead of O(B^2) small elementwise products.  This removes the
per-pair autograd tape that dominated time and memory for heavy nuclei.
"""

from __future__ import annotations

import torch

from dpl_rhf.rhf.density import NonlocalDensityBlock

# Dirac component index c = 2*mu + nu of R^{mu nu}(r,r').  Partner
# components on the source side of each Fock term besides (mu, nu):
_FLIP_BOTH = [3, 2, 1, 0]     # (1-mu, 1-nu)
_FLIP_NU = [1, 0, 3, 2]       # (mu, 1-nu)
_FLIP_MU = [2, 3, 0, 1]       # (1-mu, nu)

# Isospin factors F[s_a, s_b] with species index 0=n, 1=p, matching
# SphericalRHFFunctional._isospin_factor.
_ISOSPIN = {
    "sigma": ((1.0, 0.0), (0.0, 1.0)),
    "omega": ((1.0, 0.0), (0.0, 1.0)),
    "rho": ((1.0, 2.0), (2.0, 1.0)),
    "coulomb": ((0.0, 0.0), (0.0, 1.0)),
}


def _interior(kernel: torch.Tensor) -> torch.Tensor:
    return kernel[..., 1:-1, 1:-1]


class BatchedFockChannels:
    def __init__(self, functional, blocks: list[NonlocalDensityBlock]):
        self.functional = functional
        self.params = functional.params
        self.kappas = tuple(sorted({block.kappa for block in blocks}))
        self.index = {kappa: i for i, kappa in enumerate(self.kappas)}
        self.signature = tuple((block.suffix, block.kappa) for block in blocks)
        reference = blocks[0].matrix
        self.dtype = reference.dtype
        self.device = reference.device
        self.isospin = {
            name: torch.tensor(value, dtype=self.dtype, device=self.device)
            for name, value in _ISOSPIN.items()
        }
        self._kernels: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Kernel stacks (constant, built once per block structure).
    # ------------------------------------------------------------------
    def _stack(self, name: str, build) -> torch.Tensor:
        if name not in self._kernels:
            rows = [
                torch.stack([build(ka, kb) for kb in self.kappas])
                for ka in self.kappas
            ]
            self._kernels[name] = torch.stack(rows).contiguous()
        return self._kernels[name]

    def scalar_time(self, channel: str, mass: float) -> torch.Tensor:
        angular = self.functional.angular
        return self._stack(
            f"{channel}-time",
            lambda ka, kb: _interior(angular.scalar_time(channel, ka, kb, mass)),
        )

    def vector_space(self, channel: str, mass: float) -> torch.Tensor:
        # Shape (U, U, 4, N, N), component c = 2*mu + nu.
        angular = self.functional.angular
        return self._stack(
            f"{channel}-space",
            lambda ka, kb: torch.stack(
                [
                    _interior(angular.vector_space(channel, ka, kb, c // 2, c % 2, mass))
                    for c in range(4)
                ]
            ),
        )

    def pion(self) -> torch.Tensor:
        angular = self.functional.angular
        return self._stack(
            "pion",
            lambda ka, kb: _interior(angular.pion(ka, kb, self.params.m_pi)).flatten(0, 1),
        )

    def rho_tensor(self, part: str) -> torch.Tensor:
        # Shape (U, U, 3, 4, N, N) for (cv, ct, tt).
        angular = self.functional.angular
        method = angular.rho_tensor_time if part == "time" else angular.rho_tensor_space
        return self._stack(
            f"rho-tensor-{part}",
            lambda ka, kb: _interior(torch.stack(method(ka, kb, self.params.m_rho))).flatten(1, 2),
        )

    # ------------------------------------------------------------------
    # Densities.
    # ------------------------------------------------------------------
    def scatter(self, blocks: list[NonlocalDensityBlock]) -> torch.Tensor:
        """Return D[s, u, c, r, r'] (c = 2*mu + nu), zeros for absent blocks."""

        size = blocks[0].matrix.shape[-1]
        empty = torch.zeros((4, size, size), dtype=self.dtype, device=self.device)
        lookup = {
            (block.suffix, block.kappa): block.matrix.flatten(0, 1) for block in blocks
        }
        return torch.stack(
            [
                torch.stack(
                    [lookup.get((suffix, kappa), empty) for kappa in self.kappas]
                )
                for suffix in ("n", "p")
            ]
        )

    @staticmethod
    def _pair_sum(
        left: torch.Tensor, kernel: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        """sum_{s,u,v,c} left[s,u,c] kernel[u,v,c] right[s,v,c], pointwise in (r,r').

        The contraction is diagonal in (r,r'), so it is memory-bandwidth bound
        rather than a GEMM; elementwise products over all four Dirac
        components at once keep the number of kernels and saved tensors small.
        """

        pairs = left[0, :, None] * right[0, None, :]
        pairs = pairs + left[1, :, None] * right[1, None, :]
        return torch.sum(kernel * pairs, dim=(0, 1, 2))

    def _isospin_right(self, channel: str, density: torch.Tensor) -> torch.Tensor:
        # right[s_a] = sum_{s_b} F[s_a, s_b] D[s_b]
        return torch.einsum("st,t...->s...", self.isospin[channel], density)

    @staticmethod
    def _profile(sum_xy: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
        return torch.sum(vertices * sum_xy, dim=1)

    # ------------------------------------------------------------------
    # Channels; signatures mirror the reference loop implementations.
    # ------------------------------------------------------------------
    def scalar_exchange(self, density: torch.Tensor, coupling: torch.Tensor):
        vertex = coupling[1:-1]
        vertices = vertex[:, None] * vertex[None, :]
        kernel = self.scalar_time("sigma", self.params.m_sigma)[:, :, None]
        right = self._isospin_right("sigma", density)
        # sign(mu) * sign(nu) for G=+, F=-.
        signs = density.new_tensor([1.0, -1.0, -1.0, 1.0])
        signed = density * signs[None, None, :, None, None]
        profile = self._profile(self._pair_sum(signed, kernel, right), vertices)
        weights = self.functional.exchange_outer_weights
        energy = 0.5 * self.params.hbar_c * torch.sum(weights * profile)
        return energy, profile

    def vector_exchange(
        self, channel: str, density: torch.Tensor, coupling: torch.Tensor, mass: float
    ):
        vertex = coupling[1:-1]
        vertices = vertex[:, None] * vertex[None, :]
        time_kernel = self.scalar_time(channel, mass)[:, :, None]
        space_kernel = self.vector_space(channel, mass)
        right = self._isospin_right(channel, density)
        time_sum = -self._pair_sum(density, time_kernel, right)
        space_sum = self._pair_sum(density, space_kernel, right[:, :, _FLIP_BOTH])
        time_profile = self._profile(time_sum, vertices)
        space_profile = self._profile(space_sum, vertices)
        weights = self.functional.exchange_outer_weights
        scale = 0.5 * self.params.hbar_c
        return (
            scale * torch.sum(weights * time_profile),
            scale * torch.sum(weights * space_profile),
            time_profile,
            space_profile,
        )

    def pion_exchange(self, density: torch.Tensor, coupling: torch.Tensor):
        vertices = coupling[1:-1, None] * coupling[None, 1:-1]
        right = self._isospin_right("rho", density)
        profile = self._profile(self._pair_sum(density, self.pion(), right), vertices)
        weights = self.functional.exchange_outer_weights
        energy = 0.5 * self.params.hbar_c * torch.sum(weights * profile)
        return energy, profile

    def rho_tensor_exchange(
        self,
        density: torch.Tensor,
        vector_coupling: torch.Tensor,
        tensor_coupling: torch.Tensor,
    ):
        tt = tensor_coupling[1:-1, None] * tensor_coupling[None, 1:-1]
        vt = vector_coupling[1:-1, None] * tensor_coupling[None, 1:-1]
        tv = tensor_coupling[1:-1, None] * vector_coupling[None, 1:-1]
        time = self.rho_tensor("time")
        space = self.rho_tensor("space")
        cv_t, ct_t, rt_t = time[:, :, 0], time[:, :, 1], time[:, :, 2]
        cv_s, ct_s, rt_s = space[:, :, 0], space[:, :, 1], space[:, :, 2]
        right = self._isospin_right("rho", density)
        flip_both = right[:, :, _FLIP_BOTH]
        flip_nu = right[:, :, _FLIP_NU]
        flip_mu = right[:, :, _FLIP_MU]
        tensor_time = self._pair_sum(density, rt_t, flip_both)
        tensor_space = self._pair_sum(density, rt_s, right)
        vector_tensor = self._pair_sum(density, cv_t, flip_nu) + self._pair_sum(
            density, cv_s, flip_mu
        )
        tensor_vector = self._pair_sum(density, ct_t, flip_mu) + self._pair_sum(
            density, ct_s, flip_nu
        )
        profiles = (
            self._profile(tensor_time, tt),
            self._profile(tensor_space, tt),
            self._profile(vector_tensor, vt),
            self._profile(tensor_vector, tv),
        )
        weights = self.functional.exchange_outer_weights
        scale = 0.5 * self.params.hbar_c
        energies = tuple(scale * torch.sum(weights * profile) for profile in profiles)
        return (*energies, *profiles)
