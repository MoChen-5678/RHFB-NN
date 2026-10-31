from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from dpl_rhf.rhf.density import NonlocalDensityBlock, local_source_densities
from dpl_rhf.rhf.direct import RHFDirectFunctional
from dpl_rhf.rhf.fock_batched import BatchedFockChannels
from dpl_rhf.rhf.kernels import PKA1AngularKernels, PKO2AngularKernels, SphericalPropagators
from dpl_rhf.rhf.parameters import RHFParameters
from dpl_rhf.rhf.quadrature import simpson_weights


PKA1_DIRECT_CHANNEL_TERMS = {
    "sigma_direct": ("E_sigma",),
    "omega_direct": ("E_omega",),
    "rho_vector_direct": ("E_rho",),
    "rho_tensor_direct": ("E_rho_tensor_direct",),
    "rho_vector_tensor_direct": ("E_rho_vt_direct",),
    "coulomb_direct": ("E_coul",),
}

PKA1_FOCK_CHANNEL_TERMS = {
    "sigma_fock": ("E_sigma_fock",),
    "omega_fock": ("E_omega_fock_time", "E_omega_fock_space"),
    "rho_vector_fock": ("E_rho_fock_time", "E_rho_fock_space"),
    "rho_tensor_fock": (
        "E_rho_tensor_fock_time",
        "E_rho_tensor_fock_space",
    ),
    "rho_vector_tensor_fock": (
        "E_rho_vector_tensor_fock",
        "E_rho_tensor_vector_fock",
    ),
    "pion_pseudovector_fock": ("E_pion_fock",),
    "coulomb_fock": ("E_coul_fock_time", "E_coul_fock_space"),
}


@dataclass
class RHFEnergyEvaluation:
    total: torch.Tensor
    kinetic: torch.Tensor
    direct_terms: dict[str, torch.Tensor]
    exchange_terms: dict[str, torch.Tensor]
    densities: dict[str, torch.Tensor]
    fields: dict[str, torch.Tensor]
    observables: dict[str, torch.Tensor]
    coupling_density: torch.Tensor


class SphericalRHFFunctional:
    """Spherical PKO2/PKA1 density-dependent RHF energy functional.

    The implementation follows RHF-e Eqs. (3.94), (3.324)-(3.337).  Density
    dependence remains inside the graph, so differentiating ``total`` gives
    both the ordinary Fock operator and its rearrangement contribution.
    """

    def __init__(
        self,
        r: torch.Tensor,
        z: int,
        n: int,
        params: RHFParameters,
        *,
        recompute_fock: bool = False,
        batched_fock: bool = True,
    ):
        self.r = r
        self.z = int(z)
        self.n = int(n)
        self.a = self.z + self.n
        self.params = params
        self.recompute_fock = bool(recompute_fock)
        # Batched pair contractions give the same channel formulas as the
        # reference loops below; the loops are kept for verification.
        self.batched_fock = bool(batched_fock)
        self._batched_channels: BatchedFockChannels | None = None
        self.propagators = SphericalPropagators(r, params.hbar_c)
        self.direct = RHFDirectFunctional(
            r, z, n, params=params, propagators=self.propagators
        )
        self.h = self.direct.h
        self.angular = (
            PKA1AngularKernels(self.propagators)
            if params.model == "PKA1"
            else PKO2AngularKernels(self.propagators)
        )
        self.exchange_outer_weights = simpson_weights(
            r.numel(), self.h, device=r.device, dtype=r.dtype
        )[1:-1]

    def configure_fock_streaming(self, devices) -> None:
        """Stream Fock exchange per kappa pair over ``devices`` (None restores default)."""

        if devices is None:
            self.fock_streaming = None
            return
        from dpl_rhf.rhf.fock_streaming import StreamedFockChannels

        self.fock_streaming = StreamedFockChannels(self, devices)

    def _fock_call(self, function, *args):
        """Evaluate one exact Fock channel with optional activation recomputation."""

        if not self.recompute_fock or not torch.is_grad_enabled():
            return function(*args)
        return checkpoint(
            function,
            *args,
            use_reentrant=False,
            preserve_rng_state=False,
            determinism_check="default",
        )

    def _batched(self, blocks: list[NonlocalDensityBlock]) -> BatchedFockChannels:
        signature = tuple((block.suffix, block.kappa) for block in blocks)
        if (
            self._batched_channels is None
            or self._batched_channels.signature != signature
        ):
            self._batched_channels = BatchedFockChannels(self, blocks)
        return self._batched_channels

    def couplings(self, rho_b: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.direct.couplings(rho_b)

    def kinetic_energy(
        self,
        blocks: list[dict[str, torch.Tensor]],
        nonlocal_blocks: list[NonlocalDensityBlock],
    ) -> torch.Tensor:
        total = torch.zeros((), dtype=self.r.dtype, device=self.r.device)
        for block, density in zip(blocks, nonlocal_blocks):
            total = total + (
                density.degeneracy
                * torch.sum(block["free_hamiltonian"] * density.projector.T)
                * self.params.hbar_c
            )
        return total

    @staticmethod
    def _isospin_factor(
        channel: str, a: NonlocalDensityBlock, b: NonlocalDensityBlock
    ) -> float:
        same_species = a.suffix == b.suffix
        if channel in {"sigma", "omega"}:
            return 1.0 if same_species else 0.0
        if channel == "coulomb":
            return 1.0 if a.suffix == "p" and b.suffix == "p" else 0.0
        if channel == "rho":
            return 1.0 if same_species else 2.0
        raise ValueError(f"unknown PKO2 exchange channel {channel}")

    def _scalar_exchange(
        self,
        blocks: list[NonlocalDensityBlock],
        coupling: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        profile = torch.zeros_like(self.exchange_outer_weights)
        vertex = coupling[1:-1]
        vertices = vertex[:, None] * vertex[None, :]
        signs = (1.0, -1.0)
        for a in blocks:
            for b in blocks:
                factor = self._isospin_factor("sigma", a, b)
                if factor == 0.0:
                    continue
                kernel = self.angular.scalar_time(
                    "sigma", a.kappa, b.kappa, self.params.m_sigma
                )[1:-1, 1:-1]
                for mu in range(2):
                    for mu_prime in range(2):
                        profile = profile + torch.sum(
                            factor
                            * signs[mu]
                            * signs[mu_prime]
                            * a.matrix[mu, mu_prime]
                            * vertices
                            * kernel
                            * b.matrix[mu, mu_prime],
                            dim=1,
                        )
        energy = 0.5 * self.params.hbar_c * torch.sum(
            self.exchange_outer_weights * profile
        )
        return energy, profile

    def _vector_exchange(
        self,
        channel: str,
        blocks: list[NonlocalDensityBlock],
        coupling: torch.Tensor,
        mass_mev: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        time_profile = torch.zeros_like(self.exchange_outer_weights)
        space_profile = torch.zeros_like(time_profile)
        vertex = coupling[1:-1]
        vertices = vertex[:, None] * vertex[None, :]
        for a in blocks:
            for b in blocks:
                factor = self._isospin_factor(channel, a, b)
                if factor == 0.0:
                    continue
                time_kernel = self.angular.scalar_time(
                    channel, a.kappa, b.kappa, mass_mev
                )[1:-1, 1:-1]
                for mu in range(2):
                    for mu_prime in range(2):
                        space_kernel = self.angular.vector_space(
                            channel,
                            a.kappa,
                            b.kappa,
                            mu,
                            mu_prime,
                            mass_mev,
                        )[1:-1, 1:-1]
                        partner = b.matrix[1 - mu, 1 - mu_prime]
                        common = a.matrix[mu, mu_prime] * vertices
                        time_profile = time_profile - factor * torch.sum(
                            common * time_kernel * b.matrix[mu, mu_prime], dim=1
                        )
                        space_profile = space_profile + factor * torch.sum(
                            common * space_kernel * partner, dim=1
                        )
        scale = 0.5 * self.params.hbar_c
        time_energy = scale * torch.sum(self.exchange_outer_weights * time_profile)
        space_energy = scale * torch.sum(self.exchange_outer_weights * space_profile)
        return time_energy, space_energy, time_profile, space_profile

    def _pion_exchange(
        self, blocks: list[NonlocalDensityBlock], coupling: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        profile = torch.zeros_like(self.exchange_outer_weights)
        vertices = coupling[1:-1, None] * coupling[None, 1:-1]
        angular = self.angular
        if not isinstance(angular, PKA1AngularKernels):
            return torch.zeros((), dtype=self.r.dtype, device=self.r.device), profile
        for a in blocks:
            for b in blocks:
                factor = self._isospin_factor("rho", a, b)
                kernel = angular.pion(a.kappa, b.kappa, self.params.m_pi)[:, :, 1:-1, 1:-1]
                for mu in range(2):
                    for nu in range(2):
                        profile = profile + factor * torch.sum(
                            a.matrix[mu, nu]
                            * vertices
                            * kernel[mu, nu]
                            * b.matrix[mu, nu],
                            dim=1,
                        )
        energy = 0.5 * self.params.hbar_c * torch.sum(
            self.exchange_outer_weights * profile
        )
        return energy, profile

    def _rho_tensor_exchange(
        self,
        blocks: list[NonlocalDensityBlock],
        vector_coupling: torch.Tensor,
        tensor_coupling: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        angular = self.angular
        zero = torch.zeros((), dtype=self.r.dtype, device=self.r.device)
        zero_profile = torch.zeros_like(self.exchange_outer_weights)
        if not isinstance(angular, PKA1AngularKernels):
            return zero, zero, zero, zero, zero_profile, zero_profile, zero_profile, zero_profile
        tensor_time_profile = torch.zeros_like(zero_profile)
        tensor_space_profile = torch.zeros_like(zero_profile)
        vector_tensor_profile = torch.zeros_like(zero_profile)
        tensor_vector_profile = torch.zeros_like(zero_profile)
        tt = tensor_coupling[1:-1, None] * tensor_coupling[None, 1:-1]
        vt = vector_coupling[1:-1, None] * tensor_coupling[None, 1:-1]
        tv = tensor_coupling[1:-1, None] * vector_coupling[None, 1:-1]
        for a in blocks:
            for b in blocks:
                factor = self._isospin_factor("rho", a, b)
                cv_t, ct_t, rt_t = angular.rho_tensor_time(
                    a.kappa, b.kappa, self.params.m_rho
                )
                cv_s, ct_s, rt_s = angular.rho_tensor_space(
                    a.kappa, b.kappa, self.params.m_rho
                )
                cv_t = cv_t[:, :, 1:-1, 1:-1]; ct_t = ct_t[:, :, 1:-1, 1:-1]
                rt_t = rt_t[:, :, 1:-1, 1:-1]
                cv_s = cv_s[:, :, 1:-1, 1:-1]; ct_s = ct_s[:, :, 1:-1, 1:-1]
                rt_s = rt_s[:, :, 1:-1, 1:-1]
                for mu in range(2):
                    for nu in range(2):
                        common = factor * a.matrix[mu, nu]
                        tensor_time_profile = tensor_time_profile + torch.sum(
                            common * tt * rt_t[mu, nu] * b.matrix[1-mu, 1-nu],
                            dim=1,
                        )
                        tensor_space_profile = tensor_space_profile + torch.sum(
                            common * tt * rt_s[mu, nu] * b.matrix[mu, nu],
                            dim=1,
                        )
                        vector_tensor_profile = vector_tensor_profile + torch.sum(
                            common * vt * (
                                cv_t[mu, nu] * b.matrix[mu, 1-nu]
                                + cv_s[mu, nu] * b.matrix[1-mu, nu]
                            ),
                            dim=1,
                        )
                        tensor_vector_profile = tensor_vector_profile + torch.sum(
                            common * tv * (
                                ct_t[mu, nu] * b.matrix[1-mu, nu]
                                + ct_s[mu, nu] * b.matrix[mu, 1-nu]
                            ),
                            dim=1,
                        )
        scale = 0.5 * self.params.hbar_c
        profiles = (
            tensor_time_profile,
            tensor_space_profile,
            vector_tensor_profile,
            tensor_vector_profile,
        )
        energies = tuple(
            scale * torch.sum(self.exchange_outer_weights * profile)
            for profile in profiles
        )
        return (*energies, *profiles)

    def _fock_rearrangement(
        self,
        profiles: dict[str, torch.Tensor],
        couplings: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Core Meanfield:Rearrange exchange contraction, RHF-e Sec. 1.3."""

        interior = slice(1, -1)

        def contraction(profile: torch.Tensor, vertex_name: str) -> torch.Tensor:
            vertex = couplings[vertex_name][interior]
            return torch.where(
                vertex.abs() > 1.0e-30,
                profile / vertex,
                torch.zeros_like(profile),
            )

        numerator = (
            couplings["dgsig"][interior] * contraction(profiles["sigma"], "gsig")
            + couplings["dgome"][interior] * contraction(profiles["omega"], "gome")
            + couplings["dgrho"][interior] * contraction(profiles["rho"], "grho")
        )
        if self.params.model == "PKA1":
            numerator = numerator + (
                couplings["dfpio"][interior]
                * contraction(profiles["pion"], "fpio")
                + couplings["dgrtn"][interior]
                * contraction(profiles["rho_tensor"], "grtn")
                + couplings["dgrho"][interior]
                * contraction(profiles["rho_vector_tensor"], "grho")
                + couplings["dgrtn"][interior]
                * contraction(profiles["rho_tensor_vector"], "grtn")
            )
        result = torch.zeros_like(self.r)
        result[interior] = numerator / (
            4.0 * torch.pi * self.r[interior].square() * self.params.rho_sat
        )
        result[0] = 3.0 * (result[1] - result[2]) + result[3]
        return result

    def evaluate(
        self,
        blocks: list[dict[str, torch.Tensor]],
        nonlocal_blocks: list[NonlocalDensityBlock],
    ) -> RHFEnergyEvaluation:
        densities = local_source_densities(
            nonlocal_blocks,
            self.r,
            include_tensor=self.params.model == "PKA1",
            mass_n_mev=self.params.mass_n,
            mass_p_mev=self.params.mass_p,
            hbar_c_mev_fm=self.params.hbar_c,
        )
        rho_b = densities["rho_v_n"] + densities["rho_v_p"]
        couplings = self.couplings(rho_b)
        fields = self.direct.fields(densities, couplings)
        direct_terms = self.direct.energy_terms(densities, fields, couplings)
        kinetic = self.kinetic_energy(blocks, nonlocal_blocks)
        # The photon vertex in Eq. (1.25e) is e^2=4*pi*alpha.  The propagator
        # itself contains 1/(4*pi); our multipole kernel follows Eq. (3.15),
        # so each endpoint must carry sqrt(4*pi*alpha).
        electromagnetic_vertex = torch.full_like(
            rho_b, (4.0 * torch.pi / self.params.alpha_inv) ** 0.5
        )
        streamed = getattr(self, "fock_streaming", None)
        fock_source = nonlocal_blocks
        if self.batched_fock and streamed is None:
            batched = self._batched(nonlocal_blocks)
            fock_source = batched.scatter(nonlocal_blocks)
            call = self._fock_call
            vector_exchange = batched.vector_exchange
            scalar_exchange = batched.scalar_exchange
            pion_exchange = batched.pion_exchange
            rho_tensor_exchange = batched.rho_tensor_exchange
        elif streamed is not None:
            # Per-pair checkpoints replace the channel-level recomputation.
            call = lambda function, *args: function(*args)
            vector_exchange = streamed.vector_exchange
            scalar_exchange = streamed.scalar_exchange
            pion_exchange = streamed.pion_exchange
            rho_tensor_exchange = streamed.rho_tensor_exchange
        else:
            call = self._fock_call
            vector_exchange = self._vector_exchange
            scalar_exchange = self._scalar_exchange
            pion_exchange = self._pion_exchange
            rho_tensor_exchange = self._rho_tensor_exchange
        omega_time, omega_space, omega_time_profile, omega_space_profile = call(
            vector_exchange,
            "omega", fock_source, couplings["gome"], self.params.m_omega
        )
        rho_time, rho_space, rho_time_profile, rho_space_profile = call(
            vector_exchange,
            "rho", fock_source, couplings["grho"], self.params.m_rho
        )
        coul_time, coul_space, _, _ = call(
            vector_exchange,
            "coulomb", fock_source, electromagnetic_vertex, 0.0
        )
        sigma_fock, sigma_profile = call(
            scalar_exchange, fock_source, couplings["gsig"]
        )
        profiles = {
            "sigma": sigma_profile,
            "omega": omega_time_profile + omega_space_profile,
            "rho": rho_time_profile + rho_space_profile,
        }
        exchange_terms = {
            "E_sigma_fock": sigma_fock,
            "E_omega_fock_time": omega_time,
            "E_omega_fock_space": omega_space,
            "E_omega_fock": omega_time + omega_space,
            "E_rho_fock_time": rho_time,
            "E_rho_fock_space": rho_space,
            "E_rho_fock": rho_time + rho_space,
            "E_coul_fock_time": coul_time,
            "E_coul_fock_space": coul_space,
            "E_coul_fock": coul_time + coul_space,
        }
        if self.params.model == "PKA1":
            pion, pion_profile = call(
                pion_exchange, fock_source, couplings["fpio"]
            )
            (
                rt_time,
                rt_space,
                rvt,
                rtv,
                rt_time_profile,
                rt_space_profile,
                rvt_profile,
                rtv_profile,
            ) = call(
                rho_tensor_exchange,
                fock_source, couplings["grho"], couplings["grtn"]
            )
            profiles.update(
                {
                    "pion": pion_profile,
                    "rho_tensor": rt_time_profile + rt_space_profile,
                    "rho_vector_tensor": rvt_profile,
                    "rho_tensor_vector": rtv_profile,
                }
            )
            exchange_terms.update(
                {
                    "E_pion_fock": pion,
                    "E_rho_tensor_fock_time": rt_time,
                    "E_rho_tensor_fock_space": rt_space,
                    "E_rho_tensor_fock": rt_time + rt_space,
                    "E_rho_vector_tensor_fock": rvt,
                    "E_rho_tensor_vector_fock": rtv,
                    "E_rho_vt_fock": rvt + rtv,
                }
            )
        exchange_terms["E_fock"] = (
            exchange_terms["E_sigma_fock"]
            + exchange_terms["E_omega_fock"]
            + exchange_terms["E_rho_fock"]
            + exchange_terms["E_coul_fock"]
            + exchange_terms.get("E_pion_fock", torch.zeros_like(kinetic))
            + exchange_terms.get("E_rho_tensor_fock", torch.zeros_like(kinetic))
            + exchange_terms.get("E_rho_vt_fock", torch.zeros_like(kinetic))
        )
        if self.params.model == "PKA1":
            self._validate_pka1_channel_assembly(direct_terms, exchange_terms)
        total = kinetic + direct_terms["E_direct"] + exchange_terms["E_fock"]
        potentials = self.direct.local_self_energies(densities, fields, couplings)
        fock_rearrangement = self._fock_rearrangement(profiles, couplings)
        output_fields = {
            **fields,
            **potentials,
            "sigma_rearr_fock_explicit": fock_rearrangement,
        }
        return RHFEnergyEvaluation(
            total=total,
            kinetic=kinetic,
            direct_terms=direct_terms,
            exchange_terms=exchange_terms,
            densities=densities,
            fields=output_fields,
            observables=self.direct.observables(densities),
            coupling_density=rho_b,
        )

    @staticmethod
    def _validate_pka1_channel_assembly(
        direct_terms: dict[str, torch.Tensor],
        exchange_terms: dict[str, torch.Tensor],
    ) -> None:
        """Fail before variation if any textbook PKA1 channel is absent."""

        required_direct = {
            term
            for terms in PKA1_DIRECT_CHANNEL_TERMS.values()
            for term in terms
        }
        required_fock = {
            term
            for terms in PKA1_FOCK_CHANNEL_TERMS.values()
            for term in terms
        }
        missing_direct = sorted(required_direct.difference(direct_terms))
        missing_fock = sorted(required_fock.difference(exchange_terms))
        if missing_direct or missing_fock:
            raise RuntimeError(
                "incomplete PKA1 functional: "
                f"missing direct={missing_direct}, missing Fock={missing_fock}"
            )

        direct_sum = sum(
            (direct_terms[name] for name in required_direct),
            torch.zeros_like(direct_terms["E_direct"]),
        )
        fock_sum = sum(
            (exchange_terms[name] for name in required_fock),
            torch.zeros_like(exchange_terms["E_fock"]),
        )
        if not torch.allclose(direct_sum, direct_terms["E_direct"], rtol=1.0e-12, atol=1.0e-10):
            raise RuntimeError("PKA1 direct-channel sum does not equal E_direct")
        if not torch.allclose(fock_sum, exchange_terms["E_fock"], rtol=1.0e-12, atol=1.0e-10):
            raise RuntimeError("PKA1 Fock-channel sum does not equal E_fock")


# Compatibility import for checkpoints and external scripts created before the
# PKA1 functional was separated from the PKO2-only implementation name.
PKO2RHFFunctional = SphericalRHFFunctional
