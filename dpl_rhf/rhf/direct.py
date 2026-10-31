from __future__ import annotations

import torch

from dpl_rhf.rhf.kernels import SphericalPropagators
from dpl_rhf.rhf.parameters import RHFParameters


class RHFDirectFunctional:
    """Local part of the spherical density-dependent RHF functional.

    This is the direct Hamiltonian of RHF-e Eqs. (3.307)-(3.321), including
    the PKA1 rho-tensor and ordered rho-vector-tensor channels.  It is an RHF
    implementation and deliberately has no dependency on the RMF functional.
    """

    def __init__(
        self,
        r: torch.Tensor,
        z: int,
        n: int,
        params: RHFParameters,
        propagators: SphericalPropagators,
    ) -> None:
        self.r = r
        self.z = float(z)
        self.n = float(n)
        self.a = float(z + n)
        self.params = params
        self.propagators = propagators
        self.h = float((r[2] - r[1]).detach().cpu())
        self.r2 = r.square()
        # The ADF radial Hamiltonian uses the uniform physical-grid metric.
        # r[0] is Core's origin placeholder and does not enter contractions.
        self.orbital_w = torch.full_like(r, self.h)
        self.orbital_w[0] = 0.0
        self.measure = 4.0 * torch.pi * self.r2 * self.orbital_w

    def couplings(self, rho_b: torch.Tensor) -> dict[str, torch.Tensor]:
        """All PKA1 vertices and d/dzeta derivatives from Density.f90."""

        p = self.params
        zeta = rho_b / p.rho_sat
        x_sigma = zeta + p.d_sigma
        x_omega = zeta + p.d_omega

        def rational_vertex(
            g0: float, a: float, b: float, c: float, x: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            denominator = 1.0 + c * x.square()
            coupling = g0 * a * (1.0 + b * x.square()) / denominator
            derivative = (
                g0 * a * 2.0 * (b - c) * x / denominator.square()
            )
            return coupling, derivative

        gsig, dgsig = rational_vertex(
            p.g_sigma, p.a_sigma, p.b_sigma, p.c_sigma, x_sigma
        )
        gome, dgome = rational_vertex(
            p.g_omega, p.a_omega, p.b_omega, p.c_omega, x_omega
        )
        grho = p.g_rho * torch.exp(-p.a_rho * zeta)
        fpio = p.f_pi * torch.exp(-p.a_pi * zeta)
        grtn = p.rho_tensor_vertex * torch.exp(-p.a_rho_tensor * zeta)
        return {
            "gsig": gsig,
            "gome": gome,
            "grho": grho,
            "fpio": fpio,
            "grtn": grtn,
            "dgsig": dgsig,
            "dgome": dgome,
            "dgrho": -p.a_rho * grho,
            "dfpio": -p.a_pi * fpio,
            "dgrtn": -p.a_rho_tensor * grtn,
        }

    def fields(
        self,
        densities: dict[str, torch.Tensor],
        couplings: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """RHF-e (3.315)-(3.321), matching Core ``Meanfield:Hartree``."""

        p = self.params
        rho_s = densities["rho_s_n"] + densities["rho_s_p"]
        rho_b = densities["rho_v_n"] + densities["rho_v_p"]
        rho_3 = densities["rho_v_n"] - densities["rho_v_p"]
        sigma = -self.propagators.folded_yukawa(0, p.m_sigma) @ (
            rho_s * couplings["gsig"] * self.r2
        )
        omega = self.propagators.folded_yukawa(0, p.m_omega) @ (
            rho_b * couplings["gome"] * self.r2
        )
        rho = self.propagators.folded_yukawa(0, p.m_rho) @ (
            rho_3 * couplings["grho"] * self.r2
        )
        coul = self.propagators.folded_coulomb(0) @ (
            densities["rho_v_p"]
            * self.r2
            * (4.0 * torch.pi / p.alpha_inv)
        )
        fields = {"sigma": sigma, "omega": omega, "rho": rho, "coul": coul}
        if p.model != "PKA1":
            zero = torch.zeros_like(rho_b)
            fields.update({"rho_tensor_field": zero, "rho_vt_field": zero, "rho_tv_field": zero})
            return fields

        rho_t3 = densities["rho_t_n"] - densities["rho_t_p"]
        source_t = rho_t3 * couplings["grtn"] * self.r2
        source_v = rho_3 * couplings["grho"] * self.r2
        inverse_range = p.m_rho / p.hbar_c
        wrtn = self.propagators.folded_mixed_yukawa(1, 1, p.m_rho)
        wrvt = self.propagators.folded_mixed_yukawa(
            0, 1, p.m_rho, signed=True
        )
        wrtv = self.propagators.folded_mixed_yukawa(
            1, 0, p.m_rho, signed=True
        )
        fields.update(
            {
                "rho_tensor_field":
                    -(wrtn @ source_t) * inverse_range**2
                    + (2.0 / 3.0) * rho_t3 * couplings["grtn"],
                "rho_vt_field": +(wrvt @ source_t) * inverse_range,
                "rho_tv_field": -(wrtv @ source_v) * inverse_range,
            }
        )
        return fields

    def rearrangement_components(
        self,
        densities: dict[str, torch.Tensor],
        fields: dict[str, torch.Tensor],
        couplings: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Direct PKA1 rearrangement channels, Core Meanfield lines 225-231."""

        rho_s = densities["rho_s_n"] + densities["rho_s_p"]
        rho_b = densities["rho_v_n"] + densities["rho_v_p"]
        rho_3 = densities["rho_v_n"] - densities["rho_v_p"]
        standard = (
            rho_s * couplings["dgsig"] * fields["sigma"]
            + rho_b * couplings["dgome"] * fields["omega"]
            + rho_3 * couplings["dgrho"] * fields["rho"]
        )
        tensor = torch.zeros_like(standard)
        if self.params.model == "PKA1":
            rho_t3 = densities["rho_t_n"] - densities["rho_t_p"]
            tensor = (
                rho_t3 * couplings["dgrtn"] * fields["rho_tensor_field"]
                + rho_3 * couplings["dgrho"] * fields["rho_vt_field"]
                + rho_t3 * couplings["dgrtn"] * fields["rho_tv_field"]
            )
        scale = 1.0 / self.params.rho_sat
        return {
            "standard": standard * scale,
            "rho_tensor_and_vt": tensor * scale,
            "total": (standard + tensor) * scale,
        }

    def rearrangement(
        self,
        densities: dict[str, torch.Tensor],
        fields: dict[str, torch.Tensor],
        couplings: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.rearrangement_components(densities, fields, couplings)["total"]

    def energy_terms(
        self,
        densities: dict[str, torch.Tensor],
        fields: dict[str, torch.Tensor],
        couplings: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """RHF-e (3.58)-(3.61), (3.70), and (3.78) direct energies."""

        p = self.params
        rho_s = densities["rho_s_n"] + densities["rho_s_p"]
        rho_b = densities["rho_v_n"] + densities["rho_v_p"]
        rho_3 = densities["rho_v_n"] - densities["rho_v_p"]
        half_hbar_c = 0.5 * p.hbar_c
        terms = {
            "E_sigma": half_hbar_c * torch.sum(
                self.measure * rho_s * couplings["gsig"] * fields["sigma"]
            ),
            "E_omega": half_hbar_c * torch.sum(
                self.measure * rho_b * couplings["gome"] * fields["omega"]
            ),
            "E_rho": half_hbar_c * torch.sum(
                self.measure * rho_3 * couplings["grho"] * fields["rho"]
            ),
            "E_coul": half_hbar_c * torch.sum(
                self.measure * densities["rho_v_p"] * fields["coul"]
            ),
        }
        zero = torch.zeros_like(terms["E_sigma"])
        terms["E_rho_tensor_direct"] = zero
        terms["E_rho_vt_direct"] = zero
        if p.model == "PKA1":
            rho_t3 = densities["rho_t_n"] - densities["rho_t_p"]
            terms["E_rho_tensor_direct"] = half_hbar_c * torch.sum(
                self.measure
                * rho_t3
                * couplings["grtn"]
                * fields["rho_tensor_field"]
            )
            terms["E_rho_vt_direct"] = half_hbar_c * torch.sum(
                self.measure
                * (
                    rho_t3 * couplings["grtn"] * fields["rho_tv_field"]
                    + rho_3 * couplings["grho"] * fields["rho_vt_field"]
                )
            )
        terms["E_direct"] = sum(
            (
                terms["E_sigma"],
                terms["E_omega"],
                terms["E_rho"],
                terms["E_coul"],
                terms["E_rho_tensor_direct"],
                terms["E_rho_vt_direct"],
            ),
            zero,
        )
        sigma_rearr = self.rearrangement(densities, fields, couplings)
        terms["E_rearr_diagnostic"] = half_hbar_c * torch.sum(
            self.measure * rho_b * sigma_rearr
        )
        return terms

    def local_self_energies(
        self,
        densities: dict[str, torch.Tensor],
        fields: dict[str, torch.Tensor],
        couplings: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Explicit local RHF Hamiltonian fields in Eqs. (3.387)-(3.388)."""

        p = self.params
        sigma_s = couplings["gsig"] * fields["sigma"]
        rearrangement = self.rearrangement_components(densities, fields, couplings)
        sigma_r = rearrangement["total"]
        sigma_0 = couplings["gome"] * fields["omega"] + sigma_r
        sigma_3 = couplings["grho"] * (
            fields["rho"] + fields["rho_vt_field"]
        )
        sigma_t = couplings["grtn"] * (
            fields["rho_tensor_field"] + fields["rho_tv_field"]
        )
        return {
            "sigma_rearr_direct": sigma_r,
            "sigma_rearr_direct_standard": rearrangement["standard"],
            "sigma_rearr_tensor_direct": rearrangement["rho_tensor_and_vt"],
            "component_S_direct": sigma_s,
            "component_V0_direct": sigma_0,
            "component_V3_direct": sigma_3,
            "component_VC_direct": fields["coul"],
            "component_T3_direct": sigma_t,
            "Sigma_plus_n_direct": sigma_0 + sigma_s + sigma_3,
            "Sigma_minus_n_direct": sigma_0 - sigma_s + sigma_3,
            "Sigma_T_n_direct": sigma_t,
            "Sigma_plus_p_direct": sigma_0 + sigma_s - sigma_3 + fields["coul"],
            "Sigma_minus_p_direct": sigma_0 - sigma_s - sigma_3 + fields["coul"],
            "Sigma_T_p_direct": -sigma_t,
            "vps_n": sigma_0 + sigma_s + sigma_3,
            "vms_n": sigma_0 - sigma_s + sigma_3 - 2.0 * p.mass_n / p.hbar_c,
            "vps_p": sigma_0 + sigma_s - sigma_3 + fields["coul"],
            "vms_p": sigma_0 - sigma_s - sigma_3 + fields["coul"] - 2.0 * p.mass_p / p.hbar_c,
        }

    def observables(
        self, densities: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        rho_v_n = densities["rho_v_n"]
        rho_v_p = densities["rho_v_p"]
        rho_b = rho_v_n + rho_v_p
        n_int = torch.sum(self.measure * rho_v_n)
        z_int = torch.sum(self.measure * rho_v_p)
        a_int = n_int + z_int
        rms_n = torch.sqrt(
            torch.sum(self.measure * self.r2 * rho_v_n) / n_int.clamp_min(1.0e-12)
        )
        rms_p = torch.sqrt(
            torch.sum(self.measure * self.r2 * rho_v_p) / z_int.clamp_min(1.0e-12)
        )
        rms_m = torch.sqrt(
            torch.sum(self.measure * self.r2 * rho_b) / a_int.clamp_min(1.0e-12)
        )
        charge = torch.sqrt(
            rms_p.square() + 0.862**2 - 0.336**2 * n_int / z_int.clamp_min(1.0e-12)
        )
        return {
            "n_number": n_int,
            "z_number": z_int,
            "rms_n_no_com": rms_n,
            "rms_p_no_com": rms_p,
            "rms_matter_no_com": rms_m,
            "charge_radius_no_com": charge,
        }
