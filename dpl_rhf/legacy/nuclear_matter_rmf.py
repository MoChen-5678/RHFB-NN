from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from rmf_functional import PKDDParameters


class PKDDNuclearMatter:
    """PKDD uniform matter from Nuclear Physics Practice Eqs. (2.29)-(2.41)."""

    def __init__(self, quadrature_order: int = 96, params: PKDDParameters | None = None):
        self.params = params or PKDDParameters()
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
        self.nodes = torch.as_tensor(nodes, dtype=torch.float64)
        self.weights = torch.as_tensor(weights, dtype=torch.float64)

    def couplings(self, rho_b: torch.Tensor) -> dict[str, torch.Tensor]:
        p = self.params
        xi = rho_b / p.rho_sat

        def isoscalar(g0, a, b, c, d):
            x = xi + d
            g = g0 * a * (1.0 + b * x.square()) / (1.0 + c * x.square())
            dg_dxi = g0 * a * 2.0 * (b - c) * x / (1.0 + c * x.square()).square()
            return g, dg_dxi / p.rho_sat

        g_sigma, dg_sigma = isoscalar(p.g_sigma, p.a_sigma, p.b_sigma, p.c_sigma, p.d_sigma)
        g_omega, dg_omega = isoscalar(p.g_omega, p.a_omega, p.b_omega, p.c_omega, p.d_omega)
        g_rho = p.g_rho * torch.exp(-p.a_rho * xi)
        dg_rho = -p.a_rho * g_rho / p.rho_sat
        return {
            "g_sigma": g_sigma, "g_omega": g_omega, "g_rho": g_rho,
            "dg_sigma": dg_sigma, "dg_omega": dg_omega, "dg_rho": dg_rho,
        }

    def integrate_fermi_sphere(self, kf: torch.Tensor, values) -> torch.Tensor:
        momentum = 0.5 * kf * (self.nodes.to(kf) + 1.0)
        return 0.5 * kf * torch.sum(self.weights.to(kf) * values(momentum)) / torch.pi**2

    def densities_and_kinetic(
        self, rho_b: torch.Tensor, asymmetry: torch.Tensor, sigma: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        p = self.params
        c = self.couplings(rho_b)
        rho_n = 0.5 * rho_b * (1.0 + asymmetry)
        rho_p = 0.5 * rho_b * (1.0 - asymmetry)
        kf_n = (3.0 * torch.pi**2 * rho_n).pow(1.0 / 3.0)
        kf_p = (3.0 * torch.pi**2 * rho_p).pow(1.0 / 3.0)
        mass_n = torch.as_tensor(p.mass_n / p.hbar_c, dtype=rho_b.dtype, device=rho_b.device)
        mass_p = torch.as_tensor(p.mass_p / p.hbar_c, dtype=rho_b.dtype, device=rho_b.device)
        mstar_n = mass_n + c["g_sigma"] * sigma
        mstar_p = mass_p + c["g_sigma"] * sigma

        def channel(kf, mass, mstar):
            scalar = self.integrate_fermi_sphere(
                kf, lambda momentum: momentum.square() * mstar / torch.sqrt(momentum.square() + mstar.square())
            )
            kinetic = self.integrate_fermi_sphere(
                kf,
                lambda momentum: momentum.square()
                * (momentum.square() + mass * mstar)
                / torch.sqrt(momentum.square() + mstar.square()),
            )
            return scalar, kinetic

        rho_s_n, kinetic_n = channel(kf_n, mass_n, mstar_n)
        rho_s_p, kinetic_p = channel(kf_p, mass_p, mstar_p)
        return {
            **c,
            "rho_n": rho_n, "rho_p": rho_p, "rho_3": rho_n - rho_p,
            "rho_s_n": rho_s_n, "rho_s_p": rho_s_p,
            "rho_s": rho_s_n + rho_s_p,
            "kinetic": kinetic_n + kinetic_p,
            "kf_n": kf_n, "kf_p": kf_p,
            "mstar_n": mstar_n, "mstar_p": mstar_p,
        }

    def off_shell_energy_density(
        self, rho_b: torch.Tensor, asymmetry: torch.Tensor, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        p = self.params
        state = self.densities_and_kinetic(rho_b, asymmetry, sigma)
        m_sigma = p.m_sigma / p.hbar_c
        m_omega = p.m_omega / p.hbar_c
        m_rho = p.m_rho / p.hbar_c
        scalar = state["g_sigma"] * sigma * state["rho_s"] + 0.5 * m_sigma**2 * sigma.square()
        omega = 0.5 * state["g_omega"].square() / m_omega**2 * rho_b.square()
        rho = 0.5 * state["g_rho"].square() / m_rho**2 * state["rho_3"].square()
        total = state["kinetic"] + scalar + omega + rho
        return total, {**state, "E_kin": state["kinetic"], "E_sigma": scalar, "E_omega": omega, "E_rho": rho}

    def solve(self, rho_b_value: float, asymmetry_value: float = 0.0) -> dict[str, float]:
        p = self.params
        rho_b = torch.tensor(rho_b_value, dtype=torch.float64, requires_grad=True)
        asymmetry = torch.tensor(asymmetry_value, dtype=torch.float64)
        sigma = torch.nn.Parameter(torch.tensor(-0.2, dtype=torch.float64))
        optimizer = torch.optim.LBFGS([sigma], lr=0.5, max_iter=100, tolerance_grad=1.0e-13, tolerance_change=1.0e-15)

        def closure():
            optimizer.zero_grad()
            energy_density, _ = self.off_shell_energy_density(rho_b.detach(), asymmetry, sigma)
            energy_density.backward()
            return energy_density

        optimizer.step(closure)
        energy_density, state = self.off_shell_energy_density(rho_b, asymmetry, sigma)
        energy_per_particle = p.hbar_c * energy_density / rho_b
        rest = 0.5 * ((1.0 + asymmetry) * p.mass_n + (1.0 - asymmetry) * p.mass_p)
        binding = energy_per_particle - rest

        m_sigma = p.m_sigma / p.hbar_c
        m_omega = p.m_omega / p.hbar_c
        m_rho = p.m_rho / p.hbar_c
        sigma_self = -state["g_sigma"] * state["rho_s"] / m_sigma**2
        rearrangement = (
            sigma * state["rho_s"] * state["dg_sigma"]
            + state["g_omega"] * state["dg_omega"] / m_omega**2 * rho_b.square()
            + state["g_rho"] * state["dg_rho"] / m_rho**2 * state["rho_3"].square()
        )
        sigma_0_common = state["g_omega"].square() / m_omega**2 * rho_b + rearrangement
        sigma_0_isovector = state["g_rho"].square() / m_rho**2 * state["rho_3"]
        sigma_0_n = sigma_0_common + sigma_0_isovector
        sigma_0_p = sigma_0_common - sigma_0_isovector
        fermi_n = torch.sqrt(state["kf_n"].square() + state["mstar_n"].square()) + sigma_0_n
        fermi_p = torch.sqrt(state["kf_p"].square() + state["mstar_p"].square()) + sigma_0_p
        fixed_asymmetry_chemical = 0.5 * ((1.0 + asymmetry) * fermi_n + (1.0 - asymmetry) * fermi_p)
        chemical_ad = torch.autograd.grad(energy_density, rho_b)[0]

        return {
            "rho_b": rho_b_value,
            "asymmetry": asymmetry_value,
            "sigma_fm_inv": float(sigma.detach()),
            "sigma_field_residual": float((sigma.detach() - sigma_self.detach()).abs()),
            "mstar_n_over_m": float(state["mstar_n"].detach() / (p.mass_n / p.hbar_c)),
            "binding_mev": float(binding.detach()),
            "energy_per_particle_mev": float(energy_per_particle.detach()),
            "rearrangement_mev": float(rearrangement.detach() * p.hbar_c),
            "chemical_potential_ad_mev": float(chemical_ad.detach() * p.hbar_c),
            "fermi_energy_n_mev": float(fermi_n.detach() * p.hbar_c),
            "fermi_energy_p_mev": float(fermi_p.detach() * p.hbar_c),
            "thermodynamic_residual_mev": float(
                (chemical_ad.detach() - fixed_asymmetry_chemical.detach()).abs() * p.hbar_c
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="PKDD nuclear-matter variational PoC")
    parser.add_argument("--rho-min", type=float, default=0.04)
    parser.add_argument("--rho-max", type=float, default=0.24)
    parser.add_argument("--points", type=int, default=41)
    parser.add_argument("--asymmetry", type=float, default=0.0)
    parser.add_argument("--quadrature-order", type=int, default=96)
    parser.add_argument("--out", default="outputs/nuclear_matter_pkdd")
    args = parser.parse_args()

    solver = PKDDNuclearMatter(args.quadrature_order)
    rows = [solver.solve(float(rho), args.asymmetry) for rho in np.linspace(args.rho_min, args.rho_max, args.points)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "eos.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    minimum = min(rows, key=lambda row: row["binding_mev"])
    summary = {"model": "PKDD", "parameters": asdict(solver.params), "minimum_on_grid": minimum}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(minimum, indent=2))


if __name__ == "__main__":
    main()
