from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class PKDDParameters:
    model: str = "PKDD"
    hbar_c: float = 197.328284
    alpha_inv: float = 137.03602
    rho_sat: float = 0.149552
    mass_n: float = 939.5731
    mass_p: float = 938.2796
    m_sigma: float = 555.511236
    m_omega: float = 783.0
    m_rho: float = 763.0
    g_sigma: float = 10.738508
    g_omega: float = 13.147623
    g_rho: float = 5.164857
    a_sigma: float = 1.32742274
    b_sigma: float = 0.43512557
    c_sigma: float = 0.69166629
    d_sigma: float = 0.69421032
    a_omega: float = 1.34217027
    b_omega: float = 0.37116653
    c_omega: float = 0.61139691
    d_omega: float = 0.73837631
    a_rho: float = 0.18330476


RMF_PARAMETER_DATA = {
    "DD-ME1": {
        "rho_sat": 0.151962, "mass_n": 938.5, "mass_p": 938.5,
        "m_sigma": 549.5255, "m_omega": 783.0, "m_rho": 763.0,
        "g_sigma": 10.4434, "g_omega": 12.8939, "g_rho": 6.2789,
        "a_sigma": 1.3854, "b_sigma": 0.9781, "c_sigma": 1.5342, "d_sigma": 0.4661,
        "a_omega": 1.3879, "b_omega": 0.8525, "c_omega": 1.3566, "d_omega": 0.4957,
        "a_rho": 0.5008,
    },
    "DD-ME2": {
        "rho_sat": 0.152, "mass_n": 938.9, "mass_p": 938.9,
        "m_sigma": 550.1238, "m_omega": 783.0, "m_rho": 763.0,
        "g_sigma": 10.5396, "g_omega": 13.0189, "g_rho": 6.4792,
        "a_sigma": 1.3881, "b_sigma": 1.0943, "c_sigma": 1.7057, "d_sigma": 0.4421,
        "a_omega": 1.3892, "b_omega": 0.9240, "c_omega": 1.4620, "d_omega": 0.4775,
        "a_rho": 0.5647,
    },
    "PKDD": {
        "rho_sat": 0.149552, "mass_n": 939.5731, "mass_p": 938.2796,
        "m_sigma": 555.511236, "m_omega": 783.0, "m_rho": 763.0,
        "g_sigma": 10.738508, "g_omega": 13.147623, "g_rho": 5.164857,
        "a_sigma": 1.32742274, "b_sigma": 0.43512557, "c_sigma": 0.69166629, "d_sigma": 0.69421032,
        "a_omega": 1.34217027, "b_omega": 0.37116653, "c_omega": 0.61139691, "d_omega": 0.73837631,
        "a_rho": 0.18330476,
    },
    "TW99": {
        "rho_sat": 0.153004, "mass_n": 939.0, "mass_p": 939.0,
        "m_sigma": 550.0, "m_omega": 783.0, "m_rho": 763.0,
        "g_sigma": 10.72854, "g_omega": 13.29015, "g_rho": 6.127157,
        "a_sigma": 1.365469, "b_sigma": 0.226061, "c_sigma": 0.409704, "d_sigma": 0.901995,
        "a_omega": 1.402488, "b_omega": 0.172577, "c_omega": 0.344293, "d_omega": 0.983955,
        "a_rho": 0.515,
    },
    "DD-LZ1": {
        "rho_sat": 0.158100, "mass_n": 938.9, "mass_p": 938.9,
        "m_sigma": 538.619216, "m_omega": 783.0, "m_rho": 769.0,
        "g_sigma": 12.001429, "g_omega": 14.292525, "g_rho": 7.575467,
        "a_sigma": 1.062748, "b_sigma": 1.763627, "c_sigma": 2.308928, "d_sigma": 0.379957,
        "a_omega": 1.059181, "b_omega": 0.418273, "c_omega": 0.538663, "d_omega": 0.786649,
        "a_rho": 0.776095,
    },
}

RMF_MODEL_ALIASES = {
    "DDME1": "DD-ME1",
    "DDME2": "DD-ME2",
    "DDLZ1": "DD-LZ1",
}


def canonical_rmf_model(model: str) -> str:
    name = str(model).strip().upper()
    name = RMF_MODEL_ALIASES.get(name, name)
    if name not in RMF_PARAMETER_DATA:
        raise ValueError(
            f"unsupported RMF model {model!r}; choose {', '.join(RMF_PARAMETER_DATA)}"
        )
    return name


def rmf_parameters(model: str) -> PKDDParameters:
    name = canonical_rmf_model(model)
    return PKDDParameters(model=name, **RMF_PARAMETER_DATA[name])


def simpson_weights(n: int, h: float, *, device, dtype) -> torch.Tensor:
    """Weights matching Core-1204/RHFlib.f90:simps."""
    if n < 3:
        raise ValueError("Simpson integration requires at least three points")
    if n == 4:
        return torch.tensor([1.0, 3.0, 3.0, 1.0], device=device, dtype=dtype) * (3.0 * h / 8.0)
    weights = torch.ones(n, device=device, dtype=dtype)
    if (n - 1) % 2 == 0:
        weights[1:-1:2] = 4.0
        weights[2:-1:2] = 2.0
        return weights * (h / 3.0)

    # An odd number of panels uses Simpson 3/8 on the first three panels,
    # followed by Simpson 1/3, exactly as the Fortran implementation.
    weights.zero_()
    weights[:4] = torch.tensor([3.0, 9.0, 9.0, 3.0], device=device, dtype=dtype) * (h / 8.0)
    tail = simpson_weights(n - 3, h, device=device, dtype=dtype)
    weights[3:] += tail
    return weights


def boole_weights(n: int, h: float, *, device, dtype) -> torch.Tensor:
    """Sixth-order composite Boole quadrature with a 3/8 leading remainder."""
    if n < 5:
        return simpson_weights(n, h, device=device, dtype=dtype)
    weights = torch.zeros(n, device=device, dtype=dtype)
    panels = n - 1
    start = panels % 4
    if start == 1:
        # Keep the high-order rule by consuming five panels with a local
        # degree-five interpolatory formula.
        nodes = torch.arange(6, device=device, dtype=dtype)
        vandermonde = torch.stack([nodes.pow(k) for k in range(6)])
        moments = torch.tensor([5.0 ** (k + 1) / (k + 1) for k in range(6)], device=device, dtype=dtype)
        weights[:6] += torch.linalg.solve(vandermonde, moments) * h
        start = 5
    elif start == 2:
        weights[:3] += torch.tensor([1.0, 4.0, 1.0], device=device, dtype=dtype) * (h / 3.0)
    elif start == 3:
        weights[:4] += torch.tensor([1.0, 3.0, 3.0, 1.0], device=device, dtype=dtype) * (3.0 * h / 8.0)
    for i in range(start, panels, 4):
        weights[i : i + 5] += torch.tensor([7.0, 32.0, 12.0, 32.0, 7.0], device=device, dtype=dtype) * (2.0 * h / 45.0)
    return weights


def differentiation_matrix(x: torch.Tensor, derivative_order: int, stencil: int = 7) -> torch.Tensor:
    """Local Fornberg-equivalent polynomial differentiation matrix."""
    n = x.numel()
    matrix = torch.zeros((n, n), device=x.device, dtype=x.dtype)
    half = stencil // 2
    factorial = float(math.factorial(derivative_order))
    for i in range(n):
        lo = min(max(i - half, 0), n - stencil)
        indices = torch.arange(lo, lo + stencil, device=x.device)
        offsets = x[indices] - x[i]
        powers = torch.stack([offsets.pow(k) for k in range(stencil)])
        rhs = torch.zeros(stencil, device=x.device, dtype=x.dtype)
        rhs[derivative_order] = factorial
        matrix[i, indices] = torch.linalg.solve(powers, rhs)
    return matrix


class PKDDRMFFunctional:
    """PyTorch PKDD RMF residuals for spherical local Hartree fields.

    The residuals are embedded in the computational graph. Fortran is not used
    here; Fortran is reserved for final fixed-potential Dirac evaluation.
    """

    def __init__(self, r: torch.Tensor, z: int, n: int, params: PKDDParameters | None = None):
        self.r = r
        self.z = float(z)
        self.n = float(n)
        self.a = float(z + n)
        self.params = params or PKDDParameters()
        # Core-1204 stores an origin placeholder at r(1)=0.001 fm; the
        # equidistant physical mesh starts at r(2)=0.1 fm.
        self.h = float((r[2] - r[1]).detach().cpu())
        # The coordinate-space ADF Hamiltonian is Hermitian in the uniform
        # grid metric.  Use that same metric for every orbital bilinear and
        # source-field contraction, so <psi|V|psi> and int(rho V) are exactly
        # the same discrete quantity.  Mixing Newton-Cotes weights into only
        # the action breaks the E[H] variational derivative.
        self.w = torch.full_like(r, self.h)
        self.w[0] = 0.0
        self.orbital_w = self.w.clone()
        self.r2 = r * r
        # r[0]=0.001 fm is only an origin placeholder; all radial integrals
        # start on the equidistant physical mesh at r[1]=0.1 fm, as does the
        # orbital normalization below.
        self.measure = 4.0 * torch.pi * self.r2 * self.orbital_w
        self.d1 = differentiation_matrix(r, 1)
        self.d2 = differentiation_matrix(r, 2)
        radial_laplacian = self.d2 + torch.diag(2.0 / r.clamp(min=1.0e-8)) @ self.d1
        self._radial_laplacian = radial_laplacian

        # Build the field equations from the exact Hessian of the discretized
        # action. The embedding enforces regularity at the origin. All fields
        # retain their value at R; exact exterior actions impose Yukawa Robin
        # conditions on massive fields and the Coulomb Robin condition.
        n_grid = r.numel()
        massive_embedding = torch.zeros((n_grid, n_grid - 1), device=r.device, dtype=r.dtype)
        massive_embedding[0, 0] = 1.0
        massive_embedding[1:] = torch.eye(n_grid - 1, device=r.device, dtype=r.dtype)
        coul_embedding = torch.zeros((n_grid, n_grid - 1), device=r.device, dtype=r.dtype)
        coul_embedding[0, 0] = 1.0
        coul_embedding[1:] = torch.eye(n_grid - 1, device=r.device, dtype=r.dtype)
        measure_matrix = torch.diag(self.measure)
        gradient_kernel = self.d1.T @ measure_matrix @ self.d1
        identity_kernel = measure_matrix
        self.field_embeddings = {
            "sigma": massive_embedding,
            "omega": massive_embedding,
            "rho": massive_embedding,
            "coul": coul_embedding,
        }
        self.field_full_kernels = {}
        for name, mass in (
            ("sigma", self.params.m_sigma),
            ("omega", self.params.m_omega),
            ("rho", self.params.m_rho),
        ):
            mu = mass / self.params.hbar_c
            kernel = gradient_kernel + mu**2 * identity_kernel
            kernel = kernel.clone()
            # The source-free exterior solution is C exp(-mu r)/r. Integrating
            # its action from R to infinity gives this exact Dirichlet-to-
            # Neumann boundary term.
            kernel[-1, -1] += 4.0 * torch.pi * r[-1].square() * (
                mu + 1.0 / r[-1]
            )
            self.field_full_kernels[name] = kernel
        alpha = 1.0 / self.params.alpha_inv
        coul_kernel = gradient_kernel / (4.0 * torch.pi * alpha)
        coul_kernel = coul_kernel.clone()
        coul_kernel[-1, -1] += r[-1] / alpha
        self.field_full_kernels["coul"] = coul_kernel
        self.field_operators = {
            name: embedding.T @ self.field_full_kernels[name] @ embedding
            for name, embedding in self.field_embeddings.items()
        }
        self.split_left = torch.zeros((r.numel(), r.numel()), device=r.device, dtype=r.dtype)
        self.split_right = torch.zeros_like(self.split_left)
        for i in range(r.numel()):
            left_count = i + 1
            right_count = r.numel() - i
            if left_count == 2:
                self.split_left[i, :2] = self.h / 2.0
            elif left_count >= 3:
                self.split_left[i, :left_count] = boole_weights(left_count, self.h, device=r.device, dtype=r.dtype)
            if right_count == 2:
                self.split_right[i, i:] = self.h / 2.0
            elif right_count >= 3:
                self.split_right[i, i:] = boole_weights(right_count, self.h, device=r.device, dtype=r.dtype)

    def couplings(self, rho_b: torch.Tensor) -> dict[str, torch.Tensor]:
        p = self.params
        # Density dependence is the analytic PKDD function used in
        # Density.f90.  Clamping zeta would alter its functional derivative
        # and hence the rearrangement self-energy in the dilute tail.
        zeta = rho_b / p.rho_sat
        x_sig = zeta + p.d_sigma
        x_ome = zeta + p.d_omega
        gsig = p.g_sigma * p.a_sigma * (1.0 + p.b_sigma * x_sig.square()) / (
            1.0 + p.c_sigma * x_sig.square()
        )
        gome = p.g_omega * p.a_omega * (1.0 + p.b_omega * x_ome.square()) / (
            1.0 + p.c_omega * x_ome.square()
        )
        grho = p.g_rho * torch.exp(-p.a_rho * zeta)
        # Match Density.f90: cct%d* are derivatives with respect to zeta;
        # Rearrange later divides by rho_sat to obtain d g / d rho_b.
        dgsig = (
            p.g_sigma
            * p.a_sigma
            * 2.0
            * (p.b_sigma - p.c_sigma)
            * x_sig
            / (1.0 + p.c_sigma * x_sig.square()).square()
        )
        dgome = (
            p.g_omega
            * p.a_omega
            * 2.0
            * (p.b_omega - p.c_omega)
            * x_ome
            / (1.0 + p.c_omega * x_ome.square()).square()
        )
        dgrho = -p.a_rho * grho
        return {
            "gsig": gsig,
            "gome": gome,
            "grho": grho,
            "dgsig": dgsig,
            "dgome": dgome,
            "dgrho": dgrho,
        }

    def radial_laplacian(self, y: torch.Tensor) -> torch.Tensor:
        first = self.radial_derivative(y)
        second = torch.matmul(y, self.d2.T)
        lap = second + 2.0 * first / self.r.clamp(min=1.0e-8)
        lap = lap.clone()
        lap[..., 0] = lap[..., 1]
        return lap

    def radial_derivative(self, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(y, self.d1.T)

    def reconstruct_fields(self, state: dict[str, torch.Tensor], coup: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Solve the Euler-Lagrange equations of the discrete RMF action once."""
        p = self.params
        rho_s = state["rho_s_n"] + state["rho_s_p"]
        rho_b = state["rho_v_n"] + state["rho_v_p"]
        rho_3 = state["rho_v_n"] - state["rho_v_p"]
        sources = {
            "sigma": -coup["gsig"] * rho_s,
            "omega": coup["gome"] * rho_b,
            "rho": coup["grho"] * rho_3,
            "coul": state["rho_v_p"],
        }
        reconstructed = {}
        for name, source in sources.items():
            embedding = self.field_embeddings[name]
            rhs = embedding.T @ (self.measure * source)
            coefficients = torch.linalg.solve(self.field_operators[name], rhs)
            reconstructed[name] = embedding @ coefficients
        return reconstructed

    def field_stationarity_residuals(
        self, state: dict[str, torch.Tensor], coup: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        rho_s = state["rho_s_n"] + state["rho_s_p"]
        rho_b = state["rho_v_n"] + state["rho_v_p"]
        rho_3 = state["rho_v_n"] - state["rho_v_p"]
        sources = {
            "sigma": -coup["gsig"] * rho_s,
            "omega": coup["gome"] * rho_b,
            "rho": coup["grho"] * rho_3,
            "coul": state["rho_v_p"],
        }
        residuals = {}
        for name, source in sources.items():
            embedding = self.field_embeddings[name]
            coefficients = state[name][1:]
            residuals[name] = self.field_operators[name] @ coefficients - embedding.T @ (self.measure * source)
        return residuals

    def reconstruct_fields_green(
        self, state: dict[str, torch.Tensor], coup: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Textbook Eqs. (3.42)-(3.45), split at the Green-function cusp."""
        p = self.params
        rho_s = state["rho_s_n"] + state["rho_s_p"]
        rho_b = state["rho_v_n"] + state["rho_v_p"]
        rho_3 = state["rho_v_n"] - state["rho_v_p"]

        def yukawa(source: torch.Tensor, mass_mev: float) -> torch.Tensor:
            mass = mass_mev / p.hbar_c
            mr = mass * self.r
            left_integrand = self.r * torch.sinh(mr) * source
            right_integrand = self.r * torch.exp(-mr) * source
            left = self.split_left @ left_integrand
            right = self.split_right @ right_integrand
            return (torch.exp(-mr) * left + torch.sinh(mr) * right) / (mass * self.r.clamp(min=1.0e-8))

        sigma = yukawa(-coup["gsig"] * rho_s, p.m_sigma)
        omega = yukawa(coup["gome"] * rho_b, p.m_omega)
        rho = yukawa(coup["grho"] * rho_3, p.m_rho)
        coul_source = (4.0 * torch.pi / p.alpha_inv) * state["rho_v_p"]
        coul_left = self.split_left @ (self.r2 * coul_source)
        coul_right = self.split_right @ (self.r * coul_source)
        coul = coul_left / self.r.clamp(min=1.0e-8) + coul_right
        return {"sigma": sigma, "omega": omega, "rho": rho, "coul": coul}

    def densities_from_orbitals(self, orbitals: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Book Eqs. (3.6)-(3.7), matching Density.f90 for occupied states."""
        G = orbitals["G"]
        F = orbitals["F"]
        weight = orbitals["occupancy"] * orbitals["degeneracy"]
        neutron = orbitals["species_sign"] > 0.0
        proton = ~neutron
        denominator = 4.0 * torch.pi * self.r2.clamp(min=1.0e-12)

        def channel(mask: torch.Tensor, scalar: bool) -> torch.Tensor:
            sign = -1.0 if scalar else 1.0
            values = (G[mask].square() + sign * F[mask].square()) * weight[mask, None]
            density = values.sum(dim=0) / denominator
            # Same regular-origin extrapolation used in Density.f90.
            density = density.clone()
            density[0] = 3.0 * (density[1] - density[2]) + density[3]
            return density

        return {
            "rho_s_n": channel(neutron, True),
            "rho_s_p": channel(proton, True),
            "rho_v_n": channel(neutron, False),
            "rho_v_p": channel(proton, False),
        }

    def exact_kinetic_energy(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """Spherical RMF kinetic energy from the book Eq. (3.47), no TF approximation."""
        if "kinetic_expectation_mev" in state:
            orbital_weight = state["occupancy"] * state["degeneracy"]
            return torch.sum(orbital_weight * state["kinetic_expectation_mev"])
        p = self.params
        G = state["G"]
        F = state["F"]
        kappa = state["kappa"][:, None]
        weight = state["occupancy"] * state["degeneracy"]
        mass = torch.where(state["species_sign"] > 0.0, p.mass_n, p.mass_p)[:, None] / p.hbar_c
        radius = self.r.clamp(min=1.0e-8)[None, :]
        dG = self.radial_derivative(G)
        dF = self.radial_derivative(F)
        density = G * (-dF + kappa * F / radius + mass * G) + F * (
            dG + kappa * G / radius - mass * F
        )
        total = p.hbar_c * torch.sum(weight * torch.sum(self.orbital_w[None, :] * density, dim=1))
        rest = torch.sum(weight * torch.where(state["species_sign"] > 0.0, p.mass_n, p.mass_p))
        return total - rest

    def off_shell_rmf_action(
        self, orbital_state: dict[str, torch.Tensor], state: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Static RMF action before imposing the meson field equations.

        Scalar fields are minimizing directions; time-like vector and Coulomb
        fields are maximizing directions. At the saddle point these terms
        reduce exactly to the half source-field energies used by Expect.f90.
        """
        p = self.params
        rho_s = state["rho_s_n"] + state["rho_s_p"]
        rho_b = state["rho_v_n"] + state["rho_v_p"]
        rho_3 = state["rho_v_n"] - state["rho_v_p"]
        coup = self.couplings(rho_b)
        sigma = state["sigma"]
        omega = state["omega"]
        rho = state["rho"]
        coul = state["coul"]
        e_sigma = p.hbar_c * (
            0.5 * sigma @ self.field_full_kernels["sigma"] @ sigma
            + torch.sum(self.measure * coup["gsig"] * rho_s * sigma)
        )
        e_omega = p.hbar_c * (
            -0.5 * omega @ self.field_full_kernels["omega"] @ omega
            + torch.sum(self.measure * coup["gome"] * rho_b * omega)
        )
        e_rho = p.hbar_c * (
            -0.5 * rho @ self.field_full_kernels["rho"] @ rho
            + torch.sum(self.measure * coup["grho"] * rho_3 * rho)
        )
        e_coul = p.hbar_c * (
            -0.5 * coul @ self.field_full_kernels["coul"] @ coul
            + torch.sum(self.measure * state["rho_v_p"] * coul)
        )
        # The numerical box ends at R while the Coulomb solution continues as
        # C/r.  Integrating the source-free exterior action from R to infinity
        # gives this Robin boundary term.
        alpha = 1.0 / p.alpha_inv
        e_coul_exterior = -p.hbar_c * self.r[-1] * coul[-1].square() / (2.0 * alpha)
        e_kin = self.exact_kinetic_energy(orbital_state)
        norms = torch.sum(
            self.orbital_w[None, :] * (orbital_state["G"].square() + orbital_state["F"].square()), dim=1
        )
        orbital_weights = orbital_state["occupancy"] * orbital_state["degeneracy"]
        lagrange_normalization = -torch.sum(orbital_weights * orbital_state["epsilon"] * (norms - 1.0))
        action = e_kin + e_sigma + e_omega + e_rho + e_coul + lagrange_normalization
        return {
            "rmf_action": action,
            "rmf_action_per_A": action / self.a,
            "action_kinetic": e_kin,
            "action_sigma": e_sigma,
            "action_omega": e_omega,
            "action_rho": e_rho,
            "action_coul": e_coul,
            "action_coul_exterior": e_coul_exterior,
            "action_orbital_constraints": lagrange_normalization,
            "max_orbital_norm_residual": torch.max(torch.abs(norms - 1.0)),
        }

    def euler_lagrange_residuals(
        self, orbital_state: dict[str, torch.Tensor], state: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Book Eqs. (3.42)-(3.48) evaluated without empirical penalties."""
        p = self.params
        rho_s = state["rho_s_n"] + state["rho_s_p"]
        rho_b = state["rho_v_n"] + state["rho_v_p"]
        rho_3 = state["rho_v_n"] - state["rho_v_p"]
        coup = self.couplings(rho_b)
        residuals = {
            "sigma": -self.radial_laplacian(state["sigma"]) + (p.m_sigma / p.hbar_c) ** 2 * state["sigma"] + coup["gsig"] * rho_s,
            "omega": -self.radial_laplacian(state["omega"]) + (p.m_omega / p.hbar_c) ** 2 * state["omega"] - coup["gome"] * rho_b,
            "rho": -self.radial_laplacian(state["rho"]) + (p.m_rho / p.hbar_c) ** 2 * state["rho"] - coup["grho"] * rho_3,
            "coul": -self.radial_laplacian(state["coul"]) - (4.0 * torch.pi / p.alpha_inv) * state["rho_v_p"],
        }
        potentials = self.potentials(**state)
        G, F = orbital_state["G"], orbital_state["F"]
        kappa = orbital_state["kappa"][:, None]
        neutron = orbital_state["species_sign"] > 0.0
        vps = torch.where(neutron[:, None], potentials["vps_n"], potentials["vps_p"])
        vms = torch.where(neutron[:, None], potentials["vms_n"], potentials["vms_p"])
        epsilon = orbital_state["epsilon"][:, None] / p.hbar_c
        radius = self.r.clamp(min=1.0e-8)[None, :]
        residuals["dirac_G"] = -self.radial_derivative(F) + kappa * F / radius + vps * G - epsilon * G
        residuals["dirac_F"] = self.radial_derivative(G) + kappa * G / radius + vms * F - epsilon * F
        return residuals

    def potentials(
        self,
        rho_s_n: torch.Tensor,
        rho_s_p: torch.Tensor,
        rho_v_n: torch.Tensor,
        rho_v_p: torch.Tensor,
        sigma: torch.Tensor,
        omega: torch.Tensor,
        rho: torch.Tensor,
        coul: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        p = self.params
        rho_s = rho_s_n + rho_s_p
        rho_b = rho_v_n + rho_v_p
        rho_3 = rho_v_n - rho_v_p
        coup = self.couplings(rho_b)
        ss = coup["gsig"] * sigma
        sig_r = self.rearrangement_potential(rho_s, rho_b, rho_3, sigma, omega, rho, coup)
        vs = coup["gome"] * omega + sig_r
        vt = coup["grho"] * rho
        # Fortran convention: tauz(n)=+1, tauz(p)=-1, Coulomb only for protons.
        emcc_n = 2.0 * p.mass_n / p.hbar_c
        emcc_p = 2.0 * p.mass_p / p.hbar_c
        return {
            "vps_n": vs + ss + vt,
            "vms_n": vs - ss + vt - emcc_n,
            "vps_p": vs + ss - vt + coul,
            "vms_p": vs - ss - vt + coul - emcc_p,
            "sigma": sigma,
            "omega": omega,
            "rho": rho,
            "coul": coul,
            "sigma_rearr": sig_r,
            "scalar_self_energy": ss,
            "vector_isoscalar": vs,
            "vector_isovector": vt,
            "vector_coulomb": coul,
        }

    def rearrangement_potential(
        self,
        rho_s: torch.Tensor,
        rho_b: torch.Tensor,
        rho_3: torch.Tensor,
        sigma: torch.Tensor,
        omega: torch.Tensor,
        rho: torch.Tensor,
        coup: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Density-dependent rearrangement term SigR used by PotelHF.f90.

        For PKDD/RMF (IE=1), tensor and pion terms are zero. Density.f90 stores
        derivatives with respect to zeta=rho_b/rho_sat, and Meanfield.f90 divides
        by rho_sat at the end.
        """
        p = self.params
        coup = coup or self.couplings(rho_b)
        return (
            rho_s * coup["dgsig"] * sigma
            + rho_b * coup["dgome"] * omega
            + rho_3 * coup["dgrho"] * rho
        ) / p.rho_sat

    def direct_energy_terms(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        p = self.params
        rho_s_n = state["rho_s_n"]
        rho_s_p = state["rho_s_p"]
        rho_v_n = state["rho_v_n"]
        rho_v_p = state["rho_v_p"]
        sigma = state["sigma"]
        omega = state["omega"]
        rho = state["rho"]
        coul = state["coul"]
        rho_s = rho_s_n + rho_s_p
        rho_b = rho_v_n + rho_v_p
        rho_3 = rho_v_n - rho_v_p
        coup = self.couplings(rho_b)
        sig_r = self.rearrangement_potential(rho_s, rho_b, rho_3, sigma, omega, rho, coup)
        half_hbc_int = 0.5 * p.hbar_c

        e_sigma = half_hbc_int * torch.sum(self.measure * rho_s * coup["gsig"] * sigma)
        e_omega = half_hbc_int * torch.sum(self.measure * rho_b * coup["gome"] * omega)
        e_rho = half_hbc_int * torch.sum(self.measure * rho_3 * coup["grho"] * rho)
        e_coul = half_hbc_int * torch.sum(self.measure * rho_v_p * coul)
        e_rearr = half_hbc_int * torch.sum(self.measure * rho_b * sig_r)
        e_direct = e_sigma + e_omega + e_rho + e_coul
        return {
            "E_sigma": e_sigma,
            "E_omega": e_omega,
            "E_rho": e_rho,
            "E_coul": e_coul,
            "E_rearr": e_rearr,
            "E_direct": e_direct,
            "sigma_rearr": sig_r,
        }

    def observables(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        rho_v_n = state["rho_v_n"]
        rho_v_p = state["rho_v_p"]
        rho_b = rho_v_n + rho_v_p
        n_int = torch.sum(self.measure * rho_v_n)
        z_int = torch.sum(self.measure * rho_v_p)
        a_int = n_int + z_int
        rms_n = torch.sqrt(torch.sum(self.measure * self.r2 * rho_v_n) / n_int.clamp(min=1.0e-12))
        rms_p = torch.sqrt(torch.sum(self.measure * self.r2 * rho_v_p) / z_int.clamp(min=1.0e-12))
        rms_m = torch.sqrt(torch.sum(self.measure * self.r2 * rho_b) / a_int.clamp(min=1.0e-12))
        charge = torch.sqrt(rms_p.square() + 0.862**2 - 0.336**2 * n_int / z_int.clamp(min=1.0e-12))
        return {
            "n_number": n_int,
            "z_number": z_int,
            "rms_n_no_com": rms_n,
            "rms_p_no_com": rms_p,
            "rms_matter_no_com": rms_m,
            "charge_radius_no_com": charge,
        }
