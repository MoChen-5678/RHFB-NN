from __future__ import annotations

import numpy as np

from dpl_rhf.functionals.base import FunctionalBase, HamiltonianChannelSpec
from dpl_rhf.functionals.pkdd_action import PKDDParameters, rmf_parameters


class PKDDRMFFunctionalSpec(FunctionalBase):
    channel_spec = HamiltonianChannelSpec(
        names=("vps_n", "vms_n", "vps_p", "vms_p"),
        scales=(0.15, 0.30, 0.15, 0.30),
        functional_type="rmf",
    )

    def __init__(
        self, z: int, n: int, r: np.ndarray, params: PKDDParameters | None = None
    ):
        self.z = int(z)
        self.n = int(n)
        self.a = self.z + self.n
        self.r = np.asarray(r, dtype=np.float64)
        self.params = params or rmf_parameters("PKDD")

    def initial_hamiltonian(self) -> np.ndarray:
        """Textbook Woods-Saxon local Dirac channels used only as initialization."""
        p = self.params
        z, n, a, r = self.z, self.n, self.a, self.r
        v0, a0 = -71.28, 0.4616
        table = {
            "n": {"tau": 1.0, "av": 11.1175, "rplus": 1.2334, "rminus": 1.1443, "aplus": 0.6150, "aminus": 0.6476},
            "p": {"tau": -1.0, "av": 8.9698, "rplus": 1.2496, "rminus": 1.1400, "aplus": 0.6124, "aminus": 0.6469},
        }
        charge_radius = table["p"]["rplus"] * a ** (1.0 / 3.0)
        alpha = 1.0 / p.alpha_inv
        coul = np.where(
            r < charge_radius,
            alpha * z * (3.0 / (2.0 * charge_radius) - r * r / (2.0 * charge_radius**3)),
            alpha * z / r,
        )
        channels = []
        for species in ("n", "p"):
            q = table[species]
            factor = 1.0 - a0 * (n - z) * q["tau"] / a
            plus = v0 * factor / (1.0 + np.exp((r - q["rplus"] * a ** (1.0 / 3.0)) / q["aplus"]))
            minus = -v0 * q["av"] * factor / (1.0 + np.exp((r - q["rminus"] * a ** (1.0 / 3.0)) / q["aminus"]))
            if species == "p":
                plus = plus + p.hbar_c * coul
                minus = minus + p.hbar_c * coul
            mass = p.mass_n if species == "n" else p.mass_p
            channels.extend([plus / p.hbar_c, minus / p.hbar_c - 2.0 * mass / p.hbar_c])
        raw = np.stack(channels)
        # Woods-Saxon supplies only a starting point. Project it onto the exact
        # RMF channel manifold so a zero network correction reproduces its own
        # initialization and neutron/proton share one scalar self-energy.
        scalar_n = 0.5 * (raw[0] - raw[1] - 2.0 * p.mass_n / p.hbar_c)
        scalar_p = 0.5 * (raw[2] - raw[3] - 2.0 * p.mass_p / p.hbar_c)
        vector_n = 0.5 * (raw[0] + raw[1] + 2.0 * p.mass_n / p.hbar_c)
        vector_p_nuclear = 0.5 * (raw[2] + raw[3] + 2.0 * p.mass_p / p.hbar_c) - coul
        scalar = 0.5 * (scalar_n + scalar_p)
        vector = 0.5 * (vector_n + vector_p_nuclear)
        # The rho field is generated from rho_n-rho_p by the differentiable
        # functional. Do not seed it with the species-dependent Woods-Saxon
        # fit, which otherwise creates a large fictitious V3 even for N=Z.
        isovector = np.zeros_like(vector)
        return np.stack([
            vector + scalar + isovector,
            vector - scalar + isovector - 2.0 * p.mass_n / p.hbar_c,
            vector + scalar - isovector + coul,
            vector - scalar - isovector + coul - 2.0 * p.mass_p / p.hbar_c,
        ])
