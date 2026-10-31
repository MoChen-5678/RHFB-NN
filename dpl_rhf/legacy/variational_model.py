from __future__ import annotations

import torch
from torch import nn


class SphericalRMFNet(nn.Module):
    """Single-nucleus neural ansatz for RMF densities and local fields."""

    def __init__(self, r_grid, z: int, n: int, hidden: int = 96):
        super().__init__()
        r = torch.as_tensor(r_grid, dtype=torch.float64)
        self.register_buffer("r", r)
        self.z = float(z)
        self.n = float(n)
        self.a = float(z + n)
        x = 2.0 * r / r[-1].clamp(min=1.0e-12) - 1.0
        features = torch.stack(
            [
                x,
                torch.full_like(r, self.z / self.a),
                torch.full_like(r, self.n / self.a),
                torch.full_like(r, self.a / 208.0),
            ],
            dim=-1,
        )
        envelope = torch.exp(-(r / (0.68 * self.a ** (1.0 / 3.0) + 1.2)) ** 4)
        tail = (1.0 - (r / r[-1].clamp(min=1.0e-12)) ** 2).clamp(min=0.0)
        self.register_buffer("features", features)
        self.register_buffer("density_envelope", envelope)
        self.register_buffer("field_envelope", tail)
        self.net = nn.Sequential(
            nn.Linear(4, hidden, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(hidden, hidden, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(hidden, hidden, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(hidden, 8, dtype=torch.float64),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.05)
            self.net[-1].bias.zero_()

    def forward(self) -> dict[str, torch.Tensor]:
        raw = self.net(self.features).T
        rho_scale = 0.20
        rho_v_n = rho_scale * self.density_envelope * torch.nn.functional.softplus(raw[0])
        rho_v_p = rho_scale * self.density_envelope * torch.nn.functional.softplus(raw[1])
        scalar_ratio_n = torch.sigmoid(raw[2])
        scalar_ratio_p = torch.sigmoid(raw[3])
        rho_s_n = rho_v_n * scalar_ratio_n
        rho_s_p = rho_v_p * scalar_ratio_p

        # Fields are in fm^-1-like Fortran potential units. Sign priors follow
        # the RMF Hartree solution: sigma attractive, omega repulsive.
        sigma = -0.12 * self.field_envelope * torch.tanh(raw[4])
        omega = 0.12 * self.field_envelope * torch.tanh(raw[5])
        rho = 0.04 * self.field_envelope * torch.tanh(raw[6])
        coul = 0.06 * self.field_envelope * torch.nn.functional.softplus(raw[7])
        return {
            "rho_s_n": rho_s_n,
            "rho_s_p": rho_s_p,
            "rho_v_n": rho_v_n,
            "rho_v_p": rho_v_p,
            "sigma": sigma,
            "omega": omega,
            "rho": rho,
            "coul": coul,
        }


class StrictSphericalRMFNet(nn.Module):
    """SiLU ansatz for occupied radial Dirac orbitals and RMF fields."""

    def __init__(self, r_grid, z: int, n: int, orbital_data: dict, hidden: int = 96):
        super().__init__()
        r = torch.as_tensor(r_grid, dtype=torch.float64)
        self.register_buffer("r", r)
        self.a = float(z + n)
        x = 2.0 * r / r[-1].clamp(min=1.0e-12) - 1.0
        field_features = torch.stack(
            [
                x,
                torch.full_like(r, z / self.a),
                torch.full_like(r, n / self.a),
                torch.full_like(r, self.a / 208.0),
            ],
            dim=-1,
        )
        self.register_buffer("field_features", field_features)
        self.register_buffer("field_envelope", (1.0 - (r / r[-1]) ** 2).clamp(min=0.0))
        self.register_buffer("initial_sigma", torch.as_tensor(orbital_data["sigma"], dtype=torch.float64))
        self.register_buffer("initial_omega", torch.as_tensor(orbital_data["omega"], dtype=torch.float64))
        self.register_buffer("initial_rho", torch.as_tensor(orbital_data["rho"], dtype=torch.float64))
        self.register_buffer("initial_coul", torch.as_tensor(orbital_data["coul"], dtype=torch.float64))

        species = torch.as_tensor(orbital_data["species_sign"], dtype=torch.float64)
        kappa = torch.as_tensor(orbital_data["kappa"], dtype=torch.float64)
        principal = torch.as_tensor(orbital_data["principal"], dtype=torch.float64)
        norb = species.numel()
        orbital_features = torch.stack(
            [
                x.expand(norb, -1),
                (kappa / 8.0)[:, None].expand(-1, r.numel()),
                species[:, None].expand(-1, r.numel()),
                (principal / 8.0)[:, None].expand(-1, r.numel()),
                torch.full((norb, r.numel()), self.a / 208.0, dtype=torch.float64),
            ],
            dim=-1,
        )
        self.register_buffer("orbital_features", orbital_features)
        self.register_buffer("initial_G", torch.as_tensor(orbital_data["G"], dtype=torch.float64))
        self.register_buffer("initial_F", torch.as_tensor(orbital_data["F"], dtype=torch.float64))
        self.register_buffer("kappa", kappa)
        self.register_buffer("species_sign", species)
        self.register_buffer("occupancy", torch.as_tensor(orbital_data["occupancy"], dtype=torch.float64))
        self.register_buffer("degeneracy", torch.as_tensor(orbital_data["degeneracy"], dtype=torch.float64))
        self.epsilon = nn.Parameter(torch.as_tensor(orbital_data["energy"], dtype=torch.float64))

        self.field_nets = nn.ModuleDict({name: self._make_mlp(4, 1, hidden) for name in ("sigma", "omega", "rho", "coul")})
        self.orbital_net = nn.Sequential(
            nn.Linear(5, hidden, dtype=torch.float64), nn.SiLU(),
            nn.Linear(hidden, hidden, dtype=torch.float64), nn.SiLU(),
            nn.Linear(hidden, hidden, dtype=torch.float64), nn.SiLU(),
            nn.Linear(hidden, 2, dtype=torch.float64),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for net in (*self.field_nets.values(), self.orbital_net):
            for module in net:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
            with torch.no_grad():
                net[-1].weight.mul_(0.02)
                net[-1].bias.zero_()

    @staticmethod
    def _make_mlp(inputs: int, outputs: int, hidden: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(inputs, hidden, dtype=torch.float64), nn.SiLU(),
            nn.Linear(hidden, hidden, dtype=torch.float64), nn.SiLU(),
            nn.Linear(hidden, hidden, dtype=torch.float64), nn.SiLU(),
            nn.Linear(hidden, outputs, dtype=torch.float64),
        )

    def forward(self) -> dict[str, torch.Tensor]:
        field_raw = {name: net(self.field_features).squeeze(-1) for name, net in self.field_nets.items()}
        orbital_raw = self.orbital_net(self.orbital_features).permute(0, 2, 1)
        endpoint = (self.r / self.r[-1]) * (1.0 - self.r / self.r[-1])
        G = self.initial_G + endpoint * orbital_raw[:, 0]
        F = self.initial_F + 0.25 * endpoint * orbital_raw[:, 1]

        # The strict functional performs high-order hard normalization. Output
        # transforms below are physical parameterizations; hidden activations
        # are SiLU and no SCF field is present in the independent initialization.
        sigma = self.initial_sigma + 0.25 * self.field_envelope * torch.tanh(field_raw["sigma"])
        omega = self.initial_omega + 0.25 * self.field_envelope * torch.tanh(field_raw["omega"])
        rho = self.initial_rho + 0.08 * self.field_envelope * torch.tanh(field_raw["rho"])
        coul = self.initial_coul + 0.08 * self.field_envelope * torch.tanh(field_raw["coul"])
        return {
            "G": G,
            "F": F,
            "epsilon": self.epsilon,
            "kappa": self.kappa,
            "species_sign": self.species_sign,
            "occupancy": self.occupancy,
            "degeneracy": self.degeneracy,
            "sigma": sigma,
            "omega": omega,
            "rho": rho,
            "coul": coul,
        }
