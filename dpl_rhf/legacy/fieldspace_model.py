from __future__ import annotations

import torch
from torch import nn


class FieldSpaceRMFNet(nn.Module):
    """Radial network for the four local Dirac potentials, not wave functions."""

    def __init__(self, r_grid, z: int, n: int, initial_stack, hidden: int = 96):
        super().__init__()
        r = torch.as_tensor(r_grid, dtype=torch.float64)
        a = float(z + n)
        x = 2.0 * r / r[-1] - 1.0
        features = torch.stack(
            [x, torch.full_like(r, z / a), torch.full_like(r, n / a), torch.full_like(r, a / 208.0)], dim=-1
        )
        self.register_buffer("features", features)
        self.register_buffer("initial_stack", torch.as_tensor(initial_stack, dtype=torch.float64))
        # A regular spherical scalar/vector potential may be finite at the
        # origin. Only the outer correction must vanish so that the nuclear
        # fields retain their asymptotic boundary condition (and the proton
        # channel retains the analytic Coulomb tail).
        self.register_buffer("envelope", (1.0 - r / r[-1]).square())
        self.networks = nn.ModuleList([self._mlp(hidden) for _ in range(4)])
        self.register_buffer("scales", torch.tensor([0.15, 0.30, 0.15, 0.30], dtype=torch.float64))
        self.reset_parameters()

    @staticmethod
    def _mlp(hidden: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(4, hidden, dtype=torch.float64), nn.SiLU(),
            nn.Linear(hidden, hidden, dtype=torch.float64), nn.SiLU(),
            nn.Linear(hidden, hidden, dtype=torch.float64), nn.SiLU(),
            nn.Linear(hidden, 1, dtype=torch.float64),
        )

    def reset_parameters(self) -> None:
        for network in self.networks:
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
            with torch.no_grad():
                network[-1].weight.mul_(0.01)

    def forward(self) -> torch.Tensor:
        corrections = torch.stack([network(self.features).squeeze(-1) for network in self.networks])
        return self.initial_stack + self.scales[:, None] * self.envelope[None, :] * torch.tanh(corrections)
