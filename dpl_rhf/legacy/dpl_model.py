from __future__ import annotations

import torch
from torch import nn


class PotentialNet(nn.Module):
    """Small radial network for RMF local Dirac potentials.

    The network predicts a bounded correction around an initial Core-1204
    potential. Keeping the asymptotic tail anchored avoids unphysical box-edge
    drift during early DPL iterations.
    """

    def __init__(
        self,
        r_grid,
        base_potentials,
        z: int,
        n: int,
        hidden: int = 64,
        max_delta: float = 0.75,
    ):
        super().__init__()
        r = torch.as_tensor(r_grid, dtype=torch.float64)
        base = torch.as_tensor(base_potentials, dtype=torch.float64)
        if base.ndim != 2 or base.shape[0] != 4:
            raise ValueError("base_potentials must have shape (4, n_grid)")
        if base.shape[1] != r.numel():
            raise ValueError("base_potentials grid length does not match r_grid")

        a = float(z + n)
        features = torch.stack(
            [
                2.0 * r / r[-1].clamp(min=1e-12) - 1.0,
                torch.full_like(r, float(z) / a),
                torch.full_like(r, float(n) / a),
                torch.full_like(r, a / 300.0),
            ],
            dim=-1,
        )
        envelope = (1.0 - (r / r[-1].clamp(min=1e-12)) ** 2).clamp(min=0.0)

        self.register_buffer("features", features)
        self.register_buffer("base", base)
        self.register_buffer("envelope", envelope)
        self.max_delta = float(max_delta)
        self.net = nn.Sequential(
            nn.Linear(4, hidden, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(hidden, hidden, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(hidden, 4, dtype=torch.float64),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(1.0e-3)
            self.net[-1].bias.zero_()

    def forward(self) -> torch.Tensor:
        correction = self.max_delta * self.envelope.unsqueeze(0) * torch.tanh(self.net(self.features).T)
        return self.base + correction


class RMFPhysicsLoss(nn.Module):
    """Unsupervised RMF physics loss embedded as a model component.

    The target potential is not a label. It is rebuilt from the current
    network-generated wave functions/densities through the RMF equations in
    Core-1204. This module owns the physical residual used for optimization.
    """

    def __init__(
        self,
        reference_potentials,
        smooth_weight: float = 1.0e-3,
        boundary_weight: float = 1.0e-2,
        scale_floor: float = 0.05,
    ):
        super().__init__()
        base = torch.as_tensor(reference_potentials, dtype=torch.float64)
        if base.ndim != 2:
            raise ValueError("reference_potentials must have shape (channels, n_grid)")
        scale = base.abs().amax(dim=1, keepdim=True).clamp(min=scale_floor)
        self.register_buffer("scale", scale)
        self.smooth_weight = float(smooth_weight)
        self.boundary_weight = float(boundary_weight)

    def forward(self, predicted: torch.Tensor, rmf_rebuilt: torch.Tensor) -> dict[str, torch.Tensor]:
        if predicted.shape != rmf_rebuilt.shape:
            raise ValueError(f"predicted/target shape mismatch: {predicted.shape} vs {rmf_rebuilt.shape}")

        normalized = (predicted - rmf_rebuilt) / self.scale
        fixed_point_loss = (normalized ** 2).mean()
        smooth_loss = (((predicted[:, 2:] - 2.0 * predicted[:, 1:-1] + predicted[:, :-2]) / self.scale) ** 2).mean()
        boundary_loss = (
            (((predicted[:, :1] - rmf_rebuilt[:, :1]) / self.scale) ** 2).mean()
            + (((predicted[:, -1:] - rmf_rebuilt[:, -1:]) / self.scale) ** 2).mean()
        )
        total = fixed_point_loss + self.smooth_weight * smooth_loss + self.boundary_weight * boundary_loss
        return {
            "loss": total,
            "residual": fixed_point_loss.sqrt(),
            "fixed_point_loss": fixed_point_loss,
            "smooth_loss": smooth_loss,
            "boundary_loss": boundary_loss,
        }
