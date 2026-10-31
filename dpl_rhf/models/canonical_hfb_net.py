from __future__ import annotations

import torch
from torch import nn

from dpl_rhf.models.density_matrix_net import DensityBlockSpec, DensityMatrixNet


class CanonicalHFBNet(DensityMatrixNet):
    """Density network plus one occupation logit per canonical state.

    The radial factors come from the shared SiLU network exactly as in the RHF
    solver; the logits parameterize v^2 = sigmoid(logit + Fermi shift).
    """

    def __init__(self, r_grid: torch.Tensor, block_specs: list[DensityBlockSpec],
                 initial_logits: list[torch.Tensor], **kwargs):
        super().__init__(r_grid, block_specs, **kwargs)
        if len(initial_logits) != len(block_specs):
            raise ValueError("one logit vector is required per canonical block")
        self.occupation_logits = nn.ParameterList(
            nn.Parameter(logit.detach().clone().to(torch.float64)) for logit in initial_logits
        )

    def logits(self) -> list[torch.Tensor]:
        return list(self.occupation_logits)
