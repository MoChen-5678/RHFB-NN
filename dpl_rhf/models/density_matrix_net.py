from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DensityBlockSpec:
    species_sign: float
    kappa: int
    degeneracy: float
    principals: tuple[int, ...]
    occupations: torch.Tensor
    initial_factors: torch.Tensor
    free_hamiltonian: torch.Tensor
    tail_energies_mev: torch.Tensor


class _DensityBlock(nn.Module):
    def __init__(self, spec: DensityBlockSpec):
        super().__init__()
        self.species_sign = float(spec.species_sign)
        self.kappa = int(spec.kappa)
        self.degeneracy = float(spec.degeneracy)
        self.principals = tuple(int(value) for value in spec.principals)
        self.register_buffer("occupations", spec.occupations.clone())
        self.register_buffer("initial_factors", spec.initial_factors.clone())
        self.register_buffer("free_hamiltonian", spec.free_hamiltonian.clone())
        self.register_buffer("tail_energies_mev", spec.tail_energies_mev.clone())
        split = spec.initial_factors.shape[0] // 2
        radius = torch.arange(
            1,
            split + 1,
            dtype=spec.initial_factors.dtype,
            device=spec.initial_factors.device,
        )
        dilation_parts = []
        for component in (
            spec.initial_factors[:split],
            spec.initial_factors[split:],
        ):
            derivative = torch.gradient(component, dim=0, edge_order=2)[0]
            dilation_parts.append(-0.5 * component - radius[:, None] * derivative)
        self.register_buffer("dilation_tangent", torch.cat(dilation_parts, dim=0))
        self.raw_dilation = nn.Parameter(
            torch.zeros(len(self.principals), dtype=spec.initial_factors.dtype)
        )

    @property
    def name(self) -> str:
        species = "n" if self.species_sign > 0.0 else "p"
        return f"{species}_kappa_{self.kappa:+d}"

    def project(
        self,
        raw_coordinate: torch.Tensor,
        positive_projector: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Both tensors live in the fixed, complete interior Dirac grid space.
        # QR is a Stiefel retraction only; it never changes the primitive basis.
        # Radial unitary dilation u_s(r)=s^(-1/2)u(r/s) supplies the global
        # size direction that a small pointwise correction represents poorly.
        # SiLU is the only nonlinear map; QR restores exact orthonormality.
        dilation = 0.25 * torch.nn.functional.silu(self.raw_dilation)
        candidate = (
            self.initial_factors
            + raw_coordinate
            + self.dilation_tangent * dilation[None, :]
        )
        if positive_projector is not None:
            # No-sea constraint of the RHFB solver: restrict the factors to the
            # positive-energy subspace of the (detached) Dirac Hamiltonian.
            candidate = positive_projector @ candidate
        factors, _ = torch.linalg.qr(candidate, mode="reduced")
        return {
            "species_sign": factors.new_tensor(self.species_sign),
            "kappa": torch.tensor(self.kappa, device=factors.device, dtype=torch.int64),
            "degeneracy": factors.new_tensor(self.degeneracy),
            "principals": torch.tensor(self.principals, device=factors.device, dtype=torch.int64),
            "occupations": self.occupations,
            "coefficients": factors,
            "factors": factors,
            "free_hamiltonian": self.free_hamiltonian,
            "tail_energies_mev": self.tail_energies_mev,
        }


class DensityMatrixNet(nn.Module):
    """Shared SiLU network producing full-space density projectors.

    The trainable output is a low-rank factor of the one-body density matrix,
    not a set of independently optimized eigenstates.  The primitive Dirac
    grid is fixed for the entire run. QR enforces orthonormality, while the
    no-sea branch is selected by the inverse-Hamiltonian physics loss.
    """

    def __init__(
        self,
        r_grid: torch.Tensor,
        block_specs: list[DensityBlockSpec],
        *,
        hidden: int = 64,
        depth: int = 3,
        radial_basis_size: int = 16,
        correction_scale: float = 1.0e-2,
    ):
        super().__init__()
        if hidden < 8 or depth < 1 or radial_basis_size < 4:
            raise ValueError("density network requires hidden>=8, depth>=1, radial_basis_size>=4")
        if not block_specs:
            raise ValueError("at least one occupied density block is required")
        r = torch.as_tensor(r_grid, dtype=torch.float64)
        scaled = r / r[-1]
        frequencies = torch.arange(
            1, radial_basis_size + 1, dtype=torch.float64, device=r.device
        )
        features = torch.cat(
            [
                scaled[:, None],
                scaled.square()[:, None],
                torch.sinc(scaled[:, None] * frequencies[None, :]),
            ],
            dim=1,
        )
        self.register_buffer("radial_features", features)
        self.register_buffer("scaled_radius", scaled)
        self.correction_scale = float(correction_scale)
        layers: list[nn.Module] = [nn.Linear(features.shape[1], hidden, dtype=torch.float64), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden, dtype=torch.float64), nn.SiLU()])
        self.radial_encoder = nn.Sequential(*layers)
        self.species_embedding = nn.Embedding(2, hidden, dtype=torch.float64)
        kappas = [int(spec.kappa) for spec in block_specs]
        kappa_min = min(-7, min(kappas))
        kappa_max = max(7, max(kappas))
        self.kappa_offset = -kappa_min
        self.kappa_embedding = nn.Embedding(
            kappa_max - kappa_min + 1, hidden, dtype=torch.float64
        )
        max_principal = max(
            int(principal)
            for spec in block_specs
            for principal in spec.principals
        )
        self.principal_embedding = nn.Embedding(
            max(8, max_principal + 1), hidden, dtype=torch.float64
        )
        self.condition_activation = nn.SiLU()
        self.blocks = nn.ModuleList([_DensityBlock(spec) for spec in block_specs])
        # Stored no-sea projectors keyed by block name (RHF trainer); not part
        # of the state dict.  An explicit forward() argument takes precedence.
        self.positive_projectors: dict[str, torch.Tensor] | None = None
        self.output_heads = nn.ModuleList(
            nn.Linear(hidden, 2, dtype=torch.float64)
            for spec in block_specs
            for _ in spec.principals
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.radial_encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.species_embedding.weight, std=0.05)
        nn.init.normal_(self.kappa_embedding.weight, std=0.05)
        nn.init.normal_(self.principal_embedding.weight, std=0.05)
        for head in self.output_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self, positive_projectors: list[torch.Tensor | None] | None = None
    ) -> list[dict[str, torch.Tensor]]:
        radial = self.radial_encoder(self.radial_features)
        results = []
        head_index = 0
        for block_index, block in enumerate(self.blocks):
            species_index = 0 if block.species_sign > 0.0 else 1
            contexts = []
            for principal in block.principals:
                context = (
                    self.species_embedding.weight[species_index]
                    + self.kappa_embedding.weight[block.kappa + self.kappa_offset]
                    + self.principal_embedding.weight[principal]
                )
                conditioned = self.condition_activation(radial + context[None, :])
                contexts.append(self.output_heads[head_index](conditioned))
                head_index += 1
            raw = torch.stack(contexts, dim=-1) * self.correction_scale
            # Exact radial regularity: G~r^(l_G+1), F~r^(l_F+1).  The finite
            # box factor fixes the endpoint; the analytic continuation outside
            # the box supplies the physical exponential tail.
            if block.kappa < 0:
                l_upper, l_lower = -block.kappa - 1, -block.kappa
            else:
                l_upper, l_lower = block.kappa, block.kappa - 1
            boundary = (1.0 - self.scaled_radius).clamp_min(0.0)
            upper_envelope = self.scaled_radius.pow(l_upper + 1) * boundary
            lower_envelope = self.scaled_radius.pow(l_lower + 1) * boundary
            raw = raw * torch.stack([upper_envelope, lower_envelope], dim=1)[:, :, None]
            raw_coordinate = torch.cat([raw[1:-1, 0], raw[1:-1, 1]], dim=0)
            if positive_projectors is not None:
                projector = positive_projectors[block_index]
            elif self.positive_projectors is not None:
                projector = self.positive_projectors[block.name]
            else:
                projector = None
            results.append(block.project(raw_coordinate, projector))
        return results

    def set_positive_projectors(
        self, projectors: dict[str, torch.Tensor] | None
    ) -> None:
        if projectors is None:
            self.positive_projectors = None
            return
        missing = {block.name for block in self.blocks}.difference(projectors)
        if missing:
            raise ValueError(f"missing no-sea projectors for {sorted(missing)}")
        self.positive_projectors = {
            name: value.detach() for name, value in projectors.items()
        }

    def density_blocks(self) -> list[dict[str, torch.Tensor]]:
        return self.forward()

    def refresh_positive_spaces(self, specs: list[DensityBlockSpec]) -> None:
        raise RuntimeError("subspace refresh is forbidden in the no-refresh solver")
