from __future__ import annotations

import math

import torch
from torch import nn

from dpl_rhf.functionals.pkdd_action import PKDDParameters


PHYSICAL_COMPONENT_NAMES = ("S", "V0", "V3", "VC")


def compose_hamiltonian(components: torch.Tensor, params: PKDDParameters | None = None) -> torch.Tensor:
    """Map independent RMF self-energies to shifted neutron/proton Dirac channels."""
    p = params or PKDDParameters()
    scalar, vector, isovector, coulomb = components
    return torch.stack([
        vector + scalar + isovector,
        vector - scalar + isovector - 2.0 * p.mass_n / p.hbar_c,
        vector + scalar - isovector + coulomb,
        vector - scalar - isovector + coulomb - 2.0 * p.mass_p / p.hbar_c,
    ])


def decompose_hamiltonian(
    stack: torch.Tensor, params: PKDDParameters | None = None, coulomb: torch.Tensor | None = None
) -> torch.Tensor:
    """Project shifted Dirac channels onto the four independent RMF coordinates."""
    p = params or PKDDParameters()
    scalar_n = 0.5 * (stack[0] - stack[1] - 2.0 * p.mass_n / p.hbar_c)
    scalar_p = 0.5 * (stack[2] - stack[3] - 2.0 * p.mass_p / p.hbar_c)
    vector_n = 0.5 * (stack[0] + stack[1] + 2.0 * p.mass_n / p.hbar_c)
    vector_p = 0.5 * (stack[2] + stack[3] + 2.0 * p.mass_p / p.hbar_c)
    scalar = 0.5 * (scalar_n + scalar_p)
    # vector_n=V0+V3 and vector_p=V0-V3+VC. VC is fixed by the
    # physical representation and is supplied separately when exact inversion
    # is required. For an arbitrary reconstructed H, use the proton-only
    # difference after taking the isoscalar average; this minimum-norm inverse
    # is used only for PRL residual coordinates.
    vc = torch.zeros_like(vector_n) if coulomb is None else coulomb
    vector = 0.5 * (vector_n + vector_p - vc)
    isovector = 0.5 * (vector_n - vector_p + vc)
    return torch.stack([scalar, vector, isovector, vc])


def physical_component_gradient(grad_h: torch.Tensor) -> torch.Tensor:
    """Pull dE/dH back through the exact RMF channel assembly."""
    return torch.stack([
        grad_h[0] - grad_h[1] + grad_h[2] - grad_h[3],
        grad_h.sum(dim=0),
        grad_h[0] + grad_h[1] - grad_h[2] - grad_h[3],
        grad_h[2] + grad_h[3],
    ])


class PhysicalHamiltonianRepresentation(nn.Module):
    """Hard spherical RMF map shared by direct and neural variation."""

    def __init__(
        self,
        r_grid,
        z: int,
        n: int,
        initial_hamiltonian=None,
        params: PKDDParameters | None = None,
    ):
        super().__init__()
        r = torch.as_tensor(r_grid, dtype=torch.float64)
        a = float(z + n)
        t = (r / r[-1]).square()
        self.register_buffer("r", r)
        self.register_buffer("radial_feature", 2.0 * t - 1.0)
        self.z = int(z)
        self.a = int(a)

        p = params or PKDDParameters()
        self.params = p
        # Outside the source, each massive field is C exp(-mu r)/r. The
        # correction envelope has zero value and slope at R, while the three
        # tail bases obey phi'(R) + (mu + 1/R) phi(R) = 0 exactly. Using r^2
        # in both factors retains regularity at the origin.
        massive_envelope = (1.0 - t.pow(8)).square()
        massive_mu = torch.tensor(
            (p.m_sigma, p.m_omega, p.m_rho), dtype=torch.float64
        ) / p.hbar_c
        massive_tail_exponent = 0.5 * (massive_mu * r[-1] + 1.0)
        massive_tail_basis = torch.exp(
            -massive_tail_exponent[:, None] * t[None, :]
        )
        self.register_buffer("massive_envelope", massive_envelope)
        self.register_buffer("massive_mu", massive_mu)
        self.register_buffer("massive_tail_basis", massive_tail_basis)
        self.register_buffer("coulomb_envelope", (1.0 - t).square())
        charge_radius = 1.2496 * a ** (1.0 / 3.0)
        alpha_z = z / p.alpha_inv
        coulomb_base = torch.where(
            r < charge_radius,
            alpha_z * (3.0 / (2.0 * charge_radius) - r.square() / (2.0 * charge_radius**3)),
            alpha_z / r,
        )
        self.register_buffer("coulomb_base", coulomb_base)
        if initial_hamiltonian is None:
            from dpl_rhf.functionals.pkdd_rmf import PKDDRMFFunctionalSpec

            initial_hamiltonian = PKDDRMFFunctionalSpec(
                z, n, r.cpu().numpy(), params=p
            ).initial_hamiltonian()
        initial_stack = torch.as_tensor(initial_hamiltonian, dtype=torch.float64)
        initial_components = decompose_hamiltonian(
            initial_stack, params=p, coulomb=coulomb_base
        )
        # The initial Woods-Saxon profile is a starting point only. Its tail is
        # projected onto the homogeneous member of the Yukawa boundary family;
        # training remains free to vary the exterior amplitude.
        initial_components = initial_components.clone()
        initial_components[:3] *= self.massive_envelope
        initial_components[3] = coulomb_base
        self.register_buffer("initial_components", initial_components)

    def constrain(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.shape != (4, self.r.numel()):
            raise ValueError(f"expected raw physical profiles (4,{self.r.numel()}), got {tuple(raw.shape)}")
        # These factors impose only exact spherical boundary conditions.
        # A Woods-Saxon profile and the global (N-Z)/A factor are initial
        # guesses, not consequences of the RMF Euler-Lagrange equations.
        tail_amplitude = raw[:3, -1, None]
        massive = (
            self.massive_envelope * raw[:3]
            + tail_amplitude * self.massive_tail_basis
        )
        scalar, vector, isovector = massive
        coulomb = self.coulomb_base + self.coulomb_envelope * raw[3]
        correction = torch.stack([scalar, vector, isovector, coulomb - self.coulomb_base])
        return self.initial_components + correction

    def constraint_diagnostics(self, components: torch.Tensor) -> dict[str, float]:
        p = self.params
        h = float(self.r[2] - self.r[1])
        first_interval_slopes = (components[:, 2] - components[:, 1]) / h
        massive_boundary = components[:3, -1]
        massive_derivative = (components[:3, -1] - components[:3, -2]) / h
        massive_robin = massive_derivative + (
            self.massive_mu + 1.0 / self.r[-1]
        ) * massive_boundary
        # The correction and its derivative vanish at R, so the analytic
        # Coulomb base obeys the exterior Robin condition exactly.
        coul_derivative = (components[3, -1] - components[3, -2]) / h
        discrete_robin = coul_derivative + components[3, -1] / self.r[-1]
        stack = compose_hamiltonian(components, p)
        scalar_n = 0.5 * (stack[0] - stack[1] - 2.0 * p.mass_n / p.hbar_c)
        scalar_p = 0.5 * (stack[2] - stack[3] - 2.0 * p.mass_p / p.hbar_c)
        return {
            "analytic_center_derivative_fm_minus_2": 0.0,
            "max_first_interval_slope_fm_minus_2": float(first_interval_slopes.abs().max().detach().cpu()),
            "max_massive_boundary_fm_minus_1": float(massive_boundary.abs().max().detach().cpu()),
            "analytic_massive_yukawa_robin_fm_minus_2": 0.0,
            "max_discrete_massive_yukawa_robin_fm_minus_2": float(
                massive_robin.abs().max().detach().cpu()
            ),
            "analytic_coulomb_robin_fm_minus_2": 0.0,
            "discrete_coulomb_robin_fm_minus_2": float(discrete_robin.detach().cpu()),
            "max_scalar_identity_mev": float(((scalar_n - scalar_p).abs().max() * p.hbar_c).detach().cpu()),
        }


class LocalHamiltonianNet(PhysicalHamiltonianRepresentation):
    """DeepH-E3-style Hamiltonian graph network in the spherical 0e sector.

    The PRL network alternates vertex and edge message-passing updates before
    constructing Hamiltonian blocks.  Spherical RMF local self-energies carry
    only the even scalar irrep 0e; in this sector an E(3)-equivariant tensor
    product reduces exactly to scalar multiplication.  The radial graph uses
    the same three-neighbor reach as the seven-point ADF Dirac operator.
    """

    def __init__(
        self,
        r_grid,
        z: int,
        n: int,
        initial_hamiltonian,
        hidden: int = 64,
        depth: int = 3,
        radial_basis_size: int = 16,
        message_neighbors: int = 3,
        params: PKDDParameters | None = None,
    ):
        super().__init__(r_grid, z, n, initial_hamiltonian, params=params)
        if depth < 1:
            raise ValueError("network depth must be positive")
        if radial_basis_size < 4:
            raise ValueError("radial basis size must be at least four")
        if message_neighbors < 1:
            raise ValueError("message neighborhood must be positive")
        scaled_radius = self.r / self.r[-1]
        frequencies = torch.arange(
            1, radial_basis_size + 1, dtype=torch.float64
        )
        # The spherical-Bessel l=0 radial embedding is an input representation,
        # not a trainable variational basis.  The trainable map is the shared
        # nonlinear Hamiltonian network below.
        radial_embedding = torch.sinc(
            scaled_radius[:, None] * frequencies[None, :]
        )
        structural_features = torch.stack(
            [
                scaled_radius,
                scaled_radius.square(),
            ],
            dim=-1,
        )
        positive_features = torch.cat(
            [structural_features, radial_embedding], dim=-1
        )
        # Mirror the graph about the origin.  Identical even-parity features
        # and a reflection-symmetric edge set make every scalar output exactly
        # even, enforcing the regular spherical center condition without
        # multiplying away the interior variational freedom.
        node_features = torch.cat(
            [torch.flip(positive_features, dims=(0,)), positive_features], dim=0
        )
        graph_coordinate = torch.cat(
            [-torch.flip(self.r, dims=(0,)), self.r], dim=0
        )
        self.register_buffer("node_features", node_features)
        self.positive_node_offset = self.r.numel()

        targets = []
        sources = []
        size = graph_coordinate.numel()
        for target in range(size):
            lower = max(0, target - message_neighbors)
            upper = min(size, target + message_neighbors + 1)
            for source in range(lower, upper):
                targets.append(target)
                sources.append(source)
        edge_index = torch.tensor([targets, sources], dtype=torch.long)
        edge_distance = torch.abs(
            graph_coordinate[edge_index[0]] - graph_coordinate[edge_index[1]]
        )
        cutoff = float(
            self.r[min(message_neighbors, self.r.numel() - 1)] - self.r[0]
        )
        centers = torch.linspace(0.0, cutoff, radial_basis_size, dtype=torch.float64)
        spacing = max(float(centers[1] - centers[0]), 1.0e-12)
        edge_embedding = torch.exp(
            -0.5 * ((edge_distance[:, None] - centers[None, :]) / spacing).square()
        )
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_embedding", edge_embedding)
        degree = torch.bincount(edge_index[0], minlength=size).to(torch.float64)
        self.register_buffer("node_degree", degree.clamp_min(1.0)[:, None])

        self.activation_name = "silu"
        self.node_embedding = nn.Linear(
            radial_basis_size + structural_features.shape[-1],
            hidden,
            dtype=torch.float64,
        )
        self.edge_embedding_layer = nn.Linear(
            radial_basis_size, hidden, dtype=torch.float64
        )
        self.message_blocks = nn.ModuleList(
            [ScalarEquivariantMessageBlock(hidden) for _ in range(depth)]
        )
        self.output_heads = nn.ModuleList(
            [nn.Linear(hidden, 1, dtype=torch.float64) for _ in range(4)]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.node_embedding.weight)
        nn.init.zeros_(self.node_embedding.bias)
        nn.init.xavier_uniform_(self.edge_embedding_layer.weight)
        nn.init.zeros_(self.edge_embedding_layer.bias)
        for block in self.message_blocks:
            block.reset_parameters()
        # Zero heads make the initial Hamiltonian exactly equal to the analytic
        # seed without subtracting a second, parameter-dependent network pass.
        for head in self.output_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def full_latent(self) -> torch.Tensor:
        node = torch.nn.functional.silu(self.node_embedding(self.node_features))
        edge = torch.nn.functional.silu(
            self.edge_embedding_layer(self.edge_embedding)
        )
        for block in self.message_blocks:
            node, edge = block(
                node, edge, self.edge_index, self.node_degree
            )
        return node

    def latent(self) -> torch.Tensor:
        return self.full_latent()[self.positive_node_offset:]

    def components(self) -> torch.Tensor:
        latent = self.latent()
        raw = torch.stack([head(latent).squeeze(-1) for head in self.output_heads])
        return self.constrain(raw)

    def forward(self) -> torch.Tensor:
        return compose_hamiltonian(self.components(), self.params)


class ScalarEquivariantMessageBlock(nn.Module):
    """One vertex update and one edge update for even scalar irreps."""

    def __init__(self, hidden: int):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(3 * hidden, hidden, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(hidden, hidden, dtype=torch.float64),
            nn.SiLU(),
        )
        self.node_update = nn.Sequential(
            nn.Linear(2 * hidden, hidden, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(hidden, hidden, dtype=torch.float64),
        )
        self.edge_update = nn.Sequential(
            nn.Linear(3 * hidden, hidden, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(hidden, hidden, dtype=torch.float64),
        )
        self.node_norm = nn.LayerNorm(hidden, dtype=torch.float64)
        self.edge_norm = nn.LayerNorm(hidden, dtype=torch.float64)
        self.activation = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        node: torch.Tensor,
        edge: torch.Tensor,
        edge_index: torch.Tensor,
        node_degree: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target, source = edge_index
        edge_delta = self.edge_update(
            torch.cat([edge, node[target], node[source]], dim=-1)
        )
        edge = self.activation(
            self.edge_norm(edge + edge_delta / math.sqrt(2.0))
        )
        message = self.message(
            torch.cat([node[target], node[source], edge], dim=-1)
        )
        aggregate = torch.zeros_like(node)
        aggregate.index_add_(0, target, message)
        aggregate = aggregate / node_degree
        node_delta = self.node_update(torch.cat([node, aggregate], dim=-1))
        node = self.activation(
            self.node_norm(node + node_delta / math.sqrt(2.0))
        )
        return node, edge


class DirectHamiltonianParameterization(PhysicalHamiltonianRepresentation):
    """Network-free Chebyshev coordinates for validating variational RMF."""

    def __init__(
        self,
        r_grid,
        z: int,
        n: int,
        order: int = 32,
        params: PKDDParameters | None = None,
    ):
        super().__init__(r_grid, z, n, params=params)
        if order < 4:
            raise ValueError("direct physical basis needs order >= 4")
        theta = torch.acos(self.radial_feature.clamp(-1.0, 1.0))
        basis = torch.stack([torch.cos(k * theta) for k in range(order)], dim=0)
        self.register_buffer("basis", basis)
        coefficients = torch.zeros((4, order), dtype=torch.float64)
        self.coefficients = nn.Parameter(coefficients)

    def components(self) -> torch.Tensor:
        return self.constrain(self.coefficients @ self.basis)

    def forward(self) -> torch.Tensor:
        return compose_hamiltonian(self.components(), self.params)
