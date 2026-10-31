"""Full-primitive variational factors and canonical HFB coordinates.

The active generalized density is kept small (M by M); its primitive-space
normal and anomalous densities remain represented by Q factors.  No dense
primitive-space density or eigensystem is formed here.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .basis import AngularRadialBasis
from .manifold import GeneralizedDensity, generalized_density_matrix, canonical_generator
from .manifold import orthonormalize_factors
from .bogoliubov_blocking import BogoliubovState, slater_bogoliubov_frame, state_from_frame
from dpl_rhf.rhf.dirac import RHFRadialDiracOperator


class GeneralizedDensityNet(nn.Module):
    """SiLU model for full-space Q and an even or one-blocked HFB density.

    ``raw_coordinates`` supplies an unrestricted complex D by M tangent,
    while a shared smooth network adds corrections conditioned on radius,
    kappa, m, Dirac component, and active-state embeddings.  An analytic
    component-specific envelope enforces the regular origin and finite-box
    boundary.  Reduced complex QR returns an orthonormal Q.

    Canonical A/B coordinates act on a supplied pure Slater reference.  The
    small B seed creates a nonzero anomalous density whenever at least two
    eligible occupied or empty states exist; the blocked orbital is excluded
    from every seed pair.  The returned generalized density is active-space
    sized, so primitive rho/chi can be applied through :class:`DensityFactors`
    without constructing D by D matrices.
    """

    def __init__(
        self,
        basis: AngularRadialBasis,
        initial_q: torch.Tensor,
        initial_q_labels: tuple[str, ...] | list[str],
        reference: GeneralizedDensity,
        *,
        blocked_index: int | None = None,
        hidden: int = 32,
        depth: int = 2,
        radial_basis_size: int = 12,
        correction_scale: float = 1.0e-2,
        pairing_seed: float = 5.0e-3,
        channel_chunk_size: int = 8,
    ) -> None:
        super().__init__()
        if initial_q.dtype != torch.complex128 or initial_q.ndim != 2:
            raise TypeError("initial_q must be a complex128 (D,M) matrix")
        d, m = initial_q.shape
        if d != basis.dimension or d < m:
            raise ValueError("initial_q does not match the tall primitive basis")
        labels = tuple(str(value) for value in initial_q_labels)
        if len(labels) != m:
            raise ValueError("one initial-state label is required per Q column")
        if hidden < 8 or depth < 1 or radial_basis_size < 4:
            raise ValueError("network requires hidden>=8, depth>=1, radial_basis_size>=4")
        if correction_scale < 0.0 or pairing_seed < 0.0:
            raise ValueError("correction_scale and pairing_seed must be nonnegative")
        if channel_chunk_size < 1:
            raise ValueError("channel_chunk_size must be positive")
        if reference.rho.shape != (m, m) or reference.chi.shape != (m, m):
            raise ValueError("active reference dimension must match Q columns")
        if reference.rho.dtype != torch.complex128 or reference.chi.dtype != torch.complex128:
            raise TypeError("active reference densities must use complex128")
        if reference.matrix.dtype != torch.complex128 or reference.matrix.shape != (2 * m, 2 * m):
            raise TypeError("reference.matrix must be a complex128 (2M,2M) matrix")
        if reference.fermion_parity not in {-1, 1}:
            raise ValueError("reference must be an even or blocked odd pure state")
        if reference.fermion_parity == -1 and blocked_index is None:
            raise ValueError("an odd reference requires its blocked_index")
        if blocked_index is not None and not 0 <= int(blocked_index) < m:
            raise IndexError("blocked_index lies outside the active space")
        if not torch.allclose(reference.rho.mH, reference.rho, atol=1.0e-10, rtol=0.0):
            raise ValueError("reference rho must be Hermitian")
        if not torch.allclose(reference.chi.mT, -reference.chi, atol=1.0e-10, rtol=0.0):
            raise ValueError("reference chi must be antisymmetric")
        expected_reference = generalized_density_matrix(reference.rho, reference.chi)
        if not torch.allclose(reference.matrix, expected_reference, atol=1.0e-10, rtol=0.0):
            raise ValueError("reference matrix does not match rho/chi")
        if not torch.allclose(
            reference.matrix @ reference.matrix,
            reference.matrix,
            atol=1.0e-8,
            rtol=0.0,
        ):
            raise ValueError("reference generalized density must be pure and idempotent")
        identity = torch.eye(m, dtype=initial_q.dtype, device=initial_q.device)
        if not torch.allclose(initial_q.mH @ initial_q, identity, atol=1.0e-10, rtol=0.0):
            raise ValueError("initial_q columns must be orthonormal")

        self.basis = basis
        self.active_dimension = m
        self.blocked_index = None if blocked_index is None else int(blocked_index)
        self.initial_q_labels = labels
        self.correction_scale = float(correction_scale)
        self.pairing_seed = float(pairing_seed)
        self.channel_chunk_size = int(channel_chunk_size)
        self.kappa_max = basis.kappa_max
        self.hidden_size = int(hidden)
        self.network_depth = int(depth)
        self.radial_basis_size = int(radial_basis_size)

        self.register_buffer("initial_q", initial_q.detach().clone())
        self.register_buffer(
            "reference_rho", reference.rho.detach().to(initial_q.device).clone()
        )
        self.register_buffer(
            "reference_chi", reference.chi.detach().to(initial_q.device).clone()
        )
        self.register_buffer(
            "reference_matrix", reference.matrix.detach().to(initial_q.device).clone()
        )
        # Accumulated right Thouless transport in the fixed reference qp
        # frame.  It is state, not a trainable parameter: number retraction
        # changes this buffer while preserving Q and optimizer parameter IDs.
        self.register_buffer(
            "canonical_number_transport",
            torch.eye(2 * m, dtype=initial_q.dtype, device=initial_q.device),
        )
        self._number_transport_active = False
        self.reference_parity = int(reference.fermion_parity)

        radial_features = self._radial_features(
            basis.x,
            basis.radius_fm,
            radial_basis_size,
            device=initial_q.device,
        )
        self.register_buffer("radial_features", radial_features)
        self.radial_encoder = self._make_mlp(
            radial_features.shape[1], hidden, depth, device=initial_q.device
        )
        self.kappa_embedding = nn.Embedding(
            2 * basis.kappa_max, hidden, dtype=torch.float64, device=initial_q.device
        )
        self.m_embedding = nn.Embedding(
            2 * basis.kappa_max, hidden, dtype=torch.float64, device=initial_q.device
        )
        self.component_embedding = nn.Embedding(
            2, hidden, dtype=torch.float64, device=initial_q.device
        )
        self.state_embedding = nn.Embedding(
            m, hidden, dtype=torch.float64, device=initial_q.device
        )
        self.condition_activation = nn.SiLU()
        self.output_head = nn.Linear(
            hidden, 4, dtype=torch.float64, device=initial_q.device
        )

        # The unrestricted complex coordinates guarantee directions outside
        # span(initial_q); the shared network adds a smooth, channel-aware prior.
        self.raw_coordinates = nn.Parameter(torch.zeros_like(initial_q))
        self.raw_a = nn.Parameter(torch.zeros((m, m), dtype=initial_q.dtype, device=initial_q.device))
        self.raw_b = nn.Parameter(torch.zeros((m, m), dtype=initial_q.dtype, device=initial_q.device))

        envelope = self._make_envelope(basis).to(initial_q.device)
        self.register_buffer("envelope", envelope)
        self.reset_parameters()

    @staticmethod
    def _make_mlp(
        in_features: int,
        hidden: int,
        depth: int,
        *,
        device: torch.device,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Linear(in_features, hidden, dtype=torch.float64, device=device),
            nn.SiLU(),
        ]
        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.Linear(hidden, hidden, dtype=torch.float64, device=device),
                    nn.SiLU(),
                ]
            )
        return nn.Sequential(*layers)

    @staticmethod
    def _radial_features(
        radii: np.ndarray | torch.Tensor,
        radius_fm: float,
        basis_size: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        r = torch.as_tensor(radii, dtype=torch.float64, device=device)
        scaled = r / float(radius_fm)
        frequencies = torch.arange(1, basis_size + 1, dtype=torch.float64, device=device)
        phase = torch.pi * scaled[:, None] * frequencies[None, :]
        return torch.cat(
            [scaled[:, None], scaled.square()[:, None], torch.sin(phase)], dim=1
        )

    @staticmethod
    def _make_envelope(basis: AngularRadialBasis) -> torch.Tensor:
        scaled = torch.as_tensor(basis.x / basis.radius_fm, dtype=torch.float64)
        boundary = (1.0 - scaled).clamp_min(0.0)
        parts = []
        for channel in basis.channels:
            l_upper, l_lower = RHFRadialDiracOperator.component_angular_momenta(
                channel.kappa
            )
            upper = scaled.pow(l_upper + 1) * boundary
            lower = scaled.pow(l_lower + 1) * boundary
            # Normalize each envelope by its maximum to keep high-l channels
            # numerically usable without changing the origin or endpoint zeros.
            upper = upper / upper.max().clamp_min(torch.finfo(scaled.dtype).tiny)
            lower = lower / lower.max().clamp_min(torch.finfo(scaled.dtype).tiny)
            parts.append(torch.cat((upper, lower)))
        return torch.cat(parts)

    def reset_parameters(self) -> None:
        for module in self.radial_encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.kappa_embedding.weight, std=0.05)
        nn.init.normal_(self.m_embedding.weight, std=0.05)
        nn.init.normal_(self.component_embedding.weight, std=0.05)
        nn.init.normal_(self.state_embedding.weight, std=0.05)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
        self._initialize_pairing_seed()

    @staticmethod
    def _silu_complex(value: torch.Tensor) -> torch.Tensor:
        return torch.complex(F.silu(value.real), F.silu(value.imag))

    def _initialize_pairing_seed(self) -> None:
        occupations = self.reference_rho.diagonal().real
        pure = bool(
            torch.all(
                (torch.abs(occupations) < 1.0e-8)
                | (torch.abs(occupations - 1.0) < 1.0e-8)
            )
        )
        if pure:
            candidates = [
                index for index, value in enumerate(occupations) if float(value) < 0.5
            ]
            if len(candidates) < 2:
                candidates = [
                    index for index, value in enumerate(occupations) if float(value) > 0.5
                ]
        else:
            candidates = list(range(self.active_dimension))
        if self.blocked_index is not None:
            candidates = [index for index in candidates if index != self.blocked_index]
        if len(candidates) < 2 or self.pairing_seed == 0.0:
            return
        amplitude = self.pairing_seed / math.sqrt(len(candidates) * (len(candidates) - 1) / 2)
        with torch.no_grad():
            for position, i in enumerate(candidates):
                for j in candidates[position + 1 :]:
                    phase = ((i + 2 * j) % 7 - 3) / 7.0
                    value = amplitude * complex(1.0, phase)
                    self.raw_b[i, j] = value
                    self.raw_b[j, i] = -value

    def _network_correction(self) -> torch.Tensor:
        radial = self.radial_encoder(self.radial_features)
        states = self.state_embedding.weight
        blocks = []
        k_offset = self.kappa_max
        for channel_start in range(0, len(self.basis.channels), self.channel_chunk_size):
            channel_group = self.basis.channels[
                channel_start : channel_start + self.channel_chunk_size
            ]
            k_indices = torch.as_tensor(
                [
                    channel.kappa + k_offset - (1 if channel.kappa > 0 else 0)
                    for channel in channel_group
                ],
                dtype=torch.long,
                device=radial.device,
            )
            m_indices = torch.as_tensor(
                [(channel.m2 + 2 * self.kappa_max - 1) // 2 for channel in channel_group],
                dtype=torch.long,
                device=radial.device,
            )
            channel_context = self.kappa_embedding(k_indices) + self.m_embedding(m_indices)
            for local_index, channel in enumerate(channel_group):
                l_upper, l_lower = RHFRadialDiracOperator.component_angular_momenta(
                    channel.kappa
                )
                context = (
                    radial[:, None, :]
                    + channel_context[local_index][None, None, :]
                    + self.component_embedding.weight[0][None, None, :]
                    + states[None, :, :]
                )
                upper_hidden = self.condition_activation(context)
                upper_raw = self.output_head(upper_hidden)
                upper = torch.complex(upper_raw[..., 0], upper_raw[..., 1])
                upper_env = self._component_envelope(l_upper)
                upper = upper * upper_env[:, None]

                lower_context = (
                    radial[:, None, :]
                    + channel_context[local_index][None, None, :]
                    + self.component_embedding.weight[1][None, None, :]
                    + states[None, :, :]
                )
                lower_hidden = self.condition_activation(lower_context)
                lower_raw = self.output_head(lower_hidden)
                lower = torch.complex(lower_raw[..., 2], lower_raw[..., 3])
                lower_env = self._component_envelope(l_lower)
                lower = lower * lower_env[:, None]
                blocks.append(torch.cat((upper, lower), dim=0))
        return torch.cat(blocks, dim=0)

    def _component_envelope(self, ell: int) -> torch.Tensor:
        scaled = self.radial_features[:, 0]
        boundary = (1.0 - scaled).clamp_min(0.0)
        envelope = scaled.pow(ell + 1) * boundary
        return envelope / envelope.max().clamp_min(torch.finfo(envelope.dtype).tiny)

    def _canonical_transform(self) -> torch.Tensor:
        a_coordinate = self._silu_complex(self.raw_a)
        b_coordinate = self._silu_complex(self.raw_b)
        a = 0.5 * (a_coordinate - a_coordinate.mH)
        b = 0.5 * (b_coordinate - b_coordinate.mT)
        learned = torch.matrix_exp(canonical_generator(a, b))
        return self._apply_number_transport(learned)

    def _apply_number_transport(self, learned: torch.Tensor) -> torch.Tensor:
        """Apply the persisted right-frame correction to a learned transform."""
        if not self._number_transport_active:
            return learned
        # If W_ref is the constructor's Slater frame and C is the accumulated
        # right Thouless transport, then T_eff W_ref = T_learned W_ref C.
        # Conjugating C by W_ref lets the ordinary network forward map produce
        # the corrected U/V frame on its next call without refreshing Q.
        reference_frame = self._slater_reference_frame()
        return (
            learned
            @ reference_frame
            @ self.canonical_number_transport
            @ reference_frame.mH
        )

    def _canonical_density(self, transform=None) -> GeneralizedDensity:
        transform = self._canonical_transform() if transform is None else transform
        matrix = transform @ self.reference_matrix @ transform.mH
        m = self.active_dimension
        return GeneralizedDensity(matrix[:m, :m], matrix[:m, m:], matrix,
                                  self.reference_parity)

    def _slater_reference_frame(self) -> torch.Tensor:
        """Return W_ref for a zero/one Slater reference, without diagonalizing."""
        occupations_complex = self.reference_rho.diagonal()
        if not torch.allclose(
            self.reference_rho,
            torch.diag(occupations_complex),
            atol=1.0e-10,
            rtol=0.0,
        ) or not torch.allclose(
            self.reference_chi,
            torch.zeros_like(self.reference_chi),
            atol=1.0e-10,
            rtol=0.0,
        ):
            raise ValueError(
                "a complete reference Bogoliubov frame is currently constructed "
                "only for diagonal zero/one Slater references"
            )
        occupations = occupations_complex.real
        if not bool(torch.all((occupations == 0.0) | (occupations == 1.0))):
            raise ValueError("reference occupations must be zero/one Slater values")
        return slater_bogoliubov_frame(occupations)

    def forward(self) -> tuple[torch.Tensor, GeneralizedDensity]:
        """Return an orthonormal full-primitive Q and active generalized density."""

        correction = self.raw_coordinates + self._network_correction()
        candidate = self.initial_q + self.correction_scale * self.envelope[:, None] * correction
        q = orthonormalize_factors(candidate)
        return q, self._canonical_density()

    def forward_with_bogoliubov_frame(
        self,
    ) -> tuple[torch.Tensor, GeneralizedDensity, BogoliubovState]:
        """Return Q, density, and the complete active Bogoliubov frame.

        This additive interface uses the same canonical transform as
        :meth:`forward`.  It currently requires the network's supplied
        reference to be a diagonal zero/one Slater determinant, which is the
        reference chart used for explicit odd blocking.  For an odd reference,
        ``state.blocked_mode`` is the already-occupied reference Nambu column
        transported by the transform; no second blocking swap is performed.

        The frame preserves the full active U/V information needed by a later
        Hamiltonian qp analysis.  Its selected blocked column is only a state
        construction tag and does not certify instantaneous qp identity.
        """
        # Honor subclass orbital charts and their QR phase convention. A
        # second, independently chosen Q would change the primitive density
        # when the active rho/chi are not diagonal (including odd O17).
        q, density = self.forward()
        transform = self._canonical_transform()
        frame = transform @ self._slater_reference_frame()
        blocked_column = None
        if self.blocked_index is not None:
            m = self.active_dimension
            # W_ref already contains the Slater particle/hole interchange.
            # Its occupied plane is always the final M columns, independent
            # of whether the selected reference orbital was filled or empty.
            blocked_column = m + self.blocked_index
        state = state_from_frame(
            frame,
            fermion_parity=self.reference_parity,
            blocked_index=self.blocked_index,
            blocked_mode_column=blocked_column,
        )
        # Keep the established R-transform route as the reference result and
        # verify that the U/V construction has the same density algebraically.
        if not torch.allclose(state.density.matrix, density.matrix, atol=2.0e-9, rtol=0.0):
            raise RuntimeError("U/V frame density disagrees with canonical reference transform")
        return q, density, state

    def forward_with_blocked_mode(self):
        """Return the transported blocked Nambu vector, including A/B mixing.

        A Q column alone is not the blocked quasiparticle after a canonical
        transform. The occupied member of the reference Nambu pair is
        transported by exp(K), then lifted with diag(Q,Q*). It belongs to the
        occupied generalized-density subspace and has norm one, with no
        eigensystem or dense primitive Nambu matrix required.
        """
        q, gd, state = self.forward_with_bogoliubov_frame()
        mode = None
        if state.blocked_mode is not None:
            m = self.active_dimension
            mode = torch.cat(
                (q @ state.blocked_mode[:m], q.conj() @ state.blocked_mode[m:])
            )
        return q, gd, mode

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "format": "dpl-rhfb-generalized-density-net-v2",
            "canonical_number_transport": "right-thouless-reference-frame-v1",
            "active_dimension": self.active_dimension,
            "primitive_dimension": self.initial_q.shape[0],
            "labels": list(self.initial_q_labels),
            "blocked_index": self.blocked_index,
            "fermion_parity": self.reference_parity,
            "hidden": self.hidden_size,
            "depth": self.network_depth,
            "radial_basis_size": self.radial_basis_size,
            "channel_chunk_size": self.channel_chunk_size,
            "correction_scale": self.correction_scale,
            "pairing_seed": self.pairing_seed,
        }

    def get_extra_state(self) -> dict[str, Any]:
        return self.checkpoint_metadata()

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if state != self.checkpoint_metadata():
            raise RuntimeError("GeneralizedDensityNet checkpoint metadata mismatch")

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys,
        unexpected_keys, error_msgs,
    ):
        transport_key = prefix + "canonical_number_transport"
        if transport_key not in state_dict:
            raise RuntimeError(
                "checkpoint is missing canonical_number_transport; legacy network "
                "states require an explicit seed conversion"
            )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs,
        )
        if transport_key in state_dict:
            try:
                transport = state_dict[transport_key]
                if not isinstance(transport, torch.Tensor):
                    raise TypeError("canonical_number_transport must be a tensor")
                self._validate_number_transport(transport)
                self._number_transport_active = not torch.equal(
                    self.canonical_number_transport,
                    torch.eye(
                        self.canonical_number_transport.shape[0],
                        dtype=self.canonical_number_transport.dtype,
                        device=self.canonical_number_transport.device,
                    ),
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                error_msgs.append(f"invalid canonical_number_transport: {exc}")

    def _validate_number_transport(self, value: torch.Tensor) -> None:
        if value.dtype != torch.complex128 or value.shape != (2 * self.active_dimension,) * 2:
            raise ValueError("transport must be complex128 with shape (2M,2M)")
        state_from_frame(value, fermion_parity=1)
        # Canonical Bogoliubov transformations have two disconnected
        # components.  The determinant distinguishes the parity-preserving
        # component (det=+1) from a particle-hole swap (det=-1); checking the
        # phase of slogdet avoids forming a potentially ill-scaled determinant.
        determinant_phase, _logabsdet = torch.linalg.slogdet(value)
        if not torch.isclose(
            determinant_phase,
            torch.ones((), dtype=value.dtype, device=value.device),
            atol=1.0e-8,
            rtol=0.0,
        ):
            raise ValueError(
                "transport must remain in the parity-preserving canonical "
                "component (determinant phase +1)"
            )


def upgrade_legacy_seed_state(
    model: GeneralizedDensityNet,
    legacy_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Explicitly convert one compatible v1 network *seed* to the v2 format.

    This helper is for deliberate seed conversion only; it must not be used to
    resume an optimizer/training run. It requires an exact v1 state-dict key
    set and exact architecture/physics metadata match, then adds the identity
    transport buffer and current v2 metadata. There is no permissive migration
    or inferred shape conversion here.
    """
    if not isinstance(model, GeneralizedDensityNet):
        raise TypeError("model must be a GeneralizedDensityNet")
    if not isinstance(legacy_state, Mapping):
        raise TypeError("legacy seed state must be a state-dict mapping")
    transport_key = "canonical_number_transport"
    if transport_key in legacy_state:
        raise ValueError("state already contains number transport; it is not a v1 seed")

    expected_state = model.state_dict()
    expected_keys = set(expected_state) - {transport_key}
    if set(legacy_state) != expected_keys:
        missing = sorted(expected_keys - set(legacy_state))
        extra = sorted(set(legacy_state) - expected_keys)
        raise ValueError(
            "legacy seed state keys do not exactly match this model "
            f"(missing={missing}, extra={extra})"
        )

    legacy_metadata = legacy_state.get("_extra_state")
    current_metadata = model.checkpoint_metadata()
    if not isinstance(legacy_metadata, Mapping):
        raise ValueError("legacy seed state lacks network metadata")
    current_format = current_metadata.get("format")
    if not isinstance(current_format, str) or not current_format.endswith("-v2"):
        raise ValueError("target network does not declare the v2 transport format")
    expected_legacy_format = current_format[:-3] + "-v1"
    if legacy_metadata.get("format") != expected_legacy_format:
        raise ValueError(
            "legacy seed format does not match this network family "
            f"(expected {expected_legacy_format!r})"
        )

    def comparable(metadata):
        return {
            key: value
            for key, value in metadata.items()
            if key not in {"format", "canonical_number_transport"}
        }

    if comparable(legacy_metadata) != comparable(current_metadata):
        raise ValueError("legacy seed metadata differs from the target model")

    upgraded = dict(legacy_state)
    upgraded[transport_key] = torch.eye(
        2 * model.active_dimension,
        dtype=model.canonical_number_transport.dtype,
        device=model.canonical_number_transport.device,
    )
    upgraded["_extra_state"] = dict(current_metadata)
    return upgraded
