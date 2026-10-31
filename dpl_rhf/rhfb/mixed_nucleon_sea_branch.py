"""Fail-closed adapter for a conditional, full-Nambu nucleon/sea branch.

The finite-window mixed-sign map is an algebraic derivation under an explicit
complete-band separation assumption, motivated by DIRHB's mixed qp-sector
convention.  It is not an algorithm established by DIRHB for arbitrary
continuum fields.  This adapter does not pick a divider or certify that a
physical nucleon/Dirac-sea band gap exists.  A caller must provide a finite
band witness; an exact-H witness is hash-bound, and a self-consistent field
family witness still requires an external proof that this code cannot verify.
The result never claims physical no-sea certification.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import math
from typing import Any, Callable, Mapping

import torch

from .bdg_branch_filter import (
    BandBoundSource,
    BandEnclosureCertificate,
    BranchFilterResult,
    mixed_sign_bdg_projector,
)
from .full_branch_stationarity import nambu_hamiltonian
from .parity_diagnostic import number_parity_from_density


class NoSeaBranchEvidenceError(RuntimeError):
    """A mixed nucleon/sea branch cannot be built without band evidence."""


class EvidenceScope(str, Enum):
    EXACT_HAMILTONIAN = "exact_hamiltonian"
    SELFCONSISTENT_FAMILY = "selfconsistent_field_family"


@dataclass(frozen=True)
class BlockedModeEvidence:
    """Current-field positive-energy qp mode to block on an odd branch.

    ``mode`` must be a normalized eigenmode of the supplied full H_mu in the
    base branch's unoccupied particle sector. The construction swaps its
    occupation with its particle-hole partner. An exact-H instance is suitable
    only for that fixed matrix; changing fields requires a caller-supplied
    provider returning fresh, graph-connected evidence on each evaluation.
    A transported occupied-plane/tag mode is not sufficient.
    """

    mode: torch.Tensor
    scope: EvidenceScope
    proof_id: str
    hamiltonian_sha256: str | None = None
    field_family_id: str | None = None

    def fingerprint(self) -> str:
        payload = {
            "format": "blocked-mode-evidence-v1",
            "scope": self.scope.value,
            "proof_id": self.proof_id,
            "hamiltonian_sha256": self.hamiltonian_sha256,
            "field_family_id": self.field_family_id,
            "mode_sha256": _tensor_sha256(self.mode),
            "physical_identity_certified": False,
        }
        return hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()

    def validate_for(
        self,
        h_nambu: torch.Tensor,
        base_density: torch.Tensor,
        *,
        tolerance: float = 1.0e-8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.as_tensor(h_nambu)
        b = torch.as_tensor(self.mode, dtype=h.dtype, device=h.device)
        if self.scope is EvidenceScope.EXACT_HAMILTONIAN:
            if self.field_family_id is not None:
                raise ValueError("exact blocked-mode evidence cannot claim a field family")
            if self.hamiltonian_sha256 != hamiltonian_sha256(h):
                raise NoSeaBranchEvidenceError("blocked qp mode is bound to a different H_mu")
        elif self.scope is EvidenceScope.SELFCONSISTENT_FAMILY:
            if self.hamiltonian_sha256 is not None or not (
                isinstance(self.field_family_id, str) and self.field_family_id.strip()
            ):
                raise NoSeaBranchEvidenceError("blocked mode lacks a valid field-family scope")
            if h.requires_grad and not b.requires_grad:
                raise NoSeaBranchEvidenceError(
                    "field-family blocked mode must retain an autograd path when H_mu is trainable"
                )
            if h.requires_grad:
                # A requires_grad leaf unrelated to H is not a field response.
                # Use a fixed real scalar probe over every mode component;
                # zero derivatives are valid (e.g. an exact symmetry-protected
                # mode represented as b + 0*H), but an unused H is not.
                weights = torch.arange(
                    1, b.numel() + 1, dtype=torch.float64, device=b.device
                )
                probe = (b.real * weights).sum() + math.sqrt(2.0) * (b.imag * weights).sum()
                try:
                    connected, = torch.autograd.grad(
                        probe, h, retain_graph=True, allow_unused=True,
                    )
                except RuntimeError as error:
                    raise NoSeaBranchEvidenceError(
                        "unable to audit blocked-mode autograd connectivity to current H_mu"
                    ) from error
                if connected is None:
                    raise NoSeaBranchEvidenceError(
                        "field-family blocked mode is disconnected from current H_mu autograd graph"
                    )
        else:
            raise TypeError("blocked mode scope must be an EvidenceScope")
        if not isinstance(self.proof_id, str) or not self.proof_id.strip():
            raise NoSeaBranchEvidenceError("blocked mode requires a source/proof identifier")
        if h.dtype != torch.complex128 or h.ndim != 2 or h.shape[0] != h.shape[1]:
            raise TypeError("H_mu must be square complex128")
        if b.ndim != 1 or b.numel() != h.shape[0] or not bool(torch.isfinite(b).all()):
            raise ValueError("blocked mode must be a finite full-Nambu vector")
        if base_density.shape != h.shape:
            raise ValueError("base branch density must match H_mu")
        dim = h.shape[0]
        norm = torch.linalg.vector_norm(b)
        if abs(float(norm.detach()) - 1.0) > tolerance:
            raise ValueError("blocked qp mode must be normalized")
        partner = particle_hole_vector(b)
        overlap = torch.vdot(b, partner)
        if abs(complex(overlap.detach().cpu())) > tolerance:
            raise ValueError("blocked qp mode and its particle-hole partner must be orthogonal")
        energy = torch.vdot(b, h @ b).real
        eigen_residual = h @ b - energy * b
        h_scale = max(1.0, float(torch.linalg.matrix_norm(h).detach()))
        if float(torch.linalg.vector_norm(eigen_residual).detach()) > tolerance * h_scale:
            raise NoSeaBranchEvidenceError("blocked mode is not an invariant qp eigenmode of current H_mu")
        if float(energy.detach()) <= tolerance:
            raise NoSeaBranchEvidenceError("blocked mode must be a positive-energy qp mode")
        occupied = torch.vdot(b, base_density @ b).real
        partner_occupied = torch.vdot(partner, base_density @ partner).real
        if abs(float(occupied.detach())) > tolerance or abs(float(partner_occupied.detach()) - 1.0) > tolerance:
            raise NoSeaBranchEvidenceError(
                "blocked mode must lie in the base unoccupied plane and its PH partner in the occupied plane"
            )
        return b, partner


def particle_hole_vector(mode: torch.Tensor) -> torch.Tensor:
    """Canonical C mode for particle/hole Nambu ordering."""
    if mode.ndim != 1 or mode.numel() % 2:
        raise ValueError("Nambu mode must be a one-dimensional even-length vector")
    d = mode.numel() // 2
    return torch.cat((mode[d:].conj(), mode[:d].conj()))


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value)
    if tensor.dtype != torch.complex128:
        raise TypeError("evidence tensors must use complex128")
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class MixedNambuBandEvidence:
    """Caller-supplied proof witness for a finite mixed-sign BdG window.

    For ``EXACT_HAMILTONIAN``, ``hamiltonian_sha256`` must bind the certificate
    to the matrix being projected. For ``SELFCONSISTENT_FAMILY``, the cited
    proof must establish the enclosure for every Hamiltonian in the requested
    field family; a one-field spectrum or a free-form claim is insufficient.
    This value is evidence metadata, not a machine-verifiable proof object.
    """

    divider_mev: float
    certificate: BandEnclosureCertificate
    scope: EvidenceScope
    proof_id: str
    hamiltonian_sha256: str | None = None
    field_family_id: str | None = None
    physical_identity_certified: bool = False

    def to_payload(self) -> dict[str, Any]:
        cert = self.certificate
        return {
            "format": "mixed-nambu-band-evidence-v1",
            "divider_mev": float(self.divider_mev),
            "certificate": {
                **cert.__dict__,
                "source": cert.source.value,
            },
            "scope": self.scope.value,
            "proof_id": self.proof_id,
            "hamiltonian_sha256": self.hamiltonian_sha256,
            "field_family_id": self.field_family_id,
            "physical_identity_certified": False,
        }

    def fingerprint(self) -> str:
        canonical = json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MixedNambuBandEvidence":
        if not isinstance(payload, Mapping) or payload.get("format") != "mixed-nambu-band-evidence-v1":
            raise ValueError("unsupported mixed-Nambu evidence payload")
        expected = {"format", "divider_mev", "certificate", "scope", "proof_id",
                    "hamiltonian_sha256", "field_family_id", "physical_identity_certified"}
        if set(payload) != expected:
            raise ValueError("mixed-Nambu evidence payload fields do not match v1")
        raw_cert = payload["certificate"]
        if not isinstance(raw_cert, Mapping):
            raise TypeError("certificate payload must be a mapping")
        cert_fields = set(BandEnclosureCertificate.__dataclass_fields__)
        if set(raw_cert) != cert_fields:
            raise ValueError("band certificate fields do not match the current schema")
        cert_values = dict(raw_cert)
        cert_values["source"] = BandBoundSource(cert_values["source"])
        return cls(
            divider_mev=float(payload["divider_mev"]),
            certificate=BandEnclosureCertificate(**cert_values),
            scope=EvidenceScope(payload["scope"]),
            proof_id=str(payload["proof_id"]),
            hamiltonian_sha256=payload["hamiltonian_sha256"],
            field_family_id=payload["field_family_id"],
            physical_identity_certified=payload["physical_identity_certified"],
        )

    def validate_for(self, h: torch.Tensor) -> None:
        if not isinstance(self.scope, EvidenceScope):
            raise TypeError("scope must be an EvidenceScope")
        if not isinstance(self.certificate, BandEnclosureCertificate):
            raise TypeError("certificate must be a BandEnclosureCertificate")
        if not isinstance(self.proof_id, str) or not self.proof_id.strip():
            raise NoSeaBranchEvidenceError("an external band proof identifier is required")
        if self.physical_identity_certified is not False:
            raise ValueError("this adapter cannot certify Hamiltonian quasiparticle identity")
        self.certificate.validate(h.shape[0], self.divider_mev)
        if self.scope is EvidenceScope.EXACT_HAMILTONIAN:
            if self.field_family_id is not None:
                raise ValueError("exact-H evidence cannot also claim a field-family scope")
            expected = hamiltonian_sha256(h)
            if self.hamiltonian_sha256 != expected:
                raise NoSeaBranchEvidenceError(
                    "exact-H band evidence is absent or bound to a different matrix"
                )
        elif self.scope is EvidenceScope.SELFCONSISTENT_FAMILY:
            if self.hamiltonian_sha256 is not None:
                raise ValueError("field-family evidence must not be an exact-H hash")
            if not isinstance(self.field_family_id, str) or not self.field_family_id.strip():
                raise NoSeaBranchEvidenceError(
                    "a uniform self-consistent field-family proof identifier is required"
                )
            if self.certificate.source is BandBoundSource.INDEPENDENT_REFERENCE:
                raise NoSeaBranchEvidenceError(
                    "a single independent reference spectrum cannot certify a field family"
                )


@dataclass(frozen=True)
class MixedNucleonSeaDensity:
    generalized_density: torch.Tensor
    rho: torch.Tensor
    chi: torch.Tensor
    divider_mev: float
    evidence_scope: str
    evidence_id: str
    construction: BranchFilterResult
    h_nambu: torch.Tensor
    evidence_fingerprint: str
    physical_no_sea_certified: bool = False
    physical_identity_certified: bool = False

    def operator_action(self, probe: torch.Tensor) -> torch.Tensor:
        """Apply the constructed full-Nambu density to a vector/matrix probe."""
        v = torch.as_tensor(probe, dtype=self.generalized_density.dtype,
                            device=self.generalized_density.device)
        if v.ndim not in (1, 2) or v.shape[0] != self.generalized_density.shape[0]:
            raise ValueError("probe must have shape (2D,) or (2D,K)")
        return self.generalized_density @ v

    def commutator_action(self, probe: torch.Tensor) -> torch.Tensor:
        """Apply [H,R] to a probe as a fixed-field operator consistency check."""
        v = torch.as_tensor(probe, dtype=self.h_nambu.dtype, device=self.h_nambu.device)
        if v.ndim not in (1, 2) or v.shape[0] != self.h_nambu.shape[0]:
            raise ValueError("probe must have shape (2D,) or (2D,K)")
        r = self.generalized_density
        return self.h_nambu @ (r @ v) - r @ (self.h_nambu @ v)


def block_mixed_nucleon_sea_density(
    branch: MixedNucleonSeaDensity,
    evidence: BlockedModeEvidence,
    *,
    tolerance: float = 1.0e-8,
) -> MixedNucleonSeaDensity:
    """Swap a verified positive qp mode with its PH partner on the base R.

    The update is ``R_odd = R_base + |b><b| - |Cb><Cb|``.  It is valid only
    when b is a normalized current-H eigenmode in the base unoccupied sector
    and Cb is its occupied partner.  This changes the branch parity by one
    and leaves every other Nambu qp occupation in the full mixed sum intact.
    """
    if not isinstance(evidence, BlockedModeEvidence):
        raise TypeError("odd branch requires explicit BlockedModeEvidence")
    h = branch.h_nambu
    base = branch.generalized_density
    mode, partner = evidence.validate_for(h, base, tolerance=tolerance)
    odd = base + torch.outer(mode, mode.conj()) - torch.outer(partner, partner.conj())
    dim = odd.shape[0]
    n = dim // 2
    eye = torch.eye(dim, dtype=odd.dtype, device=odd.device)
    ph_swap = torch.cat((
        torch.cat((torch.zeros((n, n), dtype=odd.dtype, device=odd.device),
                   torch.eye(n, dtype=odd.dtype, device=odd.device)), dim=1),
        torch.cat((torch.eye(n, dtype=odd.dtype, device=odd.device),
                   torch.zeros((n, n), dtype=odd.dtype, device=odd.device)), dim=1),
    ), dim=0)
    herm_error = float(torch.linalg.matrix_norm(odd - odd.mH).detach())
    purity_error = float(torch.linalg.matrix_norm(odd @ odd - odd).detach())
    ph_error = float(torch.linalg.matrix_norm(ph_swap @ odd.conj() @ ph_swap - (eye - odd)).detach())
    if max(herm_error, purity_error, ph_error) > tolerance * max(1, dim):
        raise NoSeaBranchEvidenceError("blocked density failed Hermitian/purity/PH structure")
    rho, chi = odd[:n, :n], odd[:n, n:]
    chi_error = float(torch.linalg.matrix_norm(chi + chi.mT).detach())
    if chi_error > tolerance * max(1, dim):
        raise NoSeaBranchEvidenceError("blocked pairing density failed antisymmetry")
    diagnostics = replace(
        branch.construction.diagnostics,
        hermiticity_error=herm_error,
        purity_error=purity_error,
        particle_hole_error=ph_error,
        chi_skew_error=chi_error,
    )
    construction = replace(
        branch.construction, generalized_density=odd, rho=rho, chi=chi,
        diagnostics=diagnostics,
    )
    combined = hashlib.sha256(
        f"{branch.evidence_fingerprint}:{evidence.fingerprint()}".encode("utf-8")
    ).hexdigest()
    return replace(
        branch, generalized_density=odd, rho=rho, chi=chi,
        construction=construction,
        evidence_id=f"{branch.evidence_id}+blocked:{evidence.proof_id}",
        evidence_fingerprint=combined,
    )


@dataclass(frozen=True)
class MixedBranchFixedPointLoss:
    """Differentiable density-map residual for a caller-built H_mu."""

    loss: torch.Tensor
    density_residual: torch.Tensor
    branch_density: MixedNucleonSeaDensity


def mixed_branch_fixed_point_loss(
    h_nambu: torch.Tensor,
    neural_density: torch.Tensor,
    *,
    evidence: MixedNambuBandEvidence | None,
    blocked_mode: BlockedModeEvidence | None = None,
    quadrature_order: int = 161,
    log_step: float = 0.35,
) -> MixedBranchFixedPointLoss:
    """Return ``mean(|R_NN - R_branch(H_mu)|^2)`` with full response.

    ``h_nambu`` must already include the caller's chemical-potential
    convention, e.g. ``nambu_hamiltonian(h, Delta, mu)``.  The sign actions
    are evaluated from the current H on every call and stay in autograd, so
    gradients include ``H -> R_branch``.  This is an optional finite-space
    fixed-point target; a nonzero residual is a training loss, never a
    pre-training rejection gate.  Evidence limitations and the production
    continuum gap problem are unchanged.
    """
    branch = construct_mixed_nucleon_sea_density(
        h_nambu, evidence=evidence,
        quadrature_order=quadrature_order, log_step=log_step,
    )
    if blocked_mode is not None:
        branch = block_mixed_nucleon_sea_density(branch, blocked_mode)
    candidate = torch.as_tensor(
        neural_density, dtype=branch.generalized_density.dtype,
        device=branch.generalized_density.device,
    )
    if candidate.shape != branch.generalized_density.shape:
        raise ValueError("neural_density must match the full Nambu branch density")
    if not bool(torch.isfinite(candidate).all()):
        raise ValueError("neural_density contains nonfinite entries")
    residual = candidate - branch.generalized_density
    return MixedBranchFixedPointLoss(
        loss=residual.abs().square().mean(),
        density_residual=residual,
        branch_density=branch,
    )


class MixedNucleonSeaBranchPolicy:
    """Adapter for trainer-style ``(states, evaluation, mus)`` callbacks.

    `states` must map n/p to objects exposing ``q``, ``rho_active``, and
    ``chi_active``. `evaluation` must expose ``normal_fields`` and
    ``delta_actions`` mappings; the latter are materialized with
    ``apply(I)`` so no pairing channel is omitted. `chemical_potentials` must
    contain the actual n/p multipliers used by the evaluation. Both species
    require explicit evidence. This class only assembles the current-field
    target; it cannot verify the provenance of an external family proof.
    """

    def __init__(
        self,
        evidence_by_species: Mapping[str, MixedNambuBandEvidence],
        *,
        blocked_species: tuple[str, ...] | set[str],
        blocked_mode_sources: Mapping[str, BlockedModeEvidence | Callable] | None = None,
        quadrature_order: int = 161,
        log_step: float = 0.35,
    ) -> None:
        if not isinstance(blocked_species, (tuple, set)) or not set(blocked_species).issubset({"n", "p"}):
            raise ValueError("blocked_species must explicitly identify odd n/p branches (or be empty)")
        if set(evidence_by_species) != {"n", "p"}:
            raise ValueError("branch policy requires explicit evidence for n and p")
        if any(not isinstance(evidence_by_species[s], MixedNambuBandEvidence)
               for s in ("n", "p")):
            raise TypeError("each species evidence must be MixedNambuBandEvidence")
        self.evidence_by_species = dict(evidence_by_species)
        self.blocked_species = frozenset(blocked_species)
        self.blocked_mode_sources = dict(blocked_mode_sources or {})
        if set(self.blocked_mode_sources) != set(self.blocked_species):
            if self.blocked_species or self.blocked_mode_sources:
                raise ValueError("each odd species requires exactly one blocked-mode source")
        for species, source in self.blocked_mode_sources.items():
            if isinstance(source, BlockedModeEvidence):
                if source.scope is not EvidenceScope.EXACT_HAMILTONIAN:
                    raise ValueError("fixed blocked evidence is allowed only for exact-H tests")
            elif not callable(source) or not callable(getattr(source, "checkpoint_metadata", None)):
                raise TypeError(
                    "field-varying blocked-mode providers must be callable and expose checkpoint_metadata()"
                )
        self.quadrature_order = quadrature_order
        self.log_step = log_step

    def checkpoint_metadata(self) -> dict[str, Any]:
        """Return exact JSON metadata for prescription compatibility checks."""
        evidence_payloads = {
            species: self.evidence_by_species[species].to_payload()
            for species in ("n", "p")
        }
        return {
            "format": "mixed-nucleon-sea-branch-policy-v1",
            "branch_map": "full_nambu_mixed_sign_projector",
            "chemical_potential_convention": "[[h-mu,Delta],[-Delta*,-h*+mu]]",
            "quadrature_order": int(self.quadrature_order),
            "log_step": float(self.log_step),
            "evidence": evidence_payloads,
            "evidence_fingerprints": {
                species: self.evidence_by_species[species].fingerprint()
                for species in ("n", "p")
            },
            "blocked_species": sorted(self.blocked_species),
            "blocked_mode_sources": {
                species: (source.fingerprint() if isinstance(source, BlockedModeEvidence)
                          else source.checkpoint_metadata())
                for species, source in self.blocked_mode_sources.items()
            },
            "field_family_proof_machine_verified": False,
            "blocked_mode_family_proof_machine_verified": False,
            "physical_no_sea_certified": False,
            "physical_identity_certified": False,
        }

    def evaluate(self, states, evaluation, chemical_potentials):
        """Return dimensionless branch loss and compact diagnostics.

        This is intended to replace a commutator-squared objective only when
        the caller explicitly selects the branch fixed-point objective. The
        caller may still calculate and report its usual full commutator
        diagnostic independently.
        """
        if set(states) != {"n", "p"}:
            raise ValueError("branch policy requires n and p state factors")
        if not isinstance(chemical_potentials, Mapping) or not {"n", "p"}.issubset(
            chemical_potentials
        ):
            raise ValueError("actual n/p chemical potentials are required")
        if not hasattr(evaluation, "normal_fields") or not hasattr(evaluation, "delta_actions"):
            raise TypeError("evaluation must expose normal_fields and delta_actions")
        species_losses = {}
        diagnostics = {
            "objective": "full_nambu_mixed_branch_fixed_point",
            "physical_no_sea_certified": False,
            "physical_identity_certified": False,
            "species": {},
        }
        for species in ("n", "p"):
            if species not in evaluation.normal_fields or species not in evaluation.delta_actions:
                raise ValueError(f"evaluation lacks full normal/pairing action for {species}")
            state = states[species]
            q = state.q
            d = int(q.shape[0])
            if d > 96:
                raise ValueError("dense prototype cap exceeded: primitive D must be <=96")
            h = evaluation.normal_fields[species]
            if h.shape != (d, d):
                raise ValueError("normal field must be a full primitive-space matrix")
            identity = torch.eye(d, dtype=h.dtype, device=h.device)
            delta = evaluation.delta_actions[species].apply(identity)
            if delta.shape != (d, d):
                raise ValueError("DeltaAction.apply(I) must return the full pairing matrix")
            h_mu = nambu_hamiltonian(h, delta, chemical_potentials[species])
            if torch.is_grad_enabled() and not h_mu.requires_grad:
                raise NoSeaBranchEvidenceError(
                    "branch fixed-point training requires a live H_mu autograd path"
                )
            rho = q @ state.rho_active @ q.mH
            chi = q @ state.chi_active @ q.mT
            r_nn = torch.cat((
                torch.cat((rho, chi), dim=1),
                torch.cat((-chi.conj(), identity - rho.conj()), dim=1),
            ), dim=0)
            blocked_mode = None
            branch = construct_mixed_nucleon_sea_density(
                h_mu, evidence=self.evidence_by_species[species],
                quadrature_order=self.quadrature_order, log_step=self.log_step,
            )
            if species in self.blocked_species:
                source = self.blocked_mode_sources.get(species)
                if isinstance(source, BlockedModeEvidence):
                    blocked_mode = source
                    if h_mu.requires_grad:
                        raise NoSeaBranchEvidenceError(
                            "exact-H blocked mode is fixed-matrix only; training requires a current differentiable provider"
                        )
                elif callable(source):
                    blocked_mode = source(species, h_mu, branch.generalized_density)
                    if not isinstance(blocked_mode, BlockedModeEvidence):
                        raise TypeError("blocked-mode provider must return BlockedModeEvidence")
                    if blocked_mode.scope is not EvidenceScope.SELFCONSISTENT_FAMILY:
                        raise ValueError("field-varying provider must return family-scoped blocked evidence")
                else:
                    raise NoSeaBranchEvidenceError(
                        f"odd {species} branch requires a current quasiparticle-mode witness"
                    )
                branch = block_mixed_nucleon_sea_density(branch, blocked_mode)
            parity = number_parity_from_density(
                branch.rho.detach().cpu().numpy(),
                branch.chi.detach().cpu().numpy(),
            )
            expected_parity = -1 if species in self.blocked_species else 1
            if parity["number_parity"] != expected_parity:
                raise NoSeaBranchEvidenceError(
                    f"{species} branch target parity {parity['number_parity']} disagrees with "
                    f"declared odd/even prescription {expected_parity}; refusing to relabel R"
                )
            residual = r_nn - branch.generalized_density
            species_losses[species] = residual.abs().square().mean()
            diagnostics["species"][species] = {
                "evidence_fingerprint": branch.evidence_fingerprint,
                "density_residual_rms": float(residual.abs().square().mean().sqrt().detach()),
                "commutator_rms": float(branch.commutator_action(
                    torch.eye(2 * d, dtype=h.dtype, device=h.device)
                ).abs().square().mean().sqrt().detach()),
                "divider_mev": branch.divider_mev,
                "number_parity": parity["number_parity"],
                "particle_number_from_pfaffian_audit": parity["particle_number"],
                "blocked_mode_fingerprint": (blocked_mode.fingerprint()
                                             if blocked_mode is not None else None),
                "physical_no_sea_certified": False,
                "physical_identity_certified": False,
            }
        total = 0.5 * (species_losses["n"] + species_losses["p"])
        return total, diagnostics


def hamiltonian_sha256(h_nambu: torch.Tensor) -> str:
    """Hash the exact complex128 matrix bytes for exact-H evidence binding."""
    h = torch.as_tensor(h_nambu)
    if h.dtype != torch.complex128 or h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise TypeError("H must be a square complex128 Nambu matrix")
    return hashlib.sha256(h.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def construct_mixed_nucleon_sea_density(
    h_nambu: torch.Tensor,
    *,
    evidence: MixedNambuBandEvidence | None,
    quadrature_order: int = 161,
    log_step: float = 0.35,
) -> MixedNucleonSeaDensity:
    """Construct the conditional full-Nambu mixed nucleon/sea density.

    No evidence means immediate failure; there is no default energy window,
    normal-space projection, or recovery fallback.  Given evidence, the
    delegated map is

    ``R = 1/2 [I - sign(H) sign(Ec**2 I - H**2)]``

    which selects the conventional low-energy particle/filled sectors and
    reverses occupation on the high-energy Dirac-sea sector under the stated
    complete-band enclosure. Linear solves and their gradients remain in the
    graph. The helper is branch construction only; it does not diagonalize,
    refresh a basis, solve a self-consistent field family, or certify the
    physical origin of the supplied witness.
    """
    h = torch.as_tensor(h_nambu)
    if evidence is None:
        raise NoSeaBranchEvidenceError(
            "no mixed nucleon/sea branch evidence supplied; refusing to guess Ec or use P+"
        )
    if h.dtype != torch.complex128 or h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise TypeError("H must be a square complex128 full-Nambu matrix")
    if h.shape[0] > 192 or h.shape[0] % 2:
        raise ValueError("prototype accepts primitive D<=96 (full Nambu dimension <=192)")
    if (isinstance(quadrature_order, bool) or not isinstance(quadrature_order, int)
            or quadrature_order < 3 or quadrature_order % 2 != 1):
        raise ValueError("quadrature_order must be an odd integer >= 3")
    if not math.isfinite(log_step) or log_step <= 0.0:
        raise ValueError("log_step must be finite and positive")
    if not math.isfinite(evidence.divider_mev) or evidence.divider_mev <= 0.0:
        raise NoSeaBranchEvidenceError("an explicit positive Ec divider is required")
    evidence.validate_for(h)
    branch = mixed_sign_bdg_projector(
        h,
        divider=evidence.divider_mev,
        certificate=evidence.certificate,
        quadrature_order=quadrature_order,
        log_step=log_step,
    )
    return MixedNucleonSeaDensity(
        generalized_density=branch.generalized_density,
        rho=branch.rho,
        chi=branch.chi,
        divider_mev=float(evidence.divider_mev),
        evidence_scope=evidence.scope.value,
        evidence_id=evidence.proof_id,
        construction=branch,
        h_nambu=h,
        evidence_fingerprint=evidence.fingerprint(),
    )
