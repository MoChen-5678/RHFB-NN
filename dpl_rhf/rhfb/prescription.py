"""Versioned physical-model identity for RHFB checkpoints.

The schema records a choice of equations; it does not certify that a chosen
no-sea branch is physically complete.  In particular, the source-backed
``positive_qp_vv_sum`` rule distinguishes negative-energy *basis states* from
negative-energy *occupied quasiparticle states*.  It does not provide a
matrix-sign/window prescription for arbitrary self-consistent fields.

The source distinction follows Long et al., arXiv:0812.1103, Eqs. (32)--(35),
Geng and Long, arXiv:2202.00482, Sec. III.A, and the specific no-sea density
prescription in DIRHB, arXiv:1403.4039, Eqs. (46)--(48). These references do
not certify a fixed finite spectral window for all NN-generated fields.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping


SCHEMA = "dpl-rhfb-physical-prescription"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PhysicalPrescription:
    """Complete, immutable identity of the model and branch being resumed.

    String fields are intentionally explicit instead of inferred from a
    filename or an optimizer config.  ``physical_branch_certified`` is false
    for the source transcription below: the cited implementations establish
    their finite-basis algorithm, not a general field-independent branch
    theorem.
    """

    prescription_id: str
    particle_hole_functional: str
    pairing_force: str
    pairing_channels: tuple[str, ...]
    pairing_window: str
    energy_zero: str
    chemical_potential_convention: str
    nambu_order: str
    particle_hole_conjugation: str
    normal_density_definition: str
    qp_branch_rule: str
    positive_qp_density: str
    negative_energy_basis_role: str
    negative_qp_density: str
    canonical_postprocess: str
    blocking: str
    spatial_symmetry: str
    time_reversal: str
    basis_spec: str
    basis_positive_cutoff_mev: float | None
    basis_negative_cutoff_relative_to_2m_mev: float | None
    basis_max_positive_states: int | None
    basis_max_negative_states: int | None
    source_audit: str
    no_sea_implemented: bool
    physical_branch_certified: bool = False
    schema: str = SCHEMA
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != SCHEMA or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported physical-prescription schema/version")
        required = (
            "prescription_id", "particle_hole_functional", "pairing_force",
            "pairing_window", "energy_zero", "chemical_potential_convention",
            "nambu_order", "particle_hole_conjugation", "normal_density_definition", "qp_branch_rule",
            "positive_qp_density", "negative_energy_basis_role",
            "negative_qp_density", "canonical_postprocess", "blocking",
            "spatial_symmetry", "time_reversal", "basis_spec", "source_audit",
        )
        if any(not isinstance(getattr(self, field), str) or not getattr(self, field).strip()
               for field in required):
            raise ValueError("every prescription field must be explicit and non-empty")
        if (not self.pairing_channels or
                any(not isinstance(x, str) or not x.strip() for x in self.pairing_channels)):
            raise ValueError("pairing_channels must explicitly name at least one channel")
        if len(set(self.pairing_channels)) != len(self.pairing_channels):
            raise ValueError("pairing_channels must not contain duplicates")
        if not isinstance(self.physical_branch_certified, bool):
            raise TypeError("physical_branch_certified must be a boolean")
        if not isinstance(self.no_sea_implemented, bool):
            raise TypeError("no_sea_implemented must be a boolean")
        for name in ("basis_positive_cutoff_mev", "basis_negative_cutoff_relative_to_2m_mev"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise TypeError(f"{name} must be a finite number or None")
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        for name in ("basis_max_positive_states", "basis_max_negative_states"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.qp_branch_rule == "positive_qp_vv_sum" and (
            self.negative_energy_basis_role != "expansion_basis_only" or
            self.negative_qp_density != "omitted_from_selfconsistent_density"
        ):
            raise ValueError(
                "positive_qp_vv_sum requires negative-energy basis states to be expansion-only "
                "and negative qp density to be omitted from the self-consistent source"
            )

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pairing_channels"] = list(self.pairing_channels)
        return payload

    def fingerprint(self) -> str:
        canonical = json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PhysicalPrescription":
        if not isinstance(payload, Mapping):
            raise TypeError("physical prescription payload must be a mapping")
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            missing = sorted(expected - set(payload))
            extra = sorted(set(payload) - expected)
            raise ValueError(f"prescription fields differ (missing={missing}, extra={extra})")
        values = dict(payload)
        channels = values["pairing_channels"]
        if not isinstance(channels, (list, tuple)):
            raise TypeError("pairing_channels must be a list/tuple")
        values["pairing_channels"] = tuple(channels)
        return cls(**values)


def source_dws_positive_qp_prescription(
    *, particle_hole_functional: str, energy_zero: str,
    chemical_potential_convention: str, pairing_force: str = "Gogny D1S",
    pairing_channels: tuple[str, ...] = ("nn", "pp"),
    pairing_window: str = "no explicit window for finite-range Gogny pairing",
    blocking: str = "spherical_radial_degeneracy_prescription",
    spatial_symmetry: str = "spherical_radial",
    time_reversal: str = "not_resolved_by_radial_source",
    basis_spec: str = "finite Dirac-Woods-Saxon basis with positive and negative-energy states",
    basis_positive_cutoff_mev: float | None = None,
    basis_negative_cutoff_relative_to_2m_mev: float | None = None,
    basis_max_positive_states: int | None = None,
    basis_max_negative_states: int | None = None,
    source_audit: str = "RHFB-BI and S-RHFB-C source audit; see cited file/line records",
) -> PhysicalPrescription:
    """Transcribe the main self-consistency density rule in the supplied BI/S sources.

    S-RHFB-C's negative-qp contribution to its canonical postprocessing is
    recorded separately; it is not silently substituted into the iterative
    source density.  No Ec, W, spectral threshold, or completeness claim is
    supplied by this constructor.
    """
    return PhysicalPrescription(
        prescription_id="source-dws-positive-qps-vv-density-v1",
        particle_hole_functional=particle_hole_functional,
        pairing_force=pairing_force,
        pairing_channels=tuple(pairing_channels),
        pairing_window=pairing_window,
        energy_zero=energy_zero,
        chemical_potential_convention=chemical_potential_convention,
        nambu_order="particle,hole",
        particle_hole_conjugation="C=tau_x K in particle,hole order",
        normal_density_definition=("spherical radial scalar/vector/tensor sources from selected positive qp V amplitudes; "
                                   "omitted negative-qp terms mean full-space canonical purity and exact chi skewness are not certified"),
        qp_branch_rule="positive_qp_vv_sum",
        positive_qp_density="sum over selected positive-energy qp V V^dagger",
        negative_energy_basis_role="expansion_basis_only",
        negative_qp_density="omitted_from_selfconsistent_density",
        canonical_postprocess=("S-RHFB-C has a separate negative-qp contribution in canonical "
                               "postprocessing; diagnostic/postprocess only"),
        blocking=blocking,
        spatial_symmetry=spatial_symmetry,
        time_reversal=time_reversal,
        basis_spec=basis_spec,
        basis_positive_cutoff_mev=basis_positive_cutoff_mev,
        basis_negative_cutoff_relative_to_2m_mev=basis_negative_cutoff_relative_to_2m_mev,
        basis_max_positive_states=basis_max_positive_states,
        basis_max_negative_states=basis_max_negative_states,
        source_audit=source_audit,
        no_sea_implemented=True,
        physical_branch_certified=False,
    )


def current_neural_prescription(*, basis_spec: str, blocking: str) -> PhysicalPrescription:
    """Identify the current NN candidate without attributing a no-sea selector to it."""
    return PhysicalPrescription(
        prescription_id="current-neural-generalized-density-candidate-v1",
        particle_hole_functional="PKA1 full represented particle-hole functional",
        pairing_force="Gogny D1S finite range, unscaled nn/pp channels",
        pairing_channels=("nn", "pp"),
        pairing_window="full represented J matrix; no separate qp energy window",
        energy_zero=("configured RHFB h_D-M convention; no qp spectrum is selected "
                     "by the training network"),
        chemical_potential_convention=("particle-number Lagrange multipliers for neutron and proton; "
                                       "no quasiparticle energy ordering is imposed"),
        nambu_order="particle,hole",
        particle_hole_conjugation="C=tau_x K in particle,hole order",
        normal_density_definition=("current sqrt(j_mu j^mu) covariant invariant-density extension; "
                                   "odd-state response is not externally benchmarked"),
        qp_branch_rule="not_supplied",
        positive_qp_density="not_supplied; full generalized-density state is varied directly",
        negative_energy_basis_role="not_certified",
        negative_qp_density="not_certified",
        canonical_postprocess="not supplied",
        blocking=blocking,
        spatial_symmetry="all represented magnetic substates; no spherical-radial density projection",
        time_reversal="time-odd channels are represented; no time-reversal constraint imposed",
        basis_spec=basis_spec,
        basis_positive_cutoff_mev=None,
        basis_negative_cutoff_relative_to_2m_mev=None,
        basis_max_positive_states=None,
        basis_max_negative_states=None,
        source_audit="current neural candidate metadata; no no-sea quasiparticle branch is implemented",
        no_sea_implemented=False,
        physical_branch_certified=False,
    )


def checkpoint_prescription_fields(prescription: PhysicalPrescription) -> dict[str, Any]:
    """Fields to embed in a checkpoint payload; callers must pass them explicitly."""
    if not isinstance(prescription, PhysicalPrescription):
        raise TypeError("checkpoint requires an explicit PhysicalPrescription")
    return {
        "physics_prescription": prescription.to_payload(),
        "physics_prescription_fingerprint": prescription.fingerprint(),
    }


def validate_checkpoint_prescription(
    checkpoint: Mapping[str, Any], requested: PhysicalPrescription,
) -> None:
    """Reject legacy, malformed, or physically incompatible resume payloads."""
    if not isinstance(requested, PhysicalPrescription):
        raise TypeError("requested resume prescription must be explicit")
    payload = checkpoint.get("physics_prescription")
    fingerprint = checkpoint.get("physics_prescription_fingerprint")
    if payload is None or fingerprint is None:
        raise ValueError("checkpoint lacks a versioned physical prescription; frozen/legacy state is not resumable")
    saved = PhysicalPrescription.from_payload(payload)
    if saved.fingerprint() != fingerprint:
        raise ValueError("checkpoint physical-prescription fingerprint is corrupt")
    if saved.fingerprint() != requested.fingerprint():
        raise ValueError("physical prescription mismatch; use an explicit state conversion, not optimizer resume")

