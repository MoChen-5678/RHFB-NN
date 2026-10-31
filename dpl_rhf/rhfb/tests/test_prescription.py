from __future__ import annotations

import copy

import pytest

from dpl_rhf.rhfb.prescription import (
    PhysicalPrescription,
    checkpoint_prescription_fields,
    current_neural_prescription,
    source_dws_positive_qp_prescription,
    validate_checkpoint_prescription,
)
from dpl_rhf.rhfb.checkpoint import load_checkpoint, save_checkpoint
from dpl_rhf.rhfb.config import BlockedRHFBConfig


def _source_prescription(**changes):
    values = {
        "particle_hole_functional": "PKA1 covariant RHF",
        "energy_zero": "h_D-M; particle continuum near 0, sea continuum near -2M",
        "chemical_potential_convention": "BdG blocks h-mu and -h*+mu; mu enforces mean N",
        "pairing_force": "Gogny D1S finite range",
        "pairing_channels": ("nn", "pp"),
        "pairing_window": "natural finite-range regulator; no explicit qp window",
        "blocking": "spherical_radial_degeneracy_prescription",
        "spatial_symmetry": "spherical_radial",
        "time_reversal": "not_resolved_by_radial_source",
        "basis_spec": "DWS finite basis: NBF=36,NBD=12; run-fixed counts",
        "basis_max_positive_states": 36,
        "basis_max_negative_states": 12,
        "source_audit": "RHFB-BI/Dirac.f90 and S-RHFB-C/Dirac.f90 source review",
    }
    values.update(changes)
    return source_dws_positive_qp_prescription(**values)


def test_source_rule_records_negative_basis_without_sea_occupation():
    prescription = _source_prescription()
    assert prescription.qp_branch_rule == "positive_qp_vv_sum"
    assert prescription.negative_energy_basis_role == "expansion_basis_only"
    assert prescription.negative_qp_density == "omitted_from_selfconsistent_density"
    assert "canonical postprocessing" in prescription.canonical_postprocess
    assert prescription.physical_branch_certified is False
    assert "Ec" not in prescription.to_payload()
    assert "W" not in prescription.to_payload()


def test_current_neural_preset_does_not_claim_source_qp_rule_or_no_sea():
    prescription = current_neural_prescription(
        basis_spec="actual AngularRadialBasis spec",
        blocking="explicit block records from trainer config",
    )
    assert prescription.no_sea_implemented is False
    assert prescription.physical_branch_certified is False
    assert prescription.qp_branch_rule == "not_supplied"
    assert prescription.negative_energy_basis_role == "not_certified"
    assert prescription.negative_qp_density == "not_certified"
    assert "full represented J" in prescription.pairing_window
    assert "no spherical-radial" in prescription.spatial_symmetry
    assert "not externally benchmarked" in prescription.normal_density_definition


def test_checkpoint_fields_round_trip_and_resume_requires_exact_identity():
    prescription = _source_prescription()
    checkpoint = {"network_state": {"n": {}, "p": {}},
                  **checkpoint_prescription_fields(prescription)}
    validate_checkpoint_prescription(checkpoint, prescription)
    payload = copy.deepcopy(checkpoint["physics_prescription"])
    restored = PhysicalPrescription.from_payload(payload)
    assert restored.fingerprint() == prescription.fingerprint()


@pytest.mark.parametrize("change", [
    {"energy_zero": "unshifted Dirac energy including rest mass"},
    {"chemical_potential_convention": "h+mu and -h*-mu"},
    {"pairing_force": "zero-range contact"},
    {"basis_negative_cutoff_relative_to_2m_mev": -10.0},
    {"qp_branch_rule": "unresolved_full_generalized_density"},
    {"qp_branch_rule": "conditional_mixed_sign_window",
     "negative_energy_basis_role": "selected_negative_qp_sector"},
    {"blocking": "explicit_one_quasiparticle", "spatial_symmetry": "axial",
     "time_reversal": "broken_by_blocked_state"},
    {"spatial_symmetry": "unrestricted"},
])
def test_resume_rejects_each_physics_or_symmetry_change(change):
    base = _source_prescription()
    checkpoint = checkpoint_prescription_fields(base)
    requested = PhysicalPrescription(**{
        **base.__dict__,
        **change,
    })
    with pytest.raises(ValueError, match="mismatch"):
        validate_checkpoint_prescription(checkpoint, requested)


def test_resume_rejects_legacy_payload_and_tampered_fingerprint():
    prescription = _source_prescription()
    with pytest.raises(ValueError, match="lacks a versioned"):
        validate_checkpoint_prescription({"network_state": {}}, prescription)
    checkpoint = checkpoint_prescription_fields(prescription)
    checkpoint["physics_prescription_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint is corrupt"):
        validate_checkpoint_prescription(checkpoint, prescription)


def test_schema_records_blocking_geometry_without_invented_compatibility_rule():
    # Specific angular-momentum/block geometry compatibility is source-dependent.
    exact_blocker = _source_prescription(blocking="explicit_one_quasiparticle")
    assert exact_blocker.spatial_symmetry == "spherical_radial"
    base = _source_prescription()
    with pytest.raises(ValueError, match="schema/version"):
        PhysicalPrescription(**{**base.__dict__, "schema_version": 99})


def test_payload_schema_is_closed_and_requires_all_physics_fields():
    payload = _source_prescription().to_payload()
    payload.pop("energy_zero")
    with pytest.raises(ValueError, match="fields differ"):
        PhysicalPrescription.from_payload(payload)


def test_checkpoint_api_persists_and_checks_the_physics_prescription(tmp_path):
    config = BlockedRHFBConfig(z=2, n=2)
    prescription = _source_prescription()
    path = tmp_path / "state.pt"
    save_checkpoint(
        path, config, basis_spec={"dimension": 4}, blocked_reference={"n": None, "p": None},
        physics_prescription=prescription, network_state={"n": {}, "p": {}},
    )
    payload = load_checkpoint(
        path, config, basis_spec={"dimension": 4}, blocked_reference={"n": None, "p": None},
        physics_prescription=prescription,
    )
    assert payload["physics_prescription_fingerprint"] == prescription.fingerprint()
    incompatible = _source_prescription(pairing_force="contact pairing")
    with pytest.raises(ValueError, match="physical prescription mismatch"):
        load_checkpoint(
            path, config, basis_spec={"dimension": 4}, blocked_reference={"n": None, "p": None},
            physics_prescription=incompatible,
        )

