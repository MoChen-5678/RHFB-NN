from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil

import pytest

from dpl_rhf.rhf.functional import PKA1_DIRECT_CHANNEL_TERMS, PKA1_FOCK_CHANNEL_TERMS
from dpl_rhf.rhfb.source_template import compare_source_to_python, parse_source_template


FIXTURES = Path(__file__).parent / "fixtures" / "source_templates"
CHANNELS = tuple(PKA1_DIRECT_CHANNEL_TERMS) + tuple(PKA1_FOCK_CHANNEL_TERMS)


def _startup(template, *, channels=CHANNELS, strength=1.0):
    return template.require_nn_startup(
        requested_model="PKA1",
        requested_channels=channels,
        requested_pairing_force="D1S",
        requested_pairing_strength=strength,
    )


def test_broyden_fixture_resolves_actual_interaction_and_empty_blocking():
    template = parse_source_template(FIXTURES / "RHFB-BI" / "Broyden")

    assert template.source_variant == "RHFB-BI/Broyden"
    assert template.parameter_set_index == 0
    assert template.parameter_set_name == "PKA1"
    assert template.pka1_constants["txtfor"] == "PKA1"
    assert template.pka1_constants["amu"] == (938.9, 938.9)
    assert template.pka1_constants["gsig"] == 8.372672
    assert template.pka1_constants["ie"] == 1
    assert template.d1s_constants["gr"] == (0.7, 1.2)
    assert template.d1s_constants["gw"] == (-1720.30, 103.639)
    assert template.pairing_force == "D1S"
    assert template.pairing_strength == 1.0
    assert template.blocking_n == ()
    assert template.blocking_p == ()
    assert template.effective_exchange_enabled is True
    assert template.tensor_enabled is True
    assert template.xct_positive_mev is None
    assert template.xpt_mev is None
    assert _startup(template).no_sea_occupation_rule == "not inherited from DWS basis cutoffs"


def test_spherical_fixture_preserves_xct_xpt_as_source_input_metadata():
    template = parse_source_template(FIXTURES / "S-RHFB-C")

    assert template.source_variant == "S-RHFB-C"
    assert template.parameter_set_index == 0
    assert template.pairing_force == "D1S"
    assert template.pairing_strength == 1.0
    assert template.blocking_n == ()
    assert template.blocking_p == ()
    assert template.xct_positive_mev == 350.0
    assert template.xct_negative_relative_to_2m_mev == -10.0
    assert template.xpt_mev == (400.0, 400.0)
    assert template.tensor_enabled is True
    assert template.source_encoding["Define.f90"] == "latin-1-fallback"
    basis = template.source_basis_cutoffs
    assert "never an NN sea threshold" in basis["meaning"]
    assert basis["xpt_pairing_window_mev"] == (400.0, 400.0)
    assert "not interpreted as sea occupancy" in basis["xpt_meaning"]
    startup = _startup(template)
    assert startup.source_basis_cutoffs["xpt_pairing_window_mev"] == (400.0, 400.0)


def test_original_bi_root_fixture_does_not_guess_interactive_model_or_pairing():
    template = parse_source_template(FIXTURES / "RHFB-BI")

    assert template.source_variant == "RHFB-BI"
    assert template.parameter_set_index is None
    assert template.parameter_set_name is None
    assert template.pairing_force is None
    assert template.pairing_strength is None
    assert template.blocking_n == ()
    assert template.blocking_p == ()
    mapping = compare_source_to_python(template)
    assert not mapping.d1s_active_matches
    assert mapping.d1s_active_source_to_python["ranges_fm"] == mapping.d1s_python["ranges_fm"]
    assert mapping.d1s_active_source_to_python["strength"] is None
    assert any("strength (source=None" in item for item in mapping.mismatches)
    with pytest.raises(ValueError, match="interactive|absent"):
        _startup(template)


def test_startup_rejects_cropped_channels_or_different_constants():
    assert {
        "rho_tensor_direct", "rho_vector_tensor_direct", "coulomb_direct",
        "rho_tensor_fock", "rho_vector_tensor_fock", "pion_pseudovector_fock",
        "coulomb_fock",
    } <= set(CHANNELS)
    template = parse_source_template(FIXTURES / "RHFB-BI" / "Broyden")
    with pytest.raises(ValueError, match="cropped"):
        _startup(template, channels=CHANNELS[:-1])

    changed = dict(template.pka1_constants)
    changed["gsig"] = changed["gsig"] + 1e-6
    with pytest.raises(ValueError, match="source-to-Python PKA1"):
        _startup(replace(template, pka1_constants=changed))


def test_source_constants_map_field_by_field_to_current_python_parameters():
    template = parse_source_template(FIXTURES / "RHFB-BI" / "Broyden")
    mapping = compare_source_to_python(template)

    assert mapping.pka1_matches
    assert mapping.d1s_active_matches
    assert mapping.pka1_source_to_python == mapping.pka1_python
    assert mapping.d1s_active_source_to_python == mapping.d1s_python
    assert mapping.d1s_active_source_to_python["ranges_fm"] == (0.7, 1.2)
    assert mapping.d1s_active_source_to_python["m_mev"] == (1397.60, -223.934)
    assert mapping.pka1_source_to_python["f_rho"] == 3.199491
    assert "grtn_to_f_rho" in mapping.unit_conventions
    assert "w0" not in mapping.d1s_active_source_to_python
    assert "t3" not in mapping.d1s_active_source_to_python
    from dpl_rhf.rhf.parameters import pka1_parameters
    parameters = pka1_parameters()
    expected_vertex = parameters.g_rho * 3.199491 / (
        parameters.mass_n / parameters.hbar_c + parameters.mass_p / parameters.hbar_c
    )
    assert parameters.rho_tensor_vertex == expected_vertex


def test_source_reference_blocking_is_metadata_not_nn_blocking_gate(tmp_path):
    source = FIXTURES / "RHFB-BI" / "Broyden"
    copied = tmp_path / "RHFB-BI" / "Broyden"
    shutil.copytree(source, copied)
    input_path = copied / "rhfb.dat"
    text = input_path.read_text()
    assert "0,0                         ! number of Blocking data" in text
    input_path.write_text(text.replace(
        "0,0                         ! number of Blocking data",
        "1,0                         ! number of Blocking data", 1))

    parsed = parse_source_template(copied)
    assert parsed.blocking_n == (11,)
    assert _startup(parsed).source_blocking_n == (11,)


def test_parser_reads_changed_d1s_multiplier_from_fixed_input_record(tmp_path):
    source = FIXTURES / "RHFB-BI" / "Broyden"
    copied = tmp_path / "RHFB-BI" / "Broyden"
    shutil.copytree(source, copied)
    input_path = copied / "rhfb.dat"
    text = input_path.read_text()
    assert "GNF/V0   =     1.000" in text
    input_path.write_text(text.replace("GNF/V0   =     1.000", "GNF/V0   =     0.900", 1))

    parsed = parse_source_template(copied)
    assert parsed.pairing_strength == 0.9
    with pytest.raises(ValueError, match="strengths must equal exactly 1"):
        _startup(parsed)


def test_source_blocking_keeps_signed_dirac_kappa_without_mapping_to_nn_indices(tmp_path):
    source = FIXTURES/"RHFB-BI"/"Broyden"
    copied = tmp_path/"RHFB-BI"/"Broyden"
    shutil.copytree(source,copied)
    path = copied/"rhfb.dat"
    text = path.read_text()
    text = text.replace("0,0                         ! number of Blocking", "2,0                         ! number of Blocking",1)
    text = text.replace("11,13,0,0,0", "-3,2,0,0,0",1)
    path.write_text(text)
    template = parse_source_template(copied)
    assert template.blocking_n == (-3,2)
    assert template.blocking_p == ()
    # These remain source labels; startup table comparison does not pretend
    # they are columns 3/2 in the unrestricted NN chart.
    assert _startup(template).source_blocking_n == (-3,2)


USER_ATTACHMENT_ROOT = Path("/home/ubuntu/rhf/dpl-param/reference_fortran/user-rhfb-20261015")


@pytest.mark.parametrize("relative", ["RHFB-BI", "RHFB-BI/Broyden", "S-RHFB-C"])
def test_actual_user_attachment_directories_are_readable_when_available(relative):
    directory = USER_ATTACHMENT_ROOT / relative
    if not directory.is_dir():
        pytest.skip("user-supplied read-only Fortran attachment is unavailable")
    template = parse_source_template(directory)
    assert len(template.source_sha256) == 6
    assert set(template.source_encoding) == set(template.source_sha256)
    assert template.pka1_constants["ie"] == 1
    mapping = compare_source_to_python(template)
    assert mapping.pka1_matches
    if relative == "RHFB-BI":
        assert template.pairing_strength is None
        assert not mapping.d1s_active_matches
        assert any("strength (source=None" in item for item in mapping.mismatches)
    else:
        assert template.pairing_strength == 1.0
        assert mapping.d1s_active_matches
    if relative == "S-RHFB-C":
        assert template.source_encoding["Define.f90"] == "latin-1-fallback"

