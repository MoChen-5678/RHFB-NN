import torch

from dpl_rhf.rhfb.fixed_field_dirac_branch_reference import compare_fixed_fields


def test_fixed_field_reference_tracks_whole_and_strict_projected_odd_branches():
    # One well-separated shifted Dirac-sea mode and four particle modes.
    # This is an API oracle only; the separate O(72) run supplies PKA1+D1S fields.
    energies = torch.tensor([-20., 1., 2., 3., 4.], dtype=torch.float64)
    h = torch.diag(energies).to(torch.complex128)
    delta = torch.zeros((5, 5), dtype=torch.complex128)
    delta[0, 3], delta[3, 0] = .2, -.2  # full cross-sector block to an empty particle mode
    delta[2, 3], delta[3, 2] = .3, -.3  # positive-sector pairing
    result = compare_fixed_fields(
        h, delta, mass_mev=10., target_number=3, continuation_steps=3,
        minimum_cutoff_distance_mev=1., minimum_band_separation_mev=2.,
    )
    report = result.report
    assert report["normal_dirac_split"]["safe_sector_split"]
    assert report["whole_dirac_branch"]["number_parity"] == -1
    assert report["strict_normal_positive_projected_bdg"]["number_parity"] == -1
    assert abs(report["whole_dirac_branch"]["normalized_pfaffian"] + 1.) < 1e-12
    assert report["even_unblocked_reference_parity"] == 1
    # A fixed μ HFB projector has definite odd parity but need not have the
    # target integer expectation value until its number equation is solved.
    assert report["whole_dirac_branch"]["purity_residual"] < 1e-12
    assert report["strict_normal_positive_projected_bdg"]["purity_residual"] < 1e-12
    assert report["full_d1s_pairing_cross_blocks_frobenius_mev"][
        "negative_output_from_positive_input"] > 0.
    assert report["full_d1s_pairing_cross_blocks_frobenius_mev"][
        "relative_to_full_delta"] > 0.
    assert report["strict_normal_positive_projected_bdg"][
        "same_full_field_commutator_frobenius_mev"] > 0.
    assert report["strict_normal_positive_projected_bdg"]["density_projector_trace"] == 5.
    assert report["strict_normal_positive_projected_bdg"]["embedded_canonical_density_residual"] < 1e-12
    assert report["whole_dirac_branch"][
        "full_field_commutator_frobenius_mev"] < 1e-10


def test_shifted_dirac_gap_failure_refuses_sector_label():
    h = torch.diag(torch.tensor([-10., -9.8, 1., 2.], dtype=torch.float64)).to(torch.complex128)
    delta = torch.zeros_like(h)
    try:
        compare_fixed_fields(h, delta, mass_mev=10., target_number=1,
            minimum_cutoff_distance_mev=1., minimum_band_separation_mev=5.)
    except RuntimeError as exc:
        assert "Dirac" in str(exc)
    else:
        raise AssertionError("unsafe normal spectrum was assigned Dirac sectors")


def test_even_whole_dirac_branch_matches_zero_pairing_positive_quantization():
    energies = torch.tensor([-20., 1., 2., 3., 4., 5.], dtype=torch.float64)
    h = torch.diag(energies).to(torch.complex128)
    result = compare_fixed_fields(
        h, torch.zeros_like(h), mass_mev=10., target_number=4,
        continuation_steps=1, minimum_cutoff_distance_mev=1.,
        minimum_band_separation_mev=2.,
    )
    assert result.report["whole_dirac_branch"]["number_parity"] == 1
    assert result.report["strict_normal_positive_projected_bdg"]["number_parity"] == 1
    assert torch.allclose(result.whole_rho, result.projected_rho, atol=1e-14, rtol=0.)
    assert torch.allclose(result.whole_chi, result.projected_chi, atol=1e-14, rtol=0.)


def test_odd_reference_can_preserve_a_saved_blocked_basis_identity():
    energies = torch.tensor([-20., 1., 2., 3., 4.], dtype=torch.float64)
    h = torch.diag(energies).to(torch.complex128)
    delta = torch.zeros((5, 5), dtype=torch.complex128)
    delta[1, 2], delta[2, 1] = .2, -.2
    block = torch.zeros(5, dtype=torch.complex128)
    block[4] = 1.
    result = compare_fixed_fields(
        h, delta, mass_mev=10., target_number=3, continuation_steps=2,
        blocked_basis_vector=block, minimum_cutoff_distance_mev=1.,
        minimum_band_separation_mev=2.,
    )
    assert result.report["whole_dirac_branch"]["number_parity"] == -1
    assert result.report["blocked_mode_reference"]["spectral_index"] == 4
    assert result.report["blocked_mode_reference"]["overlap_with_seed_basis_vector"] == 1.


def test_supplied_mu_supports_hole_blocking_below_mu():
    # The unblocked even reference has four occupied normal modes; blocking
    # the e=2 mode is a one-quasiparticle hole and leaves odd N=3.
    energies = torch.tensor([-20., 1., 2., 3., 4., 5.], dtype=torch.float64)
    h = torch.diag(energies).to(torch.complex128)
    block = torch.zeros(6, dtype=torch.complex128)
    block[2] = 1.
    result = compare_fixed_fields(
        h, torch.zeros_like(h), mass_mev=10., target_number=3,
        chemical_potential_mev=4.5, blocked_basis_vector=block,
        continuation_steps=1, minimum_cutoff_distance_mev=1.,
        minimum_band_separation_mev=2.,
    )
    report = result.report
    assert report["blocked_mode_reference"]["operation"] == "hole"
    assert report["blocked_mode_reference"]["spectral_index"] == 2
    assert report["unpaired_normal_occupation_count_at_mu"] == 4
    assert report["hf_limit_reference_count_matches"]
    assert report["whole_dirac_branch"]["number_parity"] == -1
    assert report["whole_dirac_branch"]["particle_number"] == 3.
    assert report["strict_normal_positive_projected_bdg"]["particle_number"] == 3.
    assert report["even_reference_energy_minimum_claim"] is False


def test_supplied_final_mu_count_mismatch_is_reported_not_rejected():
    # Pairing/final-field μ need not reproduce N±1 in the Δ=0 normal spectrum.
    energies = torch.tensor([-20., 1., 2., 3., 4., 5.], dtype=torch.float64)
    h = torch.diag(energies).to(torch.complex128)
    block = torch.zeros(6, dtype=torch.complex128)
    block[2] = 1.
    result = compare_fixed_fields(
        h, torch.zeros_like(h), mass_mev=10., target_number=3,
        chemical_potential_mev=3.5, blocked_basis_vector=block,
        continuation_steps=1, minimum_cutoff_distance_mev=1.,
        minimum_band_separation_mev=2.,
    )
    report = result.report
    assert report["blocked_mode_reference"]["operation"] == "hole"
    assert report["unpaired_normal_occupation_count_at_mu"] == 3
    assert report["expected_hf_limit_reference_count_for_selected_blocking"] == 4
    assert not report["hf_limit_reference_count_matches"]
    assert report["whole_dirac_branch"]["number_parity"] == -1


def test_hf_limit_hole_bracket_refuses_missing_unoccupied_mode_cleanly():
    energies = torch.tensor([-20., 1., 2., 3., 4.], dtype=torch.float64)
    h = torch.diag(energies).to(torch.complex128)
    block = torch.zeros(5, dtype=torch.complex128)
    block[1] = 1.
    try:
        compare_fixed_fields(
            h, torch.zeros_like(h), mass_mev=10., target_number=3,
            blocked_basis_vector=block, continuation_steps=1,
            minimum_cutoff_distance_mev=1., minimum_band_separation_mev=2.,
        )
    except RuntimeError as exc:
        assert "hole-reference bracket" in str(exc)
    else:
        raise AssertionError("HF-limit hole bracket should be refused without an empty mode")
