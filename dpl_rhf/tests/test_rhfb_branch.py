from __future__ import annotations

import torch

from dpl_rhf.rhfb.branch import (
    branch_stationarity_loss,
    no_sea_branch_reference,
    nambu_hamiltonian,
    select_ph_branches_reference_only,
    transform_branch_reference,
)
from dpl_rhf.rhfb.pairing import pairing_gap


def _branch_case():
    energies = torch.tensor([-1.0, 1.0, 2.0, -8.0], dtype=torch.float64)
    occupations = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    branch = no_sea_branch_reference(
        energies, occupations, dirac_cutoff=-4.0, chemical_potential=0.0
    )
    h = torch.diag(energies.to(torch.complex128))
    delta = torch.zeros_like(h)
    nambu = nambu_hamiltonian(h, delta, chemical_potential=0.0)
    return branch, h, delta, nambu


def test_no_sea_reference_selects_positive_energy_hole_for_dirac_sea():
    branch, _h, _delta, nambu = _branch_case()
    r = branch.density.matrix
    assert branch.density.fermion_parity == -1
    assert torch.trace(branch.density.rho).real.item() == 1.0
    assert branch.density.rho[3, 3].item() == 0.0
    assert branch.selected_bdg_energies.tolist() == [-1.0, -1.0, -2.0, 8.0]
    torch.testing.assert_close(nambu @ r, r @ nambu, rtol=0.0, atol=1e-14)

    # Filling the full negative BdG spectrum would occupy the negative-Dirac
    # particle state and leave its positive-energy hole empty: the wrong vacuum.
    values, vectors = torch.linalg.eigh(nambu)
    negative = vectors[:, values < 0.0]
    filled_negative = negative @ negative.mH
    assert filled_negative[3, 3].real.item() == 1.0
    assert not torch.allclose(filled_negative, r)


def test_independent_bdg_oracle_tracks_mixed_sign_pure_branch():
    branch, _h, _delta, nambu = _branch_case()
    frame, metrics = select_ph_branches_reference_only(nambu, branch.frame)
    selected = frame @ frame.mH
    torch.testing.assert_close(selected, branch.density.matrix, rtol=0.0, atol=1e-13)
    assert metrics["selected_positive_bdg_branch_count"] == 1.0
    assert metrics["particle_hole_complement_residual"] < 1e-12
    assert metrics["reference_only_eigensolves"] == 1.0


def test_reference_blocking_supports_exact_odd_addition_and_hole_parity():
    energies = torch.tensor([-1.0, 0.1, 1.2, -8.0], dtype=torch.float64)
    even_occupations = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    even = no_sea_branch_reference(
        energies, even_occupations, dirac_cutoff=-4.0, chemical_potential=0.0
    )
    added = no_sea_branch_reference(
        energies, even_occupations, dirac_cutoff=-4.0, chemical_potential=0.0,
        blocked_index=2, blocking_operation="particle",
    )
    removed = no_sea_branch_reference(
        energies, even_occupations, dirac_cutoff=-4.0, chemical_potential=0.0,
        blocked_index=1, blocking_operation="hole",
    )
    assert even.density.fermion_parity == 1
    assert added.density.fermion_parity == -1
    assert removed.density.fermion_parity == -1
    assert added.particle_number - even.particle_number == 1.0
    assert even.particle_number - removed.particle_number == 1.0
    assert removed.density.rho[1, 1].item() == 0.0


def test_branch_chart_preserves_parity_and_full_space_stationarity_without_eigs(monkeypatch):
    branch, _h, _delta, nambu = _branch_case()
    zeros = torch.zeros((4, 4), dtype=torch.complex128)
    identity = torch.eye(4, dtype=torch.complex128)
    a = 0.5 * (zeros - zeros.mH)
    b = 0.5 * (zeros - zeros.mT)
    state = transform_branch_reference(branch, a, b)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("training branch path called an eigensolver")

    monkeypatch.setattr(torch.linalg, "eigh", forbidden)
    monkeypatch.setattr(torch.linalg, "eigvalsh", forbidden)
    metrics = branch_stationarity_loss(
        nambu, state, branch, target_number=1.0
    )
    assert state.fermion_parity == -1
    assert metrics.commutator_residual.item() < 1e-14
    assert metrics.particle_number_residual.item() == 0.0
    assert metrics.branch_overlap.item() == 1.0
    assert metrics.sea_particle_occupation.item() == 0.0
    assert metrics.loss.item() < 1e-26


def test_d1s_pairing_energy_measures_effect_of_cross_branch_kappa():
    # A generic four-state spinor sample supplies nonzero D1S matrix elements
    # between positive- and negative-Dirac-labelled basis functions.
    torch.manual_seed(20261001)
    q = torch.randn((4, 4, 4), dtype=torch.complex128)
    z = torch.randn((4, 4), dtype=torch.complex128)
    chi = 0.1 * (z - z.T)
    chi[:2, :2] = 0.0
    chi[2:, 2:] = 0.0
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.7, 0.0], [0.3, 0.6, 0.2]],
        dtype=torch.float64,
    )
    weights = torch.full((4,), 0.25, dtype=torch.float64)
    q = q / torch.sqrt(torch.sum(weights[:, None, None] * torch.abs(q).square(), dim=(0, 1)))[None, None, :]
    full_energy, full_gap = pairing_gap(
        q, chi, positions, weights, species="neutron", chunk_size=2
    )
    # This is the χ=PχP^T approximation tested in the old prototype.
    projected_chi = chi.clone()
    projected_chi[2:, :] = 0.0
    projected_chi[:, 2:] = 0.0
    projected_energy, projected_gap = pairing_gap(
        q, projected_chi, positions, weights, species="neutron", chunk_size=2
    )
    assert torch.linalg.matrix_norm(chi[2:, :2]).item() > 1e-3
    assert abs(float(full_energy - projected_energy)) > 1e-5
    assert torch.linalg.matrix_norm(full_gap - projected_gap).item() > 1e-5
