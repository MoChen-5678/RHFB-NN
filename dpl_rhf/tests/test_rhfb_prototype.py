from __future__ import annotations

import json

import torch

from dpl_rhf.rhfb.prototype import (
    _optimize_moving_projector,
    _toy_model,
    _field_dependent_candidate_gradient_check,
    run_prototype_validation,
)


def test_odd_hfb_prototype_runs_and_reports_training_eigensolve_counts(tmp_path):
    report = run_prototype_validation(tmp_path / "prototype.json", epochs=12, seed=73)
    saved = json.loads((tmp_path / "prototype.json").read_text())
    # The independent moving-projector oracle optimizes the same constrained
    # functional by exact reference projectors and finite differences. The
    # separate frozen positive-space SCF remains a deliberately different
    # diagnostic and is expected not to be stationary under a moving P.
    assert report["status"] == "passed"
    assert report["status_scope"] == "finite_toy_variational_algorithm_gate_only"
    assert "standard no-sea RHFB remains unvalidated" in report["production_readiness"]
    assert report["gates"]["fixed_field_reference_converged"]
    assert report["gates"]["density_dependent_reference_converged"]
    assert report["gates"]["independent_reference_gradient_stable_fd_checked"]
    assert not report["gates"]["independent_reference_moving_stationarity_le_1e-6"]
    reference_tangent = report["cases"]["density_dependent_h"][
        "moving_projector_gradient_at_independent_reference"]
    assert reference_tangent["number_tangent_projected_gradient_norm"] > 1e-6
    oracle = report["cases"]["density_dependent_h"]["moving_projector_full_variational_oracle"]
    assert oracle["optimizer_success"]
    assert oracle["particle_number_residual"] <= 1e-6
    assert oracle["projector_fixed_point_residual"] <= 1e-10
    assert oracle["number_tangent_stationarity_fd"] <= 1e-6
    deviations = report["cases"]["density_dependent_h"]["moving_projector_variational_deviations"]
    assert deviations["physical_energy_absolute"] < 2e-3
    assert deviations["rho_relative_frobenius"] < 1e-3
    assert report["gates"]["density_dependent_self_consistent_equivalence"] is False
    assert saved["solver_counts"]["training_dirac_eigensolves"] == 0
    assert saved["solver_counts"]["training_bogoliubov_eigensolves"] == 0
    assert saved["solver_counts"]["subspace_refreshes"] == 0
    assert saved["solver_counts"]["reference_only_dirac_eigensolves"] > 0
    assert saved["solver_counts"]["reference_only_moving_variational_oracle_dirac_eigensolves"] > 0
    assert saved["solver_counts"]["reference_only_moving_variational_oracle_bogoliubov_eigensolves"] == 0
    assert "density_dependent_h" in saved["cases"]
    assert "sea_leakage_residual" in saved["maxima"]


def test_candidate_training_path_never_calls_an_eigensolver(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("candidate training path called an eigensolver")

    monkeypatch.setattr(torch.linalg, "eigh", forbidden)
    monkeypatch.setattr(torch.linalg, "eigvalsh", forbidden)
    monkeypatch.setattr(torch.linalg, "eig", forbidden)
    model = _toy_model(coupling=0.012, device=torch.device("cpu"))
    _optimize_moving_projector(model, epochs=10, seed=11, fixed_h=False)


def test_field_dependent_candidate_energy_gradient_matches_finite_difference():
    model = _toy_model(coupling=0.012, device=torch.device("cpu"))
    check = _field_dependent_candidate_gradient_check(model, seed=19)
    assert check["relative_error"] < 1.0e-3, check


def test_cross_gap_pair_kernel_field_matches_energy_finite_difference():
    model = _toy_model(
        coupling=0.0, device=torch.device("cpu"), pair_kernel_mixing=0.2
    )
    chi = torch.zeros((10, 10), dtype=torch.complex128)
    direction = torch.zeros_like(chi)
    chi[0, 1] = 0.3 + 0.2j
    chi[1, 0] = -chi[0, 1]
    direction[0, 1] = -0.2 + 0.1j
    direction[1, 0] = -direction[0, 1]
    zero_rho = torch.zeros_like(chi)
    step = 1.0e-6
    plus = model.energy(zero_rho, chi + step * direction)
    minus = model.energy(zero_rho, chi - step * direction)
    finite_difference = float(((plus - minus) / (2.0 * step)).detach())
    analytic = float(torch.sum(model.pairing_field(chi).conj() * direction).real.detach())
    assert abs(finite_difference - analytic) < 1.0e-8
    assert torch.linalg.matrix_norm(model.pairing_field(chi)[5:, :5]).item() > 1.0e-3
