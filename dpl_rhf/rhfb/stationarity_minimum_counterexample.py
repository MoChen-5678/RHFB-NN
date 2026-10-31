"""Guarded analytic odd-HF counterexample to residual-based minimum claims.

One particle occupies one of two positive one-body levels. This HF subset of
HFB already has a ground state and an excited stationary state with exactly
the same zero commutator residual. No nuclear calculation or Dirac-sea
prescription is supplied by this algebra diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

from .execution_limits import require_bounded_cgroup
from .runtime_resources import resource_snapshot
from .selfconsistent_neural_training import _forbid_spectral_calls


def _evaluate_reference(*, excited: bool) -> dict:
    angle = torch.zeros((), dtype=torch.float64, requires_grad=True)
    c, s = torch.cos(angle), torch.sin(angle)
    q = torch.stack((s, c) if excited else (c, s)).to(torch.complex128)
    rho = torch.outer(q, q.conj())
    chi = torch.zeros_like(rho)
    zero = torch.zeros_like(rho)
    identity = torch.eye(2, dtype=rho.dtype)
    h = torch.diag(torch.tensor([1.0, 2.0], dtype=torch.complex128))
    generalized = torch.cat((torch.cat((rho, chi), dim=1),
                             torch.cat((-chi.conj(), identity-rho.conj()), dim=1)), dim=0)
    h_nambu = torch.cat((torch.cat((h, zero), dim=1),
                         torch.cat((zero, -h.conj()), dim=1)), dim=0)
    commutator = h_nambu @ generalized - generalized @ h_nambu
    # Write the smooth quadratic polynomial explicitly. Differentiating
    # abs(z).square() twice at z=0 can inherit abs's nonsmooth AD rule.
    residual_squared = (commutator.real.square()+commutator.imag.square()).sum()
    energy = torch.trace(h @ rho).real
    energy_slope, = torch.autograd.grad(energy, angle, create_graph=True, retain_graph=True)
    energy_curvature, = torch.autograd.grad(energy_slope, angle, retain_graph=True)
    residual_slope, = torch.autograd.grad(residual_squared, angle, create_graph=True,
                                        retain_graph=True)
    residual_curvature, = torch.autograd.grad(residual_slope, angle)
    return {
        "energy_mev": float(energy.detach()),
        "energy_directional_derivative_mev": float(energy_slope.detach()),
        "energy_directional_curvature_mev": float(energy_curvature),
        "full_nambu_commutator_squared_mev2": float(residual_squared.detach()),
        "residual_loss_directional_derivative_mev2": float(residual_slope.detach()),
        "residual_loss_directional_curvature_mev2": float(residual_curvature),
        "particle_number": float(torch.trace(rho).real.detach()),
        "number_parity": -1,
        "purity_error": float(torch.linalg.matrix_norm(
            generalized @ generalized-generalized).detach()),
    }


def run(output: Path) -> dict:
    limits = require_bounded_cgroup()
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / 'source.py')
    report = {
        "status": "running", "scope": "analytic two-level N=1 HF subset of HFB; not a nucleus",
        "kernel_limits": limits, "physical_result_accepted": False,
        "nuclear_solver_validated": False, "eigensolver_used": False,
        "basis_refreshes": 0, "fortran_used": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "paper": "https://arxiv.org/pdf/1104.5453; Eqs.14-19 and Eq.23",
        "derivation_status": "counterexample constructed here, not quoted from the paper",
    }
    path = output / "diagnostic.json"

    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(path)

    save()
    try:
        with _forbid_spectral_calls():
            ground = _evaluate_reference(excited=False)
            excited = _evaluate_reference(excited=True)
        report.update(ground=ground, excited=excited)
        for state in (ground, excited):
            assert state["particle_number"] == 1.0
            assert state["purity_error"] <= 1e-14
            assert state["full_nambu_commutator_squared_mev2"] <= 1e-14
            assert state["residual_loss_directional_curvature_mev2"] > 0.0
        assert ground["energy_mev"] < excited["energy_mev"]
        assert ground["energy_directional_curvature_mev"] == 2.0
        assert excited["energy_directional_curvature_mev"] == -2.0
        report.update(status="passed_counterexample", ground=ground, excited=excited,
                      conclusion="Zero residual, exact N and odd parity do not certify an energy minimum.",
                      limitation="A negative admissible energy curvature disproves a local minimum; "
                                 "positive curvature along finitely many probes cannot prove one.")
    except Exception as error:
        report.update(status="failed_counterexample", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["resources"] = resource_snapshot()
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output), indent=2))


if __name__ == '__main__':
    main()
