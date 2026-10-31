"""Small orchestration smoke test; this is not an O17 or solver test."""
import json
from types import SimpleNamespace

import torch
import pytest
from torch import nn

from dpl_rhf.rhfb.manifold import GeneralizedDensity, generalized_density_matrix
from dpl_rhf.rhfb.selfconsistent_neural_training import train_joint


class _TinyNet(nn.Module):
    reference_parity = 1
    blocked_index = None

    def __init__(self):
        super().__init__()
        self.radial_encoder = nn.Parameter(torch.tensor(0.04, dtype=torch.float64))
        self.output_head = nn.Parameter(torch.tensor(-0.02, dtype=torch.float64))

    def forward(self):
        angle = self.radial_encoder + self.output_head
        q = torch.stack((torch.cos(angle), torch.sin(angle)))[:, None].to(torch.complex128)
        rho = torch.zeros((1, 1), dtype=torch.complex128)
        chi = torch.zeros_like(rho)
        return q, GeneralizedDensity(rho, chi, generalized_density_matrix(rho, chi), 1)


class _ZeroAction:
    def __init__(self, dimension):
        self.dimension = dimension

    def apply(self, value):
        return torch.zeros_like(value)


def _toy_evaluator(_basis, states, _mus, _targets, **_kwargs):
    qn, qp = states["n"].q, states["p"].q
    loss = (qn[0, 0].real - 0.72).square() + (qp[0, 0].real - 0.68).square()
    zero = loss.new_zeros(())
    identity = torch.eye(2, dtype=torch.complex128)
    return SimpleNamespace(
        loss=loss, total_energy=zero, particle_hole_energy=zero,
        pairing_energy=zero,
        normal_fields={"n": identity, "p": identity},
        active_pairing_gaps={"n": states["n"].chi_active,
                             "p": states["p"].chi_active},
        delta_actions={"n": _ZeroAction(2), "p": _ZeroAction(2)},
    )


def test_joint_training_smoke_checkpoints_and_recomputes_fields(tmp_path):
    report = train_joint(
        {"n": _TinyNet(), "p": _TinyNet()},
        SimpleNamespace(dimension=2, radius_fm=1.0, spacing_fm=0.2,
                        kappa_max=0, point_count=1),
        {"n": 0, "p": 0}, 1, tmp_path / "joint-smoke",
        train_chemical_potentials=True, learning_rate=1e-2,
        number_weight=0.0, stationarity_evaluator=_toy_evaluator,
    )
    assert report["accepted_updates"] == 1
    assert report["physical_result_accepted"] is False
    assert report["no_sea_branch_certified"] is False
    assert report["fortran_used"] is False
    assert report["eigensolver_used"] is False
    assert report["resume_supported"] is True
    assert report["field_updates"] >= 3  # initial, gradient, and full trial forward
    assert (tmp_path / "joint-smoke" / "joint_step_0001.pt").is_file()
    assert (tmp_path / "joint-smoke" / "history.json").is_file()
    progress = json.loads((tmp_path / "joint-smoke" / "progress.json").read_text())
    assert progress["status"] == "complete"
    assert progress["current_phase"] == "accepted"
    assert progress["physical_identity_certified"] is False


def test_joint_training_resume_uses_full_accepted_checkpoint(tmp_path):
    basis = SimpleNamespace(dimension=2, radius_fm=1.0, spacing_fm=0.2,
                            kappa_max=0, point_count=1)
    output = tmp_path / "joint-resume"
    first = train_joint(
        {"n": _TinyNet(), "p": _TinyNet()}, basis, {"n": 0, "p": 0},
        1, output, train_chemical_potentials=True, learning_rate=1e-2,
        number_weight=0.0, stationarity_evaluator=_toy_evaluator,
    )
    assert first["accepted_updates"] == 1
    resumed = train_joint(
        {"n": _TinyNet(), "p": _TinyNet()}, basis, {"n": 0, "p": 0},
        2, output, train_chemical_potentials=True, learning_rate=1e-2,
        number_weight=0.0, stationarity_evaluator=_toy_evaluator,
        resume_from=output / "joint_step_0001.pt",
    )
    assert resumed["accepted_updates"] >= 1
    assert resumed["resume_supported"] is True
    payload = torch.load(output / "joint_step_0001.pt", weights_only=False)
    assert payload["checkpoint_format"] == "joint-selfconsistent-neural-training-v2"
    assert payload["physical_identity_certified"] is False


def test_production_training_refuses_injected_incomplete_ph_provider(tmp_path):
    from dpl_rhf.rhf.parameters import pka1_parameters

    with pytest.raises(TypeError, match="complete PKA1 streamed normal functional"):
        train_joint(
            {"n": _TinyNet(), "p": _TinyNet()},
            SimpleNamespace(dimension=2, kappa_max=0), {"n": 0, "p": 0},
            0, tmp_path / "refused-provider",
            normal_functional=SimpleNamespace(params=pka1_parameters()),
        )


@pytest.mark.parametrize("mismatch", ["basis", "parameters", "lmax", "mass", "hbar_c"])
def test_production_training_refuses_mismatched_ph_physics(tmp_path, mismatch):
    from dataclasses import replace
    from dpl_rhf.rhfb.basis import AngularRadialBasis
    from dpl_rhf.rhfb.streamed_normal_functional import StreamedPrimitiveNormalFunctional
    from dpl_rhf.rhf.parameters import pka1_parameters

    basis = AngularRadialBasis(radius_fm=1.4, spacing_fm=0.2, kappa_max=1,
                               n_theta=3, n_phi=6, derivative_order=5)
    provider_basis = basis
    params = pka1_parameters()
    provider_lmax = 2
    expected = {
        "basis": "same primitive and angular basis",
        "parameters": "fixed PKA1 parameter values",
        "lmax": "lmax differs",
        "mass": "kinetic mass_n_mev",
        "hbar_c": "kinetic hbar_c_mev_fm",
    }[mismatch]
    if mismatch == "basis":
        # Same D/P does not mean the same physical mesh.
        provider_basis = AngularRadialBasis(radius_fm=2.8, spacing_fm=0.4, kappa_max=1,
                                            n_theta=3, n_phi=6, derivative_order=5)
        assert provider_basis.dimension == basis.dimension
        assert provider_basis.point_count == basis.point_count
    elif mismatch == "parameters":
        params = replace(params, g_sigma=params.g_sigma * 1.01)
    elif mismatch == "lmax":
        provider_lmax = 0
    elif mismatch == "mass":
        basis.mass_n_mev *= 1.01
    elif mismatch == "hbar_c":
        basis.hbar_c_mev_fm *= 1.01
    provider = StreamedPrimitiveNormalFunctional(provider_basis, params=params,
                                                 lmax=provider_lmax)
    with pytest.raises(ValueError, match=expected):
        train_joint({"n": _TinyNet(), "p": _TinyNet()}, basis, {"n": 0, "p": 0},
                    0, tmp_path / mismatch, normal_functional=provider, lmax=2)


def test_solved_multiplier_mode_is_recorded_and_cannot_resume_as_free_mu(tmp_path):
    basis = SimpleNamespace(dimension=2, radius_fm=1., spacing_fm=.2,
                            kappa_max=0, point_count=1)
    def evaluator(*args, **kwargs):
        assert kwargs["chemical_potential_mode"] == "number_tangent"
        result = _toy_evaluator(*args, **kwargs)
        result.species_diagnostics = {
            s: {"chemical_potential_mev": torch.tensor(-3.5 if s == "n" else -2.7,
                                                      dtype=torch.float64)} for s in ("n", "p")}
        return result
    output = tmp_path / "multiplier"
    report = train_joint({"n":_TinyNet(),"p":_TinyNet()},basis,{"n":0,"p":0},0,output,
                         train_chemical_potentials=False,chemical_potential_mode="number_tangent",
                         stationarity_evaluator=evaluator)
    assert report["chemical_potentials_mev"] == {"n":-3.5,"p":-2.7}
    with pytest.raises(ValueError,match="incompatible"):
        train_joint({"n":_TinyNet(),"p":_TinyNet()},basis,{"n":0,"p":0},0,output,
                    train_chemical_potentials=False, chemical_potential_mode="provided",
                    stationarity_evaluator=evaluator,resume_from=output/"joint_step_0000.pt")


def test_solved_multiplier_is_not_also_an_adam_parameter(tmp_path):
    with pytest.raises(ValueError,match="disable train_chemical_potentials"):
        train_joint({"n":_TinyNet(),"p":_TinyNet()},SimpleNamespace(dimension=2),
                    {"n":0,"p":0},0,tmp_path/"invalid",
                    chemical_potential_mode="number_tangent",train_chemical_potentials=True,
                    stationarity_evaluator=_toy_evaluator)
