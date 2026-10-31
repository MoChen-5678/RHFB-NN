import pytest
import numpy as np
import torch

from dpl_rhf.rhfb.full_branch_neural_training import (
    _model_forward,
    _objective,
    _forbid_spectra,
    _prepare_species_models,
    _resolve_device,
    _validate_settings,
)
import dpl_rhf.rhfb.full_branch_neural_training as neural_training
from dpl_rhf.rhfb.manifold import GeneralizedDensity, generalized_density_matrix
from dpl_rhf.rhfb.full_branch_stationarity import nambu_hamiltonian


class _FixedPureState(torch.nn.Module):
    def __init__(self):
        super().__init__()
        dtype = torch.complex128
        self.register_buffer("q", torch.eye(3, dtype=dtype))
        rho = torch.diag(torch.tensor([1.0, 0.5, 0.5], dtype=torch.float64)).to(dtype)
        chi = torch.zeros((3, 3), dtype=dtype)
        chi[1, 2], chi[2, 1] = 0.5, -0.5
        self.gd = GeneralizedDensity(rho, chi, generalized_density_matrix(rho, chi), -1)

    def forward(self):
        return self.q, self.gd


class _ScaledPureState(_FixedPureState):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.25, dtype=torch.float64))
        self.reference_parity = -1

    def forward(self):
        q = torch.sin(self.scale.to(torch.complex128) * self.q)
        return q, self.gd


def test_full_objective_matches_dense_nambu_commutator_and_differentiates_mu():
    dtype = torch.complex128
    model = _FixedPureState()
    h = torch.diag(torch.tensor([-0.4, 0.3, 1.2], dtype=torch.float64)).to(dtype)
    delta = torch.tensor([
        [0.0, 0.04 + 0.01j, -0.03j],
        [-0.04 - 0.01j, 0.0, 0.02],
        [0.03j, -0.02, 0.0],
    ], dtype=dtype)
    mu = torch.tensor(0.17, dtype=torch.float64, requires_grad=True)
    objective, parts = _objective(model, h, delta, mu, 2, chunk_size=2, number_weight=1.0e4)
    h_mu = nambu_hamiltonian(h, delta, mu)
    commutator = h_mu @ model.gd.matrix - model.gd.matrix @ h_mu
    expected = torch.linalg.matrix_norm(commutator).square() / float((2 * h.shape[0]) ** 2)
    torch.testing.assert_close(parts["element_mean_square_residual_mev2"], expected)
    torch.testing.assert_close(objective, expected)
    objective.backward()
    assert mu.grad is not None and torch.isfinite(mu.grad) and abs(float(mu.grad)) > 1e-10


@pytest.mark.parametrize("active,updates", [(40, 20), (42, 19), (42, 51)])
def test_training_admission_requires_m41_and_twenty_to_fifty_updates(active, updates):
    with pytest.raises(ValueError):
        _validate_settings(active, updates, 1.0e-4, 8)


def test_training_admission_rejects_invalid_stepsize_and_chunk():
    with pytest.raises(ValueError):
        _validate_settings(42, 20, 0.0, 8)
    with pytest.raises(ValueError):
        _validate_settings(42, 20, 1.0e-4, 0)


def test_non_reentrant_checkpoint_preserves_autograd_and_counts_recomputation():
    checkpointed = _ScaledPureState()
    ordinary = _ScaledPureState()
    ordinary.load_state_dict(checkpointed.state_dict())
    counts = {"checkpointed_model_forward_executions": 0,
              "backward_recompute_calls": 0}
    q, gd = _model_forward(checkpointed, checkpoint_recompute=True,
                           recompute_counts=counts)
    loss = q.square().real.sum() + gd.rho.real.sum()
    loss.backward()
    ordinary_q, ordinary_gd = _model_forward(ordinary, checkpoint_recompute=False)
    (ordinary_q.square().real.sum() + ordinary_gd.rho.real.sum()).backward()
    assert checkpointed.scale.grad is not None and torch.isfinite(checkpointed.scale.grad)
    torch.testing.assert_close(checkpointed.scale.grad, ordinary.scale.grad)
    assert counts["checkpointed_model_forward_executions"] >= 1
    assert counts["backward_recompute_calls"] >= 1


def test_cpu_device_must_be_explicitly_selected_for_small_checks():
    assert _resolve_device("cpu") == torch.device("cpu")


def test_fixed_grid_model_preparation_precedes_optimization_spectrum_guard(monkeypatch, tmp_path):
    calls = []

    def fake_build(checkpoint, density, *, species, active_dimension, device):
        calls.append((species, device))
        # np.leggauss internally calls eigvalsh to build the fixed angular grid.
        np.polynomial.legendre.leggauss(4)
        return object(), object(), {"species": species}

    monkeypatch.setattr(neural_training, "build_enlarged_o17_model", fake_build)
    prepared = _prepare_species_models(tmp_path / "checkpoint", tmp_path / "density", 42)
    assert calls == [("n", "cpu"), ("p", "cpu")]
    assert set(prepared) == {"n", "p"}
    originals, counts = _forbid_spectra()
    try:
        assert all(count == 0 for count in counts.values())
    finally:
        neural_training._restore_spectra(originals)
