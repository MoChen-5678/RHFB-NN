"""No-pairing Dirac support check, separate from quasiparticle energy signs."""
from dataclasses import replace

import torch

from dpl_rhf.rhfb.manifold import slater_reference
from dpl_rhf.rhfb.prototype import (
    _CanonicalCoordinates, _fixed_point_state, _metrics, _toy_model,
)


def test_zero_pairing_preserves_odd_slater_state_without_spectral_solves(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no-pairing candidate must not diagonalize")

    for name in ("eig", "eigh", "eigvalsh"):
        monkeypatch.setattr(torch.linalg, name, forbidden)
    model = replace(_toy_model(coupling=.012, device=torch.device("cpu")),
                    pairing_strength=0.)
    reference = slater_reference(torch.tensor([1., 1., 0., 0., 0.],
                                               dtype=torch.float64), blocked_index=2)
    coordinates = _CanonicalCoordinates(5, device=torch.device("cpu"), seed_pairing=False)
    seed = torch.eye(10, dtype=torch.complex128)[:, :5]
    state, q, residual, sign = _fixed_point_state(model, reference, coordinates,
                                                seed, iterations=8)
    metrics = _metrics(model, state, q, sign)
    assert state.fermion_parity == -1
    assert torch.count_nonzero(state.chi) == 0
    assert torch.count_nonzero(model.pairing_field(state.chi)) == 0
    assert metrics["particle_number_residual"] < 1e-10
    assert metrics["generalized_idempotency_residual"] < 1e-10
    assert metrics["sea_leakage_residual"] < 1e-8
    assert residual < 1e-10
