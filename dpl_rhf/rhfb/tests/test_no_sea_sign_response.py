"""Isolated matrix-sign response tests; execute only through the RHFB launcher."""
from __future__ import annotations

import math

import pytest
import torch

from dpl_rhf.rhfb.no_sea_sign_response import (
    positive_energy_projector,
    projector_source_contraction,
)


def _options():
    return {
        "shift": 0.0,
        "spectral_gap": 2.0,
        "spectral_bound": 2.3,
        "bound_certificate": "analytic toy-family enclosure: h=[[ -2,t ],[t,2]], |t|<=1",
        "quadrature_order": 131,
        "log_step": 0.4,
        "tolerance": 1.0e-8,
    }


def _toy_hamiltonian(coupling):
    return torch.stack((
        torch.stack((coupling.new_tensor(-2.0), coupling)),
        torch.stack((coupling, coupling.new_tensor(2.0))),
    )).to(torch.complex128)


def test_positive_projector_matches_closed_form_without_eigensolver(monkeypatch):
    coupling = torch.tensor(0.35, dtype=torch.float64)
    h = _toy_hamiltonian(coupling)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("matrix-sign projection must use shifted solves, not eigensolvers")

    for name in ("eig", "eigh", "eigvals", "eigvalsh"):
        monkeypatch.setattr(torch.linalg, name, forbidden)
    result = positive_energy_projector(h, **_options())
    energy = math.sqrt(4.0 + 0.35**2)
    expected = 0.5 * (torch.eye(2, dtype=torch.complex128) + h / energy)
    torch.testing.assert_close(result.projector, expected, atol=2.0e-8, rtol=2.0e-8)
    assert result.physical_no_sea_prescription_certified is False
    assert result.field_response_in_autograd_graph is False
    assert result.diagnostics.quadrature_error_bound <= 1.0e-8


def test_projector_source_contraction_keeps_field_response_through_solves():
    coupling = torch.tensor(0.35, dtype=torch.float64, requires_grad=True)
    h = _toy_hamiltonian(coupling)
    source = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    value, result = projector_source_contraction(h, source, **_options())
    derivative = torch.autograd.grad(value, coupling)[0]

    # For h=[[-a,t],[t,a]], Q+=(I+h/sqrt(a^2+t^2))/2.  The symmetric
    # off-diagonal source therefore has d ReTr(C Q+)/dt = a^2/E^3.
    expected = 4.0 / (4.0 + 0.35**2) ** 1.5
    assert result.field_response_in_autograd_graph
    torch.testing.assert_close(derivative, torch.tensor(expected, dtype=torch.float64),
                               atol=2.0e-8, rtol=2.0e-8)


def test_spectral_enclosure_is_mandatory_and_not_inferred():
    h = torch.diag(torch.tensor([-2.0, 2.0], dtype=torch.complex128))
    options = _options()
    options.pop("bound_certificate")
    with pytest.raises(TypeError, match="bound_certificate"):
        positive_energy_projector(h, **options)


def test_projector_source_shape_and_finiteness_are_checked():
    h = torch.diag(torch.tensor([-2.0, 2.0], dtype=torch.complex128))
    with pytest.raises(ValueError, match="same shape"):
        projector_source_contraction(h, torch.ones(3, dtype=torch.complex128), **_options())
    source = torch.full((2, 2), complex(float("nan"), 0), dtype=torch.complex128)
    with pytest.raises(ValueError, match="non-finite"):
        projector_source_contraction(h, source, **_options())
