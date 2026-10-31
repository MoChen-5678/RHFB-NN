"""Finite-dimensional regression checks for the candidate nucleon-space chart."""
import torch

from dpl_rhf.rhfb.manifold import slater_reference, transform_reference
from dpl_rhf.rhfb.no_sea_particle_space import (
    active_nambu_isometry,
    active_stationarity_residual,
    density_purity_residual,
    embed_active_density,
    positive_frame_from_projector,
    projector_frechet_derivative_reference,
    projector_diagnostics,
)


def _torch_seed(seed=17):
    torch.manual_seed(seed)
    return torch


def test_active_odd_hfb_density_embeds_as_a_pure_full_density():
    _torch_seed()
    d, m = 5, 3
    raw = torch.randn((d, m), dtype=torch.complex128)
    frame, _ = torch.linalg.qr(raw, mode="reduced")
    reference = slater_reference(torch.tensor([1., 0., 0.], dtype=torch.float64))
    a = torch.zeros((m, m), dtype=torch.complex128)
    b = torch.zeros_like(a)
    b[1, 2], b[2, 1] = 0.31, -0.31
    active = transform_reference(reference, a, b)

    full = embed_active_density(
        frame, active.rho, active.chi,
        fermion_parity=active.fermion_parity,
    )

    assert active.fermion_parity == -1
    assert torch.allclose(full.matrix @ full.matrix, full.matrix, atol=2.0e-12, rtol=0.0)
    assert float(density_purity_residual(full)) < 2.0e-12
    assert torch.allclose(full.chi.mT, -full.chi, atol=1.0e-12, rtol=0.0)
    assert abs(float(torch.trace(full.rho).real - torch.trace(active.rho).real)) < 1.0e-12


def test_active_commutator_is_projection_of_full_commutator():
    _torch_seed(29)
    d, m = 4, 2
    frame, _ = torch.linalg.qr(torch.randn((d, m), dtype=torch.complex128), mode="reduced")
    reference = slater_reference(torch.tensor([1., 0.], dtype=torch.float64))
    full = embed_active_density(frame, reference.rho, reference.chi, fermion_parity=-1)
    raw = torch.randn((2 * d, 2 * d), dtype=torch.complex128)
    h_bdg = 0.5 * (raw + raw.mH)

    rms, active_commutator = active_stationarity_residual(h_bdg, full, frame)
    w = active_nambu_isometry(frame)
    projected_full = w.mH @ (h_bdg @ full.matrix - full.matrix @ h_bdg) @ w

    assert torch.allclose(active_commutator, projected_full, atol=2.0e-12, rtol=0.0)
    assert abs(float(rms) - float(torch.sqrt(torch.mean(torch.abs(projected_full).square())))) < 1.0e-12


def test_projector_frame_chart_and_response_reference():
    d, m = 4, 2
    h = torch.diag(torch.tensor([-2.0, -0.8, 0.9, 2.4], dtype=torch.float64)).to(torch.complex128)
    exact_projector = torch.diag(torch.tensor([0., 0., 1., 1.], dtype=torch.float64)).to(torch.complex128)
    reference = torch.eye(d, dtype=torch.complex128)[:, 2:]
    frame = positive_frame_from_projector(exact_projector, reference)
    q_from_frame = frame @ frame.mH
    diag = projector_diagnostics(h, exact_projector)
    assert diag.hermiticity_error < 1.0e-12
    assert diag.idempotency_error < 1.0e-12
    assert diag.commutator_error < 1.0e-12
    assert torch.allclose(q_from_frame, exact_projector, atol=1.0e-12, rtol=0.0)

    perturbation = torch.zeros_like(h)
    perturbation[0, 2], perturbation[2, 0] = 0.7, 0.7
    derivative = projector_frechet_derivative_reference(h, perturbation)
    step = 2.0e-6
    plus = torch.linalg.eigh(h + step * perturbation)
    minus = torch.linalg.eigh(h - step * perturbation)
    q_plus = plus.eigenvectors[:, plus.eigenvalues > 0.0] @ plus.eigenvectors[:, plus.eigenvalues > 0.0].mH
    q_minus = minus.eigenvectors[:, minus.eigenvalues > 0.0] @ minus.eigenvectors[:, minus.eigenvalues > 0.0].mH
    finite_difference = (q_plus - q_minus) / (2.0 * step)
    assert torch.allclose(derivative, finite_difference, atol=3.0e-7, rtol=3.0e-7)


def test_candidate_active_chart_needs_no_eigensolver(monkeypatch):
    d, m = 3, 2
    projector = torch.diag(torch.tensor([1., 1., 0.], dtype=torch.float64)).to(torch.complex128)
    reference = torch.eye(d, dtype=torch.complex128)[:, :m]
    frame = positive_frame_from_projector(projector, reference)
    active = slater_reference(torch.tensor([1., 0.], dtype=torch.float64))

    def forbid_eigh(*_args, **_kwargs):
        raise AssertionError("candidate active-space path called eigh")

    monkeypatch.setattr(torch.linalg, "eigh", forbid_eigh)
    full = embed_active_density(frame, active.rho, active.chi, fermion_parity=-1)
    h_nambu = torch.eye(2 * d, dtype=torch.complex128)
    residual, _ = active_stationarity_residual(h_nambu, full, frame)
    assert float(density_purity_residual(full)) < 1.0e-12
    assert torch.isfinite(residual)


def test_frame_gauge_change_requires_density_transport_or_fixed_chart():
    """The same P alone does not identify the same paired blocked state."""
    _torch_seed(6221)
    d, m = 6, 4
    old_frame, _ = torch.linalg.qr(torch.randn(d, m, dtype=torch.complex128))
    gauge, _ = torch.linalg.qr(torch.randn(m, m, dtype=torch.complex128))
    new_frame = old_frame @ gauge
    reference = slater_reference(torch.tensor([1., 0., 0., 0.], dtype=torch.float64))
    a = .17 * torch.randn(m, m, dtype=torch.complex128)
    b = .23 * torch.randn(m, m, dtype=torch.complex128)
    active = transform_reference(reference, a - a.mH, b - b.mT)
    original = embed_active_density(old_frame, active.rho, active.chi, fermion_parity=-1)
    unchanged_coordinates = embed_active_density(new_frame, active.rho, active.chi,
                                                  fermion_parity=-1)
    transported = embed_active_density(
        new_frame, gauge.mH @ active.rho @ gauge,
        gauge.mH @ active.chi @ gauge.conj(), fermion_parity=-1,
    )
    torch.testing.assert_close(old_frame @ old_frame.mH, new_frame @ new_frame.mH,
                               atol=2e-12, rtol=0.)
    torch.testing.assert_close(original.matrix, transported.matrix, atol=2e-12, rtol=0.)
    assert float(torch.linalg.matrix_norm(original.matrix - unchanged_coordinates.matrix)) > .1
    assert float(torch.linalg.matrix_norm(active.chi)) > .1
    # A chart formed from P and a fixed seed is independent of the spectral
    # eigenvectors' internal phase/rotation gauge.
    first = positive_frame_from_projector(old_frame @ old_frame.mH, old_frame)
    second = positive_frame_from_projector(new_frame @ new_frame.mH, old_frame)
    torch.testing.assert_close(first, second, atol=2e-12, rtol=0.)
