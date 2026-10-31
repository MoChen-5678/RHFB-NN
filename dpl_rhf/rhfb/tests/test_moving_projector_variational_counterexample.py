"""Tiny analytic witness: active stationarity need not include moving-P response."""
import torch

from dpl_rhf.rhfb.parity_diagnostic import number_parity_from_density


def _generalized_density(rho, chi):
    eye = torch.eye(rho.shape[0], dtype=rho.dtype)
    return torch.cat((torch.cat((rho, chi), 1),
                      torch.cat((-chi.conj(), eye - rho.conj()), 1)), 0)


def test_pure_odd_active_stationary_state_has_nonzero_composed_pair_response(monkeypatch):
    spectral_calls = {name: 0 for name in ("eig", "eigvals", "eigh", "eigvalsh")}

    def forbid(name):
        def call(*_args, **_kwargs):
            spectral_calls[name] += 1
            raise AssertionError(f"analytic example called torch.linalg.{name}")
        return call

    for name in spectral_calls:
        monkeypatch.setattr(torch.linalg, name, forbid(name))
    dtype = torch.complex128
    j = torch.tensor([[0, 1], [-1, 0]], dtype=dtype)
    rho = torch.diag(torch.tensor([.2, .2, .8, .8, 1.], dtype=torch.float64)).to(dtype)
    chi = torch.zeros((5, 5), dtype=dtype)
    chi[:2, :2] = .4 * j
    chi[2:4, 2:4] = .4 * j
    active_r = _generalized_density(rho, chi)
    assert torch.allclose(active_r @ active_r, active_r, atol=1e-14, rtol=0.)
    assert torch.allclose(rho - rho @ rho, chi @ chi.mH, atol=1e-14, rtol=0.)
    assert torch.allclose(rho @ chi, chi @ rho.conj(), atol=1e-14, rtol=0.)
    assert torch.allclose(torch.trace(rho).real,
                          torch.tensor(3., dtype=torch.float64), atol=0., rtol=0.)

    xi = torch.diag(torch.tensor([.3, .3, -.3, -.3, -.5], dtype=torch.float64)).to(dtype)
    delta_active = torch.zeros((5, 5), dtype=dtype)
    delta_active[:2, :2] = -.4 * j
    delta_active[2:4, 2:4] = -.4 * j
    h_active = torch.cat((torch.cat((xi, delta_active), 1),
                          torch.cat((-delta_active.conj(), -xi.conj()), 1)), 0)
    assert torch.allclose(h_active @ active_r - active_r @ h_active,
                          torch.zeros_like(h_active), atol=1e-14, rtol=0.)

    # Number-conserving pair-angle tangent: dN_a=+1.6 t and dN_b=-1.6 t.
    t = 1.0
    delta_rho_a = .8 * t * torch.eye(2, dtype=torch.float64)
    delta_rho_b = -.8 * t * torch.eye(2, dtype=torch.float64)
    delta_number_a = float(torch.trace(delta_rho_a))
    delta_number_b = float(torch.trace(delta_rho_b))
    assert delta_number_a + delta_number_b == 0.
    # Add one excluded state s1; P selects the first five coordinates.
    p = torch.diag(torch.tensor([1., 1., 1., 1., 1., 0.], dtype=torch.float64)).to(dtype)
    h = torch.diag(torch.tensor([2., 2., 1.4, 1.4, 1.2, -2.], dtype=torch.float64)).to(dtype)
    rho_full = torch.zeros((6, 6), dtype=dtype)
    chi_full = torch.zeros_like(rho_full)
    rho_full[:5, :5] = rho
    chi_full[:5, :5] = chi
    parity = number_parity_from_density(rho_full.numpy(), chi_full.numpy())
    assert parity["number_parity"] == -1
    assert parity["particle_number"] == 3.0

    # Explicit Hermitian scalar source.  Its active diagonal samples pair a;
    # its sea/p1 matrix element moves the spectral projector.  Include that
    # projector-induced density in the source closure, not only the active
    # density tangent.
    source = torch.zeros((6, 6), dtype=dtype)
    source[0, 0] = source[1, 1] = 1.
    source[5, 0] = source[0, 5] = 1.
    x0 = torch.trace(source @ rho_full).real
    g = 10. / 41.
    h0 = h - g * x0 * source
    assert torch.allclose(h0 + g * x0 * source, h, atol=1e-14, rtol=0.)

    delta_rho_active = torch.zeros_like(rho_full)
    delta_rho_active[:2, :2] = delta_rho_a.to(dtype)
    delta_rho_active[2:4, 2:4] = delta_rho_b.to(dtype)
    delta_x_active = torch.trace(source @ delta_rho_active).real
    assert torch.allclose(delta_x_active, torch.tensor(1.6, dtype=torch.float64),
                          atol=1e-14, rtol=0.)

    # Linearized exact-projector response: the p1/sea gap is four, so
    # deltaP[sea,p1] = delta h[sea,p1]/4 = q/4.  The off-diagonal density
    # feedback contributes 2 rho_p1 deltaP[sea,p1] = 0.1 q to delta x.
    # Solve q = g (1.6 + 0.1 q), giving q=0.4 for g=10/41.
    q = g * float(delta_x_active) / (1. - .1 * g)
    assert abs(q - .4) < 1e-15
    dp = torch.zeros_like(p)
    dp[5, 0] = q / (2 * 2.)
    dp[0, 5] = q / (2 * 2.)
    assert torch.allclose(p @ dp @ p, torch.zeros_like(p), atol=0., rtol=0.)
    assert torch.allclose((torch.eye(6, dtype=dtype) - p) @ dp @
                          (torch.eye(6, dtype=dtype) - p),
                          torch.zeros_like(p), atol=0., rtol=0.)

    delta_rho_projector = dp @ rho_full @ p + p @ rho_full @ dp
    delta_rho_total = delta_rho_active + delta_rho_projector
    delta_x_projector = torch.trace(source @ delta_rho_projector).real
    delta_x_total = torch.trace(source @ delta_rho_total).real
    assert torch.allclose(delta_x_projector, torch.tensor(.04, dtype=torch.float64),
                          atol=1e-14, rtol=0.)
    assert torch.allclose(delta_x_total, torch.tensor(1.64, dtype=torch.float64),
                          atol=1e-14, rtol=0.)
    assert abs(g * float(delta_x_total) - q) < 1e-15
    delta_chi_projector = dp @ chi_full @ p.T + p @ chi_full @ dp.T
    delta_full = torch.zeros_like(chi_full)
    delta_full[:2, :2] = -.4 * j
    delta_full[2:4, 2:4] = -.4 * j
    delta_full[5, 1] = .3
    delta_full[1, 5] = -.3

    normal_response = torch.trace(h @ delta_rho_projector).real
    pairing_response = torch.sum(delta_full.conj() * delta_chi_projector).real
    assert abs(float(normal_response)) < 1e-14
    assert torch.allclose(delta_chi_projector[5, 1], torch.tensor(.04, dtype=dtype),
                          atol=1e-14, rtol=0.)
    assert torch.allclose(delta_chi_projector[1, 5], torch.tensor(-.04, dtype=dtype),
                          atol=1e-14, rtol=0.)
    assert torch.allclose(pairing_response, torch.tensor(.024, dtype=torch.float64),
                          atol=1e-14, rtol=0.)
    assert spectral_calls == {name: 0 for name in spectral_calls}
