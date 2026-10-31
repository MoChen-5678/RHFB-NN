from __future__ import annotations

import torch

from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.manifold import generalized_density_matrix
from dpl_rhf.rhfb.operators import DensityFactors
from dpl_rhf.rhfb.parity_diagnostic import number_parity_from_density
from dpl_rhf.rhfb.selfconsistent_stationarity import selfconsistent_stationarity_loss
from dpl_rhf.rhfb.streamed_normal_functional import StreamedPrimitiveNormalFunctional


class _ToyNormalFunctional:
    """Small differentiable cross-species field provider for interface tests."""

    def __init__(self, coupling=0.3):
        self.calls = 0
        self.coupling = coupling

    def differentiable_energy_and_fields(self, states):
        self.calls += 1
        rho = {s: states[s].q @ states[s].rho_active @ states[s].q.mH
               for s in ("n", "p")}
        mix = torch.zeros_like(rho["n"])
        mix[0, 2] = mix[2, 0] = 0.2
        energy = (torch.trace(rho["n"]).real + torch.trace(rho["p"]).real
                  + (rho["n"].conj() * mix).sum().real
                  + (rho["p"].conj() * mix).sum().real
                  + self.coupling * (rho["n"].conj() * rho["p"]).sum().real)
        eye = torch.eye(rho["n"].shape[0], dtype=torch.complex128,
                        device=rho["n"].device)
        return energy, {"n": eye + mix + self.coupling * rho["p"],
                        "p": eye + mix + self.coupling * rho["n"]}


class _ToyPairingAction:
    """A differentiable skew full-space action with the production API."""

    def __init__(self, basis, q, chi, **_kwargs):
        self.q, self.chi = q, chi
        self.delta = q @ chi @ q.T

    def apply(self, vector):
        return self.delta @ vector


def _pure_factors(*, q_requires_grad=False, chi_requires_grad=False):
    q = torch.zeros((3, 2), dtype=torch.complex128)
    q[:2] = torch.eye(2, dtype=torch.complex128)
    q.requires_grad_(q_requires_grad)
    rho = 0.5 * torch.eye(2, dtype=torch.complex128)
    chi = torch.tensor([[0.0, 0.5], [-0.5, 0.0]], dtype=torch.complex128)
    chi.requires_grad_(chi_requires_grad)
    return DensityFactors(q, rho, chi)


def test_joint_fields_recompute_and_preserve_cross_species_vjp(monkeypatch):
    import dpl_rhf.rhfb.selfconsistent_stationarity as module

    monkeypatch.setattr(module, "require_bounded_cgroup", lambda: None)
    normal = _ToyNormalFunctional()
    neutron = _pure_factors()
    proton = _pure_factors(q_requires_grad=True)
    states = {"n": neutron, "p": proton}
    kwargs = dict(
        normal_functional=normal,
        pairing_action_factory=_ToyPairingAction,
        residual_chunk_size=1,
        pairing_chunk_size=1,
        pairing_active_chunk_size=1,
    )

    first = selfconsistent_stationarity_loss(
        basis=type("Basis", (), {"dimension": 3, "kappa_max": 0})(),
        states=states, chemical_potentials={"n": 0.1, "p": -0.1},
        number_targets={"n": 1, "p": 1}, **kwargs)
    second = selfconsistent_stationarity_loss(
        basis=type("Basis", (), {"dimension": 3, "kappa_max": 0})(),
        states=states, chemical_potentials={"n": 0.1, "p": -0.1},
        number_targets={"n": 1, "p": 1}, **kwargs)

    assert normal.calls == 2
    assert first.physical_result_accepted is False
    assert first.no_sea_branch_certified is False
    assert first.normal_fields["n"].shape == (3, 3)
    assert first.delta_actions["n"].delta.shape == (3, 3)
    assert torch.isfinite(first.loss)
    (gradient,) = torch.autograd.grad(
        first.species_diagnostics["n"]["scaled_residual_loss"], proton.q,
        retain_graph=True)
    assert torch.isfinite(gradient).all()
    assert torch.linalg.vector_norm(gradient) > 0
    assert second.loss.requires_grad


def test_toy_residual_matches_dense_full_nambu_commutator(monkeypatch):
    import dpl_rhf.rhfb.selfconsistent_stationarity as module

    monkeypatch.setattr(module, "require_bounded_cgroup", lambda: None)
    states = {"n": _pure_factors(), "p": _pure_factors()}
    mus = {"n": 0.07, "p": -0.03}
    basis = type("Basis", (), {"dimension": 3, "kappa_max": 0})()
    result = selfconsistent_stationarity_loss(
        basis, states, mus, {"n": 1, "p": 1},
        normal_functional=_ToyNormalFunctional(),
        pairing_action_factory=_ToyPairingAction,
        residual_chunk_size=1, pairing_chunk_size=1,
        pairing_active_chunk_size=1,
    )

    species = "n"
    state = states[species]
    d = state.dimension
    eye = torch.eye(d, dtype=torch.complex128)
    delta = result.delta_actions[species].apply(eye)
    h = result.normal_fields[species]
    hamiltonian = torch.cat((torch.cat((h-mus[species]*eye, delta), 1),
                             torch.cat((-delta.conj(), -h.conj()+mus[species]*eye), 1)), 0)
    nambu_eye = torch.eye(2*d, dtype=torch.complex128)
    generalized_density = state.apply(nambu_eye)
    dense_commutator = hamiltonian @ generalized_density - generalized_density @ hamiltonian
    expected = dense_commutator.abs().square().sum() / float((2*d)**2)
    torch.testing.assert_close(result.species_diagnostics[species]["scaled_residual_loss"],
                               expected)


def test_blocked_odd_neutron_selfconsistent_field_loss_vjp_matches_finite_difference():
    """Guarded odd-A field-loss gradient test, not a nuclear solution."""
    basis = AngularRadialBasis(radius_fm=1.4, spacing_fm=0.2, kappa_max=1,
                               n_theta=3, n_phi=6, derivative_order=5)
    d = basis.dimension
    backend = StreamedPrimitiveNormalFunctional(
        basis, tile_size=8, chunk_size=32, lmax=2)

    def evaluate(direction):
        # Keep one occupied blocked neutron column fixed, rotate a paired Q
        # column, and move both species along pure M=2 BCS arcs.
        angle = torch.as_tensor(direction, dtype=torch.float64)
        zero = torch.zeros((), dtype=torch.float64)
        cosine, sine = torch.cos(0.17 * angle), torch.sin(0.17 * angle)
        column_n = torch.stack((cosine, zero, sine, *([zero] * (d - 3))))
        column_p = torch.stack((cosine, zero, sine, *([zero] * (d - 3))))
        second = torch.stack((zero, torch.ones_like(zero), *([zero] * (d - 2))))
        blocked = torch.zeros(d, dtype=torch.float64)
        blocked[3] = 1.0
        qn = torch.stack((blocked, column_n, second), dim=1).to(torch.complex128)
        qp = torch.stack((column_p, second), dim=1).to(torch.complex128)
        pair_angle = torch.as_tensor(0.61, dtype=torch.float64) + 0.13 * angle
        occupation = torch.sin(pair_angle).square()
        pair_amplitude = torch.sin(pair_angle) * torch.cos(pair_angle)
        rho_p = occupation.to(torch.complex128) * torch.eye(2, dtype=torch.complex128)
        skew_p = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.complex128)
        chi_p = pair_amplitude.to(torch.complex128) * skew_p
        rho_n = torch.diag(torch.stack((torch.ones_like(occupation), occupation,
                                        occupation))).to(torch.complex128)
        skew_n = torch.zeros((3, 3), dtype=torch.complex128)
        skew_n[1, 2], skew_n[2, 1] = 1.0, -1.0
        chi_n = pair_amplitude.to(torch.complex128) * skew_n
        states = {"n": DensityFactors(qn, rho_n, chi_n),
                  "p": DensityFactors(qp, rho_p, chi_p)}
        assert torch.allclose(generalized_density_matrix(rho_n, chi_n)
                              @ generalized_density_matrix(rho_n, chi_n),
                              generalized_density_matrix(rho_n, chi_n),
                              atol=1.0e-10, rtol=0.0)
        neutron_parity = number_parity_from_density(
            rho_n.detach().cpu().numpy(), chi_n.detach().cpu().numpy())
        proton_parity = number_parity_from_density(
            rho_p.detach().cpu().numpy(), chi_p.detach().cpu().numpy())
        assert neutron_parity["number_parity"] == -1
        assert proton_parity["number_parity"] == 1
        return selfconsistent_stationarity_loss(
            basis, states, {"n": 0.0, "p": 0.0}, {"n": 2, "p": 1},
            normal_functional=backend, normal_tile_size=8,
            normal_chunk_size=32, residual_chunk_size=4,
            pairing_chunk_size=2, pairing_active_chunk_size=1, lmax=2,
            number_weight=0.0,
        )

    direction = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    result = evaluate(direction)
    channels = {key: value.clone() for key, value in result.energy_channels.items()}
    assert all(not value.requires_grad for value in channels.values())
    torch.testing.assert_close(sum(channels.values()), result.total_energy.detach(),
                               atol=1e-10, rtol=1e-12)
    torch.testing.assert_close(channels["d1s_pairing"], result.pairing_energy.detach(),
                               atol=1e-12, rtol=1e-12)
    (ad_directional_derivative,) = torch.autograd.grad(result.loss, direction)
    for key, value in channels.items():
        torch.testing.assert_close(result.energy_channels[key], value, atol=1e-12, rtol=1e-12)
        if key != "d1s_pairing":
            torch.testing.assert_close(backend.last_energy_terms[key], value,
                                       atol=1e-12, rtol=1e-12)
    epsilon = 1.0e-4
    plus, minus = evaluate(epsilon), evaluate(-epsilon)
    finite_difference = (plus.loss - minus.loss) / (2.0 * epsilon)
    assert torch.isfinite(result.total_energy)
    assert result.physical_result_accepted is False
    assert result.no_sea_branch_certified is False
    assert abs(0.13 * torch.cos(torch.tensor(1.22)).item()) > 1.0e-3
    assert torch.isfinite(ad_directional_derivative)
    assert torch.isfinite(finite_difference)
    torch.testing.assert_close(ad_directional_derivative, finite_difference,
                               atol=1.0e-4, rtol=2.0e-5)
