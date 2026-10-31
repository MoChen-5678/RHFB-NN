import torch

from dpl_rhf.rhfb.contact import pion_pv_contact_energy, rho_tensor_contact_energy


def _random_inputs(seed=312):
    g = torch.Generator().manual_seed(seed)
    qn = torch.randn((3, 4, 2), generator=g, dtype=torch.float64).to(torch.complex128)
    qp = torch.randn((3, 4, 2), generator=g, dtype=torch.float64).to(torch.complex128)
    # Explicitly populate both upper and lower Dirac components.
    qn = qn / torch.linalg.vector_norm(qn, dim=1, keepdim=True)
    qp = qp / torch.linalg.vector_norm(qp, dim=1, keepdim=True)
    rho = torch.tensor([[0.71, 0.08j], [-0.08j, 0.36]], dtype=torch.complex128)
    rho = {"n": rho, "p": torch.tensor([[0.55, 0.04], [0.04, 0.29]], dtype=torch.complex128)}
    weights = torch.tensor([0.3, 0.8, 0.5], dtype=torch.float64)
    vertex = torch.tensor([0.7, 0.4, 0.9], dtype=torch.float64, requires_grad=True)
    q = {"n": qn.requires_grad_(), "p": qp.requires_grad_()}
    return q, rho, weights, vertex


def _loop_oracle(q, rho, weights, vertex, kind, hbar_c):
    """Tiny explicit point/state-index oracle, independent of production einsum."""
    import torch
    from dpl_rhf.rhfb.currents import _gamma_matrices

    gamma, gamma5 = _gamma_matrices(device=weights.device, dtype=torch.complex128)
    ops = []
    if kind == "rho_tensor":
        for mu in range(4):
            sign_mu = (1, -1, -1, -1)[mu]
            for axis in range(1, 4):
                sigma = 0.5j * (gamma[mu] @ gamma[axis] - gamma[axis] @ gamma[mu])
                ops.append((gamma[0] @ sigma, -sign_mu))
    else:
        ops = [(gamma[0] @ gamma5 @ gamma[k], 1.0) for k in range(1, 4)]
    species = ("n", "p")
    t3 = {"n": 1.0, "p": -1.0}
    direct = weights.new_zeros(())
    exchange = weights.new_zeros(())
    for point in range(weights.numel()):
        density = {}
        for s in species:
            d = torch.zeros((4, 4), dtype=torch.complex128)
            for a in range(rho[s].shape[0]):
                for b in range(rho[s].shape[1]):
                    d += rho[s][a, b] * torch.outer(q[s][point, :, a], q[s][point, :, b].conj())
            density[s] = d
        for op, sign in ops:
            for s in species:
                for t in species:
                    js = torch.trace(density[s] @ op).real
                    jt = torch.trace(density[t] @ op).real
                    direct += weights[point] * vertex[point].square() * (sign / 6) * t3[s] * t3[t] * js * jt
                    isof = 1 if s == t else 2
                    ex = torch.trace(density[s] @ op @ density[t] @ op).real
                    exchange -= weights[point] * vertex[point].square() * (sign / 6) * isof * ex
    return hbar_c * direct, hbar_c * exchange


def test_compensation_channels_match_explicit_oracle_and_gradients():
    q, rho, weights, vertex = _random_inputs()
    for func, kind in ((rho_tensor_contact_energy, "rho_tensor"),
                       (pion_pv_contact_energy, "pion_pv")):
        result = func(q, rho, weights, vertex, hbar_c_mev_fm=197.3)
        direct, exchange = _loop_oracle(q, rho, weights, vertex, kind, 197.3)
        torch.testing.assert_close(result["direct"], direct, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(result["exchange"], exchange, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(result["total"], direct + exchange, atol=1e-10, rtol=1e-10)
        grad = torch.autograd.grad(result["total"], (vertex, q["n"]), retain_graph=True)
        assert torch.isfinite(grad[0]).all() and torch.isfinite(grad[1]).all()
        assert torch.linalg.vector_norm(grad[1][:, 2:, :]) > 1e-8


def test_contact_is_invariant_under_state_phase_and_has_correct_vertex_derivative():
    q, rho, weights, vertex = _random_inputs()
    fun = rho_tensor_contact_energy
    reference = fun(q, rho, weights, vertex, hbar_c_mev_fm=197.3)["total"]
    phase = torch.exp(1j * torch.tensor([0.31, -0.52], dtype=torch.float64))
    rotated = {s: q[s] * phase[None, None, :] for s in q}
    u = torch.diag(phase.to(torch.complex128))
    rotated_rho = {s: u.mH @ rho[s] @ u for s in rho}
    transformed = fun(rotated, rotated_rho, weights, vertex, hbar_c_mev_fm=197.3)["total"]
    torch.testing.assert_close(reference, transformed, atol=1e-10, rtol=1e-10)
    derivative = torch.autograd.grad(reference, vertex)[0]
    direction = torch.tensor([0.2, -0.4, 0.1], dtype=torch.float64)
    eps = 1e-6
    plus = fun(q, rho, weights, vertex.detach() + eps * direction,
               hbar_c_mev_fm=197.3)["total"]
    minus = fun(q, rho, weights, vertex.detach() - eps * direction,
                hbar_c_mev_fm=197.3)["total"]
    fd = (plus - minus) / (2 * eps)
    torch.testing.assert_close(fd, torch.dot(derivative, direction), atol=2e-7, rtol=2e-7)


def test_time_even_spherical_reduction_matches_rhfe_contact_trace_identities():
    """Compare Cartesian contractions to the paper's independently reduced traces."""
    from dpl_rhf.rhfb.basis import AngularRadialBasis
    from dpl_rhf.rhfb.currents import local_bilinears

    basis = AngularRadialBasis(radius_fm=8.0, spacing_fm=0.2, kappa_max=1,
                               n_theta=3, n_phi=6)
    q = torch.zeros((basis.dimension, 2), dtype=torch.complex128)
    r = torch.as_tensor(basis.x, dtype=torch.float64)
    upper = r * torch.exp(-r)
    lower = 0.12 * r.square() * torch.exp(-r)
    norm = torch.sqrt(basis.spacing_fm * torch.sum(upper.square() + lower.square()))
    upper, lower = (upper / norm).to(torch.complex128), (lower / norm).to(torch.complex128)
    for col, m2 in enumerate((-1, 1)):
        offset = basis.channels[basis.channel_index[(-1, m2)]].offset
        q[offset:offset + basis.n_radial, col] = upper
        q[offset + basis.n_radial:offset + 2 * basis.n_radial, col] = lower
    samples, weights, positions = basis.sample(q)
    rho_one_species = torch.eye(2, dtype=torch.complex128)
    spinors = {"n": samples, "p": samples}
    rho = {"n": rho_one_species, "p": rho_one_species}
    vertex = torch.full((basis.point_count,), 0.7, dtype=torch.float64)
    bilinear = local_bilinears(samples, rho_one_species)
    baryon = bilinear["baryon_current"][:, 0]
    scalar = bilinear["scalar"]
    radial = positions / torch.linalg.vector_norm(positions, dim=1, keepdim=True)
    tensor_radial = torch.sum(bilinear["tensor"][:, 0, 1:] * radial, dim=1)

    # RHF-e Eq. (3.207) for rho-T and Eq. (3.294) for pion-PV.
    expected_rt = weights.new_zeros(())
    expected_pi = weights.new_zeros(())
    for a in ("n", "p"):
        for b in ("n", "p"):
            isospin = 1.0 if a == b else 2.0
            expected_rt -= (isospin / 8.0) * torch.sum(
                weights * vertex.square() *
                (3 * scalar.square() + baryon.square() + tensor_radial.square())
            )
            expected_pi -= (isospin / 8.0) * torch.sum(
                weights * vertex.square() *
                (scalar.square() + baryon.square() - tensor_radial.square() / 3)
            )
    observed_rt = rho_tensor_contact_energy(
        spinors, rho, weights, vertex, hbar_c_mev_fm=1.0
    )
    observed_pi = pion_pv_contact_energy(
        spinors, rho, weights, vertex, hbar_c_mev_fm=1.0
    )
    torch.testing.assert_close(observed_rt["exchange"], expected_rt, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(observed_pi["exchange"], expected_pi, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(observed_rt["direct"], weights.new_zeros(()), atol=1e-12, rtol=0)
    torch.testing.assert_close(observed_pi["direct"], weights.new_zeros(()), atol=1e-12, rtol=0)
