from types import SimpleNamespace

import torch

from dpl_rhf.rhfb.functional import _exchange_channel


class IdentityGreen:
    def apply(self, source, mass):
        return source


def _hermitian(seed, n):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn((n, n), generator=g, dtype=torch.float64).to(torch.complex128)
    return (raw + raw.mH) / 2


def _inputs(seed=871):
    g = torch.Generator().manual_seed(seed)
    q = {}
    states = {}
    for i, species in enumerate(("n", "p")):
        raw = torch.randn((5, 4, 3), generator=g, dtype=torch.float64)
        raw = raw.to(torch.complex128) + 1j * torch.randn(
            (5, 4, 3), generator=g, dtype=torch.float64
        )
        q[species] = raw
        states[species] = SimpleNamespace(rho_active=_hermitian(seed + i, 3))
    weights = torch.tensor([0.13, 0.7, 0.21, 0.9, 0.44], dtype=torch.float64)
    coupling = {
        "n": torch.tensor([0.7, 0.5, 1.1, 0.9, 0.4], dtype=torch.float64),
        "p": torch.tensor([0.2, 0.8, 0.6, 1.2, 0.3], dtype=torch.float64),
    }
    op = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0],
                       [0, 0, 0, -1]], dtype=torch.complex128)
    return q, states, weights, coupling, [(op, -1.0)]


def _explicit_oracle(q, states, weights, coupling, operators, *, isovector=False):
    result = weights.new_zeros(())
    for left in ("n", "p"):
        for right in ("n", "p"):
            iso = (1.0 if left == right else 2.0) if isovector else (1.0 if left == right else 0.0)
            if not iso:
                continue
            rho_l = states[left].rho_active
            rho_r = states[right].rho_active
            ql, qr = q[left], q[right]
            weighted_l = torch.einsum("pia,ab->pib", ql, rho_l)
            weighted_r = torch.einsum("pia,ab->pib", qr, rho_r)
            for op, metric_sign in operators:
                for point in range(weights.numel()):
                    block = torch.zeros((), dtype=torch.complex128, device=weights.device)
                    for a in range(ql.shape[-1]):
                        for b in range(qr.shape[-1]):
                            lhs = weighted_l[point, :, a].conj() @ op @ weighted_r[point, :, b]
                            rhs = qr[point, :, b].conj() @ op @ ql[point, :, a]
                            block += lhs * coupling[right][point] * rhs
                    result -= (0.5 * metric_sign * iso * weights[point] *
                               coupling[left][point] * block.real)
    return result


def test_exchange_matches_full_non_diagonal_density_oracle_and_pair_chunks():
    q, states, weights, coupling, operators = _inputs()
    oracle = _explicit_oracle(q, states, weights, coupling, operators, isovector=True)
    for pair_chunk in (1, 2, 3):
        actual = _exchange_channel(
            q, states, weights, coupling, IdentityGreen(), 770.0, operators,
            isovector=True, pair_chunk=pair_chunk,
        )
        torch.testing.assert_close(actual, oracle, atol=1e-10, rtol=1e-10)


def test_exchange_is_invariant_under_active_basis_unitary_rotation():
    q, states, weights, coupling, operators = _inputs(seed=119)
    reference = _exchange_channel(
        q, states, weights, coupling, IdentityGreen(), 770.0, operators,
        isovector=True, pair_chunk=1,
    )
    g = torch.Generator().manual_seed(928)
    rotated_q, rotated_states = {}, {}
    for i, species in enumerate(("n", "p")):
        raw = torch.randn((3, 3), generator=g, dtype=torch.float64).to(torch.complex128)
        raw = raw + 1j * torch.randn((3, 3), generator=g, dtype=torch.float64)
        unitary, _ = torch.linalg.qr(raw)
        rotated_q[species] = q[species] @ unitary
        rotated_states[species] = SimpleNamespace(
            rho_active=unitary.mH @ states[species].rho_active @ unitary
        )
    transformed = _exchange_channel(
        rotated_q, rotated_states, weights, coupling, IdentityGreen(), 770.0, operators,
        isovector=True, pair_chunk=2,
    )
    torch.testing.assert_close(reference, transformed, atol=1e-10, rtol=1e-10)


def test_particle_hole_energy_accepts_normal_density_and_has_density_derivative():
    from types import SimpleNamespace

    from dpl_rhf.rhfb.basis import AngularRadialBasis
    from dpl_rhf.rhfb.functional import (
        evaluate_particle_hole_energy,
        functional_readiness,
    )
    from dpl_rhf.rhf.parameters import pka1_parameters

    basis = AngularRadialBasis(
        radius_fm=1.6, spacing_fm=0.2, kappa_max=1, n_theta=3, n_phi=6,
    )
    q = torch.zeros((basis.dimension, 1), dtype=torch.complex128)
    q[0, 0] = 1.0
    density = torch.tensor([[0.45]], dtype=torch.complex128, requires_grad=True)
    states = {s: SimpleNamespace(q=q, rho_active=density) for s in ("n", "p")}
    result = evaluate_particle_hole_energy(
        basis, states, chunk_size=64, pair_chunk=1, params=pka1_parameters(), lmax=0,
    )
    assert "d1s_pairing" not in result.audit["included_energy_terms"]
    assert all(torch.isfinite(value) for value in result.energy_terms.values())
    assert all(key in result.energy_terms for key in (
        "rho_tensor_direct", "rho_tensor_fock", "rho_vt_direct", "rho_vt_fock",
        "rho_tv_direct", "rho_tv_fock", "pion_pv_direct", "pion_pv_fock",
        "rho_tensor_contact", "pion_pv_contact",
    ))
    derivative, = torch.autograd.grad(result.total_energy, density, retain_graph=True)
    plain_density = density.detach().clone().requires_grad_()
    plain_states = {s: SimpleNamespace(q=q, rho_active=plain_density)
                    for s in ("n", "p")}
    plain = evaluate_particle_hole_energy(
        basis, plain_states, chunk_size=64, pair_chunk=1,
        params=pka1_parameters(), lmax=0, _checkpoint_fock_blocks=False,
    )
    for name in result.energy_terms:
        torch.testing.assert_close(result.energy_terms[name], plain.energy_terms[name],
                                   atol=2e-12, rtol=2e-12)
    checkpoint_mesons = sum(result.energy_terms[name] for name in (
        "sigma_direct", "sigma_fock", "omega_direct", "omega_fock"))
    plain_mesons = sum(plain.energy_terms[name] for name in (
        "sigma_direct", "sigma_fock", "omega_direct", "omega_fock"))
    torch.testing.assert_close(checkpoint_mesons, plain_mesons, atol=2e-12, rtol=2e-12)
    plain_derivative, = torch.autograd.grad(plain.total_energy, plain_density,
                                           retain_graph=True)
    torch.testing.assert_close(derivative, plain_derivative, atol=2e-11, rtol=2e-11)
    for orientation_name in (
        "rho_vt_direct", "rho_tv_direct", "rho_vt_fock", "rho_tv_fock",
    ):
        checkpoint_gradient, = torch.autograd.grad(
            result.energy_terms[orientation_name], density, retain_graph=True,
        )
        plain_gradient, = torch.autograd.grad(
            plain.energy_terms[orientation_name], plain_density, retain_graph=True,
        )
        torch.testing.assert_close(checkpoint_gradient, plain_gradient,
                                   atol=2e-11, rtol=2e-11)
    direction = torch.tensor([[0.17]], dtype=torch.complex128)
    eps = 2e-6

    def energy_at(value):
        candidate = {s: SimpleNamespace(q=q, rho_active=value) for s in ("n", "p")}
        return evaluate_particle_hole_energy(
            basis, candidate, chunk_size=64, pair_chunk=1,
            params=pka1_parameters(), lmax=0,
        ).total_energy

    fd = (energy_at(density.detach() + eps * direction)
          - energy_at(density.detach() - eps * direction)) / (2 * eps)
    expected = torch.sum(derivative.conj() * direction).real
    torch.testing.assert_close(fd, expected, atol=3e-5, rtol=3e-5)
    manifest = functional_readiness()
    assert manifest["ready"] is False
    assert manifest["channels"]["rho_tensor_fock"]["implemented"] is True
    assert manifest["channels"]["rho_tensor_fock"]["closure_status"] == "unvalidated"


def test_derivative_vertices_match_independent_two_endpoint_kernel_oracle():
    from dpl_rhf.rhfb.basis import AngularRadialBasis
    from dpl_rhf.rhfb.currents import _gamma_matrices
    from dpl_rhf.rhfb.functional import (
        _direct_derivative_channel,
        _exchange_derivative_channel,
    )
    from dpl_rhf.rhfb.spatial import GreenOperator

    basis = AngularRadialBasis(
        radius_fm=1.6, spacing_fm=0.2, kappa_max=1, n_theta=3, n_phi=6,
    )
    green = GreenOperator(basis, lmax=0)
    g = torch.Generator().manual_seed(762)
    q, states = {}, {}
    for index, species in enumerate(("n", "p")):
        raw = torch.randn((basis.point_count, 4, 2), generator=g, dtype=torch.float64)
        raw = raw.to(torch.complex128) + 1j * torch.randn(
            raw.shape, generator=g, dtype=torch.float64
        )
        q[species] = raw
        states[species] = SimpleNamespace(rho_active=_hermitian(900 + index, 2))
    weights = torch.as_tensor(basis.weights, dtype=torch.float64)
    coupling = {
        "n": torch.linspace(0.3, 0.9, basis.point_count, dtype=torch.float64),
        "p": torch.linspace(0.8, 0.2, basis.point_count, dtype=torch.float64),
    }
    gamma, gamma5 = _gamma_matrices(device=weights.device, dtype=torch.complex128)
    g0 = gamma[0]
    tensor_ops = []
    metric_signs = []
    for mu in range(4):
        tensor_ops.append([
            g0 @ (0.5j * (gamma[i] @ gamma[mu] - gamma[mu] @ gamma[i]))
            for i in range(1, 4)
        ])
        metric_signs.append((1.0, -1.0, -1.0, -1.0)[mu])
    kernel = green.kernel_block(
        0, basis.point_count, 0, basis.point_count, 770.0,
        derivative_x=True, derivative_y=True,
    )

    # Hartree oracle directly contracts both differentiated kernel endpoints.
    direct_oracle = weights.new_zeros(())
    for ops, sign in zip(tensor_ops, metric_signs):
        species_current = []
        for species in ("n", "p"):
            current = torch.stack([
                torch.einsum("pia,ij,pjb->pab", q[species].conj(), op,
                             q[species]) for op in ops
            ], dim=1)
            species_current.append(torch.einsum(
                "ab,piba->pi", states[species].rho_active, current
            ).real)
        source = coupling["n"][:, None] * (
            species_current[0] - species_current[1]
        )
        direct_oracle += 0.5 * sign * torch.einsum(
            "x,xi,xyij,yj,y->", weights, source, kernel, source, weights
        ).real
    direct = _direct_derivative_channel(
        q, {s: states[s].rho_active for s in states}, weights, coupling["n"],
        green, 770.0, tensor_ops, metric_signs=metric_signs, prefactor=0.5,
    )
    # The production helper uses one shared pointwise coupling for the isovector
    # source; verify with the same coupling for n and p.
    torch.testing.assert_close(direct, direct_oracle, atol=2e-10, rtol=2e-10)

    rho_equal = coupling["n"]
    exchange_oracle = weights.new_zeros(())
    complex_kernel = kernel.to(torch.complex128)
    complex_weights = weights.to(torch.complex128)
    for left in ("n", "p"):
        for right in ("n", "p"):
            iso = 1.0 if left == right else 2.0
            qwl = q[left] @ states[left].rho_active
            qwr = q[right] @ states[right].rho_active
            for ops, sign in zip(tensor_ops, metric_signs):
                lhs = torch.stack([
                    torch.einsum("pia,ij,pjb->pab", qwl.conj(), op, qwr)
                    for op in ops
                ], dim=1) * rho_equal[:, None, None, None]
                rhs = torch.stack([
                    torch.einsum("pia,ij,pjb->pab", q[right].conj(), op,
                                 q[left]) for op in ops
                ], dim=1) * rho_equal[:, None, None, None]
                exchange_oracle -= 0.5 * iso * sign * torch.einsum(
                    "x,xiab,xyij,yjba,y->", complex_weights, lhs,
                    complex_kernel, rhs, complex_weights
                ).real
    exchange = _exchange_derivative_channel(
        q, states, weights, {s: rho_equal for s in states}, green, 770.0,
        list(zip(tensor_ops, metric_signs)), isovector=True, pair_chunk=1,
        prefactor=-0.5,
    )
    torch.testing.assert_close(exchange, exchange_oracle, atol=2e-10, rtol=2e-10)

    pion_ops = [[g0 @ gamma5 @ gamma[i] for i in range(1, 4)]]
    pion_kernel = green.kernel_block(
        0, basis.point_count, 0, basis.point_count, 138.0,
        derivative_x=True, derivative_y=True,
    )
    direct_pi_oracle = weights.new_zeros(())
    for ops in pion_ops:
        species_current = []
        for species in ("n", "p"):
            current = torch.stack([
                torch.einsum("pia,ij,pjb->pab", q[species].conj(), op,
                             q[species]) for op in ops
            ], dim=1)
            species_current.append(torch.einsum(
                "ab,piba->pi", states[species].rho_active, current
            ).real)
        source = rho_equal[:, None] * (species_current[0] - species_current[1])
        direct_pi_oracle -= 0.5 * torch.einsum(
            "x,xi,xyij,yj,y->", weights, source, pion_kernel, source, weights
        ).real
    direct_pi = _direct_derivative_channel(
        q, {s: states[s].rho_active for s in states}, weights, rho_equal,
        green, 138.0, pion_ops, prefactor=-0.5,
    )
    torch.testing.assert_close(direct_pi, direct_pi_oracle, atol=2e-10, rtol=2e-10)
    exchange_pi_oracle = weights.new_zeros(())
    for left in ("n", "p"):
        for right in ("n", "p"):
            iso = 1.0 if left == right else 2.0
            qwl = q[left] @ states[left].rho_active
            qwr = q[right] @ states[right].rho_active
            for ops in pion_ops:
                lhs = torch.stack([
                    torch.einsum("pia,ij,pjb->pab", qwl.conj(), op, qwr)
                    for op in ops
                ], dim=1) * rho_equal[:, None, None, None]
                rhs = torch.stack([
                    torch.einsum("pia,ij,pjb->pab", q[right].conj(), op,
                                 q[left]) for op in ops
                ], dim=1) * rho_equal[:, None, None, None]
                exchange_pi_oracle += 0.5 * iso * torch.einsum(
                    "x,xiab,xyij,yjba,y->", complex_weights, lhs,
                    pion_kernel.to(torch.complex128), rhs, complex_weights
                ).real
    exchange_pi = _exchange_derivative_channel(
        q, states, weights, {s: rho_equal for s in states}, green, 138.0,
        pion_ops, isovector=True, pair_chunk=1, prefactor=0.5,
    )
    torch.testing.assert_close(exchange_pi, exchange_pi_oracle, atol=2e-10, rtol=2e-10)


def test_mixed_vector_tensor_orders_match_independent_one_endpoint_kernel_oracle():
    from dpl_rhf.rhfb.basis import AngularRadialBasis
    from dpl_rhf.rhfb.currents import _gamma_matrices
    from dpl_rhf.rhfb.functional import (
        _direct_mixed_derivative_channel,
        _exchange_mixed_derivative_channel,
    )
    from dpl_rhf.rhfb.spatial import GreenOperator

    basis = AngularRadialBasis(
        radius_fm=1.6, spacing_fm=0.2, kappa_max=1, n_theta=3, n_phi=6,
    )
    green = GreenOperator(basis, lmax=0)
    gen = torch.Generator().manual_seed(333)
    q, states = {}, {}
    for i, species in enumerate(("n", "p")):
        r = torch.randn((basis.point_count, 4, 2), generator=gen, dtype=torch.float64)
        q[species] = r.to(torch.complex128) + 1j * torch.randn(
            r.shape, generator=gen, dtype=torch.float64
        )
        states[species] = SimpleNamespace(rho_active=_hermitian(1100 + i, 2))
    weights = torch.as_tensor(basis.weights, dtype=torch.float64)
    g = torch.linspace(0.2, 0.8, basis.point_count, dtype=torch.float64)
    f = torch.linspace(1.1, 0.4, basis.point_count, dtype=torch.float64)
    gamma, _ = _gamma_matrices(device=weights.device, dtype=torch.complex128)
    g0 = gamma[0]
    vops = [g0 @ gamma[mu] for mu in range(4)]
    tops = [[g0 @ (0.5j * (gamma[i] @ gamma[mu] - gamma[mu] @ gamma[i]))
             for i in range(1, 4)] for mu in range(4)]
    signs = (1.0, -1.0, -1.0, -1.0)
    kernel = green.kernel_block(0, basis.point_count, 0, basis.point_count, 770.)
    dx = green.kernel_block(0, basis.point_count, 0, basis.point_count, 770.,
                            derivative_x=True)
    dy = green.kernel_block(0, basis.point_count, 0, basis.point_count, 770.,
                            derivative_y=True)
    ckernel, cweights = kernel.to(torch.complex128), weights.to(torch.complex128)

    js, ts = [], []
    for species in ("n", "p"):
        species_j, species_t = [], []
        for vop, top in zip(vops, tops):
            jmat = torch.einsum("pia,ij,pjb->pab", q[species].conj(), vop, q[species])
            tmat = torch.stack([
                torch.einsum("pia,ij,pjb->pab", q[species].conj(), op,
                             q[species]) for op in top
            ], dim=1)
            species_j.append(torch.einsum("ab,pba->p", states[species].rho_active,
                                          jmat).real)
            species_t.append(torch.einsum("ab,piba->pi", states[species].rho_active,
                                          tmat).real)
        js.append(torch.stack(species_j, dim=1))
        ts.append(torch.stack(species_t, dim=1))
    j = g[:, None] * (js[0] - js[1])
    u = f[:, None, None] * (ts[0] - ts[1])
    direct_oracle = weights.new_zeros(())
    for mu, sign in enumerate(signs):
        # Keep endpoint ordering explicit: y-derivative for J(x)D_y, and
        # x-derivative for D_xJ(y).
        first = torch.einsum("x,x,xyi,yi,y->", weights, j[:, mu], dy, u[:, mu], weights)
        second = torch.einsum("x,xi,xyi,y,y->", weights, u[:, mu], dx,
                              j[:, mu], weights)
        direct_oracle += 0.5 * sign * (first + second).real
    direct = _direct_mixed_derivative_channel(
        q, {s: states[s].rho_active for s in states}, weights, g, f, green,
        770., vops, tops, signs, prefactor=0.5,
    )
    torch.testing.assert_close(direct, direct_oracle, atol=3e-10, rtol=3e-10)

    weighted = {s: q[s] @ states[s].rho_active for s in ("n", "p")}
    vt_oracle = weights.new_zeros(())
    tv_oracle = weights.new_zeros(())
    for left in ("n", "p"):
        for right in ("n", "p"):
            iso = 1.0 if left == right else 2.0
            for mu, sign in enumerate(signs):
                jl = torch.einsum("pia,ij,pjb->pab", weighted[left].conj(),
                                  vops[mu], weighted[right]) * g[:, None, None]
                tr = torch.stack([
                    torch.einsum("pia,ij,pjb->pab", q[right].conj(), op,
                                 q[left]) for op in tops[mu]
                ], dim=1) * f[:, None, None, None]
                tl = torch.stack([
                    torch.einsum("pia,ij,pjb->pab", weighted[left].conj(), op,
                                 weighted[right]) for op in tops[mu]
                ], dim=1) * f[:, None, None, None]
                jr = torch.einsum("pia,ij,pjb->pab", q[right].conj(),
                                  vops[mu], q[left]) * g[:, None, None]
                vt_oracle -= 0.5 * iso * sign * torch.einsum(
                    "x,xab,xyi,yiba,y->", cweights, jl, dy.to(torch.complex128),
                    tr, cweights
                ).real
                tv_oracle -= 0.5 * iso * sign * torch.einsum(
                    "x,xiab,xyi,yba,y->", cweights, tl,
                    dx.to(torch.complex128), jr, cweights
                ).real
    vt = _exchange_mixed_derivative_channel(
        q, states, weights, {s: g for s in states}, {s: f for s in states},
        green, 770., vops, tops, signs, derivative_on_left=False,
        pair_chunk=1, prefactor=-0.5,
    )
    tv = _exchange_mixed_derivative_channel(
        q, states, weights, {s: f for s in states}, {s: g for s in states},
        green, 770., vops, tops, signs, derivative_on_left=True,
        pair_chunk=1, prefactor=-0.5,
    )
    torch.testing.assert_close(vt, vt_oracle, atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(tv, tv_oracle, atol=3e-10, rtol=3e-10)
