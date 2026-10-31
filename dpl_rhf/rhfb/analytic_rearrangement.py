"""Independent endpoint contraction for covariant PKA1 vertex rearrangement.

The routines here are a small-space physics reference, not a production
replacement for ``functional.py``.  They differentiate the general two-body
vertices at both endpoints explicitly and only then contract with
``d g_i / d rho``.  No energy autograd or frozen-vertex subtraction is used
to form the reference source.
"""
from __future__ import annotations

import torch

from .currents import _gamma_matrices, invariant_baryon_density, local_bilinears
from .spatial import GreenOperator


def _transition(left: torch.Tensor, op: torch.Tensor,
                right: torch.Tensor) -> torch.Tensor:
    return torch.einsum("pia,ij,pjb->pab", left, op, right)


def _density_matrix(q: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ab,pia,pjb->pij", rho, q, q.conj())


def _local_transition_current(q, rho, operators):
    matrices = torch.stack([_transition(q.conj(), op, q) for op in operators], dim=1)
    return torch.einsum("ab,piba->pi", rho, matrices).real


def _couplings_and_slopes(rho, params):
    z = rho / params.rho_sat

    def rational(g0, a, b, c, d):
        x = z + d
        value = g0 * a * (1.0 + b * x.square()) / (1.0 + c * x.square())
        slope = g0 * a * (2.0 * x * (b - c) / (1.0 + c * x.square()).square())
        return value, slope / params.rho_sat

    gsig, dgsig = rational(params.g_sigma, params.a_sigma, params.b_sigma,
                           params.c_sigma, params.d_sigma)
    gome, dgome = rational(params.g_omega, params.a_omega, params.b_omega,
                           params.c_omega, params.d_omega)
    grho = params.g_rho * torch.exp(-params.a_rho * z)
    grtn = params.rho_tensor_vertex * torch.exp(-params.a_rho_tensor * z)
    fpi = params.f_pi * torch.exp(-params.a_pi * z) * params.hbar_c / params.m_pi
    return (
        {"gsig": gsig, "gome": gome, "grho": grho, "grtn": grtn, "fpi": fpi},
        {"gsig": dgsig,
         "gome": dgome,
         "grho": -params.a_rho / params.rho_sat * grho,
         "grtn": -params.a_rho_tensor / params.rho_sat * grtn,
         "fpi": -params.a_pi / params.rho_sat * fpi},
    )


def _fock_vertex_sensitivity(q, states, weights, coupling, green, mass,
                             operators, *, isovector=False, photon=False,
                             scale=1.0):
    """dE/dc(x) for a symmetric Fock vertex c(x) Gamma_x D c(y) Gamma_y.

    The two endpoint responses are evaluated separately.  The displayed
    expression is the endpoint derivative of the Fock expectation, including
    its ``-1/2`` and species/Dirac metric factors.
    """
    weighted = {s: torch.einsum("pia,ab->pib", q[s], states[s].rho_active)
                for s in ("n", "p")}
    result = weights.new_zeros(weights.shape)
    for left in ("n", "p"):
        for right in ("n", "p"):
            if photon and (left != "p" or right != "p"):
                continue
            iso = ((1.0 if left == right else 2.0) if isovector
                   else (1.0 if left == right else 0.0))
            if not iso:
                continue
            for op, metric_sign in operators:
                lhs = _transition(weighted[left].conj(), op, weighted[right])
                rhs = torch.einsum("pib,ij,pja->pba", q[right].conj(), op, q[left])
                field_rhs = green.apply((coupling[:, None, None] * rhs).reshape(
                    weights.numel(), -1), mass).reshape_as(rhs)
                field_lhs = green.apply((coupling[:, None, None] * lhs).reshape(
                    weights.numel(), -1), mass).reshape_as(lhs)
                lhs_response = (lhs * field_rhs.transpose(1, 2)).sum(dim=(1, 2)).real
                rhs_response = (rhs * field_lhs.transpose(1, 2)).sum(dim=(1, 2)).real
                result = result - (0.5 * scale * iso * metric_sign
                                   * (lhs_response + rhs_response))
    return result


def _derivative_fock_vertex_sensitivity(q, states, weights, coupling, green,
                                        mass, operator_groups, *, isovector=True,
                                        prefactor=-0.5, scale=1.0):
    """Endpoint derivative for derivative-meson Fock vertices.

    Each group is ``(Cartesian derivative operators, Lorentz sign)``.  Using
    the weighted adjoint identity gives the pointwise endpoint source
    ``J . G D G†[c J']`` at each end.
    """
    weighted = {s: torch.einsum("pia,ab->pib", q[s], states[s].rho_active)
                for s in ("n", "p")}
    result = weights.new_zeros(weights.shape)
    for left in ("n", "p"):
        for right in ("n", "p"):
            iso = (1.0 if left == right else 2.0) if isovector else (
                1.0 if left == right else 0.0)
            if not iso:
                continue
            for ops, metric_sign in operator_groups:
                j_left = torch.stack([
                    _transition(weighted[left].conj(), op, weighted[right])
                    for op in ops
                ], dim=1)
                j_right = torch.stack([
                    torch.einsum("pib,ij,pja->pba", q[right].conj(), op, q[left])
                    for op in ops
                ], dim=1)
                s_left = green.gradient_adjoint(
                    coupling[:, None, None, None] * j_left)
                s_right = green.gradient_adjoint(
                    coupling[:, None, None, None] * j_right)
                gds_right = green.gradient(green.apply(s_right, mass))
                gds_left = green.gradient(green.apply(s_left, mass))
                # The reverse transition carries the ordered state indices
                # (b,a), while the left endpoint is (a,b).  Preserve that
                # ordering explicitly at both endpoint contractions.
                left_response = (j_left * gds_right.transpose(2, 3)).sum(
                    dim=(1, 2, 3)).real
                right_response = (j_right * gds_left.transpose(2, 3)).sum(
                    dim=(1, 2, 3)).real
                result = result + (scale * prefactor * iso * metric_sign
                                   * (left_response + right_response))
    return result


def _direct_derivative_vertex_sensitivity(q, states, weights, coupling, green,
                                          mass, operator_groups, *,
                                          prefactor=0.5, scale=1.0):
    result = weights.new_zeros(weights.shape)
    for ops, metric_sign in operator_groups:
        current = _local_transition_current(q["n"], states["n"].rho_active, ops)
        current = current - _local_transition_current(q["p"], states["p"].rho_active, ops)
        source = green.gradient_adjoint(coupling[:, None] * current)
        response = green.gradient(green.apply(source, mass))
        local = (current * response).sum(dim=1).real
        result = result + 2.0 * prefactor * scale * metric_sign * local
    return result


def _direct_mixed_vertex_sensitivities(q, states, weights, gvec, ftensor,
                                       green, mass, vector_ops, tensor_groups,
                                       scale=1.0):
    dg, df = weights.new_zeros(weights.shape), weights.new_zeros(weights.shape)
    vector_n = _local_transition_current(q["n"], states["n"].rho_active, vector_ops)
    vector_p = _local_transition_current(q["p"], states["p"].rho_active, vector_ops)
    for mu, (tensor_ops, metric_sign) in enumerate(tensor_groups):
        # The Lorentz contraction pairs T_{i mu} with the same vector-current
        # component j^mu.  Keeping mu aligned is essential for time-odd
        # spatial currents and for time-reversal invariance.
        jv = vector_n[:, mu] - vector_p[:, mu]
        jt = (_local_transition_current(q["n"], states["n"].rho_active, tensor_ops)
              - _local_transition_current(q["p"], states["p"].rho_active, tensor_ops))
        source_t = green.gradient_adjoint(ftensor[:, None] * jt)
        dg = dg + scale * metric_sign * jv * green.apply(source_t, mass).reshape(-1)
        j_source = gvec * jv
        dsource_v = green.apply(j_source, mass)
        df = df + scale * metric_sign * (jt * green.gradient(dsource_v)).sum(dim=1).real
    return dg.real, df.real


def _mixed_fock_vertex_sensitivities(q, states, weights, gvec, ftensor,
                                     green, mass, vector_ops, tensor_groups,
                                     *, pair_chunk_unused=None, scale=1.0):
    """dE/dg_rho and dE/df_rho for both ordered rho-VT Fock endpoints."""
    weighted = {s: torch.einsum("pia,ab->pib", q[s], states[s].rho_active)
                for s in ("n", "p")}
    dg, df = weights.new_zeros(weights.shape), weights.new_zeros(weights.shape)
    prefactor = -0.5 * scale
    for left in ("n", "p"):
        for right in ("n", "p"):
            iso = 1.0 if left == right else 2.0
            for vop, (tops, metric_sign) in zip(vector_ops, tensor_groups):
                jv_l = _transition(weighted[left].conj(), vop, weighted[right])
                jv_r = torch.einsum("pib,ij,pja->pba", q[right].conj(), vop, q[left])
                jt_l = torch.stack([_transition(weighted[left].conj(), op,
                                                weighted[right]) for op in tops], dim=1)
                jt_r = torch.stack([torch.einsum("pib,ij,pja->pba", q[right].conj(),
                                                op, q[left]) for op in tops], dim=1)

                # V(x) T(y), with the derivative acting at y.
                st_r = green.gradient_adjoint(ftensor[:, None, None, None] * jt_r)
                d_st_r = green.apply(st_r, mass)
                dg = dg + prefactor * iso * metric_sign * (
                    jv_l * d_st_r.transpose(1, 2)).sum(dim=(1, 2)).real
                d_gjv_l = green.apply(gvec[:, None, None] * jv_l, mass)
                df = df + prefactor * iso * metric_sign * (
                    jt_r * green.gradient(d_gjv_l).transpose(2, 3)
                ).sum(dim=(1, 2, 3)).real

                # T(x) V(y), with the derivative acting at x.
                st_l = green.gradient_adjoint(ftensor[:, None, None, None] * jt_l)
                d_st_l = green.apply(st_l, mass)
                dg = dg + prefactor * iso * metric_sign * (
                    jv_r * d_st_l.transpose(1, 2)).sum(dim=(1, 2)).real
                d_gjv_r = green.apply(gvec[:, None, None] * jv_r, mass)
                # jv_r is ordered (b,a); transpose it to match jt_l (a,b).
                response = green.gradient(d_gjv_r).transpose(2, 3)
                df = df + prefactor * iso * metric_sign * (
                    jt_l * response).sum(dim=(1, 2, 3)).real
    return dg, df


def _contact_vertex_sensitivity(q, states, weights, vertex, *, kind, hbar_c):
    """Pointwise derivative of the explicit (1/6)(direct-exchange) contacts."""
    d = {s: _density_matrix(q[s], states[s].rho_active) for s in ("n", "p")}
    gamma, gamma5 = _gamma_matrices(device=q["n"].device, dtype=torch.complex128)
    g0 = gamma[0]
    if kind == "rho_tensor":
        grouped, signs = [], []
        for mu in range(4):
            mu_sign = (1.0, -1.0, -1.0, -1.0)[mu]
            grouped.append([g0 @ (0.5j * (gamma[i] @ gamma[mu]
                                            - gamma[mu] @ gamma[i]))
                            for i in range(1, 4)])
            signs.append(-mu_sign)
    elif kind == "pion_pv":
        grouped = [[g0 @ gamma5 @ gamma[i] for i in range(1, 4)]]
        signs = [1.0]
    else:
        raise ValueError("unknown compensation contact")
    direct = weights.new_zeros(weights.shape)
    exchange = weights.new_zeros(weights.shape)
    t3 = {"n": 1.0, "p": -1.0}
    for ops, group_sign in zip(grouped, signs):
        for op in ops:
            currents = {s: torch.einsum("pij,ji->p", d[s], op).real for s in ("n", "p")}
            for s in ("n", "p"):
                for t in ("n", "p"):
                    direct = direct + (group_sign / 6.0) * t3[s] * t3[t] * currents[s] * currents[t]
                    iso = 1.0 if s == t else 2.0
                    exchange = exchange - (group_sign / 6.0) * iso * torch.einsum(
                        "pij,jk,pkl,li->p", d[s], op, d[t], op).real
    return 2.0 * hbar_c * vertex * (direct + exchange)


def analytic_rearrangement_reference(basis, states, params):
    """Return explicit vertex sensitivities and the covariant rearrangement.

    ``dE_dvertex[name][x]`` is the local endpoint derivative density, such
    that ``delta E = sum_x w_x dE_dvertex[x] delta_vertex[x]``.  ``S_R`` is
    ``sum_i (d g_i/d rho_B) (dE/dg_i)`` and the vector field is
    ``Sigma_R^mu = S_R j^mu/rho_B``.  All normal, exchange, ordered mixed,
    derivative, and compensation contact terms are included.
    """
    q = {s: basis._sample_range(states[s].q, 0, basis.point_count) for s in ("n", "p")}
    weights = torch.as_tensor(basis.weights, dtype=torch.float64, device=next(iter(q.values())).device)
    bil = {s: local_bilinears(q[s], states[s].rho_active) for s in ("n", "p")}
    current = bil["n"]["baryon_current"] + bil["p"]["baryon_current"]
    rho = invariant_baryon_density(current)
    vertex, slope = _couplings_and_slopes(rho, params)
    green = GreenOperator(basis, lmax=max(2 * basis.kappa_max, 2), device=weights.device)
    gamma, gamma5 = _gamma_matrices(device=weights.device, dtype=torch.complex128)
    g0 = gamma[0]
    sens = {name: weights.new_zeros(weights.shape) for name in vertex}
    components = {}

    # Direct sigma, omega, and rho-vector endpoint contractions.
    rho_s = bil["n"]["scalar"] + bil["p"]["scalar"]
    j3 = bil["n"]["baryon_current"] - bil["p"]["baryon_current"]
    scalar_field = green.apply((vertex["gsig"] * rho_s), params.m_sigma)
    sens["gsig"] = sens["gsig"] - params.hbar_c * rho_s * scalar_field
    vector_ops = [g0 @ gamma[mu] for mu in range(4)]
    for name, source, mass in (("gome", current, params.m_omega),
                               ("grho", j3, params.m_rho)):
        weighted_source = vertex[name][:, None] * source
        field = green.apply(weighted_source, mass)
        signs = weights.new_tensor([1.0, -1.0, -1.0, -1.0])
        sens[name] = sens[name] + params.hbar_c * torch.sum(
            signs[None, :] * source * field, dim=1)

    # Fock endpoints for non-derivative scalar/vector vertices.
    # The scalar Fock expectation carries the minus sign in the RHF trace.
    scalar_ops = [(g0, -1.0)]
    metric_vector_ops = [(g0 @ gamma[mu], (1.0, -1.0, -1.0, -1.0)[mu])
                         for mu in range(4)]
    sens["gsig"] = sens["gsig"] + _fock_vertex_sensitivity(
        q, states, weights, vertex["gsig"], green, params.m_sigma,
        scalar_ops, scale=params.hbar_c)
    sens["gome"] = sens["gome"] + _fock_vertex_sensitivity(
        q, states, weights, vertex["gome"], green, params.m_omega,
        metric_vector_ops, scale=params.hbar_c)
    sens["grho"] = sens["grho"] + _fock_vertex_sensitivity(
        q, states, weights, vertex["grho"], green, params.m_rho,
        metric_vector_ops, isovector=True, scale=params.hbar_c)

    # rho-T and pion-pseudovector derivative vertices.
    tensor_groups = []
    for mu in range(4):
        ops = []
        for axis in range(1, 4):
            sigma = 0.5j * (gamma[axis] @ gamma[mu] - gamma[mu] @ gamma[axis])
            ops.append(g0 @ sigma)
        tensor_groups.append((ops, (1.0, -1.0, -1.0, -1.0)[mu]))
    pion_groups = [([g0 @ gamma5 @ gamma[axis] for axis in range(1, 4)], 1.0)]
    components["rho_tensor_direct"] = _direct_derivative_vertex_sensitivity(
        q, states, weights, vertex["grtn"], green, params.m_rho,
        tensor_groups, prefactor=0.5, scale=params.hbar_c)
    components["rho_tensor_fock"] = _derivative_fock_vertex_sensitivity(
        q, states, weights, vertex["grtn"], green, params.m_rho,
        tensor_groups, isovector=True, prefactor=-0.5, scale=params.hbar_c)
    sens["grtn"] = sens["grtn"] + components["rho_tensor_direct"] + components["rho_tensor_fock"]
    sens["fpi"] = sens["fpi"] + _direct_derivative_vertex_sensitivity(
        q, states, weights, vertex["fpi"], green, params.m_pi,
        pion_groups, prefactor=-0.5, scale=params.hbar_c)
    sens["fpi"] = sens["fpi"] + _derivative_fock_vertex_sensitivity(
        q, states, weights, vertex["fpi"], green, params.m_pi,
        pion_groups, isovector=True, prefactor=0.5, scale=params.hbar_c)

    # Ordered rho-VT and rho-TV terms have independent endpoint couplings.
    direct_g, direct_f = _direct_mixed_vertex_sensitivities(
        q, states, weights, vertex["grho"], vertex["grtn"], green,
        params.m_rho, vector_ops, tensor_groups, scale=params.hbar_c)
    fock_g, fock_f = _mixed_fock_vertex_sensitivities(
        q, states, weights, vertex["grho"], vertex["grtn"], green,
        params.m_rho, vector_ops, tensor_groups, scale=params.hbar_c)
    sens["grho"] = sens["grho"] + direct_g + fock_g
    sens["grtn"] = sens["grtn"] + direct_f + fock_f

    # Explicit local compensation Hamiltonians are quadratic in their vertex.
    components["rho_tensor_contact"] = _contact_vertex_sensitivity(
        q, states, weights, vertex["grtn"], kind="rho_tensor", hbar_c=params.hbar_c)
    sens["grtn"] = sens["grtn"] + components["rho_tensor_contact"]
    components["pion_pv_contact"] = _contact_vertex_sensitivity(
        q, states, weights, vertex["fpi"], kind="pion_pv", hbar_c=params.hbar_c)
    sens["fpi"] = sens["fpi"] + components["pion_pv_contact"]

    source = sum((slope[name] * sens[name] for name in sens), weights.new_zeros(weights.shape))
    vector = source[:, None] * current / rho[:, None]
    return {
        "rho_b": rho,
        "baryon_current": current,
        "vertices": vertex,
        "vertex_slopes": slope,
        "dE_dvertex": sens,
        "dE_dvertex_components": components,
        "S_R": source,
        "Sigma_R_contravariant": vector,
    }
