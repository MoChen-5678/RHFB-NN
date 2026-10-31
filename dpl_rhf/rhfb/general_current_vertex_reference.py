"""Independent finite-coordinate contractions of PKA1 derivative vertices.

This small diagnostic starts from the uncontracted Dirac Gamma matrices and
the GreenOperator's raw two-endpoint kernel blocks. It does not call the
production functional's current, derivative-source, or energy-contraction
helpers. The Green projection itself is shared, so this checks endpoint
algebra and signs on the declared discrete kernel, not continuum convergence.
"""
from __future__ import annotations

import torch

from .spatial import GreenOperator
from .functional import evaluate_particle_hole_energy
from dpl_rhf.rhf.parameters import pka1_parameters


def _dirac_matrices(device):
    cd = torch.complex128
    one = torch.eye(2, dtype=cd, device=device)
    zero = torch.zeros((2, 2), dtype=cd, device=device)
    sx = torch.tensor([[0, 1], [1, 0]], dtype=cd, device=device)
    sy = torch.tensor([[0, -1j], [1j, 0]], dtype=cd, device=device)
    sz = torch.tensor([[1, 0], [0, -1]], dtype=cd, device=device)
    pauli = (sx, sy, sz)
    g0 = torch.block_diag(one, -one)
    gamma = [g0]
    for sigma in pauli:
        gamma.append(torch.cat((torch.cat((zero, sigma), 1),
                                torch.cat((-sigma, zero), 1)), 0))
    g5 = 1j * gamma[0] @ gamma[1] @ gamma[2] @ gamma[3]
    sigma = [[0.5j * (gamma[mu] @ gamma[nu] - gamma[nu] @ gamma[mu])
              for nu in range(4)] for mu in range(4)]
    # Hamiltonian-space matrices: Q† (gamma^0 Gamma) Q.
    tensor = [[g0 @ sigma[axis][mu] for axis in range(1, 4)]
              for mu in range(4)]
    vector = [g0 @ gamma[mu] for mu in range(4)]
    axial_space = [g0 @ g5 @ gamma[axis] for axis in range(1, 4)]
    return vector, tensor, axial_space


def _transition(q_left, op, q_right):
    return torch.einsum("pia,ij,pjb->pab", q_left.conj(), op, q_right)


def _occupied(current_matrix, rho):
    # Tr(rho J), with J_ab = <a|Gamma|b>.
    return torch.einsum("ab,pba->p", rho, current_matrix).real


def _direct_reference(q_by_species, rho_by_species, weights, green, params):
    vector_ops, tensor_ops, axial_ops = _dirac_matrices(weights.device)
    j_iso, t_iso, a_iso = [], [], []
    j_total = []
    for species in ("n", "p"):
        q, rho = q_by_species[species], rho_by_species[species]
        j = torch.stack([_occupied(_transition(q, op, q), rho)
                         for op in vector_ops], dim=-1)
        t = torch.stack([
            torch.stack([_occupied(_transition(q, op, q), rho)
                         for op in tensor_ops[mu]], dim=-1)
            for mu in range(4)
        ], dim=1)  # point, mu, spatial-i
        a = torch.stack([_occupied(_transition(q, op, q), rho)
                         for op in axial_ops], dim=-1)
        j_total.append(j)
        j_iso.append(j * (1.0 if species == "n" else -1.0))
        t_iso.append(t * (1.0 if species == "n" else -1.0))
        a_iso.append(a * (1.0 if species == "n" else -1.0))
    j_total = j_total[0] + j_total[1]
    invariant_square = j_total[:, 0].square() - j_total[:, 1:].square().sum(dim=1)
    if bool(torch.any((j_total[:, 0] <= 0.0) | (invariant_square <= 0.0))):
        raise ValueError("explicit Gamma reference requires a future-timelike current")
    rho_b = torch.sqrt(invariant_square)
    zeta = rho_b / params.rho_sat
    g_rho = params.g_rho * torch.exp(-params.a_rho * zeta)
    g_tensor = params.rho_tensor_vertex * torch.exp(-params.a_rho_tensor * zeta)
    g_pion = params.f_pi * torch.exp(-params.a_pi * zeta) * params.hbar_c / params.m_pi
    j = j_iso[0] + j_iso[1]
    tensor = t_iso[0] + t_iso[1]
    axial = a_iso[0] + a_iso[1]
    pcount = weights.numel()
    gy = green.kernel_block(0, pcount, 0, pcount, params.m_rho,
                            derivative_y=True)
    gx = green.kernel_block(0, pcount, 0, pcount, params.m_rho,
                            derivative_x=True)
    gxy_rho = green.kernel_block(0, pcount, 0, pcount, params.m_rho,
                                 derivative_x=True, derivative_y=True)
    gxy_pi = green.kernel_block(0, pcount, 0, pcount, params.m_pi,
                                derivative_x=True, derivative_y=True)
    pair_weights = weights[:, None] * weights[None, :]

    rt_direct_components = []
    mixed_direct_vt = []
    mixed_direct_tv = []
    lorentz_sign = (1.0, -1.0, -1.0, -1.0)
    for mu in range(4):
        rt_value = 0.0
        for axis_x in range(3):
            for axis_y in range(3):
                rt_value += torch.sum(
                    pair_weights * g_tensor[:, None] * g_tensor[None, :]
                    * tensor[:, mu, axis_x, None]
                    * gxy_rho[:, :, axis_x, axis_y]
                    * tensor[None, :, mu, axis_y]
                )
        rt_direct_components.append(0.5 * params.hbar_c * lorentz_sign[mu] * rt_value)

        # One oriented V(x)T(y) endpoint. The reverse T(x)V(y) endpoint
        # carries the same real direct Hartree value for this symmetric kernel.
        mixed = torch.zeros((), dtype=torch.float64, device=weights.device)
        for axis in range(3):
            mixed += torch.sum(
                pair_weights * g_rho[:, None] * g_tensor[None, :]
                * j[:, mu, None] * gy[:, :, axis]
                * tensor[None, :, mu, axis]
            )
        mixed_direct_vt.append(0.5 * params.hbar_c * lorentz_sign[mu] * mixed)

        reverse = torch.zeros((), dtype=torch.float64, device=weights.device)
        for axis in range(3):
            reverse += torch.sum(
                pair_weights * g_tensor[:, None] * g_rho[None, :]
                * tensor[:, mu, axis, None] * gx[:, :, axis]
                * j[None, :, mu]
            )
        mixed_direct_tv.append(0.5 * params.hbar_c * lorentz_sign[mu] * reverse)

    pion_direct = torch.zeros((), dtype=torch.float64, device=weights.device)
    for axis_x in range(3):
        for axis_y in range(3):
            pion_direct -= 0.5 * params.hbar_c * torch.sum(
                pair_weights * g_pion[:, None] * g_pion[None, :]
                * axial[:, axis_x, None] * gxy_pi[:, :, axis_x, axis_y]
                * axial[None, :, axis_y]
            )
    return {
        "rho_tensor_direct_components": torch.stack(rt_direct_components),
        "rho_vt_direct_components": torch.stack(mixed_direct_vt),
        "rho_tv_direct_components": torch.stack(mixed_direct_tv),
        "pion_pv_direct": pion_direct,
        "rho_b": rho_b,
        "spatial_baryon_current": j_total[:, 1:],
        "spatial_axial_current": axial,
    }


def _fock_reference(q_by_species, rho_by_species, weights, green, params,
                    rho_b):
    """Wick-exchange terms from explicit ordered endpoint bilinears.

    For each species pair (s,t), B_ab(x)=q_{s,a}^dagger O_x q_{t,b} and
    B_ba(y)=q_{t,b}^dagger O_y q_{s,a}. The active-index contraction is
    Tr[rho_s B(x) rho_t B(y)] with B(y)'s reversed indices. Derivative
    operators are transferred to the raw propagator endpoint blocks only.
    """
    vector_ops, tensor_ops, axial_ops = _dirac_matrices(weights.device)
    zeta = rho_b / params.rho_sat
    g_rho = params.g_rho * torch.exp(-params.a_rho * zeta)
    g_tensor = params.rho_tensor_vertex * torch.exp(-params.a_rho_tensor * zeta)
    g_pion = params.f_pi * torch.exp(-params.a_pi * zeta) * params.hbar_c / params.m_pi
    pcount = weights.numel()
    pair_weights = weights[:, None] * weights[None, :]
    raw_rhot = green.kernel_block(0, pcount, 0, pcount, params.m_rho,
                                  derivative_x=True, derivative_y=True)
    raw_gx = green.kernel_block(0, pcount, 0, pcount, params.m_rho,
                                derivative_x=True)
    raw_gy = green.kernel_block(0, pcount, 0, pcount, params.m_rho,
                                derivative_y=True)
    raw_pi = green.kernel_block(0, pcount, 0, pcount, params.m_pi,
                                derivative_x=True, derivative_y=True)
    metric = (1.0, -1.0, -1.0, -1.0)
    by_mu = torch.zeros((4, 3, 3), dtype=torch.float64, device=weights.device)
    by_mu_species = torch.zeros((2, 2, 4, 3, 3), dtype=torch.float64, device=weights.device)
    vt_by_mu = torch.zeros((4,), dtype=torch.float64, device=weights.device)
    tv_by_mu = torch.zeros_like(vt_by_mu)
    vt_by_mu_axis = torch.zeros((4, 3), dtype=torch.float64, device=weights.device)
    tv_by_mu_axis = torch.zeros_like(vt_by_mu_axis)
    vt_species = torch.zeros((2, 2, 4, 3), dtype=torch.float64, device=weights.device)
    tv_species = torch.zeros_like(vt_species)
    pion_by_ij = torch.zeros((3, 3), dtype=torch.float64, device=weights.device)

    for left_index, left in enumerate(("n", "p")):
        for right_index, right in enumerate(("n", "p")):
            # tau_1 dot tau_2 exchange matrix elements in the charge basis:
            # 1 for nn/pp, 2 for the ordered np/pn charge-exchange elements.
            iso = 1.0 if left == right else 2.0
            ql, qr = q_by_species[left], q_by_species[right]
            rho_l, rho_r = rho_by_species[left], rho_by_species[right]
            weighted_l, weighted_r = ql @ rho_l, qr @ rho_r
            for mu in range(4):
                sign = metric[mu]
                for i in range(3):
                    # rho-T: B_ab^T(x) B_ba^T(y), with neither endpoint
                    # transition transposed after forming the reverse pair.
                    left_t = _transition(weighted_l, tensor_ops[mu][i], weighted_r)
                    right_t = _transition(qr, tensor_ops[mu][i], ql)
                    for j in range(3):
                        endpoint = torch.einsum("xab,yba->xy", left_t,
                                                _transition(qr, tensor_ops[mu][j], ql))
                        by_mu[mu, i, j] -= 0.5 * params.hbar_c * iso * sign * torch.sum(
                            pair_weights * g_tensor[:, None] * g_tensor[None, :]
                            * endpoint * raw_rhot[:, :, i, j]
                        ).real
                        by_mu_species[left_index, right_index, mu, i, j] -= (
                            0.5 * params.hbar_c * iso * sign * torch.sum(
                                pair_weights * g_tensor[:, None] * g_tensor[None, :]
                                * endpoint * raw_rhot[:, :, i, j]
                            ).real
                        )

                left_v = _transition(weighted_l, vector_ops[mu], weighted_r)
                for axis in range(3):
                    # Ordered V(x)T(y): derivative acts at y only.
                    right_t = _transition(qr, tensor_ops[mu][axis], ql)
                    endpoint = torch.einsum("xab,yba->xy", left_v, right_t)
                    contribution = -0.5 * params.hbar_c * iso * sign * torch.sum(
                        pair_weights * g_rho[:, None] * g_tensor[None, :]
                        * endpoint * raw_gy[:, :, axis]
                    ).real
                    vt_by_mu_axis[mu, axis] += contribution
                    vt_species[left_index, right_index, mu, axis] += contribution

                    # Ordered T(x)V(y): derivative acts at x only.
                    left_t = _transition(weighted_l, tensor_ops[mu][axis], weighted_r)
                    right_v = _transition(qr, vector_ops[mu], ql)
                    endpoint = torch.einsum("xab,yba->xy", left_t, right_v)
                    contribution = -0.5 * params.hbar_c * iso * sign * torch.sum(
                        pair_weights * g_tensor[:, None] * g_rho[None, :]
                        * endpoint * raw_gx[:, :, axis]
                    ).real
                    tv_by_mu_axis[mu, axis] += contribution
                    tv_species[left_index, right_index, mu, axis] += contribution

            for i in range(3):
                left_a = _transition(weighted_l, axial_ops[i], weighted_r)
                for j in range(3):
                    right_a = _transition(qr, axial_ops[j], ql)
                    endpoint = torch.einsum("xab,yba->xy", left_a, right_a)
                    pion_by_ij[i, j] += 0.5 * params.hbar_c * iso * torch.sum(
                        pair_weights * g_pion[:, None] * g_pion[None, :]
                        * endpoint * raw_pi[:, :, i, j]
                    ).real

    vt_by_mu = vt_by_mu_axis.sum(dim=1)
    tv_by_mu = tv_by_mu_axis.sum(dim=1)
    return {
        "rho_tensor_fock_by_muij": by_mu,
        "rho_tensor_fock_species_by_muij": by_mu_species,
        "rho_tensor_fock": by_mu.sum(),
        "rho_vt_fock_by_mu": vt_by_mu,
        "rho_vt_fock_by_mu_axis": vt_by_mu_axis,
        "rho_vt_fock_species_by_mu_axis": vt_species,
        "rho_vt_fock": vt_by_mu.sum(),
        "rho_tv_fock_by_mu": tv_by_mu,
        "rho_tv_fock_by_mu_axis": tv_by_mu_axis,
        "rho_tv_fock_species_by_mu_axis": tv_species,
        "rho_tv_fock": tv_by_mu.sum(),
        "pion_pv_fock_by_ij": pion_by_ij,
        "pion_pv_fock": pion_by_ij.sum(),
    }


def general_current_vertex_reference(basis, states, *, lmax=2):
    """Compare production derivative channels to explicit Gamma endpoint sums."""
    params = pka1_parameters()
    q_by_species = {s: basis.sample(states[s].q)[0] for s in ("n", "p")}
    rho_by_species = {s: states[s].rho_active for s in ("n", "p")}
    _, weights, _ = basis.sample(states["n"].q)
    green = GreenOperator(basis, lmax=lmax, device=weights.device)
    direct = _direct_reference(q_by_species, rho_by_species, weights, green, params)
    fock = _fock_reference(q_by_species, rho_by_species, weights, green,
                           params, direct["rho_b"])
    production = evaluate_particle_hole_energy(
        basis, states, params=params, lmax=lmax, chunk_size=128, pair_chunk=2,
    )
    return {
        "reference": direct | fock,
        "production": production.energy_terms,
        "audit": production.audit,
    }
