"""General-m RHFB PKA1 energy prototype.

All declared finite-range and compensation channels are evaluated on full
complex Dirac transition densities.  Derivative vertices use the exact
weighted discrete adjoint of the Cartesian gradient.  This implementation is
still experimental until physical and spherical-reduction closure passes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from dpl_rhf.rhf.parameters import RHFParameters, pka1_parameters
from .basis import AngularRadialBasis
from .contact import pion_pv_contact_energy, rho_tensor_contact_energy
from .currents import _gamma_matrices, invariant_baryon_density, local_bilinears
from .operators import DensityFactors
from .gaussian_pairing import GaussianPairingAction
from .spatial import GreenOperator
from .fock_checkpoint import exact_energy_block


@dataclass
class RHFBFunctionalEvaluation:
    total_energy: torch.Tensor
    energy_terms: dict[str, torch.Tensor]
    audit: dict
    stationarity: Callable


_TERM_KEYS = (
    "kinetic", "d1s_pairing", "sigma_direct", "sigma_fock",
    "omega_direct", "omega_fock", "rho_vector_direct", "rho_vector_fock",
    "rho_tensor_direct", "rho_tensor_fock", "rho_vt_direct", "rho_vt_fock",
    "rho_tv_direct", "rho_tv_fock", "pion_pv_direct", "pion_pv_fock",
    "coulomb_direct", "coulomb_fock", "rho_tensor_contact", "pion_pv_contact",
)
_CHANNELS = _TERM_KEYS + ("density_rearrangement",)


def functional_readiness() -> dict:
    """Lightweight implementation/closure manifest for trainer preflight.

    A channel marked implemented only means its Cartesian contribution is
    evaluated by this prototype.  None is closure-certified yet, so callers
    must keep the physical-training gate closed before allocating a large
    basis or evaluating the Green operators.
    """
    implemented = set(_TERM_KEYS) | {"density_rearrangement"}
    channels = {
        name: {
            "implemented": name in implemented,
            "closure_status": "unvalidated",
            "reference": "RHF-e.pdf, Eqs. (1.24)-(1.25), (3.204)-(3.298)",
        }
        for name in _CHANNELS
    }
    return {"ready": False, "channels": channels,
            "source": "experimental Cartesian RHFB implementation; physical closure pending"}


def _couplings(rho_b: torch.Tensor, p: RHFParameters) -> dict[str, torch.Tensor]:
    """PKA1 density-dependent vertices evaluated at invariant rho_B."""
    zeta = rho_b / p.rho_sat
    def rational(g0, a, b, c, shift):
        x = zeta + shift
        return g0 * a * (1 + b*x.square()) / (1 + c*x.square())
    return {
        "gsig": rational(p.g_sigma, p.a_sigma, p.b_sigma, p.c_sigma, p.d_sigma),
        "gome": rational(p.g_omega, p.a_omega, p.b_omega, p.c_omega, p.d_omega),
        "grho": p.g_rho * torch.exp(-p.a_rho*zeta),
        "fpio": p.f_pi * torch.exp(-p.a_pi*zeta),
        "grtn": p.rho_tensor_vertex * torch.exp(-p.a_rho_tensor*zeta),
    }


def _species_samples(basis, states, chunk_size):
    sampled = {}
    for species, state in states.items():
        sampled[species] = basis._sample_range(state.q, 0, basis.point_count)
    device = next(iter(states.values())).q.device
    weights = torch.as_tensor(basis.weights, dtype=torch.float64, device=device)
    positions = torch.as_tensor(basis.positions, dtype=torch.float64, device=device)
    return sampled, weights, positions


def _transition(q_left: torch.Tensor, op: torch.Tensor,
                q_right: torch.Tensor) -> torch.Tensor:
    return torch.einsum("pia,ij,pjb->pab", q_left.conj(), op, q_right)


def _exchange_channel(q, states, weights, coupling, green, mass, operators,
                      *, isovector=False, photon=False, pair_chunk=32,
                      checkpoint_blocks=True):
    """Ordered transition-density Fock trace, chunking active state pairs.

    ``operators`` are the Dirac-space matrices of the covariant vertex; the
    associated ``metric_sign`` is the Lorentz contraction. This streams over
    active pair indices and never forms a four-index orbital tensor.
    """
    zero = weights.new_zeros(())
    energy = zero
    species = ("n", "p")
    # A density matrix is generally dense in the active space.  Weight the
    # sampled orbitals once, before slicing pair chunks, so each block keeps
    # contributions from every active index (including indices outside the
    # current chunk).
    weighted = {s: torch.einsum("pia,ab->pib", q[s], states[s].rho_active)
                for s in species}
    for left in species:
        for right in species:
            if photon and (left != "p" or right != "p"):
                continue
            iso = (1.0 if left == right else 2.0) if isovector else (1.0 if left == right else 0.0)
            if not iso:
                continue
            ql, qr = q[left], q[right]
            ml, mr = ql.shape[-1], qr.shape[-1]
            for op, metric_sign in operators:
                value = zero
                for a0 in range(0, ml, pair_chunk):
                    a1 = min(ml, a0 + pair_chunk)
                    for b0 in range(0, mr, pair_chunk):
                        b1 = min(mr, b0 + pair_chunk)
                        # Checkpoint the full pair contraction. The saved
                        # scalar replaces transition, source, Green field,
                        # and pointwise contraction activations; backward
                        # recomputes the unchanged block from these factors.
                        def contract(w_left, w_right, q_right, q_left,
                                     c_left, c_right, volume_weights,
                                     op=op, green=green, mass=mass):
                            t_lr = torch.einsum("pia,ij,pjb->pab",
                                                w_left, op, w_right)
                            t_rl = torch.einsum("pib,ij,pja->pba",
                                                q_right.conj(), op, q_left)
                            source = (c_right[:, None, None] * t_rl).reshape(
                                (volume_weights.numel(), -1))
                            field = green.apply(source, mass).reshape_as(t_rl)
                            block = torch.einsum("pab,pba->p", t_lr, field)
                            return torch.sum(volume_weights * c_left * block).real

                        value = value + exact_energy_block(
                            contract,
                            weighted[left][:, :, a0:a1].conj(),
                            weighted[right][:, :, b0:b1],
                            qr[:, :, b0:b1], ql[:, :, a0:a1],
                            coupling[left], coupling[right], weights,
                            enabled=checkpoint_blocks,
                        )
                energy = energy - 0.5 * metric_sign * iso * value
    return energy


def _transition_derivative_source(q_left, q_right, operators, coupling, green):
    """Exact discrete weak derivative source ``G†[coupling * current]``."""
    current = torch.stack([
        torch.einsum("pia,ij,pjb->pab", q_left.conj(), op, q_right)
        for op in operators
    ], dim=1)
    weighted_current = current * coupling[:, None, None, None]
    return green.gradient_adjoint(weighted_current)


def _exchange_derivative_channel(q, states, weights, coupling, green, mass,
                                 operators, *, isovector=True,
                                 pair_chunk=32, prefactor=-0.5,
                                 checkpoint_blocks=True):
    """Fock contraction for derivative-on-meson vertices after integration by parts."""
    weighted = {s: torch.einsum("pia,ab->pib", q[s], states[s].rho_active)
                for s in ("n", "p")}
    energy = weights.new_zeros(())
    for left in ("n", "p"):
        for right in ("n", "p"):
            iso = (1.0 if left == right else 2.0) if isovector else (1.0 if left == right else 0.0)
            if not iso:
                continue
            ql, qr = q[left], q[right]
            for component in operators:
                if (isinstance(component, tuple) and len(component) == 2
                        and isinstance(component[1], (float, int))):
                    component_ops, component_sign = component
                else:
                    component_ops, component_sign = component, 1.0
                value = weights.new_zeros(())
                ml, mr = ql.shape[-1], qr.shape[-1]
                for a0 in range(0, ml, pair_chunk):
                    a1 = min(ml, a0 + pair_chunk)
                    for b0 in range(0, mr, pair_chunk):
                        b1 = min(mr, b0 + pair_chunk)
                        def contract(w_left, w_right, q_right, q_left,
                                     c_left, c_right, volume_weights,
                                     component_ops=component_ops, green=green,
                                     mass=mass):
                            lhs = _transition_derivative_source(
                                w_left, w_right, component_ops, c_left, green,
                            )
                            rhs = _transition_derivative_source(
                                q_right, q_left, component_ops, c_right, green,
                            )
                            field = green.apply(
                                rhs.reshape(volume_weights.numel(), -1), mass
                            ).reshape_as(rhs)
                            block = torch.einsum("pab,pba->p", lhs, field)
                            return torch.sum(volume_weights * block).real

                        value = value + exact_energy_block(
                            contract,
                            weighted[left][:, :, a0:a1],
                            weighted[right][:, :, b0:b1],
                            qr[:, :, b0:b1], ql[:, :, a0:a1],
                            coupling[left], coupling[right], weights,
                            enabled=checkpoint_blocks,
                        )
                energy = energy + prefactor * iso * component_sign * value
    return energy


def _direct_derivative_channel(q, rho, weights, coupling, green, mass,
                               operators, *, metric_signs=None, prefactor=0.5):
    """Direct derivative-source energy; components are kept fully Cartesian."""
    signs = [1.0] * len(operators) if metric_signs is None else metric_signs
    energy = weights.new_zeros(())
    for component_ops, sign in zip(operators, signs):
        species_currents = []
        for species in ("n", "p"):
            current_matrix = torch.stack([
                torch.einsum("pia,ij,pjb->pab", q[species].conj(), op,
                             q[species])
                for op in component_ops
            ], dim=1)
            species_currents.append(torch.einsum(
                "ab,piba->pi", rho[species], current_matrix
            ).real)
        # The rho and pion fields are isovectors; in a diagonal n/p normal
        # density their direct source is the third isospin component.
        total_current = coupling[:, None] * (species_currents[0] - species_currents[1])
        source = green.gradient_adjoint(total_current)
        field = green.apply(source, mass)
        energy = energy + prefactor * sign * torch.sum(weights * source * field).real
    return energy


def _exchange_mixed_derivative_channel(q, states, weights, left_coupling,
                                       right_coupling, green, mass,
                                       vector_operators, derivative_operators,
                                       metric_signs, *, derivative_on_left,
                                       pair_chunk=32, prefactor=-0.5,
                                       checkpoint_blocks=True):
    """One ordered rho-VT/rho-TV Fock vertex orientation."""
    weighted = {s: torch.einsum("pia,ab->pib", q[s], states[s].rho_active)
                for s in ("n", "p")}
    result = weights.new_zeros(())
    for left in ("n", "p"):
        for right in ("n", "p"):
            iso = 1.0 if left == right else 2.0
            ql, qr = q[left], q[right]
            for mu, (vector_op, derivative_ops, sign) in enumerate(zip(
                    vector_operators, derivative_operators, metric_signs)):
                value = weights.new_zeros(())
                for a0 in range(0, ql.shape[-1], pair_chunk):
                    a1 = min(ql.shape[-1], a0 + pair_chunk)
                    for b0 in range(0, qr.shape[-1], pair_chunk):
                        b1 = min(qr.shape[-1], b0 + pair_chunk)
                        def contract(w_left, w_right, q_right, q_left,
                                     c_left, c_right, volume_weights,
                                     derivative_on_left=derivative_on_left,
                                     vector_op=vector_op, derivative_ops=derivative_ops,
                                     green=green, mass=mass):
                            if derivative_on_left:
                                lhs = _transition_derivative_source(
                                    w_left, w_right, derivative_ops, c_left, green,
                                )
                                rhs = torch.einsum(
                                    "pia,ij,pjb->pab", q_right.conj(),
                                    vector_op, q_left,
                                )
                                rhs = c_right[:, None, None] * rhs
                            else:
                                lhs = torch.einsum(
                                    "pia,ij,pjb->pab", w_left.conj(),
                                    vector_op, w_right,
                                )
                                rhs = _transition_derivative_source(
                                    q_right, q_left, derivative_ops, c_right, green,
                                )
                                lhs = c_left[:, None, None] * lhs
                            field = green.apply(
                                rhs.reshape(volume_weights.numel(), -1), mass
                            ).reshape_as(rhs)
                            return torch.sum(
                                volume_weights * torch.einsum("pab,pba->p", lhs, field)
                            ).real

                        value = value + exact_energy_block(
                            contract,
                            weighted[left][:, :, a0:a1],
                            weighted[right][:, :, b0:b1],
                            qr[:, :, b0:b1], ql[:, :, a0:a1],
                            left_coupling[left], right_coupling[right], weights,
                            enabled=checkpoint_blocks,
                        )
                result += prefactor * iso * sign * value
    return result


def _direct_mixed_derivative_channel(q, rho, weights, vector_coupling,
                                     tensor_coupling, green, mass,
                                     vector_operators, derivative_operators,
                                     metric_signs, *, prefactor=0.5):
    energy = weights.new_zeros(())
    for vector_op, derivative_ops, sign in zip(
            vector_operators, derivative_operators, metric_signs):
        currents, tensor_currents = [], []
        for species in ("n", "p"):
            vector = torch.einsum("ab,pib,ij,pja->p", rho[species],
                                  q[species].conj(), vector_op, q[species]).real
            tensor_matrix = torch.stack([
                torch.einsum("pia,ij,pjb->pab", q[species].conj(), op,
                             q[species])
                for op in derivative_ops
            ], dim=1)
            tensor_current = torch.einsum(
                "ab,piba->pi", rho[species], tensor_matrix
            ).real
            currents.append(vector)
            tensor_currents.append(tensor_current)
        j = vector_coupling * (currents[0] - currents[1])
        transition = tensor_coupling[:, None] * (
            tensor_currents[0] - tensor_currents[1]
        )
        source = green.gradient_adjoint(transition)
        energy += sign * prefactor * torch.sum(
            weights * (j * green.apply(source, mass)
                       + source * green.apply(j, mass)).real
        )
    return energy


def _stationarity(states, *, chunk_size=32):
    """Parameter-gradient diagnostic; not a certified HFB commutator."""
    metrics = {}
    for species, state in states.items():
        metrics[f"{species}_q_parameter_gradient_norm"] = torch.full(
            (), float("nan"), dtype=torch.float64, device=state.q.device
        )
    return metrics


def evaluate_rhfb_energy(basis: AngularRadialBasis,
                         states: dict[str, DensityFactors], *,
                         chunk_size: int = 32, pair_chunk: int = 32,
                         params: RHFParameters | None = None,
                         lmax: int | None = None,
                         include_pairing: bool = True,
                         _freeze_vertex_density: bool = False,
                         _vertex_density_override: torch.Tensor | None = None,
                         _checkpoint_fock_blocks: bool = True) -> RHFBFunctionalEvaluation:
    """Evaluate the complete experimental PKA1 RHFB energy on complex spinors.

    Every sigma, omega, rho-vector, rho-tensor, rho-VT/rho-TV, pion-PV,
    Coulomb, contact, kinetic, and optional D1S term enters the scalar.  The
    audit remains closed until independent physical-reference and spherical
    closure checks pass.
    """
    if set(states) != {"n", "p"}:
        raise ValueError("provide separate neutron and proton generalized densities")
    if chunk_size < 1 or pair_chunk < 1:
        raise ValueError("chunk sizes must be positive")
    p = pka1_parameters() if params is None else params
    if p.model != "PKA1":
        raise ValueError("this general RHFB prototype currently requires PKA1")
    q, weights, positions = _species_samples(basis, states, chunk_size)
    currents = {s: local_bilinears(q[s], states[s].rho_active) for s in ("n", "p")}
    total_j = currents["n"]["baryon_current"] + currents["p"]["baryon_current"]
    rho_b = invariant_baryon_density(total_j)
    vertex_density = rho_b if _vertex_density_override is None else _vertex_density_override
    if (vertex_density.shape != rho_b.shape or vertex_density.dtype != rho_b.dtype
            or vertex_density.device != rho_b.device):
        raise ValueError("vertex-density override must match the sampled float64 field")
    vtx = _couplings(vertex_density, p)
    if _freeze_vertex_density:
        # Diagnostic coordinate split: the explicit density dependence of
        # every PH contraction remains active, while all density-dependent
        # vertices are held at their reference values.  The energy is
        # unchanged; subtracting this derivative from the full derivative
        # isolates the rearrangement generated by the vertices.
        vtx = {name: value.detach() for name, value in vtx.items()}
    gamma, gamma5 = _gamma_matrices(device=weights.device, dtype=torch.complex128)
    g0 = gamma[0]
    green = GreenOperator(basis, lmax=lmax, device=weights.device)

    # Four-current sources use contravariant bilinears and (+---) metric.
    rho_s = currents["n"]["scalar"] + currents["p"]["scalar"]
    j3 = currents["n"]["baryon_current"] - currents["p"]["baryon_current"]
    charge = currents["p"]["baryon_current"]
    source_sigma = vtx["gsig"] * rho_s
    source_omega = vtx["gome"][:, None] * total_j
    source_rho = vtx["grho"][:, None] * j3
    photon_v = (4 * torch.pi / p.alpha_inv) ** 0.5
    source_coul = photon_v * charge
    def direct_scalar(source, mass, sign):
        field = green.apply(source, mass)
        return sign * 0.5 * p.hbar_c * torch.sum(weights * source * field).real
    def direct_vector(source, mass):
        field = green.apply(source, mass)
        metric = source[:, 0] * field[:, 0] - torch.sum(source[:, 1:] * field[:, 1:], dim=1)
        return 0.5 * p.hbar_c * torch.sum(weights * metric).real

    terms = {key: weights.new_zeros(()) for key in _TERM_KEYS}
    terms["sigma_direct"] = direct_scalar(source_sigma, p.m_sigma, -1.0)
    terms["omega_direct"] = direct_vector(source_omega, p.m_omega)
    terms["rho_vector_direct"] = direct_vector(source_rho, p.m_rho)
    terms["coulomb_direct"] = direct_vector(source_coul, 0.0)
    # O=gamma^0 Gamma in Q†OQ, with vector metric signs from (+---).
    scalar_ops = [(g0 @ torch.eye(4, dtype=torch.complex128, device=g0.device), -1.0)]
    vector_ops = [(g0 @ gamma[mu], (1.0, -1.0, -1.0, -1.0)[mu]) for mu in range(4)]
    terms["sigma_fock"] = _exchange_channel(
        q, states, weights, {s: vtx["gsig"] for s in states}, green, p.m_sigma,
        scalar_ops, pair_chunk=pair_chunk,
        checkpoint_blocks=_checkpoint_fock_blocks) * p.hbar_c
    terms["omega_fock"] = _exchange_channel(
        q, states, weights, {s: vtx["gome"] for s in states}, green, p.m_omega,
        vector_ops, pair_chunk=pair_chunk,
        checkpoint_blocks=_checkpoint_fock_blocks) * p.hbar_c
    terms["rho_vector_fock"] = _exchange_channel(
        q, states, weights, {s: vtx["grho"] for s in states}, green, p.m_rho,
        vector_ops, isovector=True, pair_chunk=pair_chunk,
        checkpoint_blocks=_checkpoint_fock_blocks) * p.hbar_c
    terms["coulomb_fock"] = _exchange_channel(
        q, states, weights, {s: torch.ones_like(rho_b) * photon_v for s in states}, green,
        0.0, vector_ops, photon=True, pair_chunk=pair_chunk,
        checkpoint_blocks=_checkpoint_fock_blocks) * p.hbar_c

    # Derivative meson vertices are converted to local divergences only after
    # integration by parts.  This retains every Cartesian component and keeps
    # the coincident Green-kernel Hessian in the finite-range operator.
    rho_tensor_ops, rho_tensor_signs = [], []
    for mu in range(4):
        axis_ops = []
        for axis in range(1, 4):
            sigma = 0.5j * (gamma[axis] @ gamma[mu] - gamma[mu] @ gamma[axis])
            axis_ops.append(g0 @ sigma)
        rho_tensor_ops.append(axis_ops)
        rho_tensor_signs.append((1.0, -1.0, -1.0, -1.0)[mu])
    rho_vector_ops = [g0 @ gamma[mu] for mu in range(4)]
    pion_ops = [[g0 @ gamma5 @ gamma[axis] for axis in range(1, 4)]]
    rho_tensor_vertex = vtx["grtn"]
    pion_vertex = vtx["fpio"] * p.hbar_c / p.m_pi
    terms["rho_tensor_direct"] = p.hbar_c * _direct_derivative_channel(
        q, {s: states[s].rho_active for s in states}, weights, rho_tensor_vertex,
        green, p.m_rho, rho_tensor_ops, metric_signs=rho_tensor_signs,
        prefactor=0.5,
    )
    terms["rho_tensor_fock"] = p.hbar_c * _exchange_derivative_channel(
        q, states, weights, {s: rho_tensor_vertex for s in states}, green,
        p.m_rho, list(zip(rho_tensor_ops, rho_tensor_signs)),
        isovector=True, pair_chunk=pair_chunk, prefactor=-0.5,
        checkpoint_blocks=_checkpoint_fock_blocks,
    )
    rho_vt = p.hbar_c * _exchange_mixed_derivative_channel(
        q, states, weights,
        {s: vtx["grho"] for s in states},
        {s: rho_tensor_vertex for s in states}, green, p.m_rho,
        rho_vector_ops, rho_tensor_ops, rho_tensor_signs,
        derivative_on_left=False, pair_chunk=pair_chunk, prefactor=-0.5,
        checkpoint_blocks=_checkpoint_fock_blocks,
    )
    rho_tv = p.hbar_c * _exchange_mixed_derivative_channel(
        q, states, weights,
        {s: rho_tensor_vertex for s in states},
        {s: vtx["grho"] for s in states}, green, p.m_rho,
        rho_vector_ops, rho_tensor_ops, rho_tensor_signs,
        derivative_on_left=True, pair_chunk=pair_chunk, prefactor=-0.5,
        checkpoint_blocks=_checkpoint_fock_blocks,
    )
    terms["rho_vt_fock"] = rho_vt
    terms["rho_tv_fock"] = rho_tv
    rho_vt_hartree = p.hbar_c * _direct_mixed_derivative_channel(
        q, {s: states[s].rho_active for s in states}, weights, vtx["grho"],
        rho_tensor_vertex, green, p.m_rho, rho_vector_ops, rho_tensor_ops,
        rho_tensor_signs, prefactor=0.5,
    )
    terms["rho_vt_direct"] = 0.5 * rho_vt_hartree
    terms["rho_tv_direct"] = 0.5 * rho_vt_hartree
    terms["pion_pv_direct"] = p.hbar_c * _direct_derivative_channel(
        q, {s: states[s].rho_active for s in states}, weights, pion_vertex,
        green, p.m_pi, pion_ops, prefactor=-0.5,
    )
    terms["pion_pv_fock"] = p.hbar_c * _exchange_derivative_channel(
        q, states, weights, {s: pion_vertex for s in states}, green,
        p.m_pi, pion_ops, isovector=True, pair_chunk=pair_chunk,
        prefactor=0.5, checkpoint_blocks=_checkpoint_fock_blocks,
    )

    # Kinetic and D1S anomalous terms use the same factorized generalized density.
    kin = weights.new_zeros(())
    pairing = weights.new_zeros(())
    gaps = {}
    for species, state in states.items():
        kin = kin + torch.trace(state.rho_active @ (state.q.mH @ basis.apply_kinetic(
            state.q, species=species))).real
        if not include_pairing:
            continue
        action = GaussianPairingAction(
            basis, state.q, state.chi_active, species=species,
            chunk_size=chunk_size,
            lmax=max(2 * basis.kappa_max, 0 if lmax is None else int(lmax)),
            checkpoint_blocks=True,
        )
        ep, gap = action.energy_and_active_gap()
        pairing, gaps[species] = pairing + ep, gap
    terms["kinetic"] = kin
    terms["d1s_pairing"] = pairing
    # RHF-e Eqs. (3.204) and (3.293) add independent phenomenological contact
    # counterterms.  The finite-range Green Hessian above remains unchanged.
    rt_chunks, pi_chunks = [], []
    for start in range(0, basis.point_count, chunk_size):
        stop = min(basis.point_count, start + chunk_size)
        qs = {s: q[s][start:stop] for s in states}
        ws = weights[start:stop]
        rt_chunks.append(rho_tensor_contact_energy(qs, {s: states[s].rho_active for s in states},
                            ws, vtx["grtn"][start:stop], hbar_c_mev_fm=p.hbar_c)["total"])
        pi_v = vtx["fpio"][start:stop] * p.hbar_c / p.m_pi
        pi_chunks.append(pion_pv_contact_energy(qs, {s: states[s].rho_active for s in states},
                            ws, pi_v, hbar_c_mev_fm=p.hbar_c)["total"])
    terms["rho_tensor_contact"] = sum(rt_chunks, weights.new_zeros(()))
    terms["pion_pv_contact"] = sum(pi_chunks, weights.new_zeros(()))

    # Density dependence is part of the scalar functional; its rearrangement
    # field is generated by autograd in the normal action, not added as a
    # separate energy term.
    energy_keys = (("kinetic", "d1s_pairing") if include_pairing else ("kinetic",)) + (
                   "sigma_direct", "sigma_fock",
                   "omega_direct", "omega_fock", "rho_vector_direct", "rho_vector_fock",
                   "rho_tensor_direct", "rho_tensor_fock", "rho_vt_direct", "rho_vt_fock",
                   "rho_tv_direct", "rho_tv_fock", "pion_pv_direct", "pion_pv_fock",
                   "coulomb_direct", "coulomb_fock", "rho_tensor_contact", "pion_pv_contact")
    total = sum((terms[k] for k in energy_keys), weights.new_zeros(()))
    terms["density_rearrangement"] = weights.new_zeros(())
    evidence = {}
    for key in _CHANNELS:
        evidence[key] = {
            "implemented": key in energy_keys,
            "energy_term_names": [key] if key in terms else [],
            "energy_identity_residual": float("inf"), "identity_tolerance": 1e-10,
            "directional_fd_residual": float("inf"), "directional_fd_tolerance": 1e-6,
            "gradient_finite": False,
            "status": "unvalidated; derivative vertices/rearrangement/physical closure pending",
        }
    audit = {
        "complete": False, "channels": evidence,
        "total_energy_identity_residual": float("inf"),
        "total_energy_identity_tolerance": 1e-10,
        "normal_branch_status": "unverified",
        "anomalous_sea_branch_status": "unverified",
        "independent_physical_reference_status": "unverified",
        "no_sea_status": "experimental",
        "rearrangement_status": "vertex-density derivative is implicit in the scalar; independent field closure unvalidated",
        "space_convergence_status": "unverified",
        "included_energy_terms": energy_keys,
        "excluded_energy_terms": tuple(k for k in _TERM_KEYS if k not in energy_keys),
        "source": "RHF-e Eqs. (1.24)-(1.25), (3.204)-(3.298); experimental Cartesian implementation",
    }
    def stationarity(states, *, chunk_size=32, vector_chunk=8):
        from .fields import full_stationarity

        def normal_energy(active_basis, normal_states):
            return evaluate_particle_hole_energy(
                active_basis, normal_states, chunk_size=chunk_size,
                pair_chunk=pair_chunk, params=p, lmax=lmax,
            ).total_energy

        return full_stationarity(
            basis, states, normal_energy, chunk_size=chunk_size,
            vector_chunk=vector_chunk, lmax=lmax,
            active_chunk_size=pair_chunk,
        )

    return RHFBFunctionalEvaluation(total, terms, audit, stationarity)


def evaluate_particle_hole_energy(basis: AngularRadialBasis, states: dict,
                                  *, chunk_size: int = 32, pair_chunk: int = 32,
                                  params: RHFParameters | None = None,
                                  lmax: int | None = None,
                                  _freeze_vertex_density: bool = False,
                                  _vertex_density_override: torch.Tensor | None = None,
                                  _checkpoint_fock_blocks: bool = True) -> RHFBFunctionalEvaluation:
    """Evaluate the normal particle-hole EDF on arbitrary Hermitian densities.

    ``states`` need only expose ``q`` and ``rho_active`` for each species.
    This intentionally avoids ``DensityFactors`` purity/orthogonality checks:
    particle-hole tangent probes vary rho independently of anomalous pairing.
    The returned terms omit D1S pairing and its density derivatives.
    """
    if set(states) != {"n", "p"}:
        raise ValueError("provide separate neutron and proton normal densities")
    from types import SimpleNamespace
    ph_states = {species: SimpleNamespace(q=state.q, rho_active=state.rho_active,
                                          chi_active=None)
                 for species, state in states.items()}
    return evaluate_rhfb_energy(
        basis, ph_states, chunk_size=chunk_size, pair_chunk=pair_chunk,
        params=params, lmax=lmax, include_pairing=False,
        _freeze_vertex_density=_freeze_vertex_density,
        _vertex_density_override=_vertex_density_override,
        _checkpoint_fock_blocks=_checkpoint_fock_blocks,
    )
