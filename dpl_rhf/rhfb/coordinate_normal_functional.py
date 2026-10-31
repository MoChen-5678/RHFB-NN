"""Prototype one-pass normal fields from the coordinate-space PKA1 EDF.

The public ``CoordinateNormalFunctional`` explicitly materializes each
species' weighted nonlocal coordinate density
``R_s[p,a,q,b] = B_w[p,a,i] rho_s[i,j] B_w[q,b,j]*``. It differentiates the
complete particle-hole scalar once with respect to both R fields and projects
the result back to the primitive basis. Green kernels are streamed in pair
tiles; no four-index interaction tensor is formed. Density factors are frozen
and detached, so this is a field/action diagnostic, not an end-to-end NN
gradient or self-consistent solver. ``PrimitiveDensityNormalFunctional`` is
retained as a slower reference adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import torch

from .basis import AngularRadialBasis
from .functional import (RHFBFunctionalEvaluation, _couplings,
                         evaluate_particle_hole_energy)
from .currents import _gamma_matrices, invariant_baryon_density
from .spatial import GreenOperator
from dpl_rhf.rhf.parameters import pka1_parameters
from .execution_limits import require_small_coordinate_space


@dataclass(frozen=True)
class CoordinateNormalEvaluation:
    """Energy decomposition and full primitive-space normal fields."""

    energy: RHFBFunctionalEvaluation
    rho_primitive: dict[str, torch.Tensor]
    h: dict[str, torch.Tensor]
    diagnostics: dict


class PrimitiveDensityNormalFunctional:
    """Evaluate one complete PH energy and differentiate both species once.

    The scalar callback is injectable for controlled testing, but production
    use defaults to ``evaluate_particle_hole_energy`` with the full PKA1 term
    set.  The frozen factors are converted to primitive densities exactly;
    no occupation inversion, low-rank truncation, or spectrum is used.
    """

    def __init__(self, basis: AngularRadialBasis, *, pair_chunk: int = 32,
                 chunk_size: int = 32, params=None, lmax: int | None = None,
                 checkpoint_fock_blocks: bool = True,
                 energy: Callable | None = None):
        if pair_chunk < 1 or chunk_size < 1:
            raise ValueError("chunk sizes must be positive")
        self.basis = basis
        self.pair_chunk = int(pair_chunk)
        self.chunk_size = int(chunk_size)
        self.params = params
        self.lmax = lmax
        self.checkpoint_fock_blocks = bool(checkpoint_fock_blocks)
        self.energy = evaluate_particle_hole_energy if energy is None else energy

    def primitive_densities(self, states: dict) -> dict[str, torch.Tensor]:
        if set(states) != {"n", "p"}:
            raise ValueError("coordinate normal fields require neutron and proton states")
        out = {}
        for species, state in states.items():
            q, rho = state.q, state.rho_active
            if (q.ndim != 2 or q.shape[0] != self.basis.dimension
                    or q.dtype != torch.complex128 or rho.dtype != torch.complex128
                    or rho.shape != (q.shape[1], q.shape[1]) or q.device != rho.device):
                raise ValueError(f"invalid {species} density factors")
            if not torch.isfinite(q).all() or not torch.isfinite(rho).all():
                raise ValueError(f"non-finite {species} density factors")
            if not torch.allclose(rho, rho.mH, atol=2e-10, rtol=0):
                raise ValueError(f"{species} active density must be Hermitian")
            # Density depends on Q rho Q†. This exact contraction retains all
            # non-diagonal and complex entries without reconstructing orbitals.
            out[species] = (q @ rho @ q.mH).detach()
        return out

    def evaluate(self, states: dict, *, create_graph: bool = False) -> CoordinateNormalEvaluation:
        rho0 = self.primitive_densities(states)
        leaf = {s: r.detach().clone().requires_grad_(True) for s, r in rho0.items()}
        eye = torch.eye(self.basis.dimension, dtype=torch.complex128,
                        device=next(iter(leaf.values())).device)
        primitive_states = {s: SimpleNamespace(q=eye, rho_active=leaf[s], chi_active=None)
                            for s in ("n", "p")}
        with torch.enable_grad():
            ev = self.energy(
                self.basis, primitive_states, pair_chunk=self.pair_chunk,
                chunk_size=self.chunk_size, params=self.params, lmax=self.lmax,
                _checkpoint_fock_blocks=self.checkpoint_fock_blocks,
            )
            if not isinstance(ev, RHFBFunctionalEvaluation):
                raise TypeError("coordinate energy callback must return RHFBFunctionalEvaluation")
            grads = torch.autograd.grad(ev.total_energy, (leaf["n"], leaf["p"]),
                                        create_graph=create_graph)
        h = {}
        for species, grad in zip(("n", "p"), grads):
            # PyTorch's real-scalar complex gradient convention gives dE =
            # Re sum(conj(grad)*d rho); on Hermitian tangents this is exactly
            # Tr(h d rho).  Symmetrization only removes out-of-domain roundoff.
            hs = 0.5 * (grad + grad.mH)
            h[species] = hs if create_graph else hs.detach()
        diagnostics = {
            "representation": "primitive-coordinate density kernel R=B rho B†; streamed channel contractions",
            "derivative_scope": "full PH EDF including implicit sqrt(j_mu j^mu) vertex rearrangement",
            "frozen_density": True,
            "self_consistent": False,
            "interaction_four_tensor_materialized": False,
            "coordinate_R_four_axis_materialized": False,
            "species_energy_forwards": 1,
            "joint_species_gradient_calls": 1,
            "pair_chunk": self.pair_chunk,
            "chunk_size": self.chunk_size,
            "basis_dimension": self.basis.dimension,
            "point_count": self.basis.point_count,
            "hermiticity_residual_max": {
                s: float(torch.max(torch.abs(hs - hs.mH)).detach())
                for s, hs in h.items()
            },
            "channel_energy_terms": tuple(ev.audit.get("included_energy_terms", ())),
        }
        return CoordinateNormalEvaluation(ev, rho0, h, diagnostics)


def estimate_coordinate_normal_memory(basis: AngularRadialBasis, *,
                                      active_rank: int = 14,
                                      pair_chunk: int = 32) -> dict[str, int]:
    """Conservative resident-storage estimate, excluding Green caches/AD tape."""
    if active_rank < 1 or pair_chunk < 1:
        raise ValueError("active_rank and pair_chunk must be positive")
    p, d = basis.point_count, basis.dimension
    k = min(pair_chunk, d)
    c16, r8 = 16, 8
    return {
        "primitive_sampled_spinors_bytes": p * 4 * d * c16,
        "primitive_density_per_species_bytes": d * d * c16,
        "primitive_field_per_species_bytes": d * d * c16,
        "active_coordinate_spinors_per_species_bytes": p * 4 * active_rank * c16,
        "one_fock_transition_block_bytes": p * 4 * k * k * c16,
        "one_fock_scalar_source_block_bytes": p * k * k * c16,
        "full_coordinate_density_per_species_if_materialized_bytes": p * p * 16 * c16,
        "full_coordinate_density_two_species_bytes": 2 * p * p * 16 * c16,
        "full_coordinate_density_gradients_two_species_bytes": 2 * p * p * 16 * c16,
        "notes": "Excludes Green angular caches and autograd tape.",
    }


def _spin_trace(operator_density: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
    return torch.einsum("pab,ba->p", operator_density, operator).real


def _endpoint_trace(rxy: torch.Tensor, ryx: torch.Tensor,
                    opx: torch.Tensor, opy: torch.Tensor) -> torch.Tensor:
    """Tr[R_s(y,x) O_x R_t(x,y) O_y], with explicit ordered endpoints."""
    return torch.einsum("ybxa,ac,xcyd,db->xy", ryx, opx, rxy, opy)


class CoordinateNormalFunctional:
    """Full PKA1 normal field from one coordinate-kernel variational pass.

    Unlike the primitive-density AD adapter above, this path first forms the
    low-rank weighted coordinate kernel R_xy=B_w Q rho Q† B_w†.  It evaluates
    the Wick direct and exchange contractions with the two coordinate
    endpoints explicit, differentiates the complete scalar once with respect
    to R, and projects h_coord back as B_w† h_coord B_w.  It never expands a
    four-index two-body tensor.  Coordinate pair tiles bound Green-kernel and
    spin-trace working storage.
    """

    def __init__(self, basis: AngularRadialBasis, *, pair_chunk: int = 32,
                 chunk_size: int = 32, params=None, lmax: int | None = None,
                 checkpoint_fock_blocks: bool = True):
        if pair_chunk < 1 or chunk_size < 1:
            raise ValueError("chunk sizes must be positive")
        self.basis = basis
        self.pair_chunk = int(pair_chunk)  # used as coordinate x tile width
        self.chunk_size = int(chunk_size)
        self.params = pka1_parameters() if params is None else params
        self.lmax = lmax
        self.checkpoint_fock_blocks = bool(checkpoint_fock_blocks)

    def _weighted_basis(self, device):
        require_small_coordinate_space(self.basis)
        eye = torch.eye(self.basis.dimension, dtype=torch.complex128, device=device)
        samples, weights, _ = self.basis.sample(eye)
        sqrtw = torch.sqrt(weights).to(torch.complex128)
        c = samples * sqrtw[:, None, None]
        return c, weights

    def _coordinate_density(self, state, c):
        a = c @ state.q
        return torch.einsum("pai,ij,qbj->paqb", a, state.rho_active, a.conj())

    def _local_density(self, r):
        p = r.shape[0]
        idx = torch.arange(p, device=r.device)
        return r[idx, :, idx, :]

    def _energy_from_coordinate_density(self, rhos, weights, green, *,
                                        return_terms=True, pair_range=None,
                                        include_local=True):
        p = self.params
        device = weights.device
        gamma, gamma5 = _gamma_matrices(device=device, dtype=torch.complex128)
        g0 = gamma[0]
        dloc = {s: self._local_density(rhos[s]) / weights[:, None, None]
                for s in ("n", "p")}
        local = {}
        for s in ("n", "p"):
            local[s] = {
                "scalar": _spin_trace(dloc[s], g0),
                "j": torch.stack([_spin_trace(dloc[s], g0 @ gm) for gm in gamma], dim=-1),
                "axial": torch.stack([
                    _spin_trace(dloc[s], g0 @ gamma5 @ gamma[k]) for k in range(1, 4)
                ], dim=-1),
                "tensor": torch.stack([
                    torch.stack([
                        _spin_trace(dloc[s], g0 @ (0.5j * (gamma[axis] @ gamma[mu]
                                                           - gamma[mu] @ gamma[axis])))
                        for axis in range(1, 4)
                    ], dim=-1) for mu in range(4)
                ], dim=1),
            }
        total_j = local["n"]["j"] + local["p"]["j"]
        rho_b = invariant_baryon_density(total_j)
        vtx = _couplings(rho_b, p)
        j_iso = local["n"]["j"] - local["p"]["j"]
        t_iso = local["n"]["tensor"] - local["p"]["tensor"]
        axial_iso = local["n"]["axial"] - local["p"]["axial"]
        scalar_total = local["n"]["scalar"] + local["p"]["scalar"]
        photon_v = (4 * torch.pi / p.alpha_inv) ** 0.5

        terms = {k: weights.new_zeros(()) for k in (
            "kinetic", "sigma_direct", "sigma_fock", "omega_direct", "omega_fock",
            "rho_vector_direct", "rho_vector_fock", "rho_tensor_direct",
            "rho_tensor_fock", "rho_vt_direct", "rho_vt_fock", "rho_tv_direct",
            "rho_tv_fock", "pion_pv_direct", "pion_pv_fock", "coulomb_direct",
            "coulomb_fock", "rho_tensor_contact", "pion_pv_contact",
            "density_rearrangement",
        )}
        # The kinetic term is evaluated and differentiated in primitive space;
        # the interaction coordinate derivative is projected below.
        if pair_range is None:
            pair_start, pair_stop = 0, weights.numel()
        else:
            if (not isinstance(pair_range, tuple) or len(pair_range) != 2
                    or any(not isinstance(value, int) for value in pair_range)):
                raise ValueError("pair_range must be a pair of integer x bounds")
            pair_start, pair_stop = pair_range
            if not 0 <= pair_start <= pair_stop <= weights.numel():
                raise ValueError("pair_range must lie within the coordinate points")
        if include_local:
            terms["sigma_direct"] = -0.5 * p.hbar_c * torch.sum(
                weights * vtx["gsig"] * scalar_total
                * green.apply(vtx["gsig"] * scalar_total, p.m_sigma))
            for term, source, mass in (
                ("omega_direct", vtx["gome"][:, None] * total_j, p.m_omega),
                ("rho_vector_direct", vtx["grho"][:, None] * j_iso, p.m_rho),
                ("coulomb_direct", torch.full_like(total_j, photon_v) * local["p"]["j"], 0.0),
            ):
                field = green.apply(source, mass)
                metric = torch.tensor((1., -1., -1., -1.), dtype=torch.float64, device=device)
                terms[term] = 0.5 * p.hbar_c * torch.sum(weights * (source * field * metric).sum(-1))

        # Vertex matrices in Hamiltonian representation.
        scalar_op = g0
        vector_ops = [g0 @ gamma[mu] for mu in range(4)]
        tensor_ops = [[g0 @ (0.5j * (gamma[axis] @ gamma[mu]
                                     - gamma[mu] @ gamma[axis]))
                       for axis in range(1, 4)] for mu in range(4)]
        pion_ops = [g0 @ gamma5 @ gamma[k] for k in range(1, 4)]
        metric = (1., -1., -1., -1.)
        tensor_vtx = vtx["grtn"]
        pion_vtx = vtx["fpio"] * p.hbar_c / p.m_pi
        y0 = torch.arange(weights.numel(), device=device)
        yweights = weights
        wxy_full = None

        # Pair-tiled exchange and derivative-direct terms. Kernel derivatives
        # are raw projected endpoints (including their discrete coincidence
        # values); contact compensation below remains a separate local term.
        from torch.utils.checkpoint import checkpoint

        def fock_tile(rx_n, rx_p, ry_n, ry_p, c_left, c_right, kernels,
                      left, right, opx, opy, coefficient):
            def contract(rxn, rxp, ryn, ryp, cl, cr, kernel):
                ryx_l = ryn if left == "n" else ryp
                rxy_r = rxn if right == "n" else rxp
                return coefficient * torch.sum(
                    kernel * cl[:, None] * cr[None, :]
                    * _endpoint_trace(rxy_r, ryx_l, opx, opy)
                ).real
            args = (rx_n, rx_p, ry_n, ry_p, c_left, c_right, kernels)
            if self.checkpoint_fock_blocks and any(x.requires_grad for x in args):
                return checkpoint(contract, *args, use_reentrant=False)
            return contract(*args)

        for x0 in range(pair_start, pair_stop, self.pair_chunk):
            x1 = min(pair_stop, x0 + self.pair_chunk)
            wx = weights[x0:x1]
            # one x-tile of R(x,y) and the transposed endpoint R(y,x)
            rx = {s: rhos[s][x0:x1] for s in ("n", "p")}
            # R(y,x) with output axes [y,spin_y,x_tile,spin_x]. This is a
            # slice of the unpermuted R tensor; permuting would instead keep
            # R(x,y) and silently conjugate the wrong Wick endpoint.
            ry = {s: rhos[s][:, :, x0:x1, :] for s in ("n", "p")}
            kernel_sigma = green.kernel_block(x0, x1, 0, weights.numel(), p.m_sigma)
            kernel_omega = green.kernel_block(x0, x1, 0, weights.numel(), p.m_omega)
            kernel_rho = green.kernel_block(x0, x1, 0, weights.numel(), p.m_rho)
            kernel_coul = green.kernel_block(x0, x1, 0, weights.numel(), 0.0)
            ktt_rho = green.kernel_block(x0, x1, 0, weights.numel(), p.m_rho,
                                         derivative_x=True, derivative_y=True)
            kgy_rho = green.kernel_block(x0, x1, 0, weights.numel(), p.m_rho,
                                         derivative_y=True)
            kgx_rho = green.kernel_block(x0, x1, 0, weights.numel(), p.m_rho,
                                         derivative_x=True)
            ktt_pi = green.kernel_block(x0, x1, 0, weights.numel(), p.m_pi,
                                        derivative_x=True, derivative_y=True)
            for left in ("n", "p"):
                for right in ("n", "p"):
                    iso = (1. if left == right else 2.)
                    if left == right:
                        pass
                    # Isoscalar σ/ω exchange only within species; photon pp.
                    if left == right:
                        cs = vtx["gsig"]
                        co = vtx["gome"]
                        terms["sigma_fock"] = terms["sigma_fock"] + fock_tile(
                            rx["n"], rx["p"], ry["n"], ry["p"], cs[x0:x1], cs,
                            kernel_sigma, left, right, scalar_op, scalar_op,
                            0.5 * p.hbar_c,
                        )
                        for mu, sign in enumerate(metric):
                            terms["omega_fock"] = terms["omega_fock"] - fock_tile(
                                rx["n"], rx["p"], ry["n"], ry["p"], co[x0:x1], co,
                                kernel_omega, left, right, vector_ops[mu], vector_ops[mu],
                                0.5 * p.hbar_c * sign,
                            )
                    if left == "p" and right == "p":
                        cp = torch.full_like(rho_b, photon_v)
                        for mu, sign in enumerate(metric):
                            terms["coulomb_fock"] = terms["coulomb_fock"] - fock_tile(
                                rx["n"], rx["p"], ry["n"], ry["p"], cp[x0:x1], cp,
                                kernel_coul, left, right, vector_ops[mu], vector_ops[mu],
                                0.5 * p.hbar_c * sign,
                            )
                    # rho-vector, rho-T, ordered VT/TV, and pion charge-exchange
                    # factors use the tau_1 dot tau_2 Wick coefficients.
                    cr = vtx["grho"]
                    ct = tensor_vtx
                    cpv = pion_vtx
                    for mu, sign in enumerate(metric):
                        terms["rho_vector_fock"] = terms["rho_vector_fock"] - fock_tile(
                            rx["n"], rx["p"], ry["n"], ry["p"], cr[x0:x1], cr,
                            kernel_rho, left, right, vector_ops[mu], vector_ops[mu],
                            0.5 * p.hbar_c * iso * sign,
                        )
                        for ax in range(3):
                            for ay in range(3):
                                # Select each derivative pair after obtaining
                                # raw Cartesian endpoint blocks.
                                kval = ktt_rho[:, :, ax, ay]
                                terms["rho_tensor_fock"] = terms["rho_tensor_fock"] + fock_tile(
                                    rx["n"], rx["p"], ry["n"], ry["p"], ct[x0:x1], ct,
                                    kval, left, right, tensor_ops[mu][ax], tensor_ops[mu][ay],
                                    -0.5 * p.hbar_c * iso * sign,
                                )
                        for axis in range(3):
                            ky = kgy_rho[:, :, axis]
                            kx = kgx_rho[:, :, axis]
                            terms["rho_vt_fock"] = terms["rho_vt_fock"] + fock_tile(
                                rx["n"], rx["p"], ry["n"], ry["p"], cr[x0:x1], ct,
                                ky, left, right, vector_ops[mu], tensor_ops[mu][axis],
                                -0.5 * p.hbar_c * iso * sign,
                            )
                            terms["rho_tv_fock"] = terms["rho_tv_fock"] + fock_tile(
                                rx["n"], rx["p"], ry["n"], ry["p"], ct[x0:x1], cr,
                                kx, left, right, tensor_ops[mu][axis], vector_ops[mu],
                                -0.5 * p.hbar_c * iso * sign,
                            )
                    for ax in range(3):
                        for ay in range(3):
                            terms["pion_pv_fock"] = terms["pion_pv_fock"] + fock_tile(
                                rx["n"], rx["p"], ry["n"], ry["p"], cpv[x0:x1], cpv,
                                ktt_pi[:, :, ax, ay], left, right, pion_ops[ax], pion_ops[ay],
                                0.5 * p.hbar_c * iso,
                            )

            # Direct rho-T, mixed VT/TV and pion Hartree contractions.
            rt_kernel = ktt_rho
            pi_kernel = ktt_pi
            gx, gy = kgx_rho, kgy_rho
            jx, tx_tile, ax_tile = j_iso[x0:x1], t_iso[x0:x1], axial_iso[x0:x1]
            for mu, sign in enumerate(metric):
                for ai in range(3):
                    for aj in range(3):
                        terms["rho_tensor_direct"] = terms["rho_tensor_direct"] + (
                            0.5 * p.hbar_c * sign * torch.sum(
                                wx[:, None] * yweights[None, :]
                                * ct[x0:x1, None] * ct[None, :]
                                * tx_tile[:, mu, ai, None] * rt_kernel[:, :, ai, aj]
                                * t_iso[None, :, mu, aj]
                            ))
                        if mu == 0:
                            terms["pion_pv_direct"] = terms["pion_pv_direct"] - (
                                0.5 * p.hbar_c * torch.sum(
                                    wx[:, None] * yweights[None, :]
                                    * cpv[x0:x1, None] * cpv[None, :]
                                    * ax_tile[:, ai, None] * pi_kernel[:, :, ai, aj]
                                    * axial_iso[None, :, aj]
                                ))
                for axis in range(3):
                    terms["rho_vt_direct"] = terms["rho_vt_direct"] + (
                        0.5 * p.hbar_c * sign * torch.sum(
                            wx[:, None] * yweights[None, :]
                            * cr[x0:x1, None] * ct[None, :]
                            * jx[:, mu, None] * gy[:, :, axis]
                            * t_iso[None, :, mu, axis]
                        ))
                    terms["rho_tv_direct"] = terms["rho_tv_direct"] + (
                        0.5 * p.hbar_c * sign * torch.sum(
                            wx[:, None] * yweights[None, :]
                            * ct[x0:x1, None] * cr[None, :]
                            * tx_tile[:, mu, axis, None] * gx[:, :, axis]
                            * j_iso[None, :, mu]
                        ))

        # Independent RHF-e contact compensation, evaluated from the local
        # coordinate density.  The vertices remain density dependent.
        dcontact = dloc
        if include_local:
            for kind, vertex, ops, signs, term_name in (
                ("rt", tensor_vtx, [op for mu in range(4) for op in tensor_ops[mu]],
                 [-sign for sign in metric for _ in range(3)], "rho_tensor_contact"),
                ("pi", pion_vtx, pion_ops, [1.] * 3, "pion_pv_contact"),
            ):
                direct_density = weights.new_zeros(weights.shape)
                exchange_density = weights.new_zeros(weights.shape)
                for op, sign in zip(ops, signs):
                    currents = {s: _spin_trace(dcontact[s], op) for s in ("n", "p")}
                    for s in ("n", "p"):
                        t3s = 1. if s == "n" else -1.
                        for t in ("n", "p"):
                            t3t = 1. if t == "n" else -1.
                            iso = 1. if s == t else 2.
                            direct_density = direct_density + (sign / 6.) * t3s*t3t*currents[s]*currents[t]
                            exchange_density = exchange_density - (sign / 6.) * iso * torch.einsum(
                                "pab,bc,pcd,da->p", dcontact[s], op, dcontact[t], op
                            ).real
                terms[term_name] = p.hbar_c * torch.sum(weights * vertex.square()
                                                         * (direct_density + exchange_density))

        total = sum(terms.values(), weights.new_zeros(()))
        return total, terms

    def energy_only(self, states: dict):
        """Evaluate the coordinate-kernel scalar without constructing fields."""
        device = states["n"].q.device
        c, weights = self._weighted_basis(device)
        rhos = {s: self._coordinate_density(states[s], c) for s in ("n", "p")}
        green = GreenOperator(self.basis, lmax=self.lmax, device=device)
        interaction, terms = self._energy_from_coordinate_density(rhos, weights, green)
        eye = torch.eye(self.basis.dimension, dtype=torch.complex128, device=device)
        kinetic = weights.new_zeros(())
        for s in ("n", "p"):
            kcols = self.basis.apply_kinetic(eye, species=s)
            rho_prim = states[s].q @ states[s].rho_active @ states[s].q.mH
            kinetic = kinetic + torch.trace(rho_prim @ kcols).real
        terms["kinetic"] = kinetic
        terms["d1s_pairing"] = weights.new_zeros(())
        terms["density_rearrangement"] = weights.new_zeros(())
        return interaction + kinetic, terms

    def evaluate(self, states: dict, *, create_graph: bool = False):
        if set(states) != {"n", "p"}:
            raise ValueError("coordinate normal fields require n and p states")
        device = states["n"].q.device
        c, weights = self._weighted_basis(device)
        # Detach directly into the leaf so the O(P²) coordinate density is
        # stored once per species, rather than keeping a second cloned copy.
        rleaf = {s: self._coordinate_density(states[s], c).detach().requires_grad_(True)
                 for s in ("n", "p")}
        green = GreenOperator(self.basis, lmax=self.lmax, device=device)
        with torch.enable_grad():
            interaction_energy, terms = self._energy_from_coordinate_density(rleaf, weights, green)
            # kinetic energy lives naturally in the declared primitive basis.
            identity = torch.eye(self.basis.dimension, dtype=torch.complex128, device=device)
            kinetic = {}
            for s in ("n", "p"):
                kcols = self.basis.apply_kinetic(identity, species=s)
                rho_prim = states[s].q @ states[s].rho_active @ states[s].q.mH
                kinetic[s] = torch.trace(rho_prim @ kcols).real
            total = interaction_energy + kinetic["n"] + kinetic["p"]
            terms["kinetic"] = kinetic["n"] + kinetic["p"]
            grads = torch.autograd.grad(interaction_energy, (rleaf["n"], rleaf["p"]),
                                        create_graph=create_graph)
        cflat = c.reshape(-1, self.basis.dimension)
        h = {}
        for s, grad in zip(("n", "p"), grads):
            gmat = grad.reshape(4*weights.numel(), 4*weights.numel())
            gmat = 0.5 * (gmat + gmat.mH)
            h[s] = cflat.mH @ gmat @ cflat
            h[s] = h[s] + self.basis.apply_kinetic(
                torch.eye(self.basis.dimension, dtype=torch.complex128, device=device), species=s
            )
            if not create_graph:
                h[s] = h[s].detach()
        total_terms = dict(terms)
        total_terms["d1s_pairing"] = weights.new_zeros(())
        total_terms["density_rearrangement"] = weights.new_zeros(())
        energy_eval = RHFBFunctionalEvaluation(
            total, total_terms,
            {"coordinate_kernel": True, "complete_particle_hole": True,
             "included_energy_terms": tuple(k for k in total_terms if k not in ("d1s_pairing", "density_rearrangement")),
             "rearrangement_status": "implicit via autograd through invariant rho_B and all DD vertices"},
            lambda *_a, **_kw: {},
        )
        return CoordinateNormalEvaluation(
            energy_eval, {s: states[s].q @ states[s].rho_active @ states[s].q.mH for s in ("n", "p")}, h,
            {"representation": "weighted coordinate nonlocal density R_xy",
             "coordinate_R_four_axis_materialized": True,
             "interaction_four_tensor_materialized": False,
             "energy_forward_count": 1,
             "joint_species_gradient_count": 1,
             "density_factors_detached": True,
             "field_gradient_back_to_nn_parameters": False,
             "field_scope": "frozen-density derivative; rebuild after any rho/Q change",
             "basis_dimension": self.basis.dimension,
             "point_count": self.basis.point_count,
             "pair_tile": self.pair_chunk,
             "lmax": green.lmax,
             "derivative_order": self.basis.derivative_order,
             "ad_points": 5 if self.basis.derivative_order in (4, 5) else 7,
             "hermiticity_residual_max": {s: float((h[s]-h[s].mH).abs().max()) for s in h}},
        )
