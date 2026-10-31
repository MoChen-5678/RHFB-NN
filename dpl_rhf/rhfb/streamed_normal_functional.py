"""Bounded-memory primitive-density PKA1 particle-hole fields.

This backend evaluates the same coordinate PKA1 contractions as
``CoordinateNormalFunctional`` while constructing only one x-by-all-y density
tile at a time.  Its derivative is a frozen-density first derivative in the
full primitive space; it does not build an end-to-end neural or second-order
graph.
"""
from __future__ import annotations

from collections.abc import Iterator, Callable
import time

import torch
from torch.utils.checkpoint import checkpoint

from dpl_rhf.rhf.parameters import pka1_parameters
from .coordinate_normal_functional import CoordinateNormalEvaluation
from .currents import _gamma_matrices, invariant_baryon_density
from .execution_limits import require_bounded_cgroup
from .functional import (RHFBFunctionalEvaluation, _couplings)
from .runtime_resources import resource_snapshot
from .spatial import GreenOperator


def _spin_trace(density: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
    return torch.einsum("pab,ba->p", density, operator).real


def _endpoint_trace(rxy: torch.Tensor, ryx: torch.Tensor,
                    opx: torch.Tensor, opy: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ybxa,ac,xcyd,db->xy", ryx, opx, rxy, opy)


class StreamedPrimitiveNormalFunctional:
    """Stream complete PKA1 PH energy and full primitive normal fields.

    ``tile_size`` bounds the x dimension of ``R(x_tile,y)``.  The complete
    primitive density is O(D²), local fields are O(P), and the largest
    coordinate-density object is O(tile_size * P), independent of any active
    rank.  The caller's full physical Q/rho is contracted exactly to the
    primitive density before any coordinate work.
    """

    def __init__(self, basis, *, tile_size: int = 8, chunk_size: int = 64,
                 params=None, lmax: int | None = None, progress_callback=None):
        if tile_size < 1 or chunk_size < 1:
            raise ValueError("tile_size and chunk_size must be positive")
        self.basis = basis
        self.tile_size = int(tile_size)
        self.chunk_size = int(chunk_size)
        self.params = pka1_parameters() if params is None else params
        self.lmax = lmax
        self._fixed_samples = {}
        self.last_energy_terms = None
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable or None")
        self.progress_callback = progress_callback

    def primitive_densities(self, states):
        if set(states) != {"n", "p"}:
            raise ValueError("streamed PH fields require neutron and proton states")
        leaves = {}
        for species in ("n", "p"):
            state = states[species]
            q, occ = state.q, state.rho_active
            if (q.ndim != 2 or q.shape[0] != self.basis.dimension
                    or q.dtype != torch.complex128 or occ.dtype != torch.complex128
                    or occ.shape != (q.shape[1], q.shape[1]) or q.device != occ.device):
                raise ValueError(f"invalid {species} density factors")
            primitive = q @ occ @ q.mH
            if not torch.isfinite(primitive).all():
                raise ValueError(f"non-finite {species} primitive density")
            if not torch.allclose(primitive, primitive.mH, atol=2e-10, rtol=0):
                raise ValueError(f"{species} primitive density must be Hermitian")
            leaves[species] = primitive
        return leaves

    def _primitive_leaves(self, states):
        return {s: value.detach().requires_grad_(True)
                for s, value in self.primitive_densities(states).items()}

    def differentiable_energy_and_fields(self, states):
        """Return PH scalar and fields with block-recomputed NN loss gradients.

        Forward and backward use the same complete PH scalar. A loss depending
        on h includes the field's density Hessian-vector product, including DD
        rearrangement. No block graph is retained across the full coordinate
        traversal. This interface supports first loss gradients, not third
        derivatives of E, pairing, or a no-sea branch selection.
        """
        require_bounded_cgroup()
        from .streamed_field_autograd import streamed_energy_and_fields
        rho = self.primitive_densities(states)
        self.last_energy_terms = None
        terms = {}
        factory_calls = 0

        def reporting_blocks(primitive_rhos):
            # The custom Function invokes the same factory again in backward.
            # Collect detached scalars only on its first (forward) traversal;
            # this adds no retained energy graphs or extra field evaluations.
            nonlocal factory_calls
            collect = factory_calls == 0
            factory_calls += 1
            traversal = factory_calls
            began = time.monotonic()
            block_count = 1 + (self.basis.point_count+self.tile_size-1)//self.tile_size
            for index, block in enumerate(self.iter_energy_blocks(primitive_rhos)):
                def reporting_block(n, p, block=block, collect=collect):
                    values = block(n, p)
                    if collect:
                        for key, value in values.items():
                            detached = value.detach()
                            terms[key] = terms.get(key, torch.zeros_like(detached)) + detached
                    return values
                yield reporting_block
                # The consumer resumes this generator only after finishing the
                # block gradient/HVP, so this reports completed blocks rather
                # than just a submitted energy contraction.
                if self.progress_callback is not None:
                    self.progress_callback({
                        "stage": "PH_forward" if collect else "PH_backward",
                        "traversal": traversal, "completed_blocks": index+1,
                        "block_count": block_count,
                        "elapsed_seconds": time.monotonic()-began,
                        "resource_snapshot": resource_snapshot(),
                    })

        energy, hn, hp = streamed_energy_and_fields(
            rho['n'], rho['p'], reporting_blocks)
        self.last_energy_terms = dict(terms)
        return energy, {'n': hn, 'p': hp}

    def _sample(self, start, stop, device):
        # Fixed primitive samples are O(P*D), independent of rho/NN parameters.
        # Cache once per device, instead of sampling all y again for every tile.
        # This is a fixed-grid cache, not a Hamiltonian-dependent subspace.
        key = str(torch.device(device))
        if key not in self._fixed_samples:
            eye = torch.eye(self.basis.dimension, dtype=torch.complex128, device=device)
            physical = self.basis._sample_range(eye, 0, self.basis.point_count)
            weights = torch.as_tensor(self.basis.weights, dtype=torch.float64, device=device)
            self._fixed_samples[key] = (physical*torch.sqrt(weights)[:, None, None]).detach()
        return self._fixed_samples[key][start:stop]

    def _local_fields(self, rho, weights, gamma, gamma5):
        """Pointwise bilinears via checkpointed sample chunks (O(P) outputs)."""
        pcount = self.basis.point_count
        pieces = {s: [] for s in ("n", "p")}
        ops_tensor = [[0.5j * (gamma[a] @ gamma[m] - gamma[m] @ gamma[a])
                       for a in range(1, 4)] for m in range(4)]

        def one_species_block(leaf, start, stop):
            c = self._sample(start, stop, leaf.device)
            d = torch.einsum("pai,ij,pbj->pab", c, leaf, c.conj())
            d = d / weights[start:stop, None, None]
            scalar = _spin_trace(d, gamma[0])
            current = torch.stack([_spin_trace(d, gamma[0] @ gm) for gm in gamma], -1)
            axial = torch.stack([_spin_trace(d, gamma[0] @ gamma5 @ gamma[k])
                                 for k in range(1, 4)], -1)
            tensor = torch.stack([
                torch.stack([_spin_trace(d, gamma[0] @ ops_tensor[m][a])
                             for a in range(3)], -1) for m in range(4)
            ], 1)
            return scalar, current, axial, tensor

        for start in range(0, pcount, self.chunk_size):
            stop = min(pcount, start + self.chunk_size)
            for s in ("n", "p"):
                # Density contractions are checkpointed; primitive samples
                # are shared fixed constants of size O(P*D).
                fn = lambda leaf, a=start, b=stop: one_species_block(leaf, a, b)
                values = checkpoint(fn, rho[s], use_reentrant=False,
                                    preserve_rng_state=False)
                pieces[s].append(values)
        local = {}
        for s in ("n", "p"):
            local[s] = {name: torch.cat([entry[i] for entry in pieces[s]], dim=0)
                        for i, name in enumerate(("scalar", "j", "axial", "tensor"))}
        return local

    def _local_context(self, rho, weights, gamma, gamma5):
        local = self._local_fields(rho, weights, gamma, gamma5)
        total_j = local["n"]["j"] + local["p"]["j"]
        rho_b = invariant_baryon_density(total_j)
        vtx = _couplings(rho_b, self.params)
        tensor_ops = [[gamma[0] @ (0.5j * (gamma[a] @ gamma[m] - gamma[m] @ gamma[a]))
                       for a in range(1, 4)] for m in range(4)]
        pion_ops = [gamma[0] @ gamma5 @ gamma[k] for k in range(1, 4)]
        return local, vtx, rho_b, tensor_ops, pion_ops

    def _local_energy(self, rho, weights, green, gamma, gamma5, tile_count):
        p = self.params
        local, vtx, rho_b, tensor_ops, pion_ops = self._local_context(
            rho, weights, gamma, gamma5)
        total_j = local["n"]["j"] + local["p"]["j"]
        scalar_total = local["n"]["scalar"] + local["p"]["scalar"]
        j_iso = local["n"]["j"] - local["p"]["j"]
        photon_v = (4 * torch.pi / p.alpha_inv) ** 0.5
        source_sigma = vtx["gsig"] * scalar_total
        source_omega = vtx["gome"][:, None] * total_j
        source_rho = vtx["grho"][:, None] * j_iso
        source_coul = photon_v * local["p"]["j"]
        metric = torch.tensor((1., -1., -1., -1.), dtype=torch.float64,
                              device=weights.device)

        def direct_scalar(source, mass, sign):
            field = green.apply(source, mass)
            return sign * 0.5 * p.hbar_c * torch.sum(weights * source * field).real

        def direct_vector(source, mass):
            field = green.apply(source, mass)
            return 0.5 * p.hbar_c * torch.sum(
                weights * (source * field * metric).sum(-1)).real

        terms = {
            "sigma_direct": direct_scalar(source_sigma, p.m_sigma, -1.),
            "omega_direct": direct_vector(source_omega, p.m_omega),
            "rho_vector_direct": direct_vector(source_rho, p.m_rho),
            "coulomb_direct": direct_vector(source_coul, 0.),
        }
        # The two local RHF-e compensation contacts are independent terms.
        dloc = {}
        # Local contact traces need the 4x4 density; recompute these by chunks.
        def density_chunk(leaf, start, stop):
            c = self._sample(start, stop, leaf.device)
            return torch.einsum("pai,ij,pbj->pab", c, leaf, c.conj())

        for start in range(0, self.basis.point_count, self.chunk_size):
            stop = min(self.basis.point_count, start + self.chunk_size)
            for s in ("n", "p"):
                fn = lambda leaf, a=start, b=stop: density_chunk(leaf, a, b)
                d = checkpoint(fn, rho[s], use_reentrant=False,
                               preserve_rng_state=False)
                dloc.setdefault(s, []).append(d / weights[start:stop, None, None])
        dloc = {s: torch.cat(v, 0) for s, v in dloc.items()}
        tensor_vtx = vtx["grtn"]
        pion_vtx = vtx["fpio"] * p.hbar_c / p.m_pi
        for kind, vertex, ops, signs, name in (
            ("rt", tensor_vtx, [o for row in tensor_ops for o in row],
             [-v for v in (1., -1., -1., -1.) for _ in range(3)], "rho_tensor_contact"),
            ("pi", pion_vtx, pion_ops, [1.] * 3, "pion_pv_contact"),
        ):
            direct = weights.new_zeros(weights.shape)
            exchange = weights.new_zeros(weights.shape)
            for op, sign in zip(ops, signs):
                currents = {s: _spin_trace(dloc[s], op) for s in ("n", "p")}
                for s in ("n", "p"):
                    t3s = 1. if s == "n" else -1.
                    for t in ("n", "p"):
                        t3t = 1. if t == "n" else -1.
                        iso = 1. if s == t else 2.
                        direct = direct + sign / 6. * t3s * t3t * currents[s] * currents[t]
                        exchange = exchange - sign / 6. * iso * torch.einsum(
                            "pab,bc,pcd,da->p", dloc[s], op, dloc[t], op).real
            terms[name] = p.hbar_c * torch.sum(weights * vertex.square()
                                               * (direct + exchange))
        return terms, local, vtx, rho_b, tensor_ops, pion_ops

    def _pair_energy(self, rho, weights, start, stop, green, gamma, gamma5,
                     local, vtx, tensor_ops, pion_ops):
        """All nonlocal pair contributions for one x-by-all-y block."""
        p = self.params
        device = weights.device
        total_j = local["n"]["j"] + local["p"]["j"]
        j_iso = local["n"]["j"] - local["p"]["j"]
        t_iso = local["n"]["tensor"] - local["p"]["tensor"]
        axial_iso = local["n"]["axial"] - local["p"]["axial"]
        metric = (1., -1., -1., -1.)
        xweight = weights[start:stop]
        yweight = weights
        cx = self._sample(start, stop, device)
        cy = self._sample(0, self.basis.point_count, device)
        rxy = {s: torch.einsum("xai,ij,ybj->xayb", cx, rho[s], cy.conj())
               for s in ("n", "p")}
        # Every local Hermitian primitive density gives R(y,x)=R(x,y)^†.
        ryx = {s: rxy[s].conj().permute(2, 3, 0, 1) for s in ("n", "p")}
        # Reuse the coordinate functional's channel matrices and coefficients.
        scalar_op = gamma[0]
        vector_ops = [gamma[0] @ gamma[mu] for mu in range(4)]
        tensor_vops = [[gamma[0] @ (0.5j * (gamma[a] @ gamma[mu]
                                               - gamma[mu] @ gamma[a]))
                        for a in range(1, 4)] for mu in range(4)]
        photon = (4 * torch.pi / p.alpha_inv) ** 0.5
        cr, ct = vtx["grho"], vtx["grtn"]
        cpv = vtx["fpio"] * p.hbar_c / p.m_pi
        kernels = {
            "sigma": green.kernel_block(start, stop, 0, self.basis.point_count, p.m_sigma),
            "omega": green.kernel_block(start, stop, 0, self.basis.point_count, p.m_omega),
            "rho": green.kernel_block(start, stop, 0, self.basis.point_count, p.m_rho),
            "coul": green.kernel_block(start, stop, 0, self.basis.point_count, 0.),
            "ktt_rho": green.kernel_block(start, stop, 0, self.basis.point_count,
                                             p.m_rho, derivative_x=True, derivative_y=True),
            "kgy_rho": green.kernel_block(start, stop, 0, self.basis.point_count,
                                             p.m_rho, derivative_y=True),
            "kgx_rho": green.kernel_block(start, stop, 0, self.basis.point_count,
                                             p.m_rho, derivative_x=True),
            "ktt_pi": green.kernel_block(start, stop, 0, self.basis.point_count,
                                           p.m_pi, derivative_x=True, derivative_y=True),
        }
        result = {k: weights.new_zeros(()) for k in (
            "sigma_fock", "omega_fock", "rho_vector_fock", "rho_tensor_fock",
            "rho_vt_fock", "rho_tv_fock", "pion_pv_fock", "coulomb_fock",
            "rho_tensor_direct", "rho_vt_direct", "rho_tv_direct", "pion_pv_direct")}

        def fock(left, right, opx, opy, coeffx, coeffy, kernel, coefficient):
            return coefficient * torch.sum(
                kernel * coeffx[start:stop, None] * coeffy[None, :]
                * _endpoint_trace(rxy[right], ryx[left], opx, opy)).real

        for left in ("n", "p"):
            for right in ("n", "p"):
                iso = 1. if left == right else 2.
                if left == right:
                    result["sigma_fock"] += fock(left, right, scalar_op, scalar_op,
                        vtx["gsig"], vtx["gsig"], kernels["sigma"], .5*p.hbar_c)
                    for mu, sign in enumerate(metric):
                        result["omega_fock"] -= fock(left, right, vector_ops[mu], vector_ops[mu],
                            vtx["gome"], vtx["gome"], kernels["omega"], .5*p.hbar_c*sign)
                if left == "p" and right == "p":
                    cp = torch.full_like(vtx["grho"], photon)
                    for mu, sign in enumerate(metric):
                        result["coulomb_fock"] -= fock(left, right, vector_ops[mu], vector_ops[mu],
                            cp, cp, kernels["coul"], .5*p.hbar_c*sign)
                for mu, sign in enumerate(metric):
                    result["rho_vector_fock"] -= fock(left, right, vector_ops[mu], vector_ops[mu],
                        cr, cr, kernels["rho"], .5*p.hbar_c*iso*sign)
                    for ax in range(3):
                        for ay in range(3):
                            result["rho_tensor_fock"] += fock(left, right,
                                tensor_vops[mu][ax], tensor_vops[mu][ay], ct, ct,
                                kernels["ktt_rho"][:, :, ax, ay], -.5*p.hbar_c*iso*sign)
                    for axis in range(3):
                        result["rho_vt_fock"] += fock(left, right, vector_ops[mu],
                            tensor_vops[mu][axis], cr, ct,
                            kernels["kgy_rho"][:, :, axis], -.5*p.hbar_c*iso*sign)
                        result["rho_tv_fock"] += fock(left, right, tensor_vops[mu][axis],
                            vector_ops[mu], ct, cr,
                            kernels["kgx_rho"][:, :, axis], -.5*p.hbar_c*iso*sign)
                for ax in range(3):
                    for ay in range(3):
                        result["pion_pv_fock"] += fock(left, right, pion_ops[ax], pion_ops[ay],
                            cpv, cpv, kernels["ktt_pi"][:, :, ax, ay], .5*p.hbar_c*iso)

        wx, wy = xweight[:, None], yweight[None, :]
        tx, ax = t_iso[start:stop], axial_iso[start:stop]
        for mu, sign in enumerate(metric):
            for ai in range(3):
                for aj in range(3):
                    result["rho_tensor_direct"] += .5*p.hbar_c*sign*torch.sum(
                        wx*wy*ct[start:stop,None]*ct[None,:]
                        *tx[:,mu,ai,None]*kernels["ktt_rho"][:,:,ai,aj]
                        *t_iso[None,:,mu,aj])
                    if mu == 0:
                        result["pion_pv_direct"] -= .5*p.hbar_c*torch.sum(
                            wx*wy*cpv[start:stop,None]*cpv[None,:]
                            *ax[:,ai,None]*kernels["ktt_pi"][:,:,ai,aj]
                            *axial_iso[None,:,aj])
            for axis in range(3):
                result["rho_vt_direct"] += .5*p.hbar_c*sign*torch.sum(
                    wx*wy*cr[start:stop,None]*ct[None,:]
                    *j_iso[start:stop,mu,None]*kernels["kgy_rho"][:,:,axis]
                    *t_iso[None,:,mu,axis])
                result["rho_tv_direct"] += .5*p.hbar_c*sign*torch.sum(
                    wx*wy*ct[start:stop,None]*cr[None,:]
                    *tx[:,mu,axis,None]*kernels["kgx_rho"][:,:,axis]
                    *j_iso[None,:,mu])
        return result, tuple(rxy.values())

    def iter_energy_blocks(self, primitive_rhos, *, weights=None,
                           green=None) -> Iterator[Callable[..., dict[str, torch.Tensor]]]:
        """Yield recomputable scalar blocks for first or second derivatives.

        Each yielded callback accepts ``(rho_n, rho_p)`` and returns a
        ``dict[str, scalar_tensor]`` without detaching.  Callers may obtain a
        block gradient with ``create_graph=True``, immediately contract that
        gradient with a cotangent, and release the block before advancing.
        The first block carries Hartree, contact, and kinetic energy; each
        following block carries one nonlocal x tile. Callbacks use the fixed
        O(PD) primitive sample cache, independent of fields and NN parameters;
        no persistent coordinate density is formed.

        This lower-level iterator does not itself impose an AD memory policy.
        Production callers must use the RHFB limited-resource launcher.
        """
        require_bounded_cgroup()
        if set(primitive_rhos) != {"n", "p"}:
            raise ValueError("primitive_rhos must contain n and p")
        for s in ("n", "p"):
            value = primitive_rhos[s]
            if (value.shape != (self.basis.dimension, self.basis.dimension)
                    or value.dtype != torch.complex128):
                raise ValueError(f"invalid {s} primitive density leaf")
        device = primitive_rhos["n"].device
        if primitive_rhos["p"].device != device:
            raise ValueError("primitive densities must share a device")
        if weights is None:
            weights = torch.as_tensor(self.basis.weights, dtype=torch.float64,
                                      device=device)
        if green is None:
            green = GreenOperator(self.basis, lmax=self.lmax, device=device)
        gamma, gamma5 = _gamma_matrices(device=device, dtype=torch.complex128)
        tile_count = (self.basis.point_count + self.tile_size - 1)//self.tile_size
        eye = torch.eye(self.basis.dimension, dtype=torch.complex128, device=device)

        def local_block(rn, rp):
            rho = {"n": rn, "p": rp}
            values, *_rest = self._local_energy(rho, weights, green, gamma, gamma5,
                                                 tile_count)
            values = dict(values)
            values["kinetic"] = sum((torch.trace(
                rho[s] @ self.basis.apply_kinetic(eye, species=s)).real
                for s in ("n", "p")), weights.new_zeros(()))
            return values

        yield local_block
        for start in range(0, self.basis.point_count, self.tile_size):
            stop = min(self.basis.point_count, start+self.tile_size)

            def pair_block(rn, rp, a=start, b=stop):
                rho = {"n": rn, "p": rp}
                local, vtx, _rho_b, tops, pops = self._local_context(
                    rho, weights, gamma, gamma5)
                values, _ = self._pair_energy(rho, weights, a, b, green, gamma,
                                               gamma5, local, vtx, tops, pops)
                return values

            yield pair_block

    def evaluate(self, states: dict, *, create_graph: bool = False,
                 progress: Callable[[dict], None] | None = None):
        """Return full primitive fields; requires the enforced RHFB cgroup."""
        if create_graph:
            raise NotImplementedError(
                "streamed primitive fields support first derivatives only; create_graph=True is unsupported")
        # Check before any model-sized allocations.  The final snapshot below
        # records peak usage after the full streamed contraction has run.
        require_bounded_cgroup()
        rho = self._primitive_leaves(states)
        device = next(iter(rho.values())).device
        weights = torch.as_tensor(self.basis.weights, dtype=torch.float64, device=device)
        green = GreenOperator(self.basis, lmax=self.lmax, device=device)
        gamma, gamma5 = _gamma_matrices(device=device, dtype=torch.complex128)
        eye = torch.eye(self.basis.dimension, dtype=torch.complex128, device=device)
        terms = {k: weights.new_zeros(()) for k in (
            "kinetic", "d1s_pairing", "sigma_direct", "sigma_fock", "omega_direct",
            "omega_fock", "rho_vector_direct", "rho_vector_fock", "rho_tensor_direct",
            "rho_tensor_fock", "rho_vt_direct", "rho_vt_fock", "rho_tv_direct",
            "rho_tv_fock", "pion_pv_direct", "pion_pv_fock", "coulomb_direct",
            "coulomb_fock", "rho_tensor_contact", "pion_pv_contact",
            "density_rearrangement")}
        total_grad = {s: torch.zeros_like(rho[s]) for s in ("n", "p")}

        # Direct Hartree, DD rearrangement, local contacts and kinetic field.
        # This block is differentiated once and then released.
        with torch.enable_grad():
            direct, local, vtx, rho_b, tops, pops = self._local_energy(
                rho, weights, green, gamma, gamma5,
                (self.basis.point_count + self.tile_size - 1)//self.tile_size)
            for key, value in direct.items():
                terms[key] += value.detach()
            for s in ("n", "p"):
                terms["kinetic"] += torch.trace(rho[s] @ self.basis.apply_kinetic(
                    eye, species=s)).real.detach()
            local_scalar = sum(direct.values(), weights.new_zeros(()))
            kinetic_scalar = sum((torch.trace(rho[s] @ self.basis.apply_kinetic(
                eye, species=s)).real for s in ("n", "p")), weights.new_zeros(()))
            grads = torch.autograd.grad(local_scalar + kinetic_scalar,
                                        (rho["n"], rho["p"]))
            for s, g in zip(("n", "p"), grads):
                total_grad[s] += g.detach()
        if progress is not None:
            progress({
                "completed_x_points": 0,
                "point_count": self.basis.point_count,
                "energy_complete": False,
                "energy_terms_mev": {k: float(v.detach()) for k, v in terms.items()},
                "resource_snapshot": resource_snapshot(),
            })

        max_tile_shape = (0, 0, 4, 4)
        for start in range(0, self.basis.point_count, self.tile_size):
            stop = min(self.basis.point_count, start + self.tile_size)
            # Rebuild local/DD graph per tile; each graph is consumed immediately.
            with torch.enable_grad():
                local, vtx, rho_b, tops, pops = self._local_context(
                    rho, weights, gamma, gamma5)
                local_names = ("scalar", "j", "axial", "tensor")
                vertex_names = ("gsig", "gome", "grho", "fpio", "grtn")
                checkpoint_inputs = tuple(local[s][name] for s in ("n", "p")
                                          for name in local_names) + tuple(
                    vtx[name] for name in vertex_names)

                def pair_callback(rn, rp, *aux):
                    rebuilt_local = {"n": {}, "p": {}}
                    offset = 0
                    for species in ("n", "p"):
                        for name in local_names:
                            rebuilt_local[species][name] = aux[offset]
                            offset += 1
                    rebuilt_vtx = {name: aux[offset+i]
                                   for i, name in enumerate(vertex_names)}
                    pair_terms, _tiles = self._pair_energy(
                        {"n": rn, "p": rp}, weights, start, stop, green,
                        gamma, gamma5, rebuilt_local, rebuilt_vtx, tops, pops)
                    return tuple(pair_terms.values())

                pair_values = checkpoint(pair_callback, rho["n"], rho["p"],
                                         *checkpoint_inputs, use_reentrant=False,
                                         preserve_rng_state=False)
                if isinstance(pair_values, torch.Tensor):
                    pair_values = (pair_values,)
                pair = dict(zip(("sigma_fock", "omega_fock", "rho_vector_fock",
                                 "rho_tensor_fock", "rho_vt_fock", "rho_tv_fock",
                                 "pion_pv_fock", "coulomb_fock", "rho_tensor_direct",
                                 "rho_vt_direct", "rho_tv_direct", "pion_pv_direct"),
                                pair_values))
                # Local energy was already counted above; this tile contributes
                # only the nonlocal x/y contractions.
                pair_total = sum(pair.values(), weights.new_zeros(()))
                pg = torch.autograd.grad(pair_total, (rho["n"], rho["p"]))
                for key, value in pair.items():
                    terms[key] += value.detach()
                for s, g in zip(("n", "p"), pg):
                    total_grad[s] += g.detach()
                max_tile_shape = max(max_tile_shape,
                                     (stop-start, 4, self.basis.point_count, 4))
            if progress is not None:
                progress({
                    "completed_x_points": stop,
                    "point_count": self.basis.point_count,
                    "energy_complete": stop == self.basis.point_count,
                    "energy_terms_mev": {k: float(v.detach()) for k, v in terms.items()},
                    "resource_snapshot": resource_snapshot(),
                })

        h = {s: (0.5*(g+g.mH)).detach() for s, g in total_grad.items()}
        total = sum((terms[k] for k in terms if k not in ("density_rearrangement",)),
                    weights.new_zeros(()))
        terms["density_rearrangement"] = weights.new_zeros(())
        energy = RHFBFunctionalEvaluation(
            total, terms,
            {"coordinate_kernel": True, "complete_particle_hole": True,
             "streamed_primitive_density": True,
             "included_energy_terms": tuple(k for k in terms if k not in (
                 "d1s_pairing", "density_rearrangement")),
             "rearrangement_status": "implicit through each tile's invariant sqrt(j_mu j^mu) and PKA1 vertices"},
            lambda *_a, **_kw: {},
        )
        rho_out = {s: rho[s].detach() for s in ("n", "p")}
        return CoordinateNormalEvaluation(energy, rho_out, h, {
            "representation": "primitive rho leaves; streamed R(x_tile,y) and R(y,x_tile)",
            "coordinate_R_four_axis_materialized": False,
            "interaction_four_tensor_materialized": False,
            "max_coordinate_density_tile_shape": max_tile_shape,
            "max_coordinate_density_tile_bytes_per_species": int(
                max_tile_shape[0]*max_tile_shape[1]*max_tile_shape[2]*max_tile_shape[3]*16),
            "pair_tile": self.tile_size,
            "fixed_primitive_sample_cache_bytes": sum(
                value.numel()*value.element_size() for value in self._fixed_samples.values()),
            "chunk_size": self.chunk_size,
            "basis_dimension": self.basis.dimension,
            "point_count": self.basis.point_count,
            "full_physics": "PKA1 odd-A n,p; kinetic, sigma, omega, rho-vector, rho-tensor, VT/TV, pion-PV, Coulomb, both contacts, DD rearrangement",
            "density_factors_detached": True,
            "field_scope": "frozen-density full primitive first derivative",
            "create_graph_supported": False,
            "resource_snapshot": resource_snapshot(),
            "hermiticity_residual_max": {s: float((h[s]-h[s].mH).abs().max()) for s in h},
        })
