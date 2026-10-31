"""Memory-bounded Fock exchange: per-kappa-pair streaming over one or more devices.

The arithmetic of every channel is that of ``SphericalRHFFunctional`` (the same
kernels, vertices, signs and isospin factors); only the evaluation schedule
changes:

* each (block a, block b) contribution is computed inside its own activation
  checkpoint, so the autograd graph keeps only the pair inputs and a radial
  profile, and the pair is recomputed during backward;
* the distinct (kappa_a, kappa_b) kernel pairs are assigned round-robin to the
  given devices; each device builds and caches only its own kernels, and the
  radial profiles are summed on the primary device.

Density matrices and vertices are copied to each device once per channel call
(a few MB); the copies are differentiable, so gradients return to the primary
device.  Sums are reordered relative to the single-device code, which changes
results only at round-off level.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from dpl_rhf.rhf.kernels import PKA1AngularKernels, PKO2AngularKernels, SphericalPropagators


class StreamedFockChannels:
    def __init__(self, functional, devices):
        if not devices:
            raise ValueError("at least one device is required for streamed Fock exchange")
        self.functional = functional
        self.primary = functional.r.device
        self.devices = [torch.device(d) for d in devices]
        self._angular = {}
        self._assignment: dict[tuple[int, int], torch.device] = {}

    # ------------------------------------------------------------ placement
    def angular(self, device: torch.device):
        key = str(device)
        if key not in self._angular:
            if device == self.primary:
                self._angular[key] = self.functional.angular
            else:
                f = self.functional
                propagators = SphericalPropagators(f.r.to(device), f.params.hbar_c)
                kind = PKA1AngularKernels if isinstance(f.angular, PKA1AngularKernels) else PKO2AngularKernels
                self._angular[key] = kind(propagators)
        return self._angular[key]

    def device_for(self, kappa_a: int, kappa_b: int) -> torch.device:
        key = (int(kappa_a), int(kappa_b))
        if key not in self._assignment:
            self._assignment[key] = self.devices[len(self._assignment) % len(self.devices)]
        return self._assignment[key]

    def _run(self, function, device, *tensors):
        moved = tuple(t if t.device == device else t.to(device) for t in tensors)
        if torch.is_grad_enabled():
            out = checkpoint(function, *moved, use_reentrant=False, preserve_rng_state=False)
        else:
            out = function(*moved)
        if isinstance(out, tuple):
            return tuple(o.to(self.primary) for o in out)
        return out.to(self.primary)

    @staticmethod
    def _per_device(tensor: torch.Tensor, cache: dict, device: torch.device) -> torch.Tensor:
        key = (id(tensor), str(device))
        if key not in cache:
            cache[key] = tensor if tensor.device == device else tensor.to(device)
        return cache[key]

    # ------------------------------------------------------------- channels
    def scalar_exchange(self, blocks, coupling):
        f = self.functional
        profile = torch.zeros_like(f.exchange_outer_weights)
        vertex = coupling[1:-1]
        vertices = vertex[:, None] * vertex[None, :]
        signs = (1.0, -1.0)
        moved: dict = {}
        for a in blocks:
            for b in blocks:
                factor = f._isospin_factor("sigma", a, b)
                if factor == 0.0:
                    continue
                device = self.device_for(a.kappa, b.kappa)
                angular = self.angular(device)

                def pair(am, bm, vv, ka=a.kappa, kb=b.kappa, factor=factor, angular=angular):
                    kernel = angular.scalar_time("sigma", ka, kb, f.params.m_sigma)[1:-1, 1:-1]
                    out = None
                    for mu in range(2):
                        for mup in range(2):
                            term = torch.sum(factor * signs[mu] * signs[mup] * am[mu, mup]
                                             * vv * kernel * bm[mu, mup], dim=1)
                            out = term if out is None else out + term
                    return out

                profile = profile + self._run(
                    pair, device, self._per_device(a.matrix, moved, device),
                    self._per_device(b.matrix, moved, device),
                    self._per_device(vertices, moved, device))
        energy = 0.5 * f.params.hbar_c * torch.sum(f.exchange_outer_weights * profile)
        return energy, profile

    def vector_exchange(self, channel, blocks, coupling, mass_mev):
        f = self.functional
        time_profile = torch.zeros_like(f.exchange_outer_weights)
        space_profile = torch.zeros_like(time_profile)
        vertex = coupling[1:-1]
        vertices = vertex[:, None] * vertex[None, :]
        moved: dict = {}
        for a in blocks:
            for b in blocks:
                factor = f._isospin_factor(channel, a, b)
                if factor == 0.0:
                    continue
                device = self.device_for(a.kappa, b.kappa)
                angular = self.angular(device)

                def pair(am, bm, vv, ka=a.kappa, kb=b.kappa, factor=factor, angular=angular):
                    time_kernel = angular.scalar_time(channel, ka, kb, mass_mev)[1:-1, 1:-1]
                    t_out = s_out = None
                    for mu in range(2):
                        for mup in range(2):
                            space_kernel = angular.vector_space(
                                channel, ka, kb, mu, mup, mass_mev)[1:-1, 1:-1]
                            common = am[mu, mup] * vv
                            t = -factor * torch.sum(common * time_kernel * bm[mu, mup], dim=1)
                            s = factor * torch.sum(common * space_kernel * bm[1 - mu, 1 - mup], dim=1)
                            t_out = t if t_out is None else t_out + t
                            s_out = s if s_out is None else s_out + s
                    return t_out, s_out

                t, s = self._run(pair, device, self._per_device(a.matrix, moved, device),
                                 self._per_device(b.matrix, moved, device),
                                 self._per_device(vertices, moved, device))
                time_profile = time_profile + t
                space_profile = space_profile + s
        scale = 0.5 * f.params.hbar_c
        return (scale * torch.sum(f.exchange_outer_weights * time_profile),
                scale * torch.sum(f.exchange_outer_weights * space_profile),
                time_profile, space_profile)

    def pion_exchange(self, blocks, coupling):
        f = self.functional
        profile = torch.zeros_like(f.exchange_outer_weights)
        if not isinstance(f.angular, PKA1AngularKernels):
            return torch.zeros((), dtype=f.r.dtype, device=f.r.device), profile
        vertices = coupling[1:-1, None] * coupling[None, 1:-1]
        moved: dict = {}
        for a in blocks:
            for b in blocks:
                factor = f._isospin_factor("rho", a, b)
                device = self.device_for(a.kappa, b.kappa)
                angular = self.angular(device)

                def pair(am, bm, vv, ka=a.kappa, kb=b.kappa, factor=factor, angular=angular):
                    kernel = angular.pion(ka, kb, f.params.m_pi)[:, :, 1:-1, 1:-1]
                    out = None
                    for mu in range(2):
                        for nu in range(2):
                            term = factor * torch.sum(am[mu, nu] * vv * kernel[mu, nu] * bm[mu, nu], dim=1)
                            out = term if out is None else out + term
                    return out

                profile = profile + self._run(
                    pair, device, self._per_device(a.matrix, moved, device),
                    self._per_device(b.matrix, moved, device),
                    self._per_device(vertices, moved, device))
        energy = 0.5 * f.params.hbar_c * torch.sum(f.exchange_outer_weights * profile)
        return energy, profile

    def rho_tensor_exchange(self, blocks, vector_coupling, tensor_coupling):
        f = self.functional
        zero = torch.zeros((), dtype=f.r.dtype, device=f.r.device)
        zero_profile = torch.zeros_like(f.exchange_outer_weights)
        if not isinstance(f.angular, PKA1AngularKernels):
            return zero, zero, zero, zero, zero_profile, zero_profile, zero_profile, zero_profile
        profiles = [torch.zeros_like(zero_profile) for _ in range(4)]
        tt = tensor_coupling[1:-1, None] * tensor_coupling[None, 1:-1]
        vt = vector_coupling[1:-1, None] * tensor_coupling[None, 1:-1]
        tv = tensor_coupling[1:-1, None] * vector_coupling[None, 1:-1]
        moved: dict = {}
        for a in blocks:
            for b in blocks:
                factor = f._isospin_factor("rho", a, b)
                device = self.device_for(a.kappa, b.kappa)
                angular = self.angular(device)

                def pair(am, bm, ttd, vtd, tvd, ka=a.kappa, kb=b.kappa, factor=factor, angular=angular):
                    cv_t, ct_t, rt_t = angular.rho_tensor_time(ka, kb, f.params.m_rho)
                    cv_s, ct_s, rt_s = angular.rho_tensor_space(ka, kb, f.params.m_rho)
                    cv_t = cv_t[:, :, 1:-1, 1:-1]; ct_t = ct_t[:, :, 1:-1, 1:-1]
                    rt_t = rt_t[:, :, 1:-1, 1:-1]
                    cv_s = cv_s[:, :, 1:-1, 1:-1]; ct_s = ct_s[:, :, 1:-1, 1:-1]
                    rt_s = rt_s[:, :, 1:-1, 1:-1]
                    outs = [None, None, None, None]
                    for mu in range(2):
                        for nu in range(2):
                            common = factor * am[mu, nu]
                            terms = (
                                torch.sum(common * ttd * rt_t[mu, nu] * bm[1 - mu, 1 - nu], dim=1),
                                torch.sum(common * ttd * rt_s[mu, nu] * bm[mu, nu], dim=1),
                                torch.sum(common * vtd * (cv_t[mu, nu] * bm[mu, 1 - nu]
                                                          + cv_s[mu, nu] * bm[1 - mu, nu]), dim=1),
                                torch.sum(common * tvd * (ct_t[mu, nu] * bm[1 - mu, nu]
                                                          + ct_s[mu, nu] * bm[mu, 1 - nu]), dim=1),
                            )
                            outs = [t if o is None else o + t for o, t in zip(outs, terms)]
                    return tuple(outs)

                contributions = self._run(
                    pair, device, self._per_device(a.matrix, moved, device),
                    self._per_device(b.matrix, moved, device),
                    self._per_device(tt, moved, device), self._per_device(vt, moved, device),
                    self._per_device(tv, moved, device))
                profiles = [p + c for p, c in zip(profiles, contributions)]
        scale = 0.5 * f.params.hbar_c
        energies = tuple(scale * torch.sum(f.exchange_outer_weights * p) for p in profiles)
        return (*energies, *profiles)
