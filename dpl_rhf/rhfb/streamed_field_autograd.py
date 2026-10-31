"""Exact block-recomputed VJP for PH energy and Hermitian normal fields.

The backend supplies the complete scalar as independent, recomputable blocks.
Forward sums their first derivatives without retaining their graphs. Backward
recomputes each block, including its density-dependent vertices, and evaluates
the field Hessian-vector product. This avoids silently freezing self-consistent
fields in a neural stationarity loss. Third derivatives are not supported.

The interface is on Hermitian primitive densities and Hermitian tangents. It
does not choose a Dirac-sea branch or supply the pairing functional.
"""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from .execution_limits import require_bounded_cgroup


def _hermitian(value):
    return 0.5*(value+value.mH)


def _block_total(values, template):
    if not isinstance(values, dict) or not values:
        raise TypeError('each recomputable energy block must return a nonempty term dict')
    total = template.real.new_zeros(())
    for value in values.values():
        if (not isinstance(value, torch.Tensor) or value.shape != ()
                or value.dtype != torch.float64):
            raise TypeError('energy terms must be scalar float64 tensors')
        total = total + value
    return total


def _gradients(total, leaves, *, create_graph):
    if not total.requires_grad:
        return tuple(torch.zeros_like(leaf) for leaf in leaves)
    gradients = torch.autograd.grad(total, leaves, create_graph=create_graph,
                                    allow_unused=True)
    return tuple(torch.zeros_like(leaf) if gradient is None else gradient
                 for leaf, gradient in zip(leaves, gradients))


class _StreamedEnergyFields(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rho_n, rho_p, block_factory):
        require_bounded_cgroup()
        for rho in (rho_n, rho_p):
            if (rho.dtype != torch.complex128 or rho.ndim != 2
                    or rho.shape[0] != rho.shape[1]):
                raise ValueError('primitive normal density must be square complex128')
            if not torch.allclose(rho, rho.mH, atol=2e-10, rtol=0):
                raise ValueError('streamed field interface requires Hermitian densities')
        if rho_n.shape != rho_p.shape or rho_n.device != rho_p.device:
            raise ValueError('neutron and proton primitive spaces must agree')
        ctx.block_factory = block_factory
        ctx.save_for_backward(rho_n, rho_p)
        ctx.set_materialize_grads(False)
        energy = rho_n.real.new_zeros(())
        fields = [torch.zeros_like(rho_n), torch.zeros_like(rho_p)]
        leaves = tuple(rho.detach().requires_grad_(True) for rho in (rho_n, rho_p))
        count = 0
        with torch.enable_grad():
            for block in block_factory(dict(zip(('n', 'p'), leaves))):
                total = _block_total(block(*leaves), rho_n)
                gradients = _gradients(total, leaves, create_graph=False)
                energy = energy + total.detach()
                for index, gradient in enumerate(gradients):
                    fields[index] = fields[index] + _hermitian(gradient.detach())
                count += 1
                del total, gradients
        if count == 0:
            raise ValueError('no energy blocks supplied')
        ctx.block_count = count
        return energy, fields[0], fields[1]

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_energy, grad_h_n, grad_h_p):
        require_bounded_cgroup()
        source = ctx.saved_tensors
        leaves = tuple(rho.detach().requires_grad_(True) for rho in source)
        result = [torch.zeros_like(rho) for rho in source]
        field_seeds = (grad_h_n, grad_h_p)
        need_hvp = any(seed is not None for seed in field_seeds)
        count = 0
        with torch.enable_grad():
            for block in ctx.block_factory(dict(zip(('n', 'p'), leaves))):
                total = _block_total(block(*leaves), source[0])
                gradients = _gradients(total, leaves, create_graph=need_hvp)
                if grad_energy is not None:
                    for index, gradient in enumerate(gradients):
                        result[index] = result[index] + grad_energy * _hermitian(gradient.detach())
                if need_hvp:
                    field_dot = source[0].real.new_zeros(())
                    for seed, gradient in zip(field_seeds, gradients):
                        if seed is not None:
                            field_dot = field_dot + (seed.conj()*_hermitian(gradient)).sum().real
                    hvp = _gradients(field_dot, leaves, create_graph=False)
                    for index, value in enumerate(hvp):
                        result[index] = result[index] + _hermitian(value.detach())
                    del field_dot, hvp
                count += 1
                del total, gradients
        if count != ctx.block_count:
            raise RuntimeError('energy block decomposition changed between forward and backward')
        return result[0], result[1], None


def streamed_energy_and_fields(rho_n, rho_p, block_factory):
    """Return (E_ph, h_n, h_p) with exact first loss gradients via recomputation.

    ``block_factory({'n': rho_n, 'p': rho_p})`` must return an iterator of
    callables ``block(rho_n, rho_p) -> energy_term_dict``. Blocks must form the
    same complete PH scalar on every call, without detached density vertices.
    The density inputs retain their NN dependencies; field-loss backward
    includes second derivatives of E. Third derivatives are explicitly absent.
    """
    return _StreamedEnergyFields.apply(rho_n, rho_p, block_factory)
