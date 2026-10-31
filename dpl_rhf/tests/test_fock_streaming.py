"""Streamed (per kappa pair, optionally multi-device) Fock exchange equals the default."""
import os

import pytest
import torch

from dpl_rhf.backends.base import NucleusCase
from dpl_rhf.models.density_matrix_net import DensityMatrixNet
from dpl_rhf.rhf.backend import DensityMatrixRHFBackend
from dpl_rhf.rhf.density import nonlocal_density_from_block


def _devices():
    if os.environ.get("RHFB_CUDA_UUID") and torch.cuda.is_available():
        from dpl_rhf.rhfb.cuda_resources import configure_cuda
        configure_cuda("cuda:0", budget_mib=int(os.environ["RHFB_CUDA_BUDGET_MIB"]))
        return [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
    return [torch.device("cpu")]


DEVICES = _devices()
PRIMARY = DEVICES[0]


def _evaluate(backend, network, streaming):
    backend.functional.configure_fock_streaming(streaming)
    blocks = network()
    nonlocal_blocks = [nonlocal_density_from_block(b, backend.functional.h) for b in blocks]
    energy = backend.functional.evaluate(blocks, nonlocal_blocks)
    projectors = [d.projector for d in nonlocal_blocks]
    fock_gradient = torch.autograd.grad(energy.exchange_terms["E_fock"], projectors,
                                        retain_graph=True)
    parameter_gradient = torch.autograd.grad(
        energy.total, [p for p in network.parameters()], allow_unused=True)
    return energy, fock_gradient, parameter_gradient


def _compare(streaming):
    case = NucleusCase("PKA1", 8, 8, 16)
    backend = DensityMatrixRHFBackend(case, device=PRIMARY)
    torch.manual_seed(3)
    network = DensityMatrixNet(torch.as_tensor(backend.r, device=PRIMARY),
                               backend.block_specs).to(PRIMARY)
    with torch.no_grad():
        for head in network.output_heads:
            head.weight.normal_(std=0.3)
    reference, ref_fock, ref_params = _evaluate(backend, network, None)
    streamed, fock, params = _evaluate(backend, network, streaming)
    for name, value in reference.exchange_terms.items():
        torch.testing.assert_close(streamed.exchange_terms[name], value, rtol=1e-11, atol=1e-10,
                                   msg=name)
    torch.testing.assert_close(streamed.total, reference.total, rtol=1e-12, atol=1e-10)
    torch.testing.assert_close(streamed.fields["sigma_rearr_fock_explicit"],
                               reference.fields["sigma_rearr_fock_explicit"], rtol=1e-10, atol=1e-12)
    for a, b in zip(fock, ref_fock):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)
    for a, b in zip(params, ref_params):
        if a is None or b is None:
            assert a is None and b is None
            continue
        torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-11)


def test_streamed_single_device_matches_default():
    _compare([PRIMARY])


@pytest.mark.skipif(len(DEVICES) < 2, reason="needs two launcher-exposed GPUs")
def test_streamed_two_devices_match_default():
    _compare(DEVICES[:2])
