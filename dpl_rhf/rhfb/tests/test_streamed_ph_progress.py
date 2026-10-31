"""Completed-block progress must not change the recomputed field gradient."""
from types import SimpleNamespace

import torch

from dpl_rhf.rhfb.operators import DensityFactors
from dpl_rhf.rhfb.streamed_normal_functional import StreamedPrimitiveNormalFunctional


class _AnalyticBlocks(StreamedPrimitiveNormalFunctional):
    def iter_energy_blocks(self, rhos, **_kwargs):
        yield lambda n, p: {'quadratic': torch.trace(n@n+p@p).real}
        yield lambda n, p: {'cross': 0.1*torch.trace(n@p).real}


def test_completed_forward_backward_progress_preserves_hvp():
    rows = []
    backend = _AnalyticBlocks(SimpleNamespace(dimension=2, point_count=4),
                              tile_size=4, progress_callback=rows.append)
    theta = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    q = torch.eye(2, dtype=torch.complex128)
    skew = torch.tensor([[0, 1], [-1, 0]], dtype=torch.complex128)
    # A fractional normal occupation needs its BCS anomalous partner so that
    # the generalized density is a pure quasiparticle vacuum.
    states = {'n': DensityFactors(q, theta*q, torch.sqrt(theta*(1-theta))*skew),
              'p': DensityFactors(q, 0.3*q, (0.3*0.7)**0.5*skew)}
    energy, fields = backend.differentiable_energy_and_fields(states)
    assert [(r['stage'], r['completed_blocks']) for r in rows] == [
        ('PH_forward', 1), ('PH_forward', 2)]
    loss = energy + fields['n'].abs().square().sum()
    derivative, = torch.autograd.grad(loss, theta)
    torch.testing.assert_close(derivative, 20*theta.detach()+0.3, atol=1e-12, rtol=1e-12)
    assert [(r['stage'], r['completed_blocks']) for r in rows[2:]] == [
        ('PH_backward', 1), ('PH_backward', 2)]
    assert all(r['block_count'] == 2 and r['resource_snapshot']['enforced'] for r in rows)
