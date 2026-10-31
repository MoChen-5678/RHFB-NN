"""Check conjugate pairing cache against independent full Nambu multiplication."""
import pytest
import torch

from dpl_rhf.rhfb.factorized_commutator import FactorizedCommutator
from dpl_rhf.rhfb.manifold import slater_reference, transform_reference
from dpl_rhf.rhfb.operators import DensityFactors, NambuAction, commutator_action
from dpl_rhf.rhfb.saved_stationarity import active_pairing_cache


def test_active_input_residual_retains_external_output_and_rejects_external_input():
    generator = torch.Generator().manual_seed(6026)
    q, _ = torch.linalg.qr(torch.randn(9, 4, dtype=torch.complex128, generator=generator))
    a = torch.randn(4, 4, dtype=q.dtype, generator=generator)
    b = torch.randn(4, 4, dtype=q.dtype, generator=generator)
    reference = slater_reference(torch.tensor([1., 0., 0., 0.], dtype=torch.float64))
    gd = transform_reference(reference, .03*(a-a.mH), .08*(b-b.T))
    state = DensityFactors(q, gd.rho, gd.chi)
    h0 = torch.randn(9, 9, dtype=q.dtype, generator=generator)
    d0 = torch.randn(9, 9, dtype=q.dtype, generator=generator)
    h, delta = .5*(h0+h0.mH), .5*(d0-d0.T)
    frame = q.new_zeros((18, 8))
    frame[:9, :4], frame[9:, 4:] = q, q.conj()
    pairing = active_pairing_cache(q, delta @ q.conj())
    cached = FactorizedCommutator(state, h @ q, pairing,
        chemical_potential=.17, delta_q_star=delta @ q.conj())
    direct = NambuAction(9, lambda v: h @ v, lambda v: delta @ v, .17)
    output = cached.apply(frame)
    torch.testing.assert_close(output, commutator_action(direct, state, frame),
                               atol=3e-14, rtol=3e-14)
    active = frame.mH @ output
    external = output-frame @ active
    assert float(external.abs().square().sum()) > .1
    torch.testing.assert_close(output.abs().square().sum(),
        active.abs().square().sum()+external.abs().square().sum(), atol=1e-12, rtol=1e-12)
    probe = q.new_zeros((9, 1))
    probe[0, 0] = 1
    with pytest.raises(RuntimeError, match='outside Q'):
        pairing(probe)
