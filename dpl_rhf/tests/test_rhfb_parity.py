import numpy as np
import torch

from dpl_rhf.rhfb.manifold import slater_reference, transform_reference


def _pfaffian(matrix):
    if len(matrix) == 0:
        return 1.
    result = 0.
    for j in range(1, len(matrix)):
        keep = [k for k in range(len(matrix)) if k not in (0, j)]
        result += (-1.) ** (j + 1) * matrix[0, j] * _pfaffian(matrix[np.ix_(keep, keep)])
    return result


def _density_parity(r):
    # Independent Majorana correlation route, not the stored reference label.
    m = r.shape[0] // 2
    transform = torch.zeros_like(r)
    for i in range(m):
        transform[2*i, i], transform[2*i, m+i] = 1, 1
        transform[2*i+1, i], transform[2*i+1, m+i] = -1j, 1j
    correlations = transform @ (torch.eye(2*m, dtype=r.dtype) - r) @ transform.mH
    covariance = 1j * (correlations - torch.eye(2*m, dtype=r.dtype))
    assert covariance.imag.abs().max() < 1e-12
    return (-1.) ** m * _pfaffian(covariance.real.detach().numpy())


def test_complex_bogoliubov_motion_preserves_measured_even_and_odd_parity():
    torch.manual_seed(125)
    raw_a = torch.randn(3, 3, dtype=torch.complex128)
    raw_b = torch.randn_like(raw_a)
    a = .4 * (raw_a - raw_a.mH)
    b = .4 * (raw_b - raw_b.mT)
    for block, parity in ((None, 1), (2, -1)):
        reference = slater_reference(torch.tensor([1., 1., 0.], dtype=torch.float64),
                                     blocked_index=block)
        moved = transform_reference(reference, a, b)
        assert abs(_density_parity(reference.matrix) - parity) < 1e-12
        assert abs(_density_parity(moved.matrix) - parity) < 1e-12
        assert abs(float(torch.trace(moved.rho).real) - round(float(torch.trace(moved.rho).real))) > .001
