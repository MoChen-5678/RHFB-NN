import numpy as np
import pytest
import torch

from dpl_rhf.rhfb.manifold import slater_reference, transform_reference
from dpl_rhf.rhfb.parity_diagnostic import number_parity_from_density


@pytest.mark.parametrize("d,occupied", [(3, 0), (3, 1), (6, 2), (6, 3), (8, 5)])
def test_number_parity_survives_general_canonical_rotation(d, occupied):
    torch.manual_seed(66000 + d + occupied)
    occ = torch.zeros(d, dtype=torch.float64)
    occ[:occupied] = 1
    state = slater_reference(occ)
    a = torch.randn(d, d, dtype=torch.complex128) * .4
    b = torch.randn(d, d, dtype=torch.complex128) * .4
    rotated = transform_reference(state, a - a.mH, b - b.mT)
    data = number_parity_from_density(rotated.rho.numpy(), rotated.chi.numpy())
    assert data["number_parity"] == (-1) ** occupied


def test_average_odd_number_does_not_imply_odd_number_parity():
    # One paired BCS vacuum: 50% |0> and 50% |2>, average N=1, parity even.
    rho = np.eye(2, dtype=np.complex128) * .5
    chi = np.array([[0, .5], [-.5, 0]], dtype=np.complex128)
    data = number_parity_from_density(rho, chi)
    assert data["particle_number"] == 1
    assert data["number_parity"] == 1


def test_equal_filling_mixed_density_has_no_pure_state_parity():
    rho = np.eye(2, dtype=np.complex128) * .5
    with pytest.raises(ValueError, match="pure-state parity"):
        number_parity_from_density(rho, np.zeros_like(rho))
