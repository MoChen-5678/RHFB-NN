from __future__ import annotations

import pytest
import torch

from dpl_rhf.rhfb.manifold import GeneralizedDensity, generalized_density_matrix
from dpl_rhf.rhfb.time_reversal_average import (
    time_reversal_averaged_odd_one_body_reference,
)


def _time_reversal_matrix() -> torch.Tensor:
    # Basis order (a-up, a-down, b-up, b-down), T = J K and T^2 = -1.
    j = torch.zeros((4, 4), dtype=torch.complex128)
    j[0, 1], j[1, 0] = 1.0, -1.0
    j[2, 3], j[3, 2] = 1.0, -1.0
    return j


def _odd_paired_component() -> GeneralizedDensity:
    v2 = 0.25
    uv = (v2 * (1.0 - v2)) ** 0.5
    rho = torch.diag(torch.tensor([1.0, v2, v2, 0.0], dtype=torch.complex128))
    chi = torch.zeros((4, 4), dtype=torch.complex128)
    chi[1, 2], chi[2, 1] = uv, -uv
    return GeneralizedDensity(rho, chi, generalized_density_matrix(rho, chi), -1)


def test_reference_is_only_one_body_average_not_generalized_density_or_energy_state():
    # In Fock space, |up> and |down> average to rho=I/2 and chi=0.
    # The helper exposes those contractions without attaching a parity or
    # pretending the result is a pure Gaussian state.
    rho = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.complex128))
    chi = torch.zeros((2, 2), dtype=torch.complex128)
    blocked = GeneralizedDensity(rho, chi, generalized_density_matrix(rho, chi), -1)
    j = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.complex128)

    ref = time_reversal_averaged_odd_one_body_reference(blocked, j)

    assert torch.allclose(ref.rho, 0.5 * torch.eye(2, dtype=torch.complex128))
    assert torch.count_nonzero(ref.chi) == 0
    assert ref.weights == (0.5, 0.5)
    assert ref.sharp_fermion_parity is None
    assert ref.wick_energy_closed is False
    assert not hasattr(ref, "matrix")


def test_pairing_contraction_transforms_with_time_reversal():
    ref = time_reversal_averaged_odd_one_body_reference(
        _odd_paired_component(), _time_reversal_matrix()
    )
    ref_again = time_reversal_averaged_odd_one_body_reference(
        ref.time_reversed_component, _time_reversal_matrix()
    )
    assert torch.count_nonzero(ref.chi) > 0
    assert torch.allclose(ref.chi.T, -ref.chi)
    assert torch.allclose(ref.rho, ref_again.rho, atol=1.0e-12)
    assert torch.allclose(ref.chi, ref_again.chi, atol=1.0e-12)


def test_odd_only_fock_ensemble_is_not_wick_closed_and_differs_from_standard_efa():
    # Fock basis |0>, |1>, |2>, |12>. The odd-only mixture and the standard
    # EFA zero/one/two-qp mixture have identical one-body contractions, but
    # different exact two-body expectations.
    n1 = torch.diag(torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64))
    n2 = torch.diag(torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64))
    odd_only = torch.diag(torch.tensor([0.0, 0.5, 0.5, 0.0], dtype=torch.float64))
    standard_efa = 0.25 * torch.eye(4, dtype=torch.float64)

    assert torch.trace(odd_only @ n1).item() == pytest.approx(0.5)
    assert torch.trace(odd_only @ n2).item() == pytest.approx(0.5)
    assert torch.trace(standard_efa @ n1).item() == pytest.approx(0.5)
    assert torch.trace(standard_efa @ n2).item() == pytest.approx(0.5)
    exact_odd_only_two_body = torch.trace(odd_only @ n1 @ n2).item()
    exact_efa_two_body = torch.trace(standard_efa @ n1 @ n2).item()
    wick_from_shared_one_body = 0.5 * 0.5
    assert exact_odd_only_two_body == pytest.approx(0.0)
    assert exact_efa_two_body == pytest.approx(0.25)
    assert wick_from_shared_one_body == pytest.approx(exact_efa_two_body)
    assert exact_odd_only_two_body != wick_from_shared_one_body


def test_rejects_invalid_input_or_time_reversal_operator():
    pure = _odd_paired_component()
    with pytest.raises(ValueError, match="pure odd-parity"):
        even = GeneralizedDensity(pure.rho, pure.chi, pure.matrix, 1)
        time_reversal_averaged_odd_one_body_reference(even, _time_reversal_matrix())

    with pytest.raises(ValueError, match="square to -1"):
        time_reversal_averaged_odd_one_body_reference(
            pure, torch.eye(4, dtype=torch.complex128)
        )

    mixed_rho = 0.5 * torch.eye(4, dtype=torch.complex128)
    mixed_chi = torch.zeros((4, 4), dtype=torch.complex128)
    mixed = GeneralizedDensity(mixed_rho, mixed_chi,
                              generalized_density_matrix(mixed_rho, mixed_chi), -1)
    with pytest.raises(ValueError, match="pure generalized density"):
        time_reversal_averaged_odd_one_body_reference(mixed, _time_reversal_matrix())
