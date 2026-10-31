from __future__ import annotations

import torch

from dpl_rhf.rhfb.manifold import (
    canonical_generator,
    lift_active_density,
    require_blocking_continuity,
    slater_reference,
    transform_reference,
)


def test_complex_canonical_generator_preserves_generalized_density_and_odd_parity():
    dtype = torch.complex128
    raw_a = torch.tensor(
        [[0.1j, 0.2 + 0.1j, 0.0], [-0.2 + 0.1j, -0.2j, 0.3j], [0.0, 0.3j, 0.1j]],
        dtype=dtype,
    )
    a = 0.5 * (raw_a - raw_a.mH)
    raw_b = torch.tensor(
        [[0.0, 0.1 + 0.2j, -0.1j], [-0.1 - 0.2j, 0.0, 0.2], [0.1j, -0.2, 0.0]],
        dtype=dtype,
    )
    b = 0.5 * (raw_b - raw_b.mT)
    generator = canonical_generator(a, b)
    assert torch.allclose(generator.mH, -generator, atol=1.0e-14)

    odd = slater_reference(torch.zeros(3, dtype=torch.float64), blocked_index=1)
    transformed = transform_reference(odd, a, b)
    n = transformed.rho.shape[0]
    assert torch.linalg.matrix_norm(transformed.rho - transformed.rho.mH) < 1.0e-12
    assert torch.linalg.matrix_norm(transformed.chi + transformed.chi.mT) < 1.0e-12
    assert torch.linalg.matrix_norm(transformed.matrix @ transformed.matrix - transformed.matrix) < 1.0e-12
    assert torch.linalg.matrix_norm(
        transformed.matrix[n:, :] - torch.cat(
            [-transformed.chi.conj(), torch.eye(n, dtype=dtype) - transformed.rho.conj()], dim=1
        )
    ) < 1.0e-12
    assert transformed.fermion_parity == -1
    assert abs(float(torch.trace(transformed.rho).real) - 1.0) > 1.0e-4


def test_anomalous_lift_uses_transpose_and_keeps_density_in_factor_space():
    q0 = torch.tensor([[1, 0], [1j, 0], [0, 1], [0, 1j]], dtype=torch.complex128)
    q, _ = torch.linalg.qr(q0, mode="reduced")
    rho_a = torch.diag(torch.tensor([0.7, 0.2], dtype=torch.complex128))
    chi_a = torch.tensor([[0, 0.3 + 0.2j], [-0.3 - 0.2j, 0]], dtype=torch.complex128)
    state = lift_active_density(q, rho_a, chi_a)
    assert torch.allclose(state.rho, q @ rho_a @ q.mH)
    assert torch.allclose(state.chi, q @ chi_a @ q.mT)
    assert torch.linalg.matrix_norm(state.chi + state.chi.mT) < 1.0e-12


def test_blocking_overlap_rejects_silent_mode_switch():
    first = torch.tensor([1, 0, 0], dtype=torch.complex128)
    same = torch.tensor([1j, 0, 0], dtype=torch.complex128)
    switch = torch.tensor([0, 1, 0], dtype=torch.complex128)
    assert abs(float(require_blocking_continuity(first, same)) - 1.0) < 1.0e-14
    try:
        require_blocking_continuity(first, switch, minimum_overlap=0.2)
    except RuntimeError as error:
        assert "silent blocking switch" in str(error)
    else:
        raise AssertionError("a blocked-mode switch must fail closed")


def test_hole_blocking_toggles_an_occupied_reference_to_the_odd_branch():
    reference = slater_reference(
        torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64),
        blocked_index=1,
        blocking_operation="hole",
    )
    assert reference.fermion_parity == -1
    assert float(torch.trace(reference.rho).real) == 1.0
