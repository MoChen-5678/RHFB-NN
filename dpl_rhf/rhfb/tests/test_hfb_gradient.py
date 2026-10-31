from __future__ import annotations

import torch

from dpl_rhf.rhfb.bogoliubov_blocking import slater_bogoliubov_frame, state_from_frame
from dpl_rhf.rhfb.hfb_gradient import (
    apply_thouless_update,
    field_energy_directional_derivative,
    h20_matrix,
    n20_matrix,
    project_fields_number_tangent,
    thouless_density_derivative,
    thouless_energy_directional_derivative,
)
from dpl_rhf.rhfb.manifold import canonical_generator
from dpl_rhf.rhfb.number_constraint import number_tangent_projection
from dpl_rhf.rhfb.operators import DensityFactors


def _complex_fields() -> tuple[torch.Tensor, torch.Tensor]:
    h = torch.tensor(
        [[-1.1, 0.17 + 0.09j, 0.0, -0.04j],
         [0.17 - 0.09j, 0.6, 0.13, 0.0],
         [0.0, 0.13, 1.4, 0.11 + 0.03j],
         [0.04j, 0.0, 0.11 - 0.03j, 2.0]],
        dtype=torch.complex128,
    )
    delta = torch.tensor(
        [[0, 0.21 + 0.04j, -0.06j, 0.03],
         [-0.21 - 0.04j, 0, 0.08, 0.02j],
         [0.06j, -0.08, 0, -0.17 + 0.05j],
         [-0.03, -0.02j, 0.17 - 0.05j, 0]],
        dtype=torch.complex128,
    )
    return h, delta


def _frame_and_direction():
    m = 4
    a = torch.zeros((m, m), dtype=torch.complex128)
    a[0, 1] = 0.07 + 0.11j
    a[1, 0] = -0.07 + 0.11j
    a[2, 3] = -0.05 + 0.08j
    a[3, 2] = 0.05 + 0.08j
    b = torch.zeros_like(a)
    b[0, 2] = 0.19 + 0.06j
    b[2, 0] = -b[0, 2]
    b[1, 3] = -0.12 + 0.09j
    b[3, 1] = -b[1, 3]
    reference = slater_bogoliubov_frame(
        torch.tensor([1, 1, 0, 0], dtype=torch.float64)
    )
    frame = torch.matrix_exp(canonical_generator(a, b)) @ reference
    z = torch.zeros_like(a)
    z[0, 1] = 0.13 + 0.17j
    z[1, 0] = -z[0, 1]
    z[2, 3] = -0.09 + 0.04j
    z[3, 2] = -z[2, 3]
    return frame, z


def test_h20_n20_match_direct_nambu_frame_transform_and_number_projection():
    frame, _z = _frame_and_direction()
    m = frame.shape[0] // 2
    u, v = frame[:m, :m], frame[m:, :m]
    h, delta = _complex_fields()
    h_nambu = torch.cat(
        (torch.cat((h, delta), dim=1),
         torch.cat((-delta.conj(), -h.conj()), dim=1)),
        dim=0,
    )
    number_nambu = torch.diag(
        torch.tensor([1.0] * m + [-1.0] * m, dtype=torch.complex128)
    )
    qp_h = frame.mH @ h_nambu @ frame
    qp_n = frame.mH @ number_nambu @ frame
    h20 = h20_matrix(u, v, h, delta)
    n20 = n20_matrix(u, v)
    torch.testing.assert_close(h20, qp_h[:m, m:], atol=3e-13, rtol=3e-13)
    torch.testing.assert_close(n20, qp_n[:m, m:], atol=3e-13, rtol=3e-13)

    state = state_from_frame(frame, fermion_parity=1).density
    factors = DensityFactors(torch.eye(m, dtype=torch.complex128), state.rho, state.chi)
    full_r_projection = number_tangent_projection(factors, h, delta)
    uv_projection = project_fields_number_tangent(u, v, h, delta)
    assert uv_projection.identifiable == full_r_projection.identifiable
    torch.testing.assert_close(
        uv_projection.chemical_potential,
        full_r_projection.chemical_potential,
        atol=3e-12,
        rtol=3e-12,
    )
    torch.testing.assert_close(
        uv_projection.constrained_h20,
        h20 - full_r_projection.chemical_potential * n20,
        atol=3e-12,
        rtol=3e-12,
    )


def test_density_field_and_thouless_h20_derivatives_agree_with_selfconsistent_fd():
    frame, z = _frame_and_direction()
    m = frame.shape[0] // 2
    u, v = frame[:m, :m], frame[m:, :m]
    rho_chi = state_from_frame(frame, fermion_parity=1).density
    drho, dchi = thouless_density_derivative(u, v, z)

    # A nonlinear normal EDF and a quadratic pairing EDF exercise the
    # functional-derivative double counting independently of the H20 route.
    t, _unused = _complex_fields()
    normal_strength = 0.31
    pairing_strength = 0.73
    number = torch.trace(rho_chi.rho).real
    h = t + normal_strength * number * torch.eye(m, dtype=torch.complex128)
    delta = 2.0 * pairing_strength * rho_chi.chi
    via_fields = field_energy_directional_derivative(h, delta, drho, dchi)
    via_h20 = thouless_energy_directional_derivative(u, v, h, delta, z)
    torch.testing.assert_close(via_fields, via_h20, atol=3e-13, rtol=3e-13)

    def energy_at(step: float) -> torch.Tensor:
        moved = apply_thouless_update(
            frame,
            z,
            step=step,
            fermion_parity=1,
        ).frame
        density = state_from_frame(moved, fermion_parity=1).density
        normal = torch.trace(t @ density.rho).real
        count = torch.trace(density.rho).real
        pairing = pairing_strength * density.chi.abs().square().sum()
        return normal + 0.5 * normal_strength * count.square() + pairing

    step = 2.0e-6
    finite_difference = (energy_at(step) - energy_at(-step)) / (2.0 * step)
    torch.testing.assert_close(finite_difference, via_h20, atol=2.0e-8, rtol=2.0e-8)


def test_number_tangent_singularity_is_reported_without_inventing_mu():
    frame = slater_bogoliubov_frame(
        torch.tensor([1, 0], dtype=torch.float64)
    )
    m = 2
    u, v = frame[:m, :m], frame[m:, :m]
    h = torch.diag(torch.tensor([0.3, 1.1], dtype=torch.complex128))
    delta = torch.zeros_like(h)
    fallback = torch.tensor(-4.2, dtype=torch.float64)
    projected = project_fields_number_tangent(u, v, h, delta, fallback_mu=fallback)
    assert not projected.identifiable
    torch.testing.assert_close(projected.chemical_potential, fallback)
    torch.testing.assert_close(projected.constrained_h20, projected.h20)
