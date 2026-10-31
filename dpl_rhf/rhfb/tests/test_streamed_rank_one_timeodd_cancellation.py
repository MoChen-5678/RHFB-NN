"""Rank-one antisymmetry check in the full streamed, time-odd PKA1 PH path.

For one occupied identical fermion orbital, the direct and exchange Wick
contractions of each antisymmetrized two-body vertex are equal and opposite.
This applies to the PKA1 density-dependent vertices when both contractions
use the same endpoint couplings.  It is a channel self-interaction identity,
not a statement that a neutron-proton pair has zero interaction, and it does
not include the kinetic energy.  See Long et al., arXiv:0812.1103,
Hamiltonian (1) and Eqs. (4a-g), and Geng & Long, arXiv:2202.00482,
Eqs. (20b-c).

This test runs the production StreamedPrimitiveNormalFunctional at D=48,
P=108 with complex, parity-mixed spinors.  Central differences independently
check the rank-one orbital-direction derivative of every complete channel
group; the RHF-e contact terms are also inspected as direct/exchange pairs.
"""
from __future__ import annotations

import pytest
import torch

from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.contact import (pion_pv_contact_energy,
                                  rho_tensor_contact_energy)
from dpl_rhf.rhfb.currents import _gamma_matrices, local_bilinears
from dpl_rhf.rhfb.execution_limits import require_bounded_cgroup
from dpl_rhf.rhfb.operators import DensityFactors
from dpl_rhf.rhfb.streamed_normal_functional import StreamedPrimitiveNormalFunctional


_CHANNELS = (
    "sigma", "omega", "rho_vector", "rho_tensor", "rho_vt", "rho_tv",
    "pion_pv", "coulomb",
)


def _rank_one_spinor(basis: AngularRadialBasis, seed: int) -> torch.Tensor:
    rng = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.zeros(basis.dimension, dtype=torch.complex128)
    radii = torch.as_tensor(basis.x, dtype=torch.float64)

    def orbital_l(kappa):
        return -kappa - 1 if kappa < 0 else kappa

    for channel in basis.channels:
        # Each channel stores its own reduced upper G and lower F radial
        # coefficients in adjacent n_radial-sized blocks.  The angular
        # sampling code uses G with harmonic(kappa) and i*F with harmonic(-kappa).
        # Smooth oscillator-like reduced-radial envelopes avoid assigning
        # random values to basis slots and mistaking those slots for components.
        upper_shape = radii ** (orbital_l(channel.kappa) + 1) * torch.exp(
            -0.5 * (radii / 0.65).square()
        )
        lower_shape = radii ** (orbital_l(-channel.kappa) + 1) * torch.exp(
            -0.5 * (radii / 0.65).square()
        )
        upper_amp = torch.complex(
            torch.randn((), generator=rng, dtype=torch.float64),
            torch.randn((), generator=rng, dtype=torch.float64),
        )
        lower_amp = 0.08 * torch.complex(
            torch.randn((), generator=rng, dtype=torch.float64),
            torch.randn((), generator=rng, dtype=torch.float64),
        )
        upper = upper_amp * upper_shape
        lower = lower_amp * lower_shape
        q[channel.offset:channel.offset + basis.n_radial] = upper
        q[channel.offset + basis.n_radial:
          channel.offset + 2 * basis.n_radial] = lower
    return (q / torch.linalg.vector_norm(q)).unsqueeze(1)


def _states(basis, species, q):
    one = torch.ones((1, 1), dtype=torch.complex128)
    zero = torch.zeros_like(one)
    return {
        s: DensityFactors(q, one if s == species else zero, zero)
        for s in ("n", "p")
    }


def _channel_groups(terms):
    groups = {}
    for channel in _CHANNELS:
        value = terms[f"{channel}_direct"] + terms[f"{channel}_fock"]
        if channel == "rho_tensor":
            value = value + terms["rho_tensor_contact"]
        elif channel == "pion_pv":
            value = value + terms["pion_pv_contact"]
        groups[channel] = value
    return groups


@pytest.mark.parametrize("species", ("n", "p"))
def test_streamed_rank_one_timeodd_channel_cancellation_and_tangent(species):
    # Must run through the project's memory guard before basis/model-sized
    # arrays are allocated; the parent owns protected test execution.
    require_bounded_cgroup()
    torch.set_num_threads(2)
    basis = AngularRadialBasis(
        radius_fm=1.4, spacing_fm=0.2, kappa_max=1,
        n_theta=3, n_phi=6, derivative_order=5,
    )
    assert basis.dimension == 48
    assert basis.point_count == 108

    q = _rank_one_spinor(basis, 20261015 if species == "n" else 20261016)
    factors = _states(basis, species, q)
    sampled = basis._sample_range(q, 0, basis.point_count)
    bilinears = local_bilinears(sampled, torch.ones((1, 1), dtype=torch.complex128))
    assert float(bilinears["baryon_current"][:, 1:].abs().max()) > 1e-8
    assert float(bilinears["axial_current"][:, 1:].abs().max()) > 1e-8

    functional = StreamedPrimitiveNormalFunctional(
        basis, tile_size=8, chunk_size=32, lmax=2,
    )

    def evaluate(orbital):
        return functional.evaluate(_states(basis, species, orbital)).energy

    base = evaluate(q)
    terms = base.energy_terms
    for channel in _CHANNELS:
        direct = float(terms[f"{channel}_direct"].detach())
        fock = float(terms[f"{channel}_fock"].detach())
        assert abs(direct + fock) <= 2e-9 * max(1., abs(direct), abs(fock)), (
            species, channel, direct, fock,
        )
    # Preserve the nonzero pion Hartree contraction: cancellation is with its
    # exchange partner, not an artificial removal of the direct field.
    assert abs(float(terms["pion_pv_direct"].detach())) > 1e-8

    # Contacts are separate PKA1 RHF-e interactions.  Check their direct and
    # exchange pieces explicitly rather than folding them into finite-range
    # derivative-meson contractions.
    primitive = functional.primitive_densities(factors)
    weights = torch.as_tensor(basis.weights, dtype=torch.float64)
    gamma, gamma5 = _gamma_matrices(device=torch.device("cpu"),
                                   dtype=torch.complex128)
    _local, vertices, _rho_b, _tops, _pops = functional._local_context(
        primitive, weights, gamma, gamma5,
    )
    spinors = {s: basis._sample_range(factors[s].q, 0, basis.point_count)
               for s in ("n", "p")}
    active = {s: factors[s].rho_active for s in ("n", "p")}
    tensor = rho_tensor_contact_energy(
        spinors, active, weights, vertices["grtn"],
        hbar_c_mev_fm=functional.params.hbar_c,
    )
    pion = pion_pv_contact_energy(
        spinors, active, weights,
        vertices["fpio"] * functional.params.hbar_c / functional.params.m_pi,
        hbar_c_mev_fm=functional.params.hbar_c,
    )
    for name, contact in (("rho_tensor_contact", tensor),
                          ("pion_pv_contact", pion)):
        direct = float(contact["direct"].detach())
        exchange = float(contact["exchange"].detach())
        assert abs(direct + exchange) <= 2e-9 * max(1., abs(direct), abs(exchange)), (
            species, name, direct, exchange,
        )
        torch.testing.assert_close(
            contact["total"], terms[name], atol=2e-10, rtol=2e-10,
            msg=f"{species} {name}: separately evaluated D+E must match streamed term",
        )

    # Use a normalized tangent to the rank-one orbital manifold.  Every
    # channel's D+F(+its own contact) energy is identically zero along this
    # path, so its independently differenced directional derivative vanishes.
    rng = torch.Generator(device="cpu").manual_seed(
        20261017 if species == "n" else 20261018,
    )
    z = torch.complex(
        torch.randn(basis.dimension, generator=rng, dtype=torch.float64),
        torch.randn(basis.dimension, generator=rng, dtype=torch.float64),
    ).to(torch.complex128).unsqueeze(1)
    tangent = z - q @ (q.mH @ z)
    tangent = tangent / torch.linalg.vector_norm(tangent)
    eps = 2e-4

    def displaced(sign):
        value = q + (sign * eps) * tangent
        return value / torch.linalg.vector_norm(value)

    plus = evaluate(displaced(+1)).energy_terms
    minus = evaluate(displaced(-1)).energy_terms
    for channel in _CHANNELS:
        plus_group = _channel_groups(plus)[channel]
        minus_group = _channel_groups(minus)[channel]
        slope = float(((plus_group - minus_group) / (2 * eps)).detach())
        scale = max(1., abs(float(plus_group.detach())), abs(float(minus_group.detach())))
        assert abs(slope) < 2e-7 * scale, (species, channel, slope)

