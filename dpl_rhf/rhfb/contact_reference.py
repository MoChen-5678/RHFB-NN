"""Independent spherical 1s contact/derivative reference for RHF-e.

This module deliberately does not call ``rhfb.contact``. Compensation traces
are evaluated from the paper's radial local-density identities. The separate
legacy spherical RHF implementation is used only as a cross-implementation
comparison target for the combined channels; it is not treated as a certified
independent physical oracle. This is a small normal-density diagnostic, not a
general-m or RHFB certificate.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from dpl_rhf.rhf.density import nonlocal_density_from_block
from dpl_rhf.rhf.functional import SphericalRHFFunctional
from dpl_rhf.rhf.parameters import pka1_parameters
from .basis import AngularRadialBasis
from .currents import local_bilinears
from .functional import evaluate_particle_hole_energy


def _one_species_closed_1s(spacing_fm: float, radius_fm: float,
                           derivative_order: int = 5):
    basis = AngularRadialBasis(
        radius_fm=radius_fm, spacing_fm=spacing_fm, kappa_max=1,
        n_theta=3, n_phi=6, derivative_order=derivative_order,
    )
    r_full = torch.arange(round(radius_fm / spacing_fm) + 1,
                          dtype=torch.float64) * spacing_fm
    r_full[0] = 0.001
    upper = r_full * torch.exp(-r_full.square() / (2.0 * 1.3**2))
    lower = 0.035 * r_full * upper
    norm = torch.sqrt(torch.sum(upper[1:-1].square() + lower[1:-1].square()))
    upper, lower = upper / norm, lower / norm

    q_n = torch.zeros((basis.dimension, 2), dtype=torch.complex128)
    for column, twice_m in enumerate((-1, 1)):
        channel = basis.channels[basis.channel_index[(-1, twice_m)]]
        q_n[channel.offset:channel.offset + basis.n_radial, column] = upper[1:-1]
        q_n[channel.offset + basis.n_radial:channel.offset + 2*basis.n_radial,
            column] = lower[1:-1]
    q_p = torch.zeros_like(q_n)
    rho_n = torch.eye(2, dtype=torch.complex128)
    rho_p = torch.zeros_like(rho_n)
    states = {
        "n": SimpleNamespace(q=q_n, rho_active=rho_n),
        "p": SimpleNamespace(q=q_p, rho_active=rho_p),
    }

    # The same occupied radial 1s doublet, represented as a spherical block.
    radial = torch.cat((upper[1:-1], lower[1:-1]))[:, None]
    block = {
        "factors": radial,
        "occupations": torch.ones(1, dtype=torch.float64),
        "degeneracy": 2.0,
        "species_sign": 1.0,
        "kappa": -1,
        "principals": (1,),
    }
    spherical_blocks = [nonlocal_density_from_block(block, spacing_fm)]
    orbital_blocks = [{"free_hamiltonian": torch.zeros(
        (2*basis.n_radial, 2*basis.n_radial), dtype=torch.float64,
    )}]
    return basis, r_full, states, spherical_blocks, orbital_blocks


def _rhfe_contact_traces(basis, states, vertex, *, kind: str,
                         hbar_c: float) -> dict[str, torch.Tensor]:
    """Evaluate RHF-e Eqs. (3.205),(3.207),(3.294) without contact.py."""
    samples = {s: basis.sample(states[s].q)[0] for s in ("n", "p")}
    bil = {s: local_bilinears(samples[s], states[s].rho_active)
           for s in ("n", "p")}
    weights = torch.as_tensor(basis.weights, dtype=torch.float64)
    positions = torch.as_tensor(basis.positions, dtype=torch.float64)
    rhat = positions / torch.linalg.vector_norm(positions, dim=1, keepdim=True)
    species = ("n", "p")
    scalar = {s: bil[s]["scalar"] for s in species}
    baryon = {s: bil[s]["baryon_current"][:, 0] for s in species}
    radial_tensor = {
        s: torch.sum(bil[s]["tensor"][:, 0, 1:] * rhat, dim=1)
        for s in species
    }
    t3 = {"n": 1.0, "p": -1.0}
    exchange = weights.new_zeros(())
    if kind == "rho_tensor":
        # H-delta direct term, Eq. (3.205), and exchange trace, Eq. (3.207).
        isovector_tensor = radial_tensor["n"] - radial_tensor["p"]
        direct = -(hbar_c / 6.0) * torch.sum(
            weights * vertex.square() * isovector_tensor.square()
        )
        for left in species:
            for right in species:
                tau = 1.0 if left == right else 2.0
                trace = (3.0 * scalar[left] * scalar[right]
                         + baryon[left] * baryon[right]
                         + radial_tensor[left] * radial_tensor[right])
                exchange -= (hbar_c * tau / 8.0) * torch.sum(
                    weights * vertex.square() * trace
                )
    elif kind == "pion_pv":
        # H-delta is +1/6 of the static spatial axial-current square;
        # Eq. (3.294) gives the exchange trace after the Dirac trace reduction.
        axial3 = sum((t3[s] * bil[s]["axial_current"][:, 1:]
                      for s in species), torch.zeros_like(
                          bil["n"]["axial_current"][:, 1:]))
        direct = (hbar_c / 6.0) * torch.sum(
            weights * vertex.square() * axial3.square().sum(dim=1)
        )
        for left in species:
            for right in species:
                tau = 1.0 if left == right else 2.0
                trace = (scalar[left] * scalar[right]
                         + baryon[left] * baryon[right]
                         - radial_tensor[left] * radial_tensor[right] / 3.0)
                exchange -= (hbar_c * tau / 8.0) * torch.sum(
                    weights * vertex.square() * trace
                )
    else:
        raise ValueError("kind must be rho_tensor or pion_pv")
    return {"direct": direct, "exchange": exchange, "total": direct + exchange}


def rhfe_contact_reference(*, spacing_fm: float = 0.125,
                           radius_fm: float = 6.5,
                           derivative_order: int = 5) -> dict:
    """Compare general-m derivative+H-delta against legacy spherical RHF.

    The only state is a time-even neutron 1s closed doublet.  The comparison
    includes full Lorentz rho-T in the spherical kernel.  Pion Hartree is
    expected to vanish for this time-reversal-invariant state.
    """
    basis, r, states, spherical_blocks, orbital_blocks = _one_species_closed_1s(
        spacing_fm, radius_fm, derivative_order,
    )
    params = pka1_parameters()
    spherical = SphericalRHFFunctional(r, z=0, n=2, params=params).evaluate(
        orbital_blocks, spherical_blocks,
    )
    general = evaluate_particle_hole_energy(
        basis, states, params=params, lmax=2, chunk_size=512, pair_chunk=2,
    )
    terms = general.energy_terms
    samples = {s: basis.sample(states[s].q)[0] for s in ("n", "p")}
    # The reference state is sufficiently dilute that the actual density
    # dependence matters. Recompute the same physical local vertices from the
    # sampled current rather than substituting constant couplings.
    from .currents import invariant_baryon_density
    local = {s: local_bilinears(samples[s], states[s].rho_active)
             for s in ("n", "p")}
    total_current = local["n"]["baryon_current"] + local["p"]["baryon_current"]
    rho_b = invariant_baryon_density(total_current)
    zeta = rho_b / params.rho_sat
    rho_vertex = params.rho_tensor_vertex * torch.exp(-params.a_rho_tensor * zeta)
    pion_vertex = params.f_pi * torch.exp(-params.a_pi * zeta) * params.hbar_c / params.m_pi
    contact_ref = {
        "rho_tensor": _rhfe_contact_traces(
            basis, states, rho_vertex, kind="rho_tensor", hbar_c=params.hbar_c,
        ),
        "pion_pv": _rhfe_contact_traces(
            basis, states, pion_vertex, kind="pion_pv", hbar_c=params.hbar_c,
        ),
    }
    rho_tensor_m = sum((terms[k] for k in (
        "rho_tensor_direct", "rho_tensor_fock", "rho_tensor_contact",
    )), torch.zeros_like(general.total_energy))
    rho_tensor_s = spherical.direct_terms["E_rho_tensor_direct"] + spherical.exchange_terms[
        "E_rho_tensor_fock"
    ]
    pion_m = sum((terms[k] for k in (
        "pion_pv_direct", "pion_pv_fock", "pion_pv_contact",
    )), torch.zeros_like(general.total_energy))
    pion_s = spherical.exchange_terms["E_pion_fock"]
    out = {}
    for prefix, m_value, s_value in (
        ("rho_tensor", rho_tensor_m, rho_tensor_s),
        ("pion_pv", pion_m, pion_s),
    ):
        abs_error = (m_value - s_value).abs()
        rel_error = abs_error / s_value.abs().clamp_min(1e-12)
        out[prefix] = {
            "general_m_mev": float(m_value.detach()),
            "spherical_mev": float(s_value.detach()),
            "abs_error_mev": float(abs_error.detach()),
            "relative_error": float(rel_error.detach()),
        }
    out["rho_tensor"].update({
        "finite_range_direct_mev": float(terms["rho_tensor_direct"].detach()),
        "finite_range_fock_mev": float(terms["rho_tensor_fock"].detach()),
        "h_delta_contact_mev": float(terms["rho_tensor_contact"].detach()),
        "h_delta_reference_mev": float(contact_ref["rho_tensor"]["total"].detach()),
        "h_delta_abs_error_mev": float((terms["rho_tensor_contact"]
                                         - contact_ref["rho_tensor"]["total"]).abs().detach()),
        "h_delta_direct_reference_mev": float(contact_ref["rho_tensor"]["direct"].detach()),
        "h_delta_exchange_reference_mev": float(contact_ref["rho_tensor"]["exchange"].detach()),
        "general_m_direct_plus_hdelta_mev": float((
            terms["rho_tensor_direct"] + contact_ref["rho_tensor"]["direct"]
        ).detach()),
        "spherical_direct_mev": float(spherical.direct_terms[
            "E_rho_tensor_direct"
        ].detach()),
        "general_m_fock_plus_hdelta_mev": float((
            terms["rho_tensor_fock"] + contact_ref["rho_tensor"]["exchange"]
        ).detach()),
        "spherical_fock_mev": float(spherical.exchange_terms[
            "E_rho_tensor_fock"
        ].detach()),
    })
    out["pion_pv"].update({
        "finite_range_direct_mev": float(terms["pion_pv_direct"].detach()),
        "finite_range_fock_mev": float(terms["pion_pv_fock"].detach()),
        "h_delta_contact_mev": float(terms["pion_pv_contact"].detach()),
        "h_delta_reference_mev": float(contact_ref["pion_pv"]["total"].detach()),
        "h_delta_abs_error_mev": float((terms["pion_pv_contact"]
                                         - contact_ref["pion_pv"]["total"]).abs().detach()),
        "h_delta_direct_reference_mev": float(contact_ref["pion_pv"]["direct"].detach()),
        "h_delta_exchange_reference_mev": float(contact_ref["pion_pv"]["exchange"].detach()),
        "general_m_direct_plus_hdelta_mev": float((
            terms["pion_pv_direct"] + contact_ref["pion_pv"]["direct"]
        ).detach()),
        "general_m_fock_plus_hdelta_mev": float((
            terms["pion_pv_fock"] + contact_ref["pion_pv"]["exchange"]
        ).detach()),
        "spherical_fock_mev": float(spherical.exchange_terms[
            "E_pion_fock"
        ].detach()),
    })
    out["case"] = {
        "state": "one neutron 1s closed doublet, proton vacuum",
        "radius_fm": radius_fm,
        "spacing_fm": spacing_fm,
        "kappa_max": basis.kappa_max,
        "lmax": 2,
        "derivative_order": derivative_order,
        "adf_points": 5 if derivative_order in (4, 5) else 7,
        "adf_formal_accuracy": "fourth order" if derivative_order in (4, 5) else "sixth order",
        "contact_reference": "RHF-e Eqs. (3.205), (3.207), and (3.294), evaluated from radial bilinears",
        "finite_range_comparison": "legacy spherical PKA1 angular kernels; diagnostic only, not independently certified",
    }
    rho_direct_m = terms["rho_tensor_direct"] + contact_ref["rho_tensor"]["direct"]
    rho_direct_s = spherical.direct_terms["E_rho_tensor_direct"]
    rho_fock_m = terms["rho_tensor_fock"] + contact_ref["rho_tensor"]["exchange"]
    rho_fock_s = spherical.exchange_terms["E_rho_tensor_fock"]
    pi_fock_m = terms["pion_pv_fock"] + contact_ref["pion_pv"]["exchange"]
    pi_fock_s = spherical.exchange_terms["E_pion_fock"]
    out["cross_implementation_components"] = {
        "rho_tensor_direct_abs_error_mev": float((rho_direct_m-rho_direct_s).abs().detach()),
        "rho_tensor_fock_abs_error_mev": float((rho_fock_m-rho_fock_s).abs().detach()),
        "pion_pv_fock_abs_error_mev": float((pi_fock_m-pi_fock_s).abs().detach()),
        "interpretation": "differences are dominated by finite-range Fock comparison, not H-delta trace",
    }
    out["numerical_path_differences"] = [
        "general-m radial quadrature uses uniform h*r^2 weights on interior points; legacy spherical Fock uses Simpson weights on the interior mesh",
        "both use the same box radius and interior radii, but the spherical grid also carries endpoint entries; its direct measure is rectangular h*r^2",
        f"general-m Green derivatives use the parity-extended {5 if derivative_order in (4, 5) else 7}-point ADF and projected lmax=2 kernel; spherical exchange uses analytic folded multipole derivatives",
        "general-m keeps differentiated finite-range kernel and adds H-delta explicitly; spherical pion/rho-tensor kernels contain diagonal contact/compensation entries",
        "one spacing refinement reduces the observed difference, but does not identify which quadrature/derivative/contact-distribution detail dominates",
    ]
    return out
