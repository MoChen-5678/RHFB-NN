"""Post-variation microscopic center-of-mass correction algebra.

This module evaluates the finite-dimensional Wick contractions only.  It
requires callers to supply Hermitian active-space momentum matrices and an
independently constructed active-space ``p^2`` matrix.  It intentionally does
not construct momentum operators from the O17 ADF basis and is not connected
to the self-consistent energy or training path.

The prescription follows Geng and Long, arXiv:2202.00482, Eq. (41) and
Appendix C: the correction is reported after variation.  The matrix formula
below generalizes the Wick expectation to arbitrary (including blocked odd)
quasiparticle vacua in the repository convention ``chi_ab=<c_b c_a>``.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np


MomentumUnits = Literal["MeV", "fm^-1"]


def evaluate_center_of_mass_correction(
    densities: Mapping[str, tuple[Any, Any]],
    momentum_active: Any,
    momentum_squared_active: Any,
    *,
    mass_number: int,
    nucleon_mass_mev: float,
    momentum_units: MomentumUnits,
    hbar_c_mev_fm: float = 197.3269804,
) -> dict[str, Any]:
    r"""Evaluate ``E_cm = -<P_cm^2>/(2 A M)`` from supplied active matrices.

    Args:
        densities: Exactly ``{"n": (rho_n, chi_n), "p": (rho_p, chi_p)}``.
            The convention is ``rho[a,b]=<c_b^dagger c_a>`` and
            ``chi[a,b]=<c_b c_a>``.  Proton-neutron normal coherence and
            proton-neutron anomalous density are outside this interface.
        momentum_active: Hermitian ``p_mu`` matrices, either common with shape
            ``(3,M,M)`` or a mapping ``{"n": p_n, "p": p_p}`` of species-
            projected arrays. Species mappings are needed when neutron and
            proton active frames differ. Matrices include all desired units.
        momentum_squared_active: Independently evaluated Hermitian
            ``<a|p^2|b>`` matrix, common with shape ``(M,M)`` or supplied as a
            species mapping. It is never inferred as ``sum_mu p_mu @ p_mu``;
            doing so can insert a truncated intermediate-state projector.
        mass_number: Positive integer ``A``.
        nucleon_mass_mev: Positive mass ``M`` in MeV.
        momentum_units: Explicitly ``"MeV"`` (natural ``c=1``) or
            ``"fm^-1"``.  The latter multiplies ``<P^2>`` by
            ``(hbar_c_mev_fm)^2`` in the energy prefactor.
        hbar_c_mev_fm: Conversion used only for ``momentum_units="fm^-1"``.

    Returns:
        A serializable dictionary with signed contribution terms, their
        Cartesian/species decomposition, ``<P_cm^2>``, and post-variation
        ``E_cm`` in MeV.  The mean-momentum square uses the total neutron plus
        proton mean, so its neutron-proton cross term is retained.
    """
    if set(densities) != {"n", "p"}:
        raise ValueError("densities must contain exactly neutron and proton rho/chi pairs")
    if momentum_units not in {"MeV", "fm^-1"}:
        raise ValueError("momentum_units must be exactly 'MeV' or 'fm^-1'")
    if isinstance(mass_number, bool) or int(mass_number) != mass_number or mass_number < 1:
        raise ValueError("mass_number must be a positive integer")
    if not np.isfinite(nucleon_mass_mev) or nucleon_mass_mev <= 0:
        raise ValueError("nucleon_mass_mev must be finite and positive")
    if not np.isfinite(hbar_c_mev_fm) or hbar_c_mev_fm <= 0:
        raise ValueError("hbar_c_mev_fm must be finite and positive")

    def species_operator(value, *, is_vector: bool, name: str):
        raw = ({s: value[s] for s in ("n", "p")} if isinstance(value, Mapping)
               else {s: value for s in ("n", "p")})
        result = {s: np.asarray(raw[s], dtype=np.complex128) for s in ("n", "p")}
        for s in ("n", "p"):
            item = result[s]
            if is_vector:
                valid_shape = (item.ndim == 3 and item.shape[0] == 3
                               and item.shape[1] == item.shape[2])
                hermitian = (valid_shape and np.allclose(
                    item, item.conj().transpose(0, 2, 1), atol=1e-10, rtol=0))
            else:
                valid_shape = item.ndim == 2 and item.shape[0] == item.shape[1]
                hermitian = (valid_shape and np.allclose(
                    item, item.conj().T, atol=1e-10, rtol=0))
            if not valid_shape or not hermitian:
                shape = "(3,M,M)" if is_vector else "(M,M)"
                raise ValueError(f"{name}[{s}] must be Hermitian with shape {shape}")
            if not np.isfinite(item).all():
                raise ValueError(f"{name}[{s}] must be finite")
        return result

    p_by_species = species_operator(momentum_active, is_vector=True,
                                    name="momentum_active")
    p2_by_species = species_operator(momentum_squared_active, is_vector=False,
                                     name="momentum_squared_active")

    rho: dict[str, np.ndarray] = {}
    chi: dict[str, np.ndarray] = {}
    for species in ("n", "p"):
        pair = densities[species]
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"{species} density must be a (rho, chi) pair")
        rho[species] = np.asarray(pair[0], dtype=np.complex128)
        chi[species] = np.asarray(pair[1], dtype=np.complex128)
        dimension = p_by_species[species].shape[1]
        if (rho[species].shape != (dimension, dimension)
                or chi[species].shape != (dimension, dimension)
                or p2_by_species[species].shape != (dimension, dimension)):
            raise ValueError(f"{species} rho, chi, p, and p^2 dimensions must agree")
        if not np.isfinite(rho[species]).all() or not np.isfinite(chi[species]).all():
            raise ValueError(f"{species} rho and chi must be finite")
        if not np.allclose(rho[species], rho[species].conj().T, atol=1e-10, rtol=0):
            raise ValueError(f"{species} rho must be Hermitian")
        if not np.allclose(chi[species], -chi[species].T, atol=1e-10, rtol=0):
            raise ValueError(f"{species} chi must be antisymmetric under transpose")

    species_terms: dict[str, dict[str, Any]] = {
        s: {"one_body_p2": 0.0, "normal_exchange": 0.0,
            "anomalous_wick": 0.0, "mean_momentum": []}
        for s in ("n", "p")
    }
    components: list[dict[str, Any]] = []
    # p^2 is the full scalar operator sum_mu p_mu^2, supplied independently;
    # count its expectation once, outside the Cartesian p_mu contractions.
    one_body_by_species = {
        s: float(np.trace(rho[s] @ p2_by_species[s]).real) for s in ("n", "p")
    }
    one_body_total = sum(one_body_by_species.values())
    for s in ("n", "p"):
        species_terms[s]["one_body_p2"] = one_body_by_species[s]
    exchange_total = 0.0
    anomalous_total = 0.0
    mean_square_total = 0.0
    for axis in range(3):
        component_exchange = {
            s: float((-np.trace(rho[s] @ p_by_species[s][axis]
                                 @ rho[s] @ p_by_species[s][axis])).real)
            for s in ("n", "p")
        }
        component_anomalous = {
            s: float(np.trace(chi[s].conj().T @ p_by_species[s][axis]
                              @ chi[s] @ p_by_species[s][axis].T).real)
            for s in ("n", "p")
        }
        component_mean = {
            s: complex(np.trace(rho[s] @ p_by_species[s][axis]))
            for s in ("n", "p")
        }
        total_mean = sum(component_mean.values(), 0.0j)
        component_mean_square = float(abs(total_mean) ** 2)

        exchange_total += sum(component_exchange.values())
        anomalous_total += sum(component_anomalous.values())
        mean_square_total += component_mean_square
        for s in ("n", "p"):
            species_terms[s]["normal_exchange"] += component_exchange[s]
            species_terms[s]["anomalous_wick"] += component_anomalous[s]
            species_terms[s]["mean_momentum"].append({
                "real": float(component_mean[s].real),
                "imag": float(component_mean[s].imag),
            })
        components.append({
            "axis": ("x", "y", "z")[axis],
            "mean_momentum_by_species": {
                s: {"real": float(value.real), "imag": float(value.imag)}
                for s, value in component_mean.items()
            },
            "total_mean_momentum": {
                "real": float(total_mean.real), "imag": float(total_mean.imag),
            },
            "mean_momentum_square_total": component_mean_square,
            "normal_exchange_signed_by_species": component_exchange,
            "anomalous_wick_signed_by_species": component_anomalous,
        })

    momentum_squared = one_body_total + mean_square_total + exchange_total + anomalous_total
    unit_factor = hbar_c_mev_fm**2 if momentum_units == "fm^-1" else 1.0
    prefactor = unit_factor / (2.0 * int(mass_number) * float(nucleon_mass_mev))
    return {
        "prescription": "post-variation microscopic CoM correction; Geng-Long arXiv:2202.00482 Eq.(41), Appendix C Wick form",
        "scope": "algebraic active-space evaluator; no momentum operator construction and no self-consistent coupling",
        "momentum_units": momentum_units,
        "momentum_squared_units": "fm^-2" if momentum_units == "fm^-1" else "MeV^2",
        "mass_number": int(mass_number),
        "nucleon_mass_mev": float(nucleon_mass_mev),
        "hbar_c_mev_fm": float(hbar_c_mev_fm) if momentum_units == "fm^-1" else None,
        "one_body_p2": one_body_total,
        "mean_momentum_square_total": mean_square_total,
        "normal_exchange_signed": exchange_total,
        "anomalous_wick_signed": anomalous_total,
        "momentum_squared_expectation": momentum_squared,
        "one_body_p2_by_species": one_body_by_species,
        "energy_prefactor_mev_per_momentum2": prefactor,
        "energy_correction_mev": -prefactor * momentum_squared,
        "species_terms": species_terms,
        "cartesian_components": components,
        "operator_provenance": "momentum_active and momentum_squared_active are caller-supplied independent matrices, common or species-projected",
    }
