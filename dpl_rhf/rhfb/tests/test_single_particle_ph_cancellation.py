"""One-particle antisymmetry check of the complete implemented PH functional.

Geng--Long 2202.00482 Eqs.20b,c give opposite Hartree/Fock contractions.
For one occupied spinor those contractions are identical at both endpoints,
also with local density-dependent vertices. This is an algebraic consequence,
not a nuclear prediction or a no-sea certification.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.currents import local_bilinears
from dpl_rhf.rhfb.functional import evaluate_particle_hole_energy
from dpl_rhf.rhfb.normal_action import AutogradNormalAction
from dpl_rhf.rhfb.operators import DensityFactors


@pytest.mark.parametrize("species", ["n", "p"])
def test_one_polarized_particle_has_no_ph_self_interaction(species):
    torch.set_num_threads(2)
    basis = AngularRadialBasis(radius_fm=8., spacing_fm=.8, kappa_max=1,
                               n_theta=3, n_phi=6, derivative_order=5)
    frame, _ = basis.build_initial_q(1, species=species, nucleus_a=17,
                                     minimum=2, empty_states=0)
    q = frame[:, :1]
    one = torch.ones((1, 1), dtype=torch.complex128)
    zero = torch.zeros_like(one)
    states = {s: DensityFactors(q, one if s == species else zero, zero)
              for s in ("n", "p")}
    currents = local_bilinears(basis._sample_range(q, 0, basis.point_count), one)
    assert float(currents["axial_current"][:, 1:].abs().max()) > 1e-6
    assert float(currents["baryon_current"][:, 1:].abs().max()) > 1e-6

    def energy(b, factors):
        return evaluate_particle_hole_energy(b, factors, chunk_size=32,
                    pair_chunk=32, lmax=2, _checkpoint_fock_blocks=False)

    evaluation = energy(basis, states)
    terms = {name: float(value.detach())
             for name, value in evaluation.energy_terms.items()}
    pairs = {}
    for channel in ("sigma", "omega", "rho_vector", "rho_tensor", "rho_vt",
                    "rho_tv", "pion_pv", "coulomb"):
        direct, exchange = terms[f"{channel}_direct"], terms[f"{channel}_fock"]
        defect = direct + exchange
        scale = max(1., abs(direct), abs(exchange))
        pairs[channel] = {"direct_mev": direct, "fock_mev": exchange,
                          "sum_mev": defect}
        assert abs(defect) < 2e-11 * scale, (channel, direct, exchange)
    for channel in ("rho_tensor_contact", "pion_pv_contact"):
        assert abs(terms[channel]) < 2e-11, (channel, terms[channel])
    assert abs(terms["pion_pv_direct"]) > 1e-6
    interaction = float(evaluation.total_energy.detach()) - terms["kinetic"]
    assert abs(interaction) < 2e-10

    action = AutogradNormalAction(basis, states,
                    lambda b, factors: energy(b, factors).total_energy,
                    species=species, vector_chunk=1, create_graph=False)
    potential_action = action.apply(q) - basis.apply_kinetic(q, species=species)
    action_defect = float(torch.linalg.vector_norm(potential_action).detach())
    assert action_defect < 2e-9

    directory = os.environ.get("RHFB_SINGLE_PARTICLE_AUDIT_DIR")
    if directory:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"{species}.json").write_text(json.dumps({
            "scope": "one polarized particle; same implemented full PH functional; antisymmetry cancellation only",
            "primary_equations": "https://arxiv.org/pdf/2202.00482 Eqs.20b,c",
            "species": species, "primitive_dimension": basis.dimension,
            "adf_points": 5, "pairing": "absent for a pure one-particle Slater state",
            "hartree_fock_pairs": pairs, "all_energy_terms_mev": terms,
            "interaction_energy_defect_mev": interaction,
            "potential_action_on_occupied_spinor_norm_mev": action_defect,
            "fortran_used": False, "neural_training": False,
            "nuclear_solution_accepted": False,
        }, indent=2) + "\n")
