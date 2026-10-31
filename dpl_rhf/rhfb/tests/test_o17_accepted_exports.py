"""Export algebra on a small chart; no self-consistent nucleus is solved."""
from types import SimpleNamespace
import json

import numpy as np
import torch

from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.manifold import slater_reference
from dpl_rhf.rhfb.operators import DensityFactors
from dpl_rhf.rhfb.o17_selfconsistent_neural import (
    UnrestrictedO17DensityNet, _callback_factory, _source_parameter_audits,
)
from pathlib import Path


class _ZeroGap:
    def apply(self, columns):
        return torch.zeros_like(columns)


def test_exported_uv_and_h20_use_the_same_accepted_primitive_density(tmp_path):
    basis = AngularRadialBasis(radius_fm=4., spacing_fm=.5, kappa_max=1,
                               n_theta=3, n_phi=6, derivative_order=5)
    active = 12
    initial_q = torch.eye(basis.dimension, active, dtype=torch.complex128)
    models, states, outputs = {}, {}, {}
    diagnostics, fields, gaps, actions = {}, {}, {}, {}
    for species, target in (("n", 9), ("p", 8)):
        occupations = torch.zeros(active, dtype=torch.float64)
        occupations[:8] = 1.
        reference = slater_reference(occupations, blocked_index=8 if species == "n" else None)
        model = UnrestrictedO17DensityNet(
            basis, initial_q, [str(i) for i in range(active)], reference,
            blocked_index=8 if species == "n" else None, hidden=8, depth=1,
            radial_basis_size=4, pairing_seed=.03)
        models[species] = model
        with torch.no_grad():
            q, gd = model()
        outputs[species] = q, gd
        states[species] = DensityFactors(q, gd.rho, gd.chi)
        fields[species] = torch.eye(basis.dimension, dtype=torch.complex128)
        gaps[species] = torch.zeros((active,active), dtype=torch.complex128)
        actions[species] = _ZeroGap()
        zero = torch.zeros((), dtype=torch.float64)
        diagnostics[species] = {name: zero for name in (
            "full_commutator_frobenius_squared_mev2", "normal_commutator_squared_mev2",
            "anomalous_commutator_squared_mev2", "scaled_residual_loss",
            "number_variance_rho_identity_error")}
        diagnostics[species].update(
            number_error=states[species].particle_number()-target,
            chemical_potential_mev=zero+1., number_tangent_identifiable=zero+1.,
            number_variance=2*gd.chi.abs().square().sum())
    zero = torch.zeros((), dtype=torch.float64)
    result = SimpleNamespace(total_energy=zero, particle_hole_energy=zero,
                             pairing_energy=zero,energy_channels={},
                             normal_fields=fields, active_pairing_gaps=gaps,
                             delta_actions=actions,species_diagnostics=diagnostics)
    callback = _callback_factory(basis,models,tmp_path)
    with torch.no_grad():
        callback(0,result,states,outputs,{"event":"export_algebra_test"})
    arrays = np.load(tmp_path/"accepted_evaluations"/"step_0000.npz")
    report = json.loads((tmp_path/"accepted_evaluations"/"step_0000.json").read_text())
    assert not report["physical_result_accepted"]
    assert not report["no_sea_branch_certified"]
    for species in ("n", "p"):
        prefix = species+"_"
        q, u, v = (arrays[prefix+k] for k in ("Q","bogoliubov_U_active","bogoliubov_V_active"))
        np.testing.assert_allclose(q@(v.conj()@v.T)@q.conj().T,
                                   arrays[prefix+"rho_primitive"],atol=2e-12,rtol=0)
        np.testing.assert_allclose(q@(v.conj()@u.T)@q.T,
                                   arrays[prefix+"chi_primitive"],atol=2e-12,rtol=0)
        # The supplied test field is h=I, Delta=0, hence H20=N20 and mu=1.
        np.testing.assert_allclose(arrays[prefix+"H20_active"],
                                   arrays[prefix+"N20_active"],atol=2e-12,rtol=0)
        np.testing.assert_allclose(arrays[prefix+"constrained_H20_active"],0.,atol=2e-12,rtol=0)


def test_runner_compares_source_tables_without_inheriting_reference_occupations():
    root = Path(__file__).parent/"fixtures"/"source_templates"
    records = _source_parameter_audits([root/"RHFB-BI",root/"RHFB-BI"/"Broyden",root/"S-RHFB-C"])
    assert len(records) == 3
    assert all(record["python_parameter_mapping"]["pka1_matches"] for record in records)
    assert records[0]["python_parameter_mapping"]["d1s_active_source_to_python"]["strength"] is None
    # The source snapshots have zero blocked indices. They are retained as
    # provenance rather than prohibiting the NN's odd-neutron reference.
    assert all(record["template"]["blocking_n"] == () for record in records)
    assert all("not inherited" in record["scope"] for record in records)
