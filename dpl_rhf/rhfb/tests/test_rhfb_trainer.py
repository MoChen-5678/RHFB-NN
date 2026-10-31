from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dpl_rhf.rhfb.config import BlockedRHFBConfig
from dpl_rhf.rhfb.manifold import GeneralizedDensity, generalized_density_matrix
from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.operators import DensityFactors
from dpl_rhf.rhfb.result import save_observables
from dpl_rhf.rhfb.trainer import (
    BlockingSpec,
    REQUIRED_PKA1_CHANNELS,
    reference_for_species,
    resolve_blocking,
    prepare_networks,
    train_rhfb,
    update_blocked_mode_tracker,
    validate_functional_evaluation,
    validate_no_sea_report,
    _TrainingEigenCounter,
)


def _audited_evaluation():
    channels = {
        key: {
            "implemented": True,
            "energy_term_names": [key],
            "energy_identity_residual": 0.0,
            "identity_tolerance": 1.0e-8,
            "directional_fd_residual": 0.0,
            "directional_fd_tolerance": 1.0e-6,
            "gradient_finite": True,
        }
        for key in REQUIRED_PKA1_CHANNELS
    }
    energy = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    return {
        "total_energy": energy,
        "energy_terms": {"kinetic": energy},
        "stationarity": lambda _states: {"full_rms_mev": 0.0},
        "audit": {
            "complete": True,
            "channels": channels,
            "total_energy_identity_residual": 0.0,
            "total_energy_identity_tolerance": 1.0e-8,
            "no_sea_status": "passed",
            "normal_branch_status": "passed",
            "anomalous_sea_branch_status": "passed",
            "independent_physical_reference_status": "passed",
            "rearrangement_status": "passed",
        },
    }


class TestBlockedRHFBTrainer(unittest.TestCase):
    def test_training_eigensolver_counter_observes_actual_torch_eigh(self):
        counter = _TrainingEigenCounter(primitive_dimension=4, active_dimensions={"n": 2})
        matrix = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
        with counter:
            torch.linalg.eigh(matrix)
        self.assertEqual(counter.counts["training_dirac_eigensolves"], 1)
        self.assertEqual(counter.counts["training_bogoliubov_eigensolves"], 0)
        self.assertEqual(counter.operations[0]["operation"], "aten._linalg_eigh.default")

    def test_o17_auto_selects_requested_d5_2_m_plus_five_half(self):
        labels = (
            "1s1/2:kappa=-1:m2=-1", "1s1/2:kappa=-1:m2=1",
            "1p3/2:kappa=-2:m2=-3", "1p3/2:kappa=-2:m2=-1",
            "1p3/2:kappa=-2:m2=1", "1p3/2:kappa=-2:m2=3",
            "1p1/2:kappa=1:m2=-1", "1p1/2:kappa=1:m2=1",
            "1d5/2:kappa=-3:m2=5",
        )
        block = resolve_blocking(labels, 9, "auto", species="n", z=8, n=9)
        self.assertEqual((block.label, block.kappa, block.m2, block.index),
                         (labels[-1], -3, 5, 8))
        reference, index = reference_for_species(labels, 9, block)
        self.assertEqual(index, 8)
        self.assertEqual(reference.fermion_parity, -1)
        self.assertEqual(float(reference.rho[index, index].real), 1.0)
        self.assertEqual(float(torch.trace(reference.rho).real), 9.0)

    def test_full_functional_audit_requires_traceable_channel_closures(self):
        self.assertEqual(validate_functional_evaluation(_audited_evaluation()), [])
        evaluation = _audited_evaluation()
        evaluation["audit"]["channels"]["rho_tv_fock"]["directional_fd_residual"] = 1.0
        failures = validate_functional_evaluation(evaluation)
        self.assertTrue(any("rho_tv_fock" in failure for failure in failures))

    def test_total_energy_must_remain_differentiable(self):
        evaluation = _audited_evaluation()
        evaluation["total_energy"] = evaluation["total_energy"].detach()
        failures = validate_functional_evaluation(evaluation)
        self.assertTrue(any("detached" in failure for failure in failures))

    def test_blocked_tracking_detects_particle_hole_partner_switch_with_fixed_q(self):
        q_before = torch.eye(2, dtype=torch.complex128)
        q_after = q_before.clone()
        previous = {"n": torch.tensor([1, 0, 0, 0], dtype=torch.complex128), "p": None}
        switched = {"n": torch.tensor([0, 0, 1, 0], dtype=torch.complex128), "p": None}
        self.assertTrue(torch.equal(q_before, q_after))
        with self.assertRaisesRegex(RuntimeError, "Nambu-mode overlap"):
            update_blocked_mode_tracker(previous, switched)

    def test_prevalidation_algorithm_can_pass_while_physical_equivalence_remains_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            from dpl_rhf.rhfb.trainer import REQUIRED_NO_SEA_ALGORITHM_GATES
            gates = {key: True for key in REQUIRED_NO_SEA_ALGORITHM_GATES}
            gates["density_dependent_self_consistent_equivalence"] = False
            path.write_text(json.dumps({"status": "failed", "gates": gates}),
                            encoding="utf-8")
            prevalidation_ready, evidence = validate_no_sea_report(path)
        self.assertTrue(prevalidation_ready)
        self.assertEqual(evidence["moving_projector_physical_equivalence"], "failed")

    def test_training_checkpoint_resumes_optimizer_and_blocked_mode_history(self):
        class Net(nn.Module):
            def __init__(self, dimension, odd):
                super().__init__()
                self.dimension = dimension
                self.odd = odd
                self.anchor = nn.Parameter(torch.tensor(1.0 + 0.1j, dtype=torch.complex128))
                self.q = torch.eye(dimension, dtype=torch.complex128)
                rho = torch.eye(dimension, dtype=torch.complex128)
                if odd:
                    rho[0, 0] = 1.0
                chi = torch.zeros_like(rho)
                self.rho = rho
                self.chi = chi
                self.gd = GeneralizedDensity(rho, chi, generalized_density_matrix(rho, chi),
                                             -1 if odd else 1)

            def forward_with_blocked_mode(self):
                mode = None
                if self.odd:
                    mode = torch.zeros(2 * self.dimension, dtype=torch.complex128)
                    mode[0] = 1.0
                return self.q, self.gd, mode

        class Basis:
            radius_fm = 1.0
            spacing_fm = 0.1
            kappa_max = 1
            dimension = 2
            n_radial = 9
            channels = (1,)

        class Interrupted(RuntimeError):
            pass

        config = BlockedRHFBConfig(z=2, n=1, radius_fm=1.0, spacing_fm=0.1,
                                   kappa_max=1, active_n=1, active_p=2, epochs=2)
        networks = {"n": {"network": Net(1, True)}, "p": {"network": Net(2, False)}}
        active_networks = networks
        blocks = {"n": BlockingSpec("1d5/2:kappa=-3:m2=5", 0, -3, 5), "p": None}
        channels = _audited_evaluation()["audit"]["channels"]
        calls = {"count": 0, "interrupted": False}

        def evaluator(_basis, states, *, chunk_size):
            calls["count"] += 1
            if calls["count"] == 3 and not calls["interrupted"]:
                calls["interrupted"] = True
                raise Interrupted("simulated stop after first saved epoch")
            total = sum(torch.abs(active_networks[s]["network"].anchor).square()
                        for s in ("n", "p"))
            return {
                "total_energy": total,
                "energy_terms": {"kinetic": total},
                "stationarity": lambda _states, chunk_size=32: {
                    "primitive_q_projected_gradient_rms": torch.tensor(0.0)},
                "audit": {
                    "complete": True, "channels": channels,
                    "total_energy_identity_residual": 0.0,
                    "total_energy_identity_tolerance": 1.0e-8,
                    "no_sea_status": "experimental",
                    "normal_branch_status": "passed",
                    "anomalous_sea_branch_status": "passed",
                    "independent_physical_reference_status": "unverified",
                    "rearrangement_status": "passed",
                },
            }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(Interrupted):
                train_rhfb(config, Basis(), networks, blocks, evaluator,
                           output_dir=output, checkpoint_every=1,
                           export_observables=False)
            checkpoint_path = output / "checkpoint.pt"
            self.assertTrue(checkpoint_path.is_file())
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["trainer_state"]["epochs_completed"], 1)
            self.assertEqual(payload["trainer_state"]["number_penalty"], 50.0)
            self.assertIsNotNone(payload["trainer_state"]["previous_blocked_mode"]["n"])

            resumed_networks = {"n": {"network": Net(1, True)},
                                "p": {"network": Net(2, False)}}
            active_networks = resumed_networks
            calls["count"] = 0
            train_rhfb(config, Basis(), resumed_networks, blocks, evaluator,
                       output_dir=output, checkpoint_every=1,
                       resume_from=checkpoint_path, export_observables=False)
            lines = (output / "history.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[-1])["epoch"], 2)
            self.assertFalse(json.loads((output / "summary.json").read_text())[
                "validated"])

    def test_observable_export_writes_full_point_fields_and_moments(self):
        config = BlockedRHFBConfig(z=2, n=1, radius_fm=2.0, spacing_fm=0.2,
                                   kappa_max=1, active_n=1, active_p=2)
        basis = AngularRadialBasis(radius_fm=config.radius_fm,
                                   spacing_fm=config.spacing_fm,
                                   kappa_max=config.kappa_max, n_theta=3, n_phi=6)
        networks, _ = prepare_networks(config, basis)
        states = {}
        for species, data in networks.items():
            q, gd, _ = data["network"].forward_with_blocked_mode()
            states[species] = DensityFactors(q, gd.rho, gd.chi)
        with tempfile.TemporaryDirectory() as directory:
            summary = save_observables(basis, states, directory,
                                       targets={"n": 1, "p": 2}, chunk_size=11)
            arrays = np.load(Path(directory) / "observables.npz")
            csv_lines = (Path(directory) / "observables.csv").read_text().splitlines()
        self.assertAlmostEqual(summary["integrated_particle_numbers"]["n"], 1.0, delta=2.0e-5)
        self.assertAlmostEqual(summary["integrated_particle_numbers"]["p"], 2.0, delta=2.0e-5)
        self.assertEqual(arrays["rho_n_fm3"].shape, (basis.point_count,))
        self.assertEqual(arrays["current_total_fm3"].shape, (basis.point_count, 3))
        self.assertEqual(len(csv_lines), basis.point_count + 1)
        self.assertIn("Q20_matter", summary["quadrupole_moments_fm2"])


if __name__ == "__main__":
    unittest.main()
