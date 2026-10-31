import json

import pytest
import torch

from dpl_rhf.cli.rhfb_train import main
from dpl_rhf.rhfb.checkpoint import load_checkpoint, save_checkpoint
from dpl_rhf.rhfb.config import BlockedRHFBConfig
from dpl_rhf.rhfb.prescription import current_neural_prescription


def test_odd_and_odd_odd_parities_and_incompatible_block_rejected():
    assert BlockedRHFBConfig(8, 9).number_parities == {"n": 1, "p": 0}
    assert BlockedRHFBConfig(9, 9).number_parities == {"n": 1, "p": 1}
    with pytest.raises(ValueError, match="requires"):
        BlockedRHFBConfig(8, 9, block_n="none")
    with pytest.raises(ValueError, match="even number parity"):
        BlockedRHFBConfig(8, 8, block_n="1d5/2,m=1/2")


def test_checkpoint_preserves_complex_state_and_rejects_wrong_space(tmp_path):
    config = BlockedRHFBConfig(8, 9)
    basis = {"kappa_max": 4, "active_n": 40, "active_p": 40}
    reference = {"n": "analytic-mode-8", "p": None}
    physics_prescription = current_neural_prescription(
        basis_spec=json.dumps(basis, sort_keys=True),
        blocking=json.dumps(reference, sort_keys=True),
    )
    tensor = torch.tensor([[1 + 2j, -3j]], dtype=torch.complex128)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, config, basis_spec=basis, blocked_reference=reference,
                    physics_prescription=physics_prescription,
                    network_state={"factor": tensor}, optimizer_state={"step": 7})
    loaded = load_checkpoint(path, config, basis_spec=basis, blocked_reference=reference,
                             physics_prescription=physics_prescription)
    assert torch.equal(loaded["network_state"]["factor"], tensor)
    assert loaded["optimizer_state"]["step"] == 7
    with pytest.raises(ValueError, match="space mismatch"):
        load_checkpoint(path, config, basis_spec={**basis, "active_n": 64}, blocked_reference=reference,
                        physics_prescription=physics_prescription)
    with pytest.raises(ValueError, match="blocked-reference"):
        load_checkpoint(path, config, basis_spec=basis, blocked_reference={"n": "other", "p": None},
                        physics_prescription=physics_prescription)
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_checkpoint(path, BlockedRHFBConfig(8, 11), basis_spec=basis, blocked_reference=reference,
                        physics_prescription=physics_prescription)


def test_uncertified_nuclear_run_fails_without_fabricated_energy(tmp_path):
    with pytest.raises(SystemExit):
        main(["--z", "8", "--n", "9", "--out", str(tmp_path)])
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["status"] == "blocked"
    assert report["physical_result"] is None
    assert all(count == 0 for count in report["solver_counts"].values())


def test_unclosed_functional_preflight_stops_before_large_basis(tmp_path, monkeypatch):
    from dpl_rhf.rhfb.trainer import REQUIRED_NO_SEA_ALGORITHM_GATES

    gate_path = tmp_path / "algorithm-gates.json"
    gate_path.write_text(json.dumps({
        "status": "prevalidated",
        "gates": {key: True for key in REQUIRED_NO_SEA_ALGORITHM_GATES},
    }))

    def forbidden_basis(*args, **kwargs):
        raise AssertionError("basis construction must wait for complete functional closure")

    monkeypatch.setattr("dpl_rhf.rhfb.basis.AngularRadialBasis", forbidden_basis)
    out = tmp_path / "preflight"
    with pytest.raises(SystemExit) as exit_info:
        main(["--z", "8", "--n", "9", "--no-sea-report", str(gate_path),
              "--out", str(out)])
    assert exit_info.value.code == 2
    report = json.loads((out / "summary.json").read_text())
    assert report["status"] == "blocked"
    assert report["initialization"]["network_constructed"] is False
    assert report["gates"]["complete_pka1_energy_and_field"]["readiness"]["status"] == "blocked"
