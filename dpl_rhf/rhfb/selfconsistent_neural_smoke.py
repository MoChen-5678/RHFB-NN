"""Guarded small-box integration fixture for joint neural odd/even RHFB.

This exercises the actual ``GeneralizedDensityNet`` and joint PKA1+D1S
training path on a deliberately small algebraic box.  It is an API/integration
smoke run, not a physical O17 calculation, full-space reference, or no-sea
branch certification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from .execution_limits import require_bounded_cgroup
from .runtime_resources import resource_snapshot


FIXTURE = {
    "scope": "algebraic small-box joint neural RHFB integration fixture; no physical result",
    "basis": {
        "radius_fm": 1.4,
        "spacing_fm": 0.2,
        "kappa_max": 1,
        "n_theta": 3,
        "n_phi": 6,
        "derivative_order": 5,
        "expected_primitive_dimension": 48,
        "expected_coordinate_point_count": 108,
    },
    "active_dimension": 40,
    "active_dimension_rule": "max(40, N+32) for both species in this fixture",
    "number_targets": {"n": 3, "p": 2},
    "neutron_reference": "odd particle-addition block at active orbital 2 from a two-particle Slater reference",
    "proton_reference": "even two-particle Slater reference",
    "network": {
        "hidden": 8,
        "depth": 1,
        "radial_basis_size": 4,
        "channel_chunk_size": 1,
        "pairing_seed": 0.005,
        "correction_scale": 0.01,
        "torch_seed": 17032026,
    },
    "optimizer": {
        "updates_default": 1,
        "learning_rate_default": 1.0e-5,
        "number_weight": 1.0e4,
        "normal_tile_size": 3,
        "normal_chunk_size": 8,
        "residual_chunk_size": 2,
        "pairing_chunk_size": 1,
        "pairing_active_chunk_size": 1,
        "lmax": 2,
        "train_chemical_potentials": True,
    },
    "physics": "complete PKA1 particle-hole functional and unscaled full-space Gaussian Gogny D1S pairing",
    "device": "cpu",
    "fortran_used": False,
    "eigensolver_used": False,
    "basis_refreshes": 0,
    "physical_result_accepted": False,
    "no_sea_branch_certified": False,
    "reference_limit": "no independent full-space reference supplied; the M=40 chart is only an integration fixture",
}


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _analytic_initial_q(basis, active_dimension: int, device: torch.device) -> torch.Tensor:
    """Sample deterministic channel-aware analytic factors and reduced-QR them."""
    rows = torch.arange(basis.dimension, dtype=torch.float64, device=device)
    radial = torch.as_tensor(basis.x / basis.radius_fm, dtype=torch.float64, device=device)
    envelope = torch.empty(basis.dimension, dtype=torch.float64, device=device)
    from dpl_rhf.rhf.dirac import RHFRadialDiracOperator

    for channel in basis.channels:
        upper_l, lower_l = RHFRadialDiracOperator.component_angular_momenta(channel.kappa)
        start = channel.offset
        stop = start + 2 * basis.n_radial
        upper = radial.pow(upper_l + 1) * torch.exp(-1.5 * radial)
        lower = radial.pow(lower_l + 1) * torch.exp(-1.5 * radial)
        envelope[start : start + basis.n_radial] = upper
        envelope[start + basis.n_radial : stop] = lower

    columns = torch.arange(1, active_dimension + 1, dtype=torch.float64, device=device)
    # Distinct Fourier samples are multiplied by regular-origin radial factors.
    # The fixed row-dependent channel phases break artificial channel copies.
    phase = (2.0 * math.pi / basis.dimension) * rows[:, None] * columns[None, :]
    channel_phase = torch.as_tensor(
        [((index * 7) % 13) / 13.0 for index in range(basis.dimension)],
        dtype=torch.float64,
        device=device,
    )
    raw = envelope[:, None] * torch.exp(1j * (phase + channel_phase[:, None]))
    q, r = torch.linalg.qr(raw.to(torch.complex128), mode="reduced")
    if torch.any(torch.abs(torch.diagonal(r)) < 1.0e-12):
        raise RuntimeError("analytic fixture factors lost rank during reduced QR")
    return q


def _make_models(basis, *, hidden: int, depth: int, radial_basis_size: int,
                 channel_chunk_size: int, device: torch.device):
    from .manifold import slater_reference
    from .network import GeneralizedDensityNet

    active = int(FIXTURE["active_dimension"])
    q_n = _analytic_initial_q(basis, active, device)
    q_p = _analytic_initial_q(basis, active, device)
    labels = tuple(f"analytic_m{index:02d}" for index in range(active))

    n_occupations = torch.zeros(active, dtype=torch.float64, device=device)
    n_occupations[:2] = 1.0
    n_reference = slater_reference(n_occupations, blocked_index=2, blocking_operation="particle")
    p_occupations = torch.zeros(active, dtype=torch.float64, device=device)
    p_occupations[:2] = 1.0
    p_reference = slater_reference(p_occupations)

    model_options = {
        "hidden": hidden,
        "depth": depth,
        "radial_basis_size": radial_basis_size,
        "channel_chunk_size": channel_chunk_size,
        "pairing_seed": float(FIXTURE["network"]["pairing_seed"]),
        "correction_scale": float(FIXTURE["network"]["correction_scale"]),
    }
    return {
        "n": GeneralizedDensityNet(
            basis, q_n, labels, n_reference, blocked_index=2, **model_options
        ),
        "p": GeneralizedDensityNet(basis, q_p, labels, p_reference, **model_options),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    args = parser.parse_args(argv)
    if args.updates < 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("updates must be nonnegative and learning-rate must be finite and positive")

    # The launcher supplies the kernel-enforced cgroup. This explicit check
    # precedes creation of the grid, Q factors, networks, and optimizer state.
    guard = require_bounded_cgroup()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    report: dict[str, Any] = {
        **FIXTURE,
        "status": "initializing",
        "updates_requested": int(args.updates),
        "learning_rate": float(args.learning_rate),
        "resource_guard": {
            "enforced": bool(guard.get("enforced", False)),
            "memory_max_bytes": guard.get("memory_max_bytes"),
            "memory_swap_max_bytes": guard.get("memory_swap_max_bytes"),
        },
        "device": "cpu",
        "started_unix": time.time(),
    }
    sources = [
        Path(__file__).resolve(),
        Path(__file__).with_name("basis.py").resolve(),
        Path(__file__).with_name("manifold.py").resolve(),
        Path(__file__).with_name("network.py").resolve(),
        Path(__file__).with_name("selfconsistent_neural_training.py").resolve(),
        Path(__file__).with_name("selfconsistent_stationarity.py").resolve(),
        Path(__file__).with_name("streamed_normal_functional.py").resolve(),
    ]
    sources += [Path(__file__).with_name(name).resolve() for name in (
        "streamed_field_autograd.py", "gaussian_pairing.py", "pairing.py",
        "factorized_commutator.py", "operators.py", "currents.py",
        "coordinate_normal_functional.py", "functional.py", "spatial.py",
        "execution_limits.py", "runtime_resources.py", "parity_diagnostic.py")]
    sources += [Path(__file__).parent.parent / "rhf" / name
                for name in ("parameters.py", "dirac.py")]
    snapshot = output / "source_begin"
    snapshot.mkdir()
    for index, path in enumerate(sources):
        shutil.copy2(path, snapshot / f"{index:02d}_{path.name}")
    report["source_snapshot"] = str(snapshot)
    report["source_sha256"] = {str(path.name): _sha256(path) for path in sources}
    config_bytes = json.dumps(report, sort_keys=True, separators=(",", ":"), default=str).encode()
    report["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    _write_json(output / "provenance.json", report)

    try:
        from .basis import AngularRadialBasis
        from .selfconsistent_neural_training import train_joint

        torch.set_num_threads(1)
        torch.manual_seed(int(FIXTURE["network"]["torch_seed"]))
        device = torch.device("cpu")
        basis = AngularRadialBasis(
            radius_fm=1.4,
            spacing_fm=0.2,
            kappa_max=1,
            n_theta=3,
            n_phi=6,
            derivative_order=5,
        )
        if basis.dimension != 48 or basis.point_count != 108:
            raise RuntimeError(
                f"fixture grid mismatch: D={basis.dimension}, P={basis.point_count}; expected D=48, P=108"
            )
        models = _make_models(
            basis,
            hidden=8,
            depth=1,
            radial_basis_size=4,
            channel_chunk_size=1,
            device=device,
        )
        if any(model.active_dimension != 40 for model in models.values()):
            raise RuntimeError("fixture models must both use active M=40")

        summary = train_joint(
            models,
            basis,
            {"n": 3, "p": 2},
            int(args.updates),
            output / "training",
            train_chemical_potentials=True,
            learning_rate=float(args.learning_rate),
            number_weight=1.0e4,
            normal_tile_size=3,
            normal_chunk_size=8,
            residual_chunk_size=2,
            pairing_chunk_size=1,
            pairing_active_chunk_size=1,
            lmax=2,
        )
        report["basis_metadata"] = {
            "dimension": int(basis.dimension),
            "point_count": int(basis.point_count),
            "active_dimension": 40,
            "radius_fm": basis.radius_fm,
            "spacing_fm": basis.spacing_fm,
            "kappa_max": basis.kappa_max,
            "n_theta": basis.n_theta,
            "n_phi": basis.n_phi,
            "derivative_order": basis.derivative_order,
        }
        report["training_summary"] = summary
        report["status"] = "complete" if summary.get("status") == "complete" else "incomplete"
        report["source_files_unchanged"] = all(
            _sha256(path) == report["source_sha256"][path.name] for path in sources)
        if not report["source_files_unchanged"]:
            report["status"] = "incomplete_source_changed_during_training"
        report["physical_result_accepted"] = False
        report["no_sea_branch_certified"] = False
        report["reference_limit"] = (
            "no independent full-space reference supplied; M=40 is only an integration fixture"
        )
        report["runtime_resource_snapshot"] = resource_snapshot()
        report["elapsed_seconds"] = time.time() - report["started_unix"]
        _write_json(output / "smoke_report.json", report)
        return 0 if report["status"] == "complete" else 2
    except Exception as error:
        report["status"] = "incomplete"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        report["physical_result_accepted"] = False
        report["no_sea_branch_certified"] = False
        report["runtime_resource_snapshot"] = resource_snapshot()
        report["elapsed_seconds"] = time.time() - report["started_unix"]
        _write_json(output / "smoke_report.json", report)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
