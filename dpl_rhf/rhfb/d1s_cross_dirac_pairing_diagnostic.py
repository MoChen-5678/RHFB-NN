"""Saved-state D1S pairing action resolved in free Dirac +/- sectors.

This is a read-only diagnostic of one saved neural state.  The free Dirac
projector is only a pre-diagnostic and is not the interacting P_+(h) used by
the separate moving-projector candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .basis import AngularRadialBasis
from .experimental_run import free_sea_blocks
from .gaussian_pairing import GaussianPairingAction
from .operators import DensityFactors
from .parity_diagnostic import number_parity_from_density
from .result import atomic_json


def _apply_free_projector(basis, values, blocks, *, negative: bool):
    """Apply the channel-block free negative or positive Dirac projector."""
    out = torch.empty_like(values)
    for channel in basis.channels:
        lo = channel.offset
        hi = lo + 2 * basis.n_radial
        projector = blocks[channel.kappa]
        if not negative:
            projector = torch.eye(projector.shape[0], dtype=projector.dtype,
                                  device=projector.device) - projector
        out[lo:hi] = projector @ values[lo:hi]
    return out


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(input_directory: str | Path, output_directory: str | Path):
    source = Path(input_directory).resolve()
    out = Path(output_directory).resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.monotonic()
    run_config = json.loads((source / "run_config.json").read_text())
    config = run_config["config"]
    snapshot = source / "source_snapshot" / "rhfb"
    source_names = ("basis.py", "experimental_run.py", "gaussian_pairing.py",
                    "pairing.py", "functional.py", "spatial.py")
    source_hashes = {}
    snapshot_matches = {}
    for name in source_names:
        current = Path(__file__).parent / name
        saved = snapshot / name
        source_hashes[name] = _sha(current)
        snapshot_matches[name] = saved.exists() and _sha(saved) == source_hashes[name]

    basis = AngularRadialBasis(
        radius_fm=config["radius_fm"], spacing_fm=config["spacing_fm"],
        kappa_max=config["kappa_max"], n_theta=7, n_phi=14,
        derivative_order=5,
    )
    states = {}
    with np.load(source / "density_factors.npz") as saved:
        for species in ("n", "p"):
            states[species] = DensityFactors(*[
                torch.from_numpy(saved[f"{species}_{key}"].copy())
                for key in ("Q", "rho_active", "chi_active")
            ])
    with np.load(source / "experimental_fields.npz") as saved:
        field_data = {species: {
            "delta_active_saved": torch.from_numpy(saved[f"{species}_delta_active_mev"].copy()),
            "pair_energy_saved": float(saved[f"{species}_pair_energy_mev"]),
        } for species in ("n", "p")}

    # Track attempts only during the projector/pairing/parity calculations.
    # The fixed angular quadrature was constructed before this guard.
    counts = {name: 0 for name in ("eig", "eigh", "eigvals", "eigvalsh")}
    originals = []
    report = {
        "status": "running",
        "input_directory": str(source),
        "input_density_sha256": _sha(source / "density_factors.npz"),
        "input_fields_sha256": _sha(source / "experimental_fields.npz"),
        "source_hashes_at_diagnostic": source_hashes,
        "source_snapshot_matches": snapshot_matches,
        "source_snapshot_present": snapshot.exists(),
        "fortran_used": False,
        "optimizer_updates": 0,
        "projector_kind": "free Dirac P-/P+ polynomial-sign blocks (pre-diagnostic only)",
        "not_computed": "interacting spectral P_+(h), its density-dependent response, or a self-consistent no-sea solution",
        "spectral_api_attempts": counts,
        "species": {},
    }
    try:
        for library in (torch.linalg, np.linalg):
            for name in counts:
                original = getattr(library, name)
                def reject(*args, _name=name, **kwargs):
                    counts[_name] += 1
                    raise RuntimeError("eigensolver API is forbidden in this saved-state diagnostic: " + _name)
                originals.append((library, name, original))
                setattr(library, name, reject)

        for species, state in states.items():
            parity = number_parity_from_density(
                state.rho_active.numpy(), state.chi_active.numpy())
            pairing = GaussianPairingAction(
                basis, state.q, state.chi_active, species=species,
                chunk_size=4, active_chunk_size=4, lmax=config["lmax"],
                checkpoint_blocks=False,
            )
            with torch.no_grad():
                pminus_blocks, pminus_idempotence = free_sea_blocks(
                    basis, species, state.q.device)
                q = state.q
                qstar = q.conj()
                # The input leg of Delta transforms in the conjugate space.
                pminus_star_qstar = _apply_free_projector(
                    basis, qstar, {k: value.conj() for k, value in pminus_blocks.items()},
                    negative=True)
                pplus_star_qstar = qstar - pminus_star_qstar
                split_action = pairing.apply(torch.cat(
                    (pplus_star_qstar, pminus_star_qstar), dim=1))
                nactive = q.shape[1]
                on_plus_input = split_action[:, :nactive]
                on_minus_input = split_action[:, nactive:]
                dpp = _apply_free_projector(basis, on_plus_input, pminus_blocks,
                                             negative=False)
                dmp = _apply_free_projector(basis, on_plus_input, pminus_blocks,
                                             negative=True)
                dpm = _apply_free_projector(basis, on_minus_input, pminus_blocks,
                                             negative=False)
                dmm = _apply_free_projector(basis, on_minus_input, pminus_blocks,
                                             negative=True)
                delta_qstar = on_plus_input + on_minus_input
                active_gap_raw = q.mH @ delta_qstar
                active_gap = 0.5 * (active_gap_raw - active_gap_raw.T)
                active_external = delta_qstar - q @ active_gap_raw
                saved_gap = field_data[species]["delta_active_saved"]
                gap_difference = active_gap - saved_gap
                chi_norm = torch.linalg.vector_norm(state.chi_active)
                delta_norm = torch.linalg.vector_norm(delta_qstar)
                block_norms = {
                    "positive_output_from_positive_input": torch.linalg.vector_norm(dpp),
                    "negative_output_from_positive_input": torch.linalg.vector_norm(dmp),
                    "positive_output_from_negative_input": torch.linalg.vector_norm(dpm),
                    "negative_output_from_negative_input": torch.linalg.vector_norm(dmm),
                }
                row = {
                    "dimension": basis.dimension,
                    "active_dimension": q.shape[1],
                    "particle_number": state.particle_number(),
                    "number_parity": parity["number_parity"],
                    "number_parity_pfaffian": parity["normalized_pfaffian"],
                    "chi_frobenius": chi_norm,
                    "chi_nonzero": float(chi_norm) > 0.0,
                    "free_negative_projector_idempotence_max": pminus_idempotence,
                    "delta_Q_star_frobenius_mev": delta_norm,
                    "delta_Q_star_external_to_Q_frobenius_mev": torch.linalg.vector_norm(active_external),
                    "external_to_Q_fraction_of_delta_Q_star": torch.linalg.vector_norm(active_external) / delta_norm.clamp_min(1e-300),
                    "active_gap_frobenius_mev": torch.linalg.vector_norm(active_gap),
                    "active_gap_antisymmetry_max_mev": (active_gap_raw + active_gap_raw.T).abs().max(),
                    "active_gap_vs_saved_max_abs_mev": gap_difference.abs().max(),
                    "active_gap_vs_saved_frobenius_mev": torch.linalg.vector_norm(gap_difference),
                    "saved_pair_energy_mev": field_data[species]["pair_energy_saved"],
                    "reconstructed_pair_energy_mev": 0.5 * torch.sum(state.chi_active.conj() * active_gap).real,
                    "cross_dirac_block_frobenius_mev": {
                        "positive_output_from_negative_input": block_norms["positive_output_from_negative_input"],
                        "negative_output_from_positive_input": block_norms["negative_output_from_positive_input"],
                    },
                    "all_four_sector_block_frobenius_mev": block_norms,
                    "all_four_block_squares_sum_mev2": sum(v.square() for v in block_norms.values()),
                    "four_block_norms_over_full_delta_Q_star": {
                        key: value / delta_norm.clamp_min(1e-300)
                        for key, value in block_norms.items()
                    },
                    "free_negative_component_of_Qstar_frobenius": torch.linalg.vector_norm(pminus_star_qstar),
                    "free_positive_component_of_Qstar_frobenius": torch.linalg.vector_norm(pplus_star_qstar),
                    "interpretation_limit": "action sampled on the saved active hole-frame only; free +/- labels are not interacting no-sea sectors",
                }
                report["species"][species] = row
                np.savez_compressed(
                    out / f"{species}_d1s_pairing_sector_action.npz",
                    delta_Q_star=delta_qstar.numpy(), active_gap=active_gap.numpy(),
                    active_external=active_external.numpy(),
                    delta_pp=dpp.numpy(), delta_mp=dmp.numpy(),
                    delta_pm=dpm.numpy(), delta_mm=dmm.numpy(),
                    source_positive_conjugate=pplus_star_qstar.numpy(),
                    source_negative_conjugate=pminus_star_qstar.numpy(),
                )
        report["status"] = "free_dirac_sector_pairing_diagnostic_completed_not_a_no_sea_or_moving_projector_verification"
    except Exception as exc:
        report["status"] = "failed_with_partial_data"
        report["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for library, name, original in reversed(originals):
            setattr(library, name, original)
        report["spectral_api_attempts"] = counts
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(out / "diagnostic.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    run(args.input_directory, args.output_directory)
