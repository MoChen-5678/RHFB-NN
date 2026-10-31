"""Bounded full-primitive normal-field diagnostic for a frozen O17 density.

This is an independent fixed-field diagnostic, not an optimizer/solver.  It
retains every saved Q/rho/chi component and obtains columns of the normal PKA1
action by differentiating the complete particle-hole scalar functional.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np
import torch

from .basis import AngularRadialBasis
from .cached_normal_action import evaluate_two_species_normal_columns
from .functional import evaluate_particle_hole_energy
from .operators import DensityFactors


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, obj):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, default=lambda x: float(x), allow_nan=False) + '\n')
    temporary.replace(path)


def _rss_gib() -> float:
    # Linux ru_maxrss is KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024**2


def run(source: Path, output: Path, *, block: int = 4, stop_after_columns: int | None = None):
    source, output = source.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapdir = output / 'source_begin'
    snapdir.mkdir(exist_ok=True)
    module_dir = Path(__file__).resolve().parent
    copied = []
    for name in ('basis.py', 'cached_normal_action.py', 'functional.py', 'operators.py',
                 'green.py', 'spatial.py', 'parameters.py', 'normal_action.py'):
        candidates = [module_dir / name, module_dir.parent / 'rhf' / name]
        src = next((p for p in candidates if p.exists()), None)
        if src is not None:
            shutil.copy2(src, snapdir / src.name)
            copied.append(src)
    shutil.copy2(Path(__file__), snapdir / Path(__file__).name)

    density_path = source / 'density_factors.npz'
    config_path = source / 'run_config.json'
    metadata = json.loads(config_path.read_text())
    cfg = metadata['config']
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    started = time.monotonic()
    report_path = output / 'diagnostic.json'
    report = {
        'status': 'running', 'nucleus': 'O17', 'scope': 'frozen saved density; full primitive normal PKA1 field only',
        'physical_result_accepted': False, 'fortran_used': False, 'neural_updates': 0,
        'source_density': str(density_path), 'input_sha256': sha256(density_path),
        'run_config_sha256': sha256(config_path), 'source_snapshot': str(snapdir),
        'source_hashes': {p.name: sha256(p) for p in copied},
        'basis': {'radius_fm': cfg['radius_fm'], 'spacing_fm': cfg['spacing_fm'],
                  'kappa_max': cfg['kappa_max'], 'n_theta': 7, 'n_phi': 14,
                  'lmax': cfg['lmax'], 'derivative_order': 5, 'adf_points': 5},
        'density_dimensions': {}, 'algorithm': {
            'normal_energy': 'evaluate_particle_hole_energy, complete PKA1 PH terms including density dependence/rearrangement derivative',
            'column_oracle': 'evaluate_two_species_normal_columns; joint n/p tangent gradients',
            'vector_chunk': block, 'pair_chunk': 32, 'fock_checkpoint_blocks': False,
            'density_compression': False, 'spectral_solve': False,
        },
        'resource_limits': {'wall_seconds': 1800, 'peak_rss_gib': 24},
        'progress': {'completed_columns': 0, 'blocks': [], 'peak_rss_gib': _rss_gib()},
        'limitations': [
            'The frozen neural density is not a self-consistent or accepted O17 solution.',
            'Normal field only; D1S anomalous field is not included in this matrix.',
            'A dense field enables independent finite-dimensional sector inspection, not NN training.',
            'The implemented functional still has separate physical-validation limits.',
        ],
    }
    _atomic_json(report_path, report)

    basis = AngularRadialBasis(radius_fm=cfg['radius_fm'], spacing_fm=cfg['spacing_fm'],
        kappa_max=cfg['kappa_max'], n_theta=7, n_phi=14, derivative_order=5)
    with np.load(density_path) as data:
        states = {s: DensityFactors(*[torch.from_numpy(data[f'{s}_{k}'].copy())
                     for k in ('Q', 'rho_active', 'chi_active')]) for s in ('n', 'p')}
    report['density_dimensions'] = {s: {'Q': list(st.q.shape), 'rho': list(st.rho_active.shape),
                                        'chi': list(st.chi_active.shape),
                                        'particle_number': float(st.particle_number().detach())}
                                    for s, st in states.items()}
    normal_energy = lambda b, st: evaluate_particle_hole_energy(
        b, st, chunk_size=128, pair_chunk=32, lmax=cfg['lmax'],
        _checkpoint_fock_blocks=False).total_energy

    d = basis.dimension
    fields = {}
    for s in ('n', 'p'):
        file = output / f'h_{s}_columns.npy'
        if file.exists():
            fields[s] = np.lib.format.open_memmap(file, mode='r+')
        else:
            fields[s] = np.lib.format.open_memmap(file, mode='w+', dtype=np.complex128,
                                                   shape=(d, d))
            fields[s][:] = np.nan + 1j*np.nan
    done_path = output / 'completed_columns.npy'
    done = np.load(done_path).astype(bool) if done_path.exists() else np.zeros(d, dtype=bool)
    identity = torch.eye(d, dtype=torch.complex128)
    for start in range(0, d, block):
        stop = min(d, start + block)
        if done[start:stop].all():
            continue
        if time.monotonic() - started > 1800:
            report['status'] = 'stopped_at_wall_limit_with_partial_columns'
            break
        probes = identity[:, start:stop]
        t0 = time.monotonic()
        columns = evaluate_two_species_normal_columns(
            basis, states, normal_energy, probes, vector_chunk=block, create_graph=False)
        dt = time.monotonic() - t0
        for s in ('n', 'p'):
            fields[s][:, start:stop] = columns[s].detach().cpu().numpy()
            fields[s].flush()
        done[start:stop] = True
        np.save(done_path, done)
        elapsed = time.monotonic() - started
        rec = {'start': start, 'stop': stop, 'seconds': dt, 'elapsed_seconds': elapsed,
               'rss_peak_gib': _rss_gib()}
        report['progress']['blocks'].append(rec)
        report['progress']['completed_columns'] = int(done.sum())
        report['progress']['peak_rss_gib'] = max(report['progress']['peak_rss_gib'], rec['rss_peak_gib'])
        report['estimated_full_seconds_from_mean_block'] = elapsed + (d - int(done.sum())) * dt / (stop - start)
        _atomic_json(report_path, report)
        print(json.dumps(rec), flush=True)
        if stop_after_columns is not None and int(done.sum()) >= stop_after_columns:
            report['status'] = 'pilot_block_completed'
            break
        if rec['rss_peak_gib'] > 24:
            report['status'] = 'stopped_peak_rss_limit'
            break

    for field in fields.values():
        field.flush()
    if done.all():
        results = {}
        for s in ('n', 'p'):
            h = np.asarray(fields[s])
            herm = h - h.conj().T
            results[s] = {
                'hermiticity_max_abs_mev': float(np.max(np.abs(herm))),
                'hermiticity_frobenius_mev': float(np.linalg.norm(herm)),
                'operator_frobenius_mev': float(np.linalg.norm(h)),
                'max_entry_mev': float(np.max(np.abs(h))),
            }
        report['matrix_diagnostics'] = results
        report['status'] = 'completed_full_primitive_normal_fields_physical_review_pending'
    report['progress']['completed_columns'] = int(done.sum())
    report['progress']['peak_rss_gib'] = max(report['progress']['peak_rss_gib'], _rss_gib())
    report['elapsed_seconds'] = time.monotonic() - started
    _atomic_json(report_path, report)
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--block', type=int, default=4)
    ap.add_argument('--stop-after-columns', type=int)
    args = ap.parse_args()
    run(args.source, args.output, block=args.block, stop_after_columns=args.stop_after_columns)
