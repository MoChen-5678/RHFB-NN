"""Same-grid O17 PH field comparison using the bounded-memory backend.

This evaluates the archived neural density, not a new self-consistent state.
It compares every primitive normal-field entry and PH energy channel against
archived independent factorized calculations, without launching the dense
coordinate prototype or recomputing its memory-heavy reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .basis import AngularRadialBasis
from .execution_limits import require_bounded_cgroup
from .full_branch_neural_prototype import _forbid_spectra, _restore_spectra
from .operators import DensityFactors
from .runtime_resources import resource_snapshot
from .streamed_normal_functional import StreamedPrimitiveNormalFunctional


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _write(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def run(source, output, *, energy_summary, energy_density, tile_size=4):
    limits = require_bounded_cgroup()
    torch.set_num_threads(2)
    source, output = Path(source).resolve(), Path(output).resolve()
    energy_summary, energy_density = Path(energy_summary).resolve(), Path(energy_density).resolve()
    density_path = source/'density_seed.npz'
    source_report = json.loads((source/'diagnostic.json').read_text())
    density_hash = _sha(density_path)
    if density_hash != source_report['input_sha256'] or density_hash != _sha(energy_density):
        raise ValueError('normal-field/energy reference density provenance mismatch')
    reference_terms = json.loads(energy_summary.read_text())['final']['energy_terms_mev']
    output.mkdir(parents=True, exist_ok=False)
    report_path = output/'diagnostic.json'
    source_paths = [Path(__file__).resolve()]
    module = Path(__file__).parent
    source_paths += [module/name for name in ('streamed_normal_functional.py',
        'coordinate_normal_functional.py', 'basis.py', 'spatial.py', 'currents.py',
        'functional.py', 'operators.py', 'execution_limits.py', 'runtime_resources.py')]
    source_paths += [module.parent/'rhf/parameters.py', density_path,
                     source/'diagnostic.json', energy_summary,
                     source/'h_n_columns.npy', source/'h_p_columns.npy']
    snapshot = output/'source_begin'
    snapshot.mkdir()
    hashes = {}
    for index, path in enumerate(source_paths):
        copied = snapshot/f'{index:02d}_{path.name}'
        shutil.copy2(path, copied)
        hashes[str(path)] = _sha(copied)
    report = {
        'status': 'running_streamed_O17_frozen_PH_comparison',
        'nucleus': 'O17', 'physical_result_accepted': False,
        'scope': 'same archived NN density; full PKA1 PH field/energy closure, no optimizer or self-consistency',
        'fortran_used': False, 'neural_updates': 0,
        'density_sha256': density_hash, 'source_hashes': hashes,
        'source_snapshot': str(snapshot), 'kernel_limits': limits,
        'tile_size': tile_size, 'coordinate_R_materialized': False,
        'interaction_four_tensor_materialized': False, 'started_unix': time.time(),
        'energy_reference': str(energy_summary),
        'energy_reference_density_hash_matches': True,
        'reference_fields': {s: str(source/f'h_{s}_columns.npy') for s in ('n', 'p')},
        'basis': source_report['basis'],
    }
    _write(report_path, report)
    config = report['basis']
    basis = AngularRadialBasis(**{key: config[key] for key in
        ('radius_fm', 'spacing_fm', 'kappa_max', 'n_theta', 'n_phi', 'derivative_order')})
    if basis.point_count != 882 or basis.dimension != 432:
        raise ValueError('unexpected archived O17 physical grid; no reduced space substitution')
    with np.load(density_path) as arrays:
        states = {s: DensityFactors(*[torch.from_numpy(arrays[f'{s}_{name}'].copy())
                  for name in ('Q', 'rho_active', 'chi_active')]) for s in ('n', 'p')}
    report['particle_numbers_of_source'] = {s: float(st.particle_number()) for s, st in states.items()}
    began = time.monotonic()
    def progress(values):
        report['progress'] = values
        report['elapsed_seconds'] = time.monotonic()-began
        _write(report_path, report)
    originals, counts = _forbid_spectra()
    try:
        result = StreamedPrimitiveNormalFunctional(basis, tile_size=tile_size,
            chunk_size=64, lmax=config['lmax']).evaluate(states, progress=progress)
        comparison = {}
        for species in ('n', 'p'):
            reference = torch.from_numpy(np.array(np.load(source/f'h_{species}_columns.npy',
                                                        mmap_mode='r'), copy=True))
            difference = result.h[species]-reference
            comparison[species] = {
                'entries_compared': reference.numel(),
                'max_abs_difference_mev': float(difference.abs().max()),
                'relative_frobenius': float(torch.linalg.matrix_norm(difference)
                                             / torch.linalg.matrix_norm(reference)),
                'hermiticity_max_abs_mev': float((result.h[species]-result.h[species].mH).abs().max()),
            }
            np.save(output/f'h_{species}_streamed.npy', result.h[species].cpu().numpy())
        report['full_primitive_field_comparison'] = comparison
        report['PH_energy_terms_mev'] = {key: float(value) for key, value in result.energy.energy_terms.items()}
        report['PH_energy_term_differences_mev'] = {
            key: float(value)-reference_terms[key]
            for key, value in result.energy.energy_terms.items() if key != 'd1s_pairing'}
        report['PH_total_energy_mev'] = float(result.energy.total_energy)
        report['full_field_closure_pass'] = all(
            entry['max_abs_difference_mev'] <= 2e-7 and entry['relative_frobenius'] <= 2e-10
            for entry in comparison.values())
        report['PH_energy_closure_pass'] = max(abs(value) for value in
            report['PH_energy_term_differences_mev'].values()) <= 2e-7
        report['backend_diagnostics'] = result.diagnostics
        report['status'] = ('completed_streamed_O17_PH_closure' if report['full_field_closure_pass']
                            and report['PH_energy_closure_pass'] else 'failed_streamed_O17_PH_closure')
        report['source_files_unchanged'] = all(_sha(path) == digest for path, digest in hashes.items())
        if not report['source_files_unchanged']:
            report['status'] = 'source_changed_during_comparison'
    except BaseException as exc:
        report['status'] = 'failed_streamed_O17_PH_comparison'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        _restore_spectra(originals)
        report['spectrum_attempts'] = counts
        report['basis_refreshes'] = 0
        report['elapsed_seconds'] = time.monotonic()-began
        report['runtime_resource_snapshot'] = resource_snapshot()
        _write(report_path, report)
    if report['status'] != 'completed_streamed_O17_PH_closure':
        raise RuntimeError(report['status'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--energy-summary', required=True)
    parser.add_argument('--energy-density', required=True)
    parser.add_argument('--tile-size', type=int, default=4)
    args = parser.parse_args()
    report = run(args.source, args.output, energy_summary=args.energy_summary,
                 energy_density=args.energy_density, tile_size=args.tile_size)
    print(json.dumps({'status': report['status'],
                     'comparison': report['full_primitive_field_comparison'],
                     'resources': report['runtime_resource_snapshot']}, indent=2))


if __name__ == '__main__':
    main()
