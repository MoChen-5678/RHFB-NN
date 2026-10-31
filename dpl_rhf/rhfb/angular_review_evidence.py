"""Compare archived fields of the same NN state on two angular grids.

This is bounded postprocessing of the small-box integration fixture. It
neither computes a new nucleus nor certifies angular or physical convergence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .execution_limits import require_bounded_cgroup


def run(output: Path):
    limits = require_bounded_cgroup()
    root = Path(__file__).resolve().parents[2]
    coarse_path = root / 'outputs/validation/joint_nn_cutoff_baseline_20261007'
    fine_path = root / 'outputs/validation/joint_nn_L2_ang7x14_20261007'
    cases = []
    for directory in (coarse_path, fine_path):
        data = json.loads((directory / 'diagnostic.json').read_text())
        if data['status'] != 'complete' or not data['saved_artifact_hashes_verified']:
            raise ValueError('archived angular comparison case is incomplete')
        for name, digest in data['saved_artifact_sha256'].items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'archived artifact hash changed: {directory/name}')
        cases.append(data)
    coarse, fine = cases
    if coarse['density_provenance'] != fine['density_provenance']:
        raise ValueError('cases do not declare the same neural density')
    if coarse['one_case']['lmax'] != fine['one_case']['lmax']:
        raise ValueError('L must remain fixed to isolate angular quadrature')
    with np.load(coarse_path/'density_factors.npz', allow_pickle=False) as a, \
            np.load(fine_path/'density_factors.npz', allow_pickle=False) as b:
        if a.files != b.files or any(not np.array_equal(a[key], b[key]) for key in a.files):
            raise ValueError('actual NN density factors differ between angular cases')
    fields = {}
    for species in ('n', 'p'):
        a = np.load(coarse_path/f'h_{species}.npy', allow_pickle=False)
        b = np.load(fine_path/f'h_{species}.npy', allow_pickle=False)
        if a.shape != (48, 48) or a.shape != b.shape:
            raise ValueError('expected the archived small-box full D=48 normal fields')
        difference = b-a
        norm = float(np.sqrt(np.sum(np.abs(difference)**2)))
        reference_norm = float(np.sqrt(np.sum(np.abs(b)**2)))
        fields[species] = {
            'max_abs_change_mev': float(np.max(np.abs(difference))),
            'frobenius_change_mev': norm,
            'relative_frobenius_change': norm/reference_norm,
        }
    channels = {key: fine['energy_channels_mev'][key]-value
                for key, value in coarse['energy_channels_mev'].items()}
    report = {
        'status': 'completed_comparison',
        'scope': 'fixed N3,Z2,D48,M40 NN integration fixture, not O17',
        'kernel_limits': limits,
        'coarse': str(coarse_path), 'fine': str(fine_path),
        'same_density_factors_verified': True,
        'coarse_case': coarse['one_case'], 'fine_case': fine['one_case'],
        'PH_energy_change_mev': fine['total_particle_hole_energy_mev']-coarse['total_particle_hole_energy_mev'],
        'PH_energy_channel_changes_mev': channels, 'normal_field_changes': fields,
        'angular_convergence_certified': False, 'physical_result_accepted': False,
        'limitations': 'Only one angular-grid refinement at fixed L; individual channels can cancel in the total.',
        'fortran_used': False, 'eigensolver_used': False, 'basis_refreshes': 0,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'diagnostic.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output), indent=2))


if __name__ == '__main__':
    main()
