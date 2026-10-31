"""Necessary RHFB stationarity checks on a saved neural density.

This checks [H-mu N,R] on diag(Q,Q*) and retains its output outside
that frame. It does not sample every primitive input, certify a no-sea
prescription, or rerun training. The input result is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from contextlib import contextmanager

import numpy as np
import torch

from .basis import AngularRadialBasis
from .factorized_commutator import FactorizedCommutator
from .fields import fit_number_tangent_chemical_potential
from .functional import evaluate_particle_hole_energy, evaluate_rhfb_energy
from .gaussian_pairing import GaussianPairingAction
from .normal_action import AutogradNormalAction
from .operators import DensityFactors, NambuAction
from .result import atomic_json


@contextmanager
def forbid_spectral_solutions(counts):
    originals = []
    for library in (torch.linalg, np.linalg):
        for name in ('eig', 'eigh', 'eigvals', 'eigvalsh'):
            original = getattr(library, name)
            def reject(*args, _name=name, **kwargs):
                counts[_name] += 1
                raise RuntimeError('Spectral solution forbidden in saved-density diagnostic: '+_name)
            originals.append((library, name, original))
            setattr(library, name, reject)
    try:
        yield
    finally:
        for library, name, original in originals:
            setattr(library, name, original)


def active_pairing_cache(q, delta_q_star):
    """Exact Delta action restricted to span(Q*), with an explicit domain check."""
    def apply(v):
        coefficients = q.T @ v
        remainder = v-q.conj() @ coefficients
        if float(remainder.abs().max()) > 1e-10:
            raise RuntimeError('Pairing cache received an input outside Q*')
        return delta_q_star @ coefficients
    return apply


def run(input_directory, output_directory):
    source = Path(input_directory).resolve()
    out = Path(output_directory).resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.monotonic()
    metadata = json.loads((source/'run_config.json').read_text())
    config = metadata['config']
    counts = dict.fromkeys(('eig', 'eigh', 'eigvals', 'eigvalsh'), 0)
    report = {
        'status': 'running', 'nucleus': 'O17', 'input_directory': str(source),
        'input_density_sha256': hashlib.sha256((source/'density_factors.npz').read_bytes()).hexdigest(),
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in Path(__file__).parent.glob('*.py')},
        'physical_result_accepted': False, 'fortran_used': False,
        'new_neural_optimizer_updates': 0, 'subspace_refresh_operations': 0,
        'spectral_solution_attempts': counts,
        'spectral_guard_scope': 'field and residual evaluation; fixed angular quadrature constructed beforehand',
        'scope': 'necessary unprojected RHFB residual on the saved active Nambu input frame',
        'limitations': [
            'The saved neural experiment has no self-consistent no-sea constraint.',
            'The external output norm includes Dirac-sea directions; it is not a projected nucleon residual.',
            'Pairing on inputs wholly outside the activity frame is not certified.',
            'Chemical potentials are fitted at this state, not independently converged.',
            'This is the current implemented functional, whose physical closures remain incomplete.',
            'No new neural training, ground-state assignment, or space-convergence certification.',
        ], 'species': {},
    }
    atomic_json(out/'diagnostic.json', report)
    try:
        # NumPy constructs Gauss-Legendre angular quadrature by solving its
        # Jacobi matrix. This fixed integration rule is unrelated to any
        # Dirac/Bogoliubov Hamiltonian or a refreshed orbital space.
        basis = AngularRadialBasis(radius_fm=config['radius_fm'],
            spacing_fm=config['spacing_fm'], kappa_max=config['kappa_max'],
            n_theta=7, n_phi=14, derivative_order=5)
        with forbid_spectral_solutions(counts):
            with np.load(source/'density_factors.npz') as data:
                states = {s: DensityFactors(*[torch.from_numpy(data[f'{s}_{k}'].copy())
                          for k in ('Q', 'rho_active', 'chi_active')]) for s in ('n', 'p')}
            normal_energy = lambda derivative_basis, values: evaluate_particle_hole_energy(
                derivative_basis, values, chunk_size=128, pair_chunk=4,
                lmax=config['lmax']).total_energy
            for species, state in states.items():
                print('Computing complete implemented normal field on saved '+species+' Q', flush=True)
                normal = AutogradNormalAction(basis, states, normal_energy,
                    species=species, vector_chunk=state.q.shape[1], create_graph=False)
                pairing = GaussianPairingAction(basis, state.q, state.chi_active,
                    species=species, chunk_size=128, active_chunk_size=4,
                    lmax=config['lmax'], checkpoint_blocks=False)
                with torch.no_grad():
                    hq = normal.apply(state.q).detach()
                    delta_q_star = pairing.apply(state.q.conj()).detach()
                    # Only these inputs are needed below; retain full Delta's action on them.
                    # Reusing this cache avoids repeatedly evaluating the identical D1S field.
                    d, m = state.q.shape
                    frame = state.q.new_zeros((2*d, 2*m))
                    frame[:d, :m] = state.q
                    frame[d:, m:] = state.q.conj()
                    delta_on_frame = active_pairing_cache(state.q, delta_q_star)
                    cached = FactorizedCommutator(state, hq, delta_on_frame,
                        delta_q_star=delta_q_star)
                    action = NambuAction(d, normal.apply, pairing.apply)
                    mu, identifiable = fit_number_tangent_chemical_potential(
                        action, state, cached_commutator=cached)
                    before = cached.apply(frame)
                    cached.chemical_potential = mu
                    full_output = cached.apply(frame)
                    active = frame.mH @ full_output
                    external = full_output-frame @ active
                    ha = state.q.mH @ hq
                    da = state.q.mH @ delta_q_star
                    values = {
                        'particle_number': state.particle_number(),
                        'chemical_potential_mev': mu,
                        'chemical_potential_identifiable': identifiable,
                        'input_frame_columns': 2*m, 'primitive_nambu_dimension': 2*d,
                        'before_number_fit_frobenius_mev': torch.linalg.matrix_norm(before),
                        'after_number_fit_frobenius_mev': torch.linalg.matrix_norm(full_output),
                        'active_output_frobenius_mev': torch.linalg.matrix_norm(active),
                        'external_output_frobenius_mev': torch.linalg.matrix_norm(external),
                        'active_output_max_mev': active.abs().max(),
                        'external_output_max_mev': external.abs().max(),
                        'input_column_rms_mev': torch.linalg.matrix_norm(full_output)/(2*m)**.5,
                        'active_normal_field_hermiticity_max_mev': (ha-ha.mH).abs().max(),
                        'active_pairing_field_antisymmetry_max_mev': (da+da.T).abs().max(),
                        'active_pairing_field_frobenius_mev': torch.linalg.matrix_norm(da),
                    }
                    report['species'][species] = values
                    np.savez_compressed(out/f'{species}_field_and_residual.npz',
                        hQ=hq.numpy(), delta_Q_star=delta_q_star.numpy(),
                        h_active=ha.numpy(), delta_active=da.numpy(),
                        frame_commutator=full_output.numpy(),
                        active_output=active.numpy(), external_output=external.numpy())
                report['elapsed_seconds'] = time.monotonic()-started
                atomic_json(out/'diagnostic.json', report)
                print(species+' active residual = '+str(float(values['active_output_frobenius_mev']))+
                      ' MeV; external output = '+str(float(values['external_output_frobenius_mev']))+' MeV', flush=True)
            with torch.no_grad():
                evaluation = evaluate_rhfb_energy(basis, states, chunk_size=128,
                    pair_chunk=4, lmax=config['lmax'])
                report['current_implemented_energy_mev'] = evaluation.total_energy.detach()
                report['current_energy_terms_mev'] = {k:v.detach() for k,v in evaluation.energy_terms.items()}
            saved = json.loads((source/'summary.json').read_text())['final']['experimental_total_energy_mev']
            report['saved_experimental_energy_mev'] = saved
            report['current_minus_saved_energy_mev'] = report['current_implemented_energy_mev']-saved
            report['status'] = 'diagnostic_completed_physical_acceptance_pending'
    except Exception as exc:
        report['status'] = 'failed_with_partial_data'
        report['failure'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        atomic_json(out/'diagnostic.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-directory', required=True)
    parser.add_argument('--output-directory', required=True)
    args = parser.parse_args()
    run(args.input_directory, args.output_directory)
