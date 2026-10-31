"""Fixed-field no-sea prescription audit, not a nuclear/NN solver.

DIRHB arXiv:1403.4039 Eqs.41,46--48 select particle E>0 and sea E<0
annihilators. The explicit Fock state and normal-projector comparison below
are our finite-dimensional derivation. Pairing is prescribed, not Gogny D1S.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from dpl_rhf.rhfb.parity_diagnostic import number_parity_from_density


def annihilators(m):
    operators = []
    for i in range(m):
        c = np.zeros((2**m, 2**m))
        for mask in range(2**m):
            if mask & (1 << i):
                c[mask ^ (1 << i), mask] = (-1)**(
                    (mask & ((1 << i)-1)).bit_count())
        operators.append(c)
    return operators


def nambu_density(rho, chi):
    return np.block([[rho, chi], [-chi.conj(), np.eye(len(rho))-rho.conj()]])


def test_no_sea_qp_branch_is_not_generic_strict_normal_projection():
    records = []
    explicit_eigh_calls = 0
    for sea_gap in (20., 2000.):
        for pairing in (0., .3):
            # Modes: positive-energy p, negative-energy a, blocked spectator b.
            # These are prescribed quadratic fields, NOT a fitted EDF/nucleus.
            energies = np.array([1., -sea_gap, .4])
            h = np.diag(energies)
            delta = np.zeros((3, 3))
            delta[0, 1], delta[1, 0] = pairing, -pairing
            k = np.block([[h, delta], [-delta, -h]])

            # Independent reference diagonalization, outside all NN training.
            eigenvalues, vectors = np.linalg.eigh(k)
            explicit_eigh_calls += 1
            selected = [int(np.argmax(abs(vectors[i])**2)) for i in range(3)]
            assert len(set(selected)) == 3
            assert eigenvalues[selected[0]] > 0
            assert eigenvalues[selected[1]] < 0  # no-sea annihilator convention
            assert eigenvalues[selected[2]] > 0
            w = vectors[:, selected].copy()
            # Designated spectator blocking swaps precisely its PH partner.
            w[:, 2] = np.concatenate((w[3:, 2], w[:3, 2])).conj()
            u, v = w[:3], w[3:]
            rho, chi = v.conj() @ v.T, v.conj() @ u.T
            r = nambu_density(rho, chi)
            np.testing.assert_allclose(r @ r, r, atol=2e-14)
            np.testing.assert_allclose(k @ r, r @ k, atol=2e-13)
            parity = number_parity_from_density(rho, chi)
            assert parity['number_parity'] == -1

            # Explicit odd Fock state: u0|b> + v0|p a b>. This is the
            # stationary even-pair branch continuously connected to empty
            # p/a, with one independently blocked positive-energy spectator.
            pair_sum = energies[0] + energies[1]
            scale = np.hypot(pair_sum, 2*pairing)
            # Stable evaluation avoids subtracting two nearly equal numbers.
            v_squared = (2*pairing**2/(scale*(scale-pair_sum))
                         if pairing else 0.)
            fock_state = np.zeros(8)
            fock_state[4], fock_state[7] = np.sqrt(1-v_squared), np.sqrt(v_squared)
            c = annihilators(3)
            rho_fock = np.array([[np.vdot(c[j]@fock_state, c[i]@fock_state)
                                  for j in range(3)] for i in range(3)])
            chi_fock = np.array([[np.vdot(fock_state, c[j]@c[i]@fock_state)
                                  for j in range(3)] for i in range(3)])
            np.testing.assert_allclose(rho, rho_fock, atol=3e-14)
            np.testing.assert_allclose(chi, chi_fock, atol=3e-14)
            fock_h = sum(e*c[i].T@c[i] for i, e in enumerate(energies))
            fock_h += pairing*(c[0].T@c[1].T+c[1]@c[0])
            np.testing.assert_allclose(fock_h@fock_state,
                float(fock_state@fock_h@fock_state)*fock_state, atol=3e-13)

            # Strict support in the normal positive-energy space excludes a.
            p = np.diag([1., 0., 1.])
            normal_sea = float(np.trace((np.eye(3)-p)@rho))
            anomalous_sea = float(np.linalg.norm(chi-p@chi@p.T))
            # Projection alone does not retain canonical purity in general.
            r_after_projection = nambu_density(p@rho@p, p@chi@p.T)
            # The canonical blocked state in this projected model has no pair.
            projected_canonical_r = nambu_density(np.diag([0., 0., 1.]),
                                                  np.zeros((3, 3)))

            # DIRHB's further omission of sea-annihilator V_a contributions
            # in BOTH qp sums is distinct from strict normal projection.
            rho_drop = v[:, [0, 2]].conj()@v[:, [0, 2]].T
            chi_drop = v[:, [0, 2]].conj()@u[:, [0, 2]].T
            drop_skew_error = float(np.linalg.norm(chi_drop+chi_drop.T))
            if pairing:
                assert normal_sea > 0 and anomalous_sea > 0
                assert np.linalg.norm(r_after_projection@r_after_projection-
                                      r_after_projection) > 1e-9
                assert np.linalg.norm(k@projected_canonical_r-
                                      projected_canonical_r@k) > .5
                assert drop_skew_error > 1e-5
                with pytest.raises(ValueError):
                    number_parity_from_density(rho_drop, chi_drop)
            else:
                assert normal_sea == anomalous_sea == drop_skew_error == 0

            records.append({
                'normal_energies_mev': energies.tolist(),
                'prescribed_pairing_p_a_mev': pairing,
                'selected_unblocked_annihilator_energies_mev': eigenvalues[selected].tolist(),
                'particle_number': parity['particle_number'],
                'computed_number_parity': parity['number_parity'],
                'purity_frobenius': parity['purity_error'],
                'full_bdg_commutator_frobenius_mev': float(np.linalg.norm(k@r-r@k)),
                'normal_sea_trace': normal_sea,
                'normal_sea_fraction': normal_sea/parity['particle_number'],
                'anomalous_outside_normal_projection_frobenius': anomalous_sea,
                'projection_only_purity_frobenius': float(np.linalg.norm(
                    r_after_projection@r_after_projection-r_after_projection)),
                'omit_sea_qp_chi_skew_frobenius': drop_skew_error,
                'fock_energy_mev': float(fock_state@fock_h@fock_state),
                'canonical_projected_state_energy_mev': energies[2],
                'rho': rho.tolist(), 'chi': chi.tolist(),
            })
    artifact = Path(__file__).resolve().parents[3]/'outputs/validation/no_sea_branch_vs_normal_projection_20261006.json'
    artifact.write_text(json.dumps({
        'source_equations': 'https://arxiv.org/pdf/1403.4039 Eqs.41,46-48',
        'scope': 'Our fixed-field analytical/Fock counterexample; not D1S, PKA1, self-consistency, or NN training.',
        'explicit_numpy_eigh_calls_in_this_reference': explicit_eigh_calls,
        'cases': records,
    }, indent=2)+'\n')
