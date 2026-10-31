"""Full primitive normal/pairing actions and independent RHFB stationarity.

This composes the normal EDF derivative with the finite-range D1S action.
Chemical potentials are fitted to the number tangent, rather than assuming
that zero chemical potential is the constrained HFB stationarity condition.
"""
from __future__ import annotations

import torch

from .normal_action import AutogradNormalAction
from .operators import NambuAction, commutator_action, commutator_metrics
from .gaussian_pairing import GaussianPairingAction
from .factorized_commutator import FactorizedCommutator


def make_nambu_actions(basis, states, normal_energy, *, chunk_size=32,
                       vector_chunk=8, chemical_potentials=None, create_graph=True,
                       lmax=None, active_chunk_size=4):
    """Compose actions from a caller-certified complete normal EDF."""
    actions = {}
    for species, state in states.items():
        h = AutogradNormalAction(basis, states, normal_energy, species=species,
                                 vector_chunk=vector_chunk, create_graph=create_graph)
        delta = GaussianPairingAction(
            basis, state.q, state.chi_active, species=species,
            chunk_size=chunk_size, active_chunk_size=active_chunk_size,
            lmax=lmax, checkpoint_blocks=create_graph,
        )
        mu = 0. if chemical_potentials is None else chemical_potentials[species]
        actions[species] = NambuAction(basis.dimension, h.apply, delta.apply, mu)
    return actions


def fit_number_tangent_chemical_potential(action, state, *, cached_commutator=None):
    """Least-squares lambda removing the particle-number commutator tangent.

    [N,R] has support entirely inside diag(Q,Q*). Therefore its inner product
    with the full commutator needs only that small input frame, while the
    resulting Hamiltonian retains its full primitive-space field action.
    """
    q = state.q
    d, m = q.shape
    frame = q.new_zeros((2*d, 2*m))
    frame[:d, :m] = q
    frame[d:, m:] = q.conj()
    number = NambuAction(d, lambda v: v, lambda v: torch.zeros_like(v), 0.)
    number_commutator = frame.mH @ commutator_action(number, state, frame)
    physical_commutator = frame.mH @ (
        commutator_action(action, state, frame) if cached_commutator is None
        else cached_commutator.apply(frame)
    )
    square = number_commutator.abs().square().sum()
    if float(square.detach()) <= 1e-24:
        # Particle-number gauge is unidentifiable in the zero-pairing limit.
        return q.real.new_zeros(()), False
    chemical_potential = (number_commutator.conj()*physical_commutator).sum().real/square
    return chemical_potential, True


def full_stationarity(basis, states, normal_energy, *, chunk_size=32,
                      vector_chunk=8, lmax=None, active_chunk_size=4):
    """Stream exact full-space residuals, including outside the activity space.

    This certification is intentionally expensive. A cheaper sampled residual
    used during training must be reported separately and cannot certify it.
    """
    actions = make_nambu_actions(basis, states, normal_energy,
                                chunk_size=chunk_size, vector_chunk=vector_chunk,
                                create_graph=False, lmax=lmax,
                                active_chunk_size=active_chunk_size)
    result = {}
    for species, state in states.items():
        action = actions[species]
        # Normal h is needed only on Q. Pairing remains an action on every
        # primitive unit vector, including pairing wholly outside Q.
        with torch.no_grad():
            cached = FactorizedCommutator(state, action.h_apply(state.q),
                                          action.delta_apply)
            mu, identified = fit_number_tangent_chemical_potential(
                action, state, cached_commutator=cached,
            )
            cached.chemical_potential = mu
            result[species] = {
                'chemical_potential_mev': mu.detach(),
                'chemical_potential_identifiable': identified,
                **{key: value.detach() for key, value in
                   commutator_metrics(cached.apply, state, chunk_size=chunk_size).items()},
                'scope': 'all primitive Nambu unit vectors, streamed; exact hQ cache',
                'normal_field_input_columns': state.q.shape[1],
            }
    return result
