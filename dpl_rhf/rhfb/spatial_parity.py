"""Optional parity-conserving candidates for the unrestricted RHFB backend.

Dirac spatial parity acts as beta psi(-r); both reduced radial coefficient
components therefore have the parity (-1)^l of the UPPER spinor harmonic.
This is distinct from particle-number parity. No angular-momentum or time-
reversal symmetry is imposed by this module.
"""
from __future__ import annotations

import torch

from .manifold import canonical_generator, orthonormalize_factors
from .network import GeneralizedDensityNet
from .parity_diagnostic import number_parity_from_density


def primitive_spatial_parity(basis, *, device='cpu'):
    values = torch.empty(basis.dimension, dtype=torch.float64, device=device)
    for channel in basis.channels:
        upper_l = -channel.kappa-1 if channel.kappa < 0 else channel.kappa
        values[channel.offset:channel.offset+2*basis.n_radial] = (-1.)**upper_l
    return values


def adapted_column_parities(basis, q, *, tolerance=1e-10):
    """Identify definite-parity columns, rejecting an unadapted orbital frame."""
    rows = primitive_spatial_parity(basis, device=q.device)
    diagonal = (q.conj()*(rows[:, None]*q)).sum(0).real
    columns = torch.where(diagonal >= 0, torch.ones_like(diagonal), -torch.ones_like(diagonal))
    error = (rows[:, None]*q-q*columns[None, :]).abs().max()
    if float(error.detach()) > tolerance:
        raise ValueError('spatial parity diagnostic requires a parity-adapted orbital frame')
    return rows, columns, error


def parity_adapted_density_diagnostic(basis, state, *, tolerance=1e-10):
    """Independent spatial parity via (-1)^N_minus on a pure adapted state.

When rho and chi do not couple opposite spatial parities, the negative-
parity orbital sector is itself a pure Gaussian state. Its number parity
is the many-body spatial parity. This deduction is specific to this
factorized symmetry sector; it is not a diagnostic of arbitrary frames.
    """
    _, columns, frame_error = adapted_column_parities(basis, state.q, tolerance=tolerance)
    rho, chi = state.rho_active, state.chi_active
    sign = columns[:, None]*columns[None, :]
    rho_error = float((rho-rho*sign).abs().max().detach())
    chi_error = float((chi-chi*sign).abs().max().detach())
    if max(rho_error, chi_error) > tolerance:
        raise ValueError('density couples opposite spatial parities')
    negative = torch.nonzero(columns < 0).flatten()
    if negative.numel():
        block_rho = rho[negative[:, None], negative[None, :]].detach().cpu().numpy()
        block_chi = chi[negative[:, None], negative[None, :]].detach().cpu().numpy()
        sector = number_parity_from_density(block_rho, block_chi, tolerance=tolerance)
        parity = sector['number_parity']
    else:
        sector = {'dimension': 0, 'number_parity': 1, 'particle_number': 0.}
        parity = 1
    return {'many_body_spatial_parity': parity,
            'column_parities': columns.detach().cpu().to(torch.int64).tolist(),
            'frame_parity_max_error': float(frame_error.detach()),
            'normal_opposite_parity_max_error': rho_error,
            'anomalous_opposite_parity_max_error': chi_error,
            'negative_parity_sector': sector,
            'scope': 'pure state in a parity-adapted frame; does not assign angular momentum'}


class ParityConservingDensityNet(GeneralizedDensityNet):
    """Additional candidate with structural spatial parity and general m mixing.

The unrestricted network remains available. This candidate permits all
radial/channel directions within each parity sector, including changes
outside the initial orbital span; it fixes neither spherical nor axial
symmetry and it does not average time-reversed blocked states.
    """
    def __init__(self, basis, initial_q, initial_q_labels, reference, **kwargs):
        rows, columns, _ = adapted_column_parities(basis, initial_q)
        sign = columns[:, None]*columns[None, :]
        if max(float((reference.rho-reference.rho*sign).abs().max()),
               float((reference.chi-reference.chi*sign).abs().max())) > 1e-10:
            raise ValueError('reference must conserve spatial parity')
        # A plain tuple is available during the base pairing-seed initializer.
        self._parity_labels = tuple(int(v) for v in columns.detach().cpu().tolist())
        super().__init__(basis, initial_q, initial_q_labels, reference, **kwargs)
        self.register_buffer('primitive_parities', rows.to(initial_q.device))
        self.register_buffer('orbital_parities', columns.to(initial_q.device))

    def _active_parity_mask(self):
        signs = torch.tensor(self._parity_labels, device=self.raw_b.device)
        return signs[:, None] == signs[None, :]

    def _initialize_pairing_seed(self):
        super()._initialize_pairing_seed()
        with torch.no_grad():
            self.raw_b.mul_(self._active_parity_mask())

    def _canonical_transform(self):
        a0 = self._silu_complex(self.raw_a)*self._active_parity_mask()
        b0 = self._silu_complex(self.raw_b)*self._active_parity_mask()
        a, b = .5*(a0-a0.mH), .5*(b0-b0.mT)
        learned = torch.matrix_exp(canonical_generator(a, b))
        return self._apply_number_transport(learned)

    def _orbital_factor(self):
        mask = self.primitive_parities[:, None] == self.orbital_parities[None, :]
        correction = (self.raw_coordinates+self._network_correction())*mask
        candidate = self.initial_q+self.correction_scale*self.envelope[:, None]*correction
        result = torch.zeros_like(candidate)
        for parity in (-1, 1):
            rows = torch.nonzero(self.primitive_parities == parity).flatten()
            columns = torch.nonzero(self.orbital_parities == parity).flatten()
            if not columns.numel():
                continue
            block = candidate[rows[:, None], columns[None, :]]
            q = orthonormalize_factors(block)
            result = result.index_put((rows[:, None], columns[None, :]), q)
        return result

    def forward(self):
        return self._orbital_factor(), self._canonical_density()

    def forward_with_blocked_mode(self):
        return super().forward_with_blocked_mode()

    def checkpoint_metadata(self):
        return {**super().checkpoint_metadata(),
                'format': 'dpl-rhfb-parity-conserving-density-net-v2',
                'spatial_parity_constraint': 'conserved; general m and time-odd fields',
                'orbital_parities': list(self._parity_labels)}
