"""Particle-number tangent multiplier in the full pure HFB density manifold.

Robledo and Bertsch, PRC 84, 014312 (2011), Eqs. (16)--(18), project the
energy gradient against the constraint gradients. Here the same projection
is expressed in the project's density convention, without U/V eigenstates:

  C12(mu) = B0 - 2 mu chi,
  mu = Re <B0,chi> / (2 <chi,chi>).

The low-rank contraction below is exact, including a full-space pairing
field. It does not project Delta onto the active space for residual actions.
It neither corrects a pre-existing number error nor supplies a no-sea
branch. The caller still solves Tr rho = N separately. At pairing collapse
this tangent cannot identify mu; no artificial susceptibility is added.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .operators import DensityFactors


@dataclass
class NumberTangentProjection:
    chemical_potential: torch.Tensor
    identifiable: bool
    number_variance: torch.Tensor
    number_variance_from_rho: torch.Tensor
    constraint_gram: torch.Tensor
    tangent_inner_product_after_projection: torch.Tensor
    numerical_gram_tolerance: float


def number_tangent_projection(
    state: DensityFactors,
    hq: torch.Tensor,
    delta_q_star: torch.Tensor,
    *,
    fallback_mu: torch.Tensor | float = 0.0,
    numerical_gram_tolerance: float = 64 * torch.finfo(torch.float64).eps,
) -> NumberTangentProjection:
    """Solve the one-constraint Gram equation using D-by-M field actions.

    n/p constraints are independent only because this model excludes pn
    mixing/pairing. This projection uses the full canonical HFB tangent;
    a future restricted no-sea tangent needs its own constraint Gram matrix.
    ``numerical_gram_tolerance`` detects a singular numerical solve, not a
    physical pairing cutoff; neither density nor Delta is modified.
    """
    if not math.isfinite(numerical_gram_tolerance) or numerical_gram_tolerance < 0:
        raise ValueError("numerical Gram tolerance must be finite and nonnegative")
    for name, value in (("hQ", hq), ("Delta Q*", delta_q_star)):
        if (value.shape != state.q.shape or value.dtype != torch.complex128
                or value.device != state.q.device or not torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite complex128 columns matching Q")
    mu0 = torch.as_tensor(fallback_mu, dtype=torch.float64, device=state.q.device)
    if mu0.shape != () or not torch.isfinite(mu0):
        raise ValueError("fallback chemical potential must be a finite scalar")
    rho, chi, q = state.rho_active, state.chi_active, state.q
    ha, gap = q.mH @ hq, q.mH @ delta_q_star
    b0 = ha @ chi + chi @ ha.conj() + gap - gap @ rho.conj() - rho @ gap
    chi_square = chi.abs().square().sum()
    gram = 8.0 * chi_square
    numerator = 4.0 * (b0.conj() * chi).sum().real
    identifiable = bool(gram.detach() > numerical_gram_tolerance)
    # This is the usual singular-constraint case for a Slater determinant,
    # where the average number is already invariant in every first-order
    # pure-HFB tangent. Preserve the supplied mu instead of inventing a root.
    mu = numerator / gram if identifiable else mu0
    if not torch.isfinite(mu):
        raise FloatingPointError("particle-number tangent multiplier became nonfinite")
    return NumberTangentProjection(
        chemical_potential=mu,
        identifiable=identifiable,
        number_variance=2.0 * chi_square,
        number_variance_from_rho=2.0 * torch.trace(rho - rho @ rho).real,
        constraint_gram=gram,
        tangent_inner_product_after_projection=numerator - mu * gram,
        numerical_gram_tolerance=float(numerical_gram_tolerance),
    )
