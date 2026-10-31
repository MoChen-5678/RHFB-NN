"""Algebraic full-Nambu stationarity identities for fixed-branch HFB states.

This module contains convention-level finite-dimensional identities only. It
does not select a physical no-sea branch, initialize quasiparticles from a
spectrum, or solve the self-consistent field equations.
"""
from __future__ import annotations

import torch

from .manifold import canonical_generator


def nambu_hamiltonian(
    h: torch.Tensor,
    delta: torch.Tensor,
    chemical_potential: float | torch.Tensor,
) -> torch.Tensor:
    """Return H_mu=[[h-mu,Delta],[-Delta*, -h*+mu]] in package convention."""
    h = torch.as_tensor(h)
    delta = torch.as_tensor(delta, device=h.device)
    if h.dtype != torch.complex128 or h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise TypeError("h must be square complex128")
    if delta.dtype != torch.complex128 or delta.shape != h.shape:
        raise TypeError("Delta must be complex128 and match h")
    if not bool(torch.isfinite(h).all() and torch.isfinite(delta).all()):
        raise ValueError("h and Delta must be finite")
    if not torch.allclose(h, h.mH, atol=1e-10, rtol=0):
        raise ValueError("h must be Hermitian")
    if not torch.allclose(delta, -delta.mT, atol=1e-10, rtol=0):
        raise ValueError("Delta must be antisymmetric under transpose")
    mu = torch.as_tensor(chemical_potential, dtype=torch.float64, device=h.device)
    if mu.numel() != 1 or not bool(torch.isfinite(mu)):
        raise ValueError("chemical potential must be a finite scalar")
    d = h.shape[0]
    eye = torch.eye(d, dtype=h.dtype, device=h.device)
    return torch.cat((
        torch.cat((h - mu * eye, delta), dim=1),
        torch.cat((-delta.conj(), -h.conj() + mu * eye), dim=1),
    ), dim=0)


def canonical_tangent(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Build the PH-compatible anti-Hermitian Nambu Thouless generator."""
    return canonical_generator(a, b)


def tangent_density(r: torch.Tensor, generator: torch.Tensor) -> torch.Tensor:
    """Exact first variation dR=[K,R] for R(t)=exp(tK)R exp(-tK)."""
    r = torch.as_tensor(r)
    k = torch.as_tensor(generator, dtype=r.dtype, device=r.device)
    if r.dtype != torch.complex128 or r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise TypeError("R must be square complex128")
    if k.shape != r.shape:
        raise ValueError("generator must match R")
    if not torch.allclose(k.mH, -k, atol=1e-10, rtol=0):
        raise ValueError("canonical generator must be anti-Hermitian")
    return k @ r - r @ k


def full_stationarity_residual(h_mu: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Return the unprojected Euler residual [H_mu,R] on the full space."""
    h_mu = torch.as_tensor(h_mu)
    r = torch.as_tensor(r, dtype=h_mu.dtype, device=h_mu.device)
    if h_mu.dtype != torch.complex128 or h_mu.ndim != 2 or h_mu.shape[0] != h_mu.shape[1]:
        raise TypeError("H_mu must be square complex128")
    if r.shape != h_mu.shape:
        raise ValueError("H_mu and R must have the same shape")
    if not bool(torch.isfinite(h_mu).all() and torch.isfinite(r).all()):
        raise ValueError("H_mu and R must be finite")
    if not torch.allclose(h_mu, h_mu.mH, atol=1e-9, rtol=0):
        raise ValueError("H_mu must be Hermitian")
    return h_mu @ r - r @ h_mu


def frame_h20_residual(h_mu: torch.Tensor, occupied_frame: torch.Tensor) -> torch.Tensor:
    """Return H20=(I-R)H F without forming an eigenbasis.

    For an orthonormal frame F spanning a pure generalized density R=FF†,
    ||[H,R]||_F² = 2 ||(I-R) H F||_F². This measures the occupied/unoccupied
    tangent block. Its zero is equivalent to full-space commutator stationarity.
    """
    h_mu = torch.as_tensor(h_mu)
    frame = torch.as_tensor(occupied_frame, dtype=h_mu.dtype, device=h_mu.device)
    if h_mu.dtype != torch.complex128 or h_mu.ndim != 2 or h_mu.shape[0] != h_mu.shape[1]:
        raise TypeError("H_mu must be square complex128")
    if frame.ndim != 2 or frame.shape[0] != h_mu.shape[0]:
        raise ValueError("occupied frame must have the same row count as H_mu")
    gram = frame.mH @ frame
    eye = torch.eye(frame.shape[1], dtype=frame.dtype, device=frame.device)
    if not torch.allclose(gram, eye, atol=1e-9, rtol=0):
        raise ValueError("occupied frame must be orthonormal")
    r = frame @ frame.mH
    complement = torch.eye(r.shape[0], dtype=r.dtype, device=r.device) - r
    return complement @ h_mu @ frame


def tangent_energy_derivative(h_mu: torch.Tensor, delta_r: torch.Tensor) -> torch.Tensor:
    """First derivative for L=1/2 Tr(H_mu R), in this Nambu convention."""
    value = 0.5 * torch.trace(h_mu @ delta_r)
    if abs(float(value.imag)) > 1e-9 * max(1.0, abs(float(value.real))):
        raise ValueError("Hermitian tangent produced a non-real energy derivative")
    return value.real
