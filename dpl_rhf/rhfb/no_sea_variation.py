"""First variations for a declared nucleon-space HFB prescription.

These algebraic identities do not select or certify a nuclear no-sea model.
They distinguish a fixed particle space from a field-dependent particle
space, whose implicit projector response must also be solved by the caller.
The anomalous convention is chi_ab=<c_b c_a>, Delta=.5 Vbar chi.
"""
from __future__ import annotations

import torch


def orbital_first_variation_gradient(h, delta, frame, rho_active, chi_active,
                                     *, chemical_potential=0.0):
    """Return G with delta(E-mu*N)=2 Re Tr(G† delta B).

    For rho=B rho_a B† and chi=B chi_a B^T, this includes both
    endpoints of the anomalous density: G=(h-mu) B rho_a+Delta B* chi_a†.
    The identity assumes the fields really are derivatives of the same EDF.
    """
    b = frame
    d, m = b.shape
    values = (h, delta, b, rho_active, chi_active)
    if any(x.dtype != torch.complex128 for x in values):
        raise TypeError("all matrices must be complex128")
    if h.shape != (d, d) or delta.shape != (d, d):
        raise ValueError("fields must match the primitive particle dimension")
    if rho_active.shape != (m, m) or chi_active.shape != (m, m):
        raise ValueError("active densities must match frame column dimension")
    mu = torch.as_tensor(chemical_potential, dtype=torch.float64, device=b.device)
    return (h @ b - mu * b) @ rho_active + delta @ b.conj() @ chi_active.mH


def external_projector_response_source(frame, orbital_gradient):
    """Hermitian C such that delta L_external=Tr(C delta P).

    P=B B† is held at fixed rank. Its allowed tangent obeys P delta P P=0.
    Thus only (1-P)G contributes, and delta B_external=(delta P)B.
    Substitution delta P=D sign(h)/2[delta h] must retain the response of h
    to rho, including the implicit dependence of rho on P.
    """
    b, g = frame, orbital_gradient
    if b.dtype != torch.complex128 or g.dtype != torch.complex128 or g.shape != b.shape:
        raise ValueError("frame and orbital gradient must be matching complex128 matrices")
    g_external = g - b @ (b.mH @ g)
    return b @ g_external.mH + g_external @ b.mH
