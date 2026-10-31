"""Time-reversal-averaged one-body reference from two pure odd HFB states.

This module averages the one-body normal/anomalous contractions of a pure
blocked state and its time-reversed partner.  It returns a deliberately
separate reference type, not ``GeneralizedDensity``: these contractions do
not specify the ensemble's higher-order correlations and must not be sent to
a pure-state energy path as if Wick factorization were established.  This is
also not the standard finite-temperature/statistical EFA density operator.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .manifold import GeneralizedDensity


@dataclass(frozen=True)
class TimeReversalAveragedOddOneBodyReference:
    """One-body contractions averaged over two pure odd components.

    ``rho`` and ``chi`` are useful for one-body diagnostics and comparisons.
    ``sharp_fermion_parity`` is intentionally ``None`` because these
    contractions alone do not specify the full statistical density operator.
    ``wick_energy_closed`` is always false: a two-body expectation cannot in
    general be reconstructed from this average by Gaussian Wick factorization.
    """

    rho: torch.Tensor
    chi: torch.Tensor
    blocked_component: GeneralizedDensity
    time_reversed_component: GeneralizedDensity
    weights: tuple[float, float] = (0.5, 0.5)
    sharp_fermion_parity: None = None
    wick_energy_closed: bool = False
    physical_identity_certified: bool = False


def time_reversal_averaged_odd_one_body_reference(
    blocked: GeneralizedDensity,
    time_reversal_unitary: torch.Tensor,
    *,
    tolerance: float = 1.0e-9,
) -> TimeReversalAveragedOddOneBodyReference:
    """Average one-body contractions of an odd pure state and its T partner.

    ``time_reversal_unitary`` is ``J`` in the antiunitary one-particle
    operator ``T = J K``.  It must be unitary and obey ``J J* = -I``. The
    induced contractions are ``rho_T=J rho* J†`` and ``chi_T=J chi* Jᵀ``.

    This records a two-component one-body average only. It does not construct
    the statistical density operator of standard EFA (whose quasiparticle
    ensemble includes zero-, one-, and two-quasiparticle sectors), and it
    does not define a two-body energy. Use the original pure component for
    pure-state energy/stationarity calculations.
    """

    if not isinstance(blocked, GeneralizedDensity):
        raise TypeError("blocked must be a GeneralizedDensity")
    if blocked.fermion_parity != -1:
        raise ValueError("the input component must be a pure odd-parity state")
    if not isinstance(tolerance, (int, float)) or tolerance <= 0:
        raise ValueError("tolerance must be positive")
    rho = _complex_matrix(blocked.rho, "rho")
    chi = _complex_matrix(blocked.chi, "chi")
    matrix = _complex_matrix(blocked.matrix, "generalized-density matrix")
    j = _complex_matrix(time_reversal_unitary, "time-reversal unitary")
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or chi.shape != rho.shape:
        raise ValueError("rho and chi must be square matrices of identical shape")
    m = rho.shape[0]
    if j.shape != (m, m):
        raise ValueError("time_reversal_unitary must match the one-particle dimension")
    if matrix.shape != (2 * m, 2 * m):
        raise ValueError("generalized-density matrix must have shape (2M,2M)")
    if any(x.device != rho.device for x in (chi, matrix, j)):
        raise ValueError("all density and time-reversal tensors must share a device")

    eye = torch.eye(m, dtype=rho.dtype, device=rho.device)
    expected = torch.cat(
        [torch.cat([rho, chi], dim=1),
         torch.cat([-chi.conj(), eye - rho.conj()], dim=1)],
        dim=0,
    )
    if not torch.allclose(matrix, expected, atol=tolerance, rtol=0.0):
        raise ValueError("matrix is inconsistent with rho and chi")
    if not torch.allclose(rho.mH, rho, atol=tolerance, rtol=0.0):
        raise ValueError("rho must be Hermitian")
    if not torch.allclose(chi.mT, -chi, atol=tolerance, rtol=0.0):
        raise ValueError("chi must be skew-symmetric")
    if not torch.allclose(j.mH @ j, eye, atol=tolerance, rtol=0.0):
        raise ValueError("time_reversal_unitary must be unitary")
    if not torch.allclose(j @ j.conj(), -eye, atol=tolerance, rtol=0.0):
        raise ValueError("time reversal must square to -1 for this fermion basis")
    if not torch.allclose(matrix.mH, matrix, atol=tolerance, rtol=0.0):
        raise ValueError("generalized density must be Hermitian")
    if not torch.allclose(matrix @ matrix, matrix, atol=10 * tolerance, rtol=0.0):
        raise ValueError("input component must be a pure generalized density")

    partner_rho = j @ rho.conj() @ j.mH
    partner_chi = j @ chi.conj() @ j.mT
    partner_matrix = torch.cat(
        [torch.cat([partner_rho, partner_chi], dim=1),
         torch.cat([-partner_chi.conj(), eye - partner_rho.conj()], dim=1)],
        dim=0,
    )
    partner = GeneralizedDensity(partner_rho, partner_chi, partner_matrix, -1)
    return TimeReversalAveragedOddOneBodyReference(
        rho=0.5 * (rho + partner_rho),
        chi=0.5 * (chi + partner_chi),
        blocked_component=blocked,
        time_reversed_component=partner,
    )


def _complex_matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    result = torch.as_tensor(value)
    if result.dtype != torch.complex128:
        raise TypeError(f"{name} must use complex128")
    if result.ndim != 2 or not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be a finite matrix")
    return result
