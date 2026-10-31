"""Matrix-free rational positive-branch projector for small RHFB prototypes.

No spectral decomposition is performed here.  The caller must provide a
defensible enclosure ``spec_gap <= |eig(h-W)| <= spec_bound``.  The prototype
uses analytic Weyl bounds; passing a guessed gap is explicitly unsupported.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import math
import torch


TensorMap = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class SignDiagnostics:
    quadrature_error_bound: float
    max_linear_solve_residual: torch.Tensor
    quadrature_order: int
    spectral_gap: float
    spectral_bound: float


class RationalSignOperator:
    """Apply ``sign(h - shift)`` using log-spaced rational quadrature and CG.

    ``h`` can be a dense matrix or a matrix-free action.  Dense matrices are
    accepted for the finite toy model; no eigensolver or spectral basis is
    constructed.  The quadrature truncation error is bounded over the supplied
    spectral enclosure.  The enclosure itself remains the caller's proof
    obligation.
    """

    def __init__(
        self,
        h: torch.Tensor | TensorMap,
        dimension: int,
        *,
        shift: float,
        spectral_gap: float,
        spectral_bound: float,
        bound_certificate: str,
        quadrature_order: int = 65,
        log_step: float = 0.4,
        tolerance: float = 1.0e-9,
        solve_tolerance: float = 1.0e-11,
        max_cg_iterations: int | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if not math.isfinite(shift):
            raise ValueError("shift must be finite")
        if not (math.isfinite(spectral_gap) and math.isfinite(spectral_bound)):
            raise ValueError("spectral bounds must be finite")
        if spectral_gap <= 0.0 or spectral_bound <= spectral_gap:
            raise ValueError("require 0 < spectral_gap < spectral_bound")
        if not bound_certificate.strip():
            raise ValueError("a source for the spectral enclosure is required")
        if quadrature_order < 3 or quadrature_order % 2 != 1:
            raise ValueError("quadrature_order must be an odd number of at least three nodes")
        if not math.isfinite(log_step) or log_step <= 0.0:
            raise ValueError("log_step must be finite and positive")
        if tolerance <= 0.0 or solve_tolerance <= 0.0:
            raise ValueError("tolerances must be positive")
        self.dimension = int(dimension)
        self.shift = float(shift)
        self.spectral_gap = float(spectral_gap)
        self.spectral_bound = float(spectral_bound)
        self.bound_certificate = bound_certificate.strip()
        self.quadrature_order = int(quadrature_order)
        self.log_step = float(log_step)
        self.tolerance = float(tolerance)
        self.solve_tolerance = float(solve_tolerance)
        self.max_cg_iterations = (
            max(4 * dimension, 128)
            if max_cg_iterations is None
            else int(max_cg_iterations)
        )
        if isinstance(h, torch.Tensor):
            if h.shape != (dimension, dimension):
                raise ValueError("dense h must have shape (dimension, dimension)")
            if h.dtype != torch.complex128:
                raise TypeError("h must use complex128")
            if not bool(torch.isfinite(h).all()):
                raise ValueError("h contains non-finite entries")
            hermitian_error = torch.linalg.matrix_norm(h - h.mH)
            if float(hermitian_error.detach()) > 1.0e-10 * max(1.0, float(torch.linalg.matrix_norm(h).detach())):
                raise ValueError("dense h must be Hermitian")
            self._h = h
            self._h_apply = lambda x: h @ x
            self._prototype = h
        else:
            self._h = None
            self._h_apply = h
            self._prototype = torch.empty((), dtype=torch.complex128)
        if self.max_cg_iterations <= 0:
            raise ValueError("max_cg_iterations must be positive")
        if self._prototype.dtype != torch.complex128:
            raise TypeError("the operator's reference tensor must use complex128")
        self.quadrature_error_bound = self._log_trapezoid_error_bound()
        if self.quadrature_error_bound > self.tolerance:
            raise ValueError(
                "quadrature order does not satisfy the enclosure-wide error bound: "
                f"{self.quadrature_error_bound:.3e} > {self.tolerance:.3e}; "
                "increase quadrature_order"
            )
        self.last_max_residual = self._prototype.real.new_tensor(float("nan"))

    def _log_trapezoid_error_bound(self) -> float:
        """Uniform log-trapezoid discretization plus finite-tail error bound.

        With ``t=exp(s)``, the normalized sign integrand is a translated
        ``sech``. Poisson summation bounds the infinite trapezoid error; the
        remaining two tails are bounded geometrically over the declared
        spectral interval.
        """

        h = self.log_step
        fourier = math.pi**2 / h
        discretization = 4.0 * math.exp(-fourier) / (1.0 - math.exp(-fourier))
        ratio = self.spectral_bound / self.spectral_gap
        half_log_ratio = 0.5 * math.log(ratio)
        half_nodes = self.quadrature_order // 2
        tail = (
            4.0 * h / math.pi
            * math.exp(half_log_ratio - (half_nodes + 1) * h)
            / (1.0 - math.exp(-h))
        )
        return discretization + tail

    def _a_apply(self, x: torch.Tensor) -> torch.Tensor:
        return self._h_apply(x) - self.shift * x

    def _normalized_a_apply(self, x: torch.Tensor) -> torch.Tensor:
        scale = math.sqrt(self.spectral_gap * self.spectral_bound)
        return self._a_apply(x) / scale

    @staticmethod
    def _inner_columns(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            return torch.sum(x.conj() * y).real.reshape(1)
        return torch.sum(x.conj() * y, dim=0).real

    def _cg_solve(self, matvec: TensorMap, rhs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiable CG on one RHS; stop tests use detached residuals."""

        x = torch.zeros_like(rhs)
        r = rhs - matvec(x)
        p = r
        rr = self._inner_columns(r, r)
        rhs_norm = torch.linalg.vector_norm(rhs, dim=0).clamp_min(1.0e-30)
        relative = torch.sqrt(rr) / rhs_norm
        iteration = 0
        while float(relative.detach().max()) > self.solve_tolerance and iteration < self.max_cg_iterations:
            ap = matvec(p)
            denominator = self._inner_columns(p, ap)
            active = relative.detach() > self.solve_tolerance
            if bool(torch.any(denominator.detach()[active] <= 0.0)):
                raise RuntimeError("CG operator lost positive definiteness")
            alpha = rr / denominator.clamp_min(1.0e-300)
            alpha = torch.where(active, alpha, torch.zeros_like(alpha))
            if p.ndim == 2:
                alpha = alpha[None, :]
            x = x + alpha * p
            r = r - alpha * ap
            rr_new = self._inner_columns(r, r)
            beta = rr_new / rr.clamp_min(1.0e-300)
            if p.ndim == 2:
                beta = beta[None, :]
            p = r + beta * p
            rr = rr_new
            relative = torch.sqrt(rr) / rhs_norm
            iteration += 1
        if float(relative.detach().max()) > self.solve_tolerance:
            raise RuntimeError(
                f"rational sign CG failed to converge: residual={float(relative.detach().max()):.3e}"
            )
        return x, relative.max()

    def apply_sign(self, vectors: torch.Tensor) -> torch.Tensor:
        """Apply the rational sign approximation to vector(s), preserving gradients."""

        vectors = torch.as_tensor(vectors)
        if vectors.dtype != torch.complex128:
            raise TypeError("sign input must use complex128")
        if vectors.shape[0] != self.dimension or vectors.ndim not in (1, 2):
            raise ValueError("sign input must have shape (D,) or (D,K)")
        was_vector = vectors.ndim == 1
        rhs_columns = vectors[:, None] if was_vector else vectors
        normalized_rhs = self._normalized_a_apply(rhs_columns)
        norms = []
        results = []
        order = self.quadrature_order
        half_nodes = order // 2
        step = self.log_step
        for index in range(-half_nodes, half_nodes + 1):
            t = math.exp(index * step)
            t_sq = t * t

            def shifted(value: torch.Tensor) -> torch.Tensor:
                av = self._normalized_a_apply(value)
                return self._normalized_a_apply(av) + t_sq * value

            if self._h is not None:
                # Small dense prototype path: an exact shifted linear solve is
                # faster than Python-level CG and remains differentiable.  It
                # is still a rational operator action (no eigenbasis/cache).
                scale = math.sqrt(self.spectral_gap * self.spectral_bound)
                shifted_h = (self._h - self.shift * torch.eye(
                    self.dimension, dtype=self._h.dtype, device=self._h.device
                )) / scale
                identity = torch.eye(self.dimension, dtype=self._h.dtype, device=self._h.device)
                matrix = shifted_h @ shifted_h + t_sq * identity
                solved = torch.linalg.solve(matrix, normalized_rhs)
                residual_vec = matrix @ solved - normalized_rhs
                residual = (
                    torch.linalg.vector_norm(residual_vec, dim=0)
                    / torch.linalg.vector_norm(normalized_rhs, dim=0).clamp_min(1.0e-30)
                ).max()
                if float(residual.detach()) > self.solve_tolerance:
                    raise RuntimeError(
                        f"dense rational shifted solve residual={float(residual.detach()):.3e}"
                    )
            else:
                solved, residual = self._cg_solve(shifted, normalized_rhs)
            results.append(t * solved)
            norms.append(residual)
        coefficient = (2.0 / math.pi) * step
        signed = coefficient * torch.stack(results, dim=0).sum(dim=0)
        if norms:
            self.last_max_residual = torch.stack(norms).max()
        return signed[:, 0] if was_vector else signed

    def project_positive(self, vectors: torch.Tensor) -> torch.Tensor:
        """Apply ``(I + sign(h-shift))/2`` without materializing a projector."""

        identity_action = vectors
        return 0.5 * (identity_action + self.apply_sign(vectors))

    def project_factors(self, factors: torch.Tensor) -> torch.Tensor:
        """Project factor columns into the positive Dirac branch, without basis caching."""

        return self.project_positive(factors)

    def project_density(self, rho: torch.Tensor, chi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project active normal/anomalous tensors, retaining the full hole identity."""

        rho = torch.as_tensor(rho)
        chi = torch.as_tensor(chi)
        if rho.shape != (self.dimension, self.dimension) or chi.shape != rho.shape:
            raise ValueError("rho and chi must match the normal-space dimension")
        p_rho = self.project_positive(rho)
        rho_projected = self.project_positive(p_rho.mH).mH
        p_chi = self.project_positive(chi)
        chi_projected = self.project_positive(p_chi.mT).mT
        return rho_projected, chi_projected

    def diagnostics(self) -> SignDiagnostics:
        return SignDiagnostics(
            quadrature_error_bound=self.quadrature_error_bound,
            max_linear_solve_residual=self.last_max_residual,
            quadrature_order=self.quadrature_order,
            spectral_gap=self.spectral_gap,
            spectral_bound=self.spectral_bound,
        )
