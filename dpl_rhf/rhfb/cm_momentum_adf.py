"""Candidate ADF Cartesian momentum operators for post-variation CoM reports.

The three momentum matrices use the integration-by-parts weak form
``p_mu = -i/2 (<a|d_mu|b> - <d_mu a|b>)`` with vanishing surface flux.
The scalar ``p^2`` matrix is evaluated separately
as ``sum_mu <d_mu a|d_mu b>``.  It is deliberately not formed as a sum of
products of the finite active-space momentum matrices: that would insert an
active-space projector between the two momenta.

The momentum uses the split weak form obtained from continuum integration by
parts with vanishing surface flux. It is a different finite-grid prescription
from the one-sided ADF derivative; the raw-versus-weak and IBP defects are
reported and require convergence checks. Algebraically, the split form equals
the Hermitian part of the raw one-sided matrix; that choice is explicit and
must not conceal a nonconvergent defect. These matrices are only candidates.
This module has no connection to the variational Hamiltonian or trainer.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np

from .center_of_mass import evaluate_center_of_mass_correction


@dataclass(frozen=True)
class BasisMomentumOperators:
    """Independent ADF p and p² matrices, in inverse-fm units."""

    momentum_fm_inverse: np.ndarray
    momentum_squared_fm_inverse2: np.ndarray
    raw_momentum_fm_inverse: np.ndarray
    integration_by_parts_defect_fm_inverse: np.ndarray
    basis_signature: str
    operator_sha256: str
    hbar_c_mev_fm: float
    diagnostics: Mapping[str, Any]
    status: str = "candidate_unvalidated"


_REQUIRED_VALIDATION = (
    "momentum_hermiticity",
    "integration_by_parts_defect_convergence",
    "direct_p2_hermiticity",
    "boundary_flux",
    "radial_order_convergence",
    "box_radius_convergence",
    "angular_cutoff_convergence",
    "spherical_reference",
    "direct_p2_reference",
)


def _basis_signature(basis) -> str:
    metadata = {
        "dimension": int(basis.dimension),
        "radius_fm": float(basis.radius_fm),
        "spacing_fm": float(basis.spacing_fm),
        "kappa_max": int(basis.kappa_max),
        "derivative_order": int(basis.derivative_order),
        "n_theta": int(basis.n_theta),
        "n_phi": int(basis.n_phi),
        "hbar_c_mev_fm": float(basis.hbar_c_mev_fm),
        "channels": [(int(c.kappa), int(c.m2), int(c.offset)) for c in basis.channels],
    }
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for name in ("x", "weights", "angular_weights"):
        digest.update(np.ascontiguousarray(getattr(basis, name)).view(np.uint8))
    return digest.hexdigest()


def _matrix_residual(matrix: np.ndarray) -> tuple[float, float]:
    difference = matrix - matrix.conj().T
    absolute = float(np.max(np.abs(difference))) if matrix.size else 0.0
    scale = max(1.0, float(np.max(np.abs(matrix))) if matrix.size else 0.0)
    return absolute, absolute / scale


def _reduced_spinor_gradient(basis, green, reduced_derivatives, coefficients):
    """Cartesian gradient using reduced radial amplitudes and exact 1/r terms.

    ``basis._sample_range`` returns physical spinors G(r)Omega/r. Multiplying
    by r before the ADF radial differentiation avoids applying a finite
    difference directly to the product G/r. The angular derivative remains
    analytic in the spherical-harmonic expansion, and the radial/angular
    pieces are then combined with the continuum product rule.
    """
    import torch

    point_count = int(basis.point_count)
    spinors = basis._sample_range(coefficients, 0, point_count)
    radii = torch.as_tensor(
        np.repeat(basis.x, basis.n_angular_points),
        dtype=torch.float64, device=spinors.device,
    )
    reduced = spinors * radii[:, None, None]
    angular = green._analyze(reduced)
    derivative = torch.einsum("lij,jlc->ilc", reduced_derivatives, angular)
    radial_r = torch.einsum("al,rlc,ai->raic", green.y, derivative, green.er)
    radial_value = torch.einsum("al,rlc,ai->raic", green.y, angular, green.er)
    radius_grid = radii.reshape(basis.n_radial, basis.n_angular_points)[:, :, None, None]
    radial = radial_r / radius_grid - radial_value / radius_grid.square()
    angular_part = torch.einsum("ali,rlc->raic", green.angular_gradient, angular)
    angular_part = angular_part / radius_grid.square()
    return (radial + angular_part).reshape(point_count, 3, 4, coefficients.shape[1]), spinors


def basis_momentum_operators(
    basis,
    *,
    column_chunk_size: int = 8,
    device: str = "cpu",
    hermiticity_tolerance: float = 1.0e-8,
    max_gradient_bytes: int = 512 * 1024 * 1024,
) -> BasisMomentumOperators:
    """Construct ADF-coordinate p_mu and independent weak-form p² candidates.

    The basis sampler, the same ADF radial stencil with reduced-amplitude
    parity, and GreenOperator's analytic angular expansion are used. The
    exact continuum ``1/r`` product rule is then applied. For ``p²``, all
    derivative components are integrated before any finite-space projection;
    no intermediate truncated p_mu matrix is used.

    The continuum integration-by-parts identification of this weak p² with
    ``-nabla²`` assumes vanishing surface flux at the origin and outer box.
    This routine reports the matrices as unvalidated candidates and cannot
    certify that assumption or mesh convergence.
    """
    # Keep the repository's enforced memory admission ahead of tensor/matrix
    # allocation.  This is a numerical diagnostic entry point.
    from .execution_limits import require_bounded_cgroup
    resource_evidence = require_bounded_cgroup()

    if isinstance(column_chunk_size, bool) or int(column_chunk_size) != column_chunk_size or column_chunk_size < 1:
        raise ValueError("column_chunk_size must be a positive integer")
    if isinstance(max_gradient_bytes, bool) or int(max_gradient_bytes) != max_gradient_bytes or max_gradient_bytes < 1:
        raise ValueError("max_gradient_bytes must be a positive integer")
    if not np.isfinite(hermiticity_tolerance) or hermiticity_tolerance < 0:
        raise ValueError("hermiticity_tolerance must be finite and nonnegative")

    import torch
    from .spatial import GreenOperator
    from .spatial import _scalar_adf

    dimension = int(basis.dimension)
    point_count = int(basis.point_count)
    gradient_bytes = point_count * (3 * 4 + 4) * dimension * np.dtype(np.complex128).itemsize
    if gradient_bytes > max_gradient_bytes:
        raise MemoryError(
            "direct p^2 and weak p require retaining coordinate gradients/spinors "
            f"contractions ({gradient_bytes} bytes exceeds cap {max_gradient_bytes}); "
            "increase the cap only after reviewing the enforced memory budget"
        )
    device_obj = torch.device(device)
    green = GreenOperator(basis, lmax=2 * int(basis.kappa_max), device=device_obj)
    stencil_points = 5 if basis.derivative_order in (4, 5) else 7
    reduced_derivatives = torch.as_tensor(np.stack([
        _scalar_adf(int(basis.n_radial), float(basis.spacing_fm), int(l) + 1, stencil_points)
        for l, _m in green.orders
    ]), dtype=torch.complex128, device=device_obj)
    identity = torch.eye(dimension, dtype=torch.complex128, device=device_obj)
    p = np.zeros((3, dimension, dimension), dtype=np.complex128)
    raw_p = np.zeros_like(p)
    ibp_defect = np.zeros_like(p)
    gradients = np.empty((point_count, 3, 4, dimension), dtype=np.complex128)
    spinors_all = np.empty((point_count, 4, dimension), dtype=np.complex128)

    # Bound derivative temporary tensors by batching columns; retain gradients
    # for both the independent p² and integration-by-parts momentum forms.
    for start in range(0, dimension, int(column_chunk_size)):
        stop = min(start + int(column_chunk_size), dimension)
        gradient, spinors = _reduced_spinor_gradient(
            basis, green, reduced_derivatives, identity[:, start:stop])
        gradients[:, :, :, start:stop] = gradient.detach().cpu().numpy()
        spinors_all[:, :, start:stop] = spinors.detach().cpu().numpy()

    # Appendix-C one-body p² is evaluated independently in the weak form,
    # before any active-space momentum projection.
    p2 = np.einsum(
        "pica,picb,p->ab", gradients.conj(), gradients,
        np.asarray(basis.weights, dtype=np.float64), optimize=True,
    )

    # Weak momentum follows continuum integration by parts with vanishing
    # surface flux: -i/2 ( <a|d_mu|b> - <d_mu a|b> ). This split weak form is
    # not the same finite-grid operator as the one-sided ADF derivative.
    # Retain raw p and the discrete IBP defect so convergence can be assessed.
    raw_momentum_residuals = []
    ibp_defect_absmax = []
    volume_weights = np.asarray(basis.weights, dtype=np.float64)
    for axis in range(3):
        right = np.einsum("pca,pcb,p->ab", spinors_all.conj(),
                          gradients[:, axis], volume_weights, optimize=True)
        left = np.einsum("pca,pcb,p->ab", gradients[:, axis].conj(),
                         spinors_all, volume_weights, optimize=True)
        raw_p[axis] = -1j * right
        ibp_defect[axis] = right + left
        p[axis] = (-0.5j) * (right - left)
        raw_momentum_residuals.append(_matrix_residual(raw_p[axis]))
        ibp_defect_absmax.append(float(np.max(np.abs(ibp_defect[axis]))))

    p_residuals = [_matrix_residual(p[axis]) for axis in range(3)]
    p2_residual = _matrix_residual(p2)
    worst_p = max((item[1] for item in p_residuals), default=0.0)
    if worst_p > hermiticity_tolerance:
        raise ValueError(
            "ADF Cartesian momentum candidate is not Hermitian under the supplied "
            f"sampling metric/boundary treatment (relative residual {worst_p:.3e})"
        )
    if p2_residual[1] > hermiticity_tolerance:
        raise ValueError(
            "ADF direct weak-form p^2 candidate is not Hermitian under the supplied "
            f"sampling metric/boundary treatment (relative residual {p2_residual[1]:.3e})"
        )

    signature = _basis_signature(basis)
    op_hash = hashlib.sha256()
    op_hash.update(np.ascontiguousarray(p).view(np.uint8))
    op_hash.update(np.ascontiguousarray(p2).view(np.uint8))
    op_hash.update(np.ascontiguousarray(raw_p).view(np.uint8))
    op_hash.update(np.ascontiguousarray(ibp_defect).view(np.uint8))
    diagnostics = {
        "construction": "weak p_mu=-i/2(<psi|d_mu|psi>-<d_mu psi|psi>); independent p2=sum_mu<d_mu psi|d_mu psi>",
        "momentum_discretization": "split weak form -i/2(<a|d|b>-<d a|b>) from integration by parts; differs from one-sided ADF derivative",
        "split_form_algebraically_equals_hermitian_part_of_raw_one_sided_matrix": True,
        "radial_representation": "differentiate reduced G,F with ADF parity l+1, then apply exact continuum 1/r product rule and analytic spin-angular gradient",
        "column_chunk_size": int(column_chunk_size),
        "retained_gradient_and_spinor_bytes": int(gradient_bytes),
        "p_units": "fm^-1",
        "p2_units": "fm^-2",
        "p_hermiticity_absolute_by_axis": [x[0] for x in p_residuals],
        "p_hermiticity_relative_by_axis": [x[1] for x in p_residuals],
        "raw_one_sided_p_hermiticity_relative_by_axis": [x[1] for x in raw_momentum_residuals],
        "integration_by_parts_defect_absmax_by_axis": ibp_defect_absmax,
        "integration_by_parts_defect_relative_by_axis": [
            float(np.linalg.norm(ibp_defect[axis]) / max(
                1.0, np.linalg.norm(raw_p[axis])))
            for axis in range(3)
        ],
        "p2_hermiticity_absolute": p2_residual[0],
        "p2_hermiticity_relative": p2_residual[1],
        "p2_is_independent_of_compressed_p_products": True,
        "surface_flux_zero_assumed_not_certified": True,
        "surface_boundary_model": "regular parity at origin; zero exterior at outer box; independent mesh/box validation still required",
        "physical_convergence_certified": False,
        "resource_evidence": resource_evidence,
    }
    return BasisMomentumOperators(
        momentum_fm_inverse=p,
        momentum_squared_fm_inverse2=p2,
        raw_momentum_fm_inverse=raw_p,
        integration_by_parts_defect_fm_inverse=ibp_defect,
        basis_signature=signature,
        operator_sha256=op_hash.hexdigest(),
        hbar_c_mev_fm=float(basis.hbar_c_mev_fm),
        diagnostics=diagnostics,
    )


def evaluate_basis_center_of_mass_post_variation(
    states: Mapping[str, Any],
    operators: BasisMomentumOperators,
    *,
    mass_number: int,
    nucleon_mass_mev: float,
    validation_certificate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the Wick CoM correction after variation from accepted states.

    ``states`` contains neutron/proton objects exposing ``q``,
    ``rho_active``, and ``chi_active``.  A complete matching validation
    certificate is recorded as an external declaration only. This module
    does not independently verify convergence/boundary evidence, so it never
    promotes a value to ``physical_result_accepted=True``. The algebraic value
    is always a diagnostic, not an energy or field term.
    """
    if set(states) != {"n", "p"}:
        raise ValueError("states must contain exactly neutron and proton states")
    if operators.status != "candidate_unvalidated":
        raise ValueError("unexpected momentum-operator status")
    densities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for species in ("n", "p"):
        state = states[species]
        q = state.q.detach().cpu().numpy()
        rho_active = state.rho_active.detach().cpu().numpy()
        chi_active = state.chi_active.detach().cpu().numpy()
        if q.ndim != 2 or rho_active.shape != (q.shape[1], q.shape[1]) or chi_active.shape != rho_active.shape:
            raise ValueError(f"{species} state factors have incompatible dimensions")
        densities[species] = (q @ rho_active @ q.conj().T,
                              q @ chi_active @ q.T)

    common_p = operators.momentum_fm_inverse
    common_p2 = operators.momentum_squared_fm_inverse2
    if common_p.shape[1] != common_p.shape[2] or common_p2.shape != common_p.shape[1:]:
        raise ValueError("operator dimensions are inconsistent")
    value = evaluate_center_of_mass_correction(
        densities,
        common_p,
        common_p2,
        mass_number=mass_number,
        nucleon_mass_mev=nucleon_mass_mev,
        momentum_units="fm^-1",
        hbar_c_mev_fm=operators.hbar_c_mev_fm,
    )

    validation = {key: False for key in _REQUIRED_VALIDATION}
    certificate_id = None
    matched_declaration = False
    if validation_certificate is not None:
        certificate_id = validation_certificate.get("certificate_id")
        matched_declaration = (
            validation_certificate.get("basis_signature") == operators.basis_signature
            and validation_certificate.get("operator_sha256") == operators.operator_sha256
            and bool(certificate_id)
        )
        supplied = validation_certificate.get("checks", {})
        validation = {key: supplied.get(key) is True for key in _REQUIRED_VALIDATION}
    declared = matched_declaration and all(validation.values())
    value.update({
        "scope": "post-variation diagnostic; no normal or pairing fields are returned",
        "included_in_variation": False,
        "external_validation_asserted": bool(declared),
        "external_validation_machine_verified": False,
        "physical_result_accepted": False,
        "operator_status": operators.status,
        "basis_signature": operators.basis_signature,
        "operator_sha256": operators.operator_sha256,
        "validation_certificate_id": certificate_id,
        "validation_checks": validation,
        "operator_diagnostics": dict(operators.diagnostics),
    })
    return value


class PostVariationCoMObserver:
    """Checkpoint-compatible post-variation CoM observable.

    The observer hashes the exact basis/operator identity and all mass/unit
    conventions. It only reports a Wick correction; it never modifies the
    training energy or its normal/pairing fields.
    """

    def __init__(
        self,
        basis,
        operators: BasisMomentumOperators,
        *,
        mass_number: int,
        nucleon_mass_mev: float,
        validation_certificate: Mapping[str, Any] | None = None,
    ) -> None:
        if operators.basis_signature != _basis_signature(basis):
            raise ValueError("momentum operators were built for a different basis")
        if not np.isclose(operators.hbar_c_mev_fm, basis.hbar_c_mev_fm, rtol=0, atol=1e-12):
            raise ValueError("momentum operator and ADF basis use different hbar-c values")
        if (isinstance(mass_number, bool) or int(mass_number) != mass_number
                or mass_number < 1):
            raise ValueError("mass_number must be a positive integer")
        if not np.isfinite(nucleon_mass_mev) or nucleon_mass_mev <= 0:
            raise ValueError("nucleon_mass_mev must be finite and positive")
        self.basis = basis
        self.operators = operators
        self.mass_number = int(mass_number)
        self.nucleon_mass_mev = float(nucleon_mass_mev)
        self.validation_certificate = validation_certificate

    def checkpoint_metadata(self) -> dict[str, Any]:
        certificate = None if self.validation_certificate is None else dict(self.validation_certificate)
        metadata: dict[str, Any] = {
            "prescription": "post-variation Geng-Long arXiv:2202.00482 Eq.(41), Appendix C",
            "included_in_variation": False,
            "basis_signature": self.operators.basis_signature,
            "operator_sha256": self.operators.operator_sha256,
            "momentum_units": "fm^-1",
            "p2_units": "fm^-2",
            "mass_number": self.mass_number,
            "nucleon_mass_mev": self.nucleon_mass_mev,
            "hbar_c_mev_fm": float(self.basis.hbar_c_mev_fm),
            "validation_certificate": certificate,
        }
        metadata["observer_sha256"] = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return metadata

    def evaluate(self, states: Mapping[str, Any]) -> dict[str, Any]:
        return evaluate_basis_center_of_mass_post_variation(
            states,
            self.operators,
            mass_number=self.mass_number,
            nucleon_mass_mev=self.nucleon_mass_mev,
            validation_certificate=self.validation_certificate,
        )
