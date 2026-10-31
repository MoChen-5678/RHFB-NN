from __future__ import annotations

import torch

from dpl_rhf.rhfb.no_sea import RationalSignOperator


def _operator(h: torch.Tensor, *, gap: float = 0.9) -> RationalSignOperator:
    return RationalSignOperator(
        h,
        h.shape[0],
        shift=0.0,
        spectral_gap=gap,
        spectral_bound=2.1,
        bound_certificate="test-only exact diagonal spectrum enclosure",
        quadrature_order=65,
        tolerance=3.0e-3,
        solve_tolerance=1.0e-12,
    )


def test_matrix_free_sign_action_matches_dense_test_oracle_and_has_gradients():
    dtype = torch.complex128
    h = torch.diag(torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=dtype)).requires_grad_()
    operator = _operator(h)
    vector = torch.tensor([1 + 0.5j, -0.3j, 0.7, 1 - 0.2j], dtype=dtype)
    result = operator.apply_sign(vector)
    # Dense eigensolve is isolated to this independent test oracle.
    values, vectors = torch.linalg.eigh(h.detach())
    oracle = vectors @ (torch.sign(values).to(dtype) * (vectors.mH @ vector))
    assert torch.linalg.vector_norm(result - oracle) < 3.0e-3
    assert float(operator.last_max_residual) < 1.0e-10
    loss = torch.sum(torch.abs(operator.project_positive(vector)).square())
    loss.backward()
    assert h.grad is not None
    assert torch.isfinite(h.grad).all()


def test_projected_rho_and_pair_tensor_obey_positive_support():
    dtype = torch.complex128
    h = torch.diag(torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=dtype))
    operator = _operator(h)
    rho = torch.zeros((4, 4), dtype=dtype)
    rho[0, 0] = 0.8
    rho[2, 2] = 0.6
    chi = torch.zeros((4, 4), dtype=dtype)
    chi[0, 2] = 0.1 + 0.2j
    chi[2, 0] = -chi[0, 2]
    rho_p, chi_p = operator.project_density(rho, chi)
    assert torch.linalg.matrix_norm(rho_p[:2, :]) < 1.0e-10
    assert torch.linalg.matrix_norm(rho_p[:, :2]) < 1.0e-10
    assert torch.linalg.matrix_norm(chi_p[:2, :]) < 1.0e-6
    assert torch.linalg.matrix_norm(chi_p[:, :2]) < 1.0e-6
    assert torch.linalg.matrix_norm(chi_p + chi_p.mT) < 1.0e-12


def test_sign_operator_rejects_uncertified_or_gapless_domain():
    h = torch.diag(torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=torch.complex128))
    try:
        RationalSignOperator(
            h,
            4,
            shift=0.0,
            spectral_gap=0.0,
            spectral_bound=2.1,
            bound_certificate="",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("a gapless or uncertified branch must be rejected")


def test_log_rational_enclosure_remains_certified_at_spectral_ratio_100():
    dtype = torch.complex128
    h = torch.diag(torch.tensor([-1.0, -0.01, 0.01, 1.0], dtype=dtype))
    operator = RationalSignOperator(
        h,
        4,
        shift=0.0,
        spectral_gap=0.01,
        spectral_bound=1.0,
        bound_certificate="test-only exact ratio-100 spectrum enclosure",
        quadrature_order=129,
        log_step=0.4,
        tolerance=1.0e-9,
        solve_tolerance=1.0e-12,
    )
    vectors = torch.ones(4, dtype=dtype)
    result = operator.apply_sign(vectors)
    assert torch.linalg.vector_norm(result - torch.tensor([-1, -1, 1, 1], dtype=dtype)) < 2.0e-9
    assert operator.quadrature_error_bound < 1.0e-9


def test_callable_matrix_free_cg_path_has_oracle_and_field_directional_fd():
    dtype = torch.complex128
    h0 = torch.diag(torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=dtype))
    perturbation = torch.zeros((4, 4), dtype=dtype)
    perturbation[1, 2] = 0.12 + 0.04j
    perturbation[2, 1] = perturbation[1, 2].conj()
    vector = torch.tensor([0.7 + 0.2j, -0.3j, 0.8, 0.1 - 0.5j], dtype=dtype)
    direction = torch.tensor([0.2, -0.4j, 0.6, 0.1 + 0.2j], dtype=dtype)
    direction = direction / torch.linalg.vector_norm(direction)

    def evaluate(parameter: torch.Tensor, *, with_gradient: bool):
        h = h0 + parameter * perturbation
        operator = RationalSignOperator(
            lambda value: h @ value,
            4,
            shift=0.0,
            spectral_gap=0.7,
            spectral_bound=2.3,
            bound_certificate="Weyl bound from exact diagonal h0 and ||parameter*V||<=0.2",
            quadrature_order=129,
            tolerance=1.0e-9,
            solve_tolerance=1.0e-11,
        )
        projected = operator.project_positive(vector)
        observable = torch.vdot(direction, projected).real
        if with_gradient:
            gradient = torch.autograd.grad(observable, parameter)[0]
            return observable.detach(), float(gradient.detach()), operator
        return observable.detach(), operator

    parameter = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    value, automatic, operator = evaluate(parameter, with_gradient=True)
    step = 2.0e-5
    plus, _ = evaluate(torch.tensor(0.2 + step, dtype=torch.float64), with_gradient=False)
    minus, _ = evaluate(torch.tensor(0.2 - step, dtype=torch.float64), with_gradient=False)
    finite_difference = float(((plus - minus) / (2.0 * step)).detach())

    h = h0 + parameter.detach() * perturbation
    eigenvalues, eigenvectors = torch.linalg.eigh(h)
    exact = eigenvectors @ (
        (eigenvalues > 0.0).to(dtype) * (eigenvectors.mH @ vector)
    )
    dense = RationalSignOperator(
        h,
        4,
        shift=0.0,
        spectral_gap=0.7,
        spectral_bound=2.3,
        bound_certificate="test-only Weyl enclosure",
        quadrature_order=129,
        tolerance=1.0e-9,
        solve_tolerance=1.0e-11,
    ).project_positive(vector)
    assert torch.linalg.vector_norm(operator.project_positive(vector) - exact) < 1.0e-8
    assert torch.linalg.vector_norm(dense - operator.project_positive(vector)) < 1.0e-8
    assert float(operator.last_max_residual) < 1.0e-10
    assert abs(automatic - finite_difference) < 1.0e-6
