from __future__ import annotations

import numpy as np


def first_derivative_coefficients(offsets: list[int]) -> np.ndarray:
    matrix = np.array([[float(offset) ** power for offset in offsets] for power in range(len(offsets))])
    rhs = np.zeros(len(offsets))
    rhs[1] = 1.0
    return np.linalg.solve(matrix, rhs)


def asymmetric_first_derivative_row(point_count: int) -> np.ndarray:
    """Forward ADF coefficients for f'(r_i) using point_count points."""
    return first_derivative_coefficients(list(range(point_count)))


def zhang_7point_adf_coefficients() -> np.ndarray:
    """Seven-point asymmetric derivative, the sixth-order extension of 5PADF."""
    return asymmetric_first_derivative_row(7)
