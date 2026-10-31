from __future__ import annotations

import torch


def simpson_weights(n: int, h: float, *, device, dtype) -> torch.Tensor:
    """Core-1204 ``RHFlib:simps`` weights for an equally spaced mesh."""

    if n < 3:
        raise ValueError("Simpson integration requires at least three points")
    if n == 4:
        return torch.tensor(
            [1.0, 3.0, 3.0, 1.0], device=device, dtype=dtype
        ) * (3.0 * h / 8.0)
    weights = torch.ones(n, device=device, dtype=dtype)
    if (n - 1) % 2 == 0:
        weights[1:-1:2] = 4.0
        weights[2:-1:2] = 2.0
        return weights * (h / 3.0)

    weights.zero_()
    weights[:4] = torch.tensor(
        [3.0, 9.0, 9.0, 3.0], device=device, dtype=dtype
    ) * (h / 8.0)
    weights[3:] += simpson_weights(n - 3, h, device=device, dtype=dtype)
    return weights
