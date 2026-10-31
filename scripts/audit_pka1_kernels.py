#!/usr/bin/env python3
"""Compare selected native Python PKA1 kernels with Core-1204 storage."""

from __future__ import annotations

import ctypes
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str((ROOT / "../Core-1204").resolve()))

import rhf_ctypes as core  # type: ignore[import-not-found]

from dpl_rhf.rhf.kernels import PKA1AngularKernels, SphericalPropagators


NPT = 201
NBLOCK = 15
COMPONENT_SIZE = NPT * NPT
OBJECT_SIZE = 4 * COMPONENT_SIZE


def block_index(kappa: int) -> int:
    return -kappa - 1 if kappa < 0 else 7 + kappa


def core_component(symbol: str, kappa_a: int, kappa_b: int, component: int) -> np.ndarray:
    anchor = ctypes.c_double.in_dll(core._lib, symbol)
    object_index = block_index(kappa_a) + NBLOCK * block_index(kappa_b)
    offset = object_index * OBJECT_SIZE + component * COMPONENT_SIZE
    pointer = ctypes.cast(ctypes.addressof(anchor) + 8 * offset, ctypes.POINTER(ctypes.c_double))
    return np.ctypeslib.as_array(pointer, shape=(COMPONENT_SIZE,)).reshape(
        (NPT, NPT), order="F"
    )


def main() -> None:
    core.ddrhf_init(0, 8, 8, 16)
    radius = torch.arange(NPT, dtype=torch.float64) * 0.1
    radius[0] = 0.001
    kernels = PKA1AngularKernels(SphericalPropagators(radius, 197.328284))
    for kappa_a, kappa_b in ((-1, -1), (-2, -1), (-2, 1), (1, 1)):
        python = kernels.pion(kappa_a, kappa_b, 138.0).numpy()
        for component in range(4):
            native = core_component(
                "__combination_MOD_cpio", kappa_a, kappa_b, component
            )
            difference = python[component // 2, component % 2] - native
            print(
                f"CPIO ka={kappa_a:+d} kb={kappa_b:+d} component={component} "
                f"max={np.max(np.abs(difference)):.12e} "
                f"interior={np.max(np.abs(difference[1:-1, 1:-1])):.12e}"
            )


if __name__ == "__main__":
    main()
