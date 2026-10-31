"""Differentiable spherical density-matrix RHF solver.

This package is deliberately isolated from the validated RMF implementation.
It provides complete spherical PKO2 and PKA1 density-dependent functionals.
"""

from dpl_rhf.rhf.backend import DensityMatrixRHFBackend, DensityMatrixRHFResult
from dpl_rhf.rhf.functional import PKO2RHFFunctional, SphericalRHFFunctional
from dpl_rhf.rhf.parameters import (
    PKO2Parameters,
    RHFParameters,
    pka1_parameters,
    pko2_parameters,
    rhf_parameters,
)
from dpl_rhf.rhf.trainer import RHFTrainingConfig, train_rhf

__all__ = [
    "DensityMatrixRHFBackend",
    "DensityMatrixRHFResult",
    "PKO2Parameters",
    "RHFParameters",
    "PKO2RHFFunctional",
    "SphericalRHFFunctional",
    "RHFTrainingConfig",
    "pka1_parameters",
    "pko2_parameters",
    "rhf_parameters",
    "train_rhf",
]
