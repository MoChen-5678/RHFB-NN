from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class NucleusCase:
    model: str
    z: int
    n: int
    a: int


@dataclass
class BackendResult:
    energy_per_a_no_com: float
    energy_total_no_com: float
    r: np.ndarray
    hamiltonian: np.ndarray
    reconstructed_hamiltonian: np.ndarray
    grad_energy_h: np.ndarray
    densities: dict[str, np.ndarray]
    fields: dict[str, np.ndarray]
    diagnostics: dict[str, float]
    functional_type: str = "rmf"


class FixedHamiltonianBackend(Protocol):
    case: NucleusCase
    r: np.ndarray

    def evaluate(self, hamiltonian: np.ndarray) -> BackendResult:
        """Evaluate E[H], grad_H E, and reconstructed H_tilde for fixed H."""
