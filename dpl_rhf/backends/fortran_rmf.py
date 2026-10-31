from __future__ import annotations

import numpy as np

from dpl_rhf.backends.base import BackendResult, NucleusCase
from dpl_rhf.functionals.pkdd_action import rmf_parameters
from dpl_rhf.legacy.dpl_rmf_core import RHFCore


class FortranRMFBackend:
    """Fixed-Hamiltonian RMF backend for PRL-style E[H] training.

    The backend owns the Dirac solve. The trainer receives only local fields,
    densities, energies, reconstructed Hamiltonians, and Hamiltonian gradients.
    """

    def __init__(self, case: NucleusCase):
        self.case = case
        self.params = rmf_parameters(case.model)
        self.core = RHFCore()
        self.core.init(case.model, case.z, case.n, case.a)
        self.r = self.core.r.copy()

    def evaluate(self, hamiltonian: np.ndarray) -> BackendResult:
        stack = np.asarray(hamiltonian, dtype=np.float64)
        if stack.shape != (4, self.core.npt):
            raise ValueError(f"expected Hamiltonian shape (4, {self.core.npt}), got {stack.shape}")
        self.core.set_local_stack(stack)
        self.core.solve_fixed_potential()
        energy = self.core.energy()
        densities = self.core.densities()
        fields_before_rebuild = self.core.fields()
        grad_energy_h = self._native_hamiltonian_gradient()
        self.core.rebuild_rmf_potentials()
        reconstructed = self.core.local_stack()
        fields = self.core.fields()
        diagnostics = {
            "e_per_a_no_com": energy.e_per_A_no_com,
            "e_per_a_with_com": energy.e_per_A_with_com,
            "rms_n_no_com": energy.rms_n_no_com,
            "rms_p_no_com": energy.rms_p_no_com,
            "rms_matter_no_com": energy.rms_matter_no_com,
            "charge_radius_no_com": energy.charge_radius_no_com,
            "gradient_norm": float(np.linalg.norm(grad_energy_h)),
            "reconstruction_rmse": float(np.sqrt(np.mean((stack - reconstructed) ** 2))),
        }
        return BackendResult(
            energy_per_a_no_com=energy.e_per_A_no_com,
            energy_total_no_com=energy.e_total_no_com,
            r=self.r.copy(),
            hamiltonian=stack.copy(),
            reconstructed_hamiltonian=reconstructed,
            grad_energy_h=grad_energy_h,
            densities=densities,
            fields={**fields_before_rebuild, **{f"rebuilt_{k}": v for k, v in fields.items()}},
            diagnostics=diagnostics,
        )

    def _native_hamiltonian_gradient(self) -> np.ndarray:
        """Fortran-native Hellmann-Feynman gradient for local shifted channels."""
        return self.core.hamiltonian_gradient()
