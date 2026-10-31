"""Isolated finite-space reference for DIRHB Eqs. (46)--(48).

This is NOT a training operator or nuclear solver. It uses a tiny Hermitian
eigensolve as an independent oracle to quantify the omitted-negative-qp
approximation discussed in https://arxiv.org/pdf/1403.4039, pp. 12--13.
No spectral window or branch rule for arbitrary NN fields is inferred.
"""
from __future__ import annotations

import torch

from .execution_limits import require_bounded_cgroup


def _density_metrics(u, v):
    rho, chi = v.conj() @ v.T, v.conj() @ u.T
    n = rho.shape[0]
    eye = torch.eye(n, dtype=torch.complex128)
    r = torch.cat((torch.cat((rho, chi), 1),
                   torch.cat((-chi.conj(), eye-rho.conj()), 1)), 0)
    skew = .5*(chi-chi.T)
    r_repaired = torch.cat((torch.cat((rho, skew), 1),
                            torch.cat((-skew.conj(), eye-rho.conj()), 1)), 0)
    return {
        "mean_number": float(torch.trace(rho).real),
        "chi_skew_error": float(torch.linalg.vector_norm(chi+chi.T)),
        "generalized_hermiticity_error": float(torch.linalg.vector_norm(r-r.mH)),
        "generalized_purity_error": float(torch.linalg.vector_norm(r@r-r)),
        "purity_error_after_chi_skew_symmetrization": float(torch.linalg.vector_norm(r_repaired@r_repaired-r_repaired)),
    }


def completeness_reference(*, training_path: bool = False):
    """Compare full mixed-branch and truncated V sums in two fixed 8x8 cases.

    The single-particle model is declared explicitly: one sea level and three
    Fermi levels, with h_D-M energies (-1900,-5,-3,10) MeV and mu=-4 MeV.
    Its unpaired branch labels are analytic. The largest pairing row sum is
    below 3 MeV, so Weyl's bound keeps the sea pair separated from the entire
    finite Fermi band by over 1800 MeV. Index selection is justified only for
    this constructed reference, including its continuum cutoff; it cannot be
    transplanted into a general fixed-window NN selector.
    """
    if training_path:
        raise RuntimeError("reference eigensolve is forbidden on the training path")
    require_bounded_cgroup()
    h = torch.diag(torch.tensor([-1900.,-5.,-3.,10.], dtype=torch.complex128))
    mu, n = -4., 4
    eye = torch.eye(n, dtype=torch.complex128)
    results = []
    for coupling in (0., .8):
        delta = torch.zeros_like(h)
        delta[2,3] = 1.2
        delta[0,2] = coupling * (1.+.125j)
        delta = delta-delta.T
        bound = float(delta.abs().sum(1).max())
        assert 1896.-bound > 14.+bound  # proven cluster separation in this model
        ham = torch.cat((torch.cat((h-mu*eye,delta),1),
                         torch.cat((-delta.conj(),-h.conj()+mu*eye),1)),0)
        energies, vectors = torch.linalg.eigh(ham)  # isolated reference ONLY
        positive = torch.nonzero(energies>0., as_tuple=False).flatten()
        assert positive.numel() == n
        # The last positive mode is the separated sea PH partner. Three
        # remaining positive modes are Fermi qps; the first negative mode
        # is their omitted sea annihilator in the full no-sea vacuum.
        fermi_indices = positive[:-1]
        complete_indices = torch.cat((torch.tensor([0]),fermi_indices))
        full = vectors[:,complete_indices]
        truncated = vectors[:,fermi_indices]
        metrics_full = _density_metrics(full[:n],full[n:])
        metrics_truncated = _density_metrics(truncated[:n],truncated[n:])
        results.append({
            "sea_pairing_coupling_mev":coupling,
            "reference_pairing_row_bound_mev":bound,
            "full_mixed_qp_sum":metrics_full,
            "positive_fermi_qp_sum_only":metrics_truncated,
            "omitted_sea_v_squared":float(full[n:,0].abs().square().sum()),
            "minimum_abs_qp_energy_mev":float(energies.abs().min()),
        })
    return {"scope":"isolated finite-space algebraic reference, not O17 or a self-consistent nucleus",
            "reference_hamiltonian_eigensolves":2, "training_hamiltonian_eigensolves":0,
            "fortran_used":False,"basis_refreshes":0,
            "physical_result_accepted":False,"nn_no_sea_algorithm_certified":False,
            "cases":results}
