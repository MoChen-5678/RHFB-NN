from __future__ import annotations

import torch


def generalized_prl_gradient(
    hamiltonian_mev: torch.Tensor,
    reconstructed_mev: torch.Tensor,
    grad_total_energy_h_mev: torch.Tensor,
    lambda_reconstruct: float,
    energy_gradient_weight: float = 1.0,
) -> torch.Tensor:
    """PRL generalized gradient in the actual Hamiltonian coordinates.

    All Hamiltonians are in MeV and the energy gradient is taken from the
    total no-CoM energy.  Therefore lambda has units MeV^-1.  No channel-wise
    metric is introduced here; the physical component map is differentiated
    by PyTorch when the surrogate is propagated to the model parameters.
    """
    if hamiltonian_mev.shape != reconstructed_mev.shape:
        raise ValueError("trial and reconstructed Hamiltonians must have the same shape")
    if grad_total_energy_h_mev.shape != hamiltonian_mev.shape:
        raise ValueError("energy gradient must have the Hamiltonian shape")
    return (
        float(energy_gradient_weight) * grad_total_energy_h_mev
        + float(lambda_reconstruct) * (hamiltonian_mev.detach() - reconstructed_mev.detach())
    )


def surrogate_loss(hamiltonian: torch.Tensor, generalized_gradient: torch.Tensor) -> torch.Tensor:
    """Scalar whose VJP is exactly the prescribed PRL Hamiltonian gradient."""
    return torch.sum(hamiltonian * generalized_gradient.detach())
