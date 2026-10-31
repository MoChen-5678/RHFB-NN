from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RHFReservedChannels:
    """Metadata placeholder for future RHF/Fock development."""

    local_channels: tuple[str, ...] = ("vps_n", "vms_n", "vps_p", "vms_p")
    nonlocal_channels: tuple[str, ...] = ("sigma_fock", "omega_fock", "rho_tensor", "pi_pv")
    orbital_sources: tuple[str, ...] = ("X_a", "Y_a")


def require_rhf_not_implemented() -> None:
    raise NotImplementedError(
        "RHF nonlocal Fock/tensor channels are reserved for a later implementation; "
        "the current PRL trainer supports only local PKDD RMF."
    )
