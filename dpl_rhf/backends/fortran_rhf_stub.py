from __future__ import annotations


class FortranRHFBackend:
    """Reserved interface for future RHF/Fock-channel development."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "RHF backend interface is reserved but not implemented. "
            "Use FortranRMFBackend for PKDD/RMF Hartree calculations."
        )
