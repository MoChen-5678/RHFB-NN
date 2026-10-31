from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HamiltonianChannelSpec:
    names: tuple[str, ...]
    scales: tuple[float, ...]
    functional_type: str


class FunctionalBase:
    channel_spec: HamiltonianChannelSpec

    def initial_hamiltonian(self):
        raise NotImplementedError
