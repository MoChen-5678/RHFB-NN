from __future__ import annotations

import math

import numpy as np

from dpl_rhf.rhf.parameters import RHFParameters


SHELLS = [
    ("1s1/2", -1, 0, 1, 2), ("1p3/2", -2, 1, 1, 4),
    ("1p1/2", 1, 1, 1, 2), ("1d5/2", -3, 2, 1, 6),
    ("2s1/2", -1, 0, 2, 2), ("1d3/2", 2, 2, 1, 4),
    ("1f7/2", -4, 3, 1, 8), ("2p3/2", -2, 1, 2, 4),
    ("1f5/2", 3, 3, 1, 6), ("2p1/2", 1, 1, 2, 2),
    ("1g9/2", -5, 4, 1, 10), ("1g7/2", 4, 4, 1, 8),
    ("2d5/2", -3, 2, 2, 6), ("2d3/2", 2, 2, 2, 4),
    ("3s1/2", -1, 0, 3, 2), ("1h11/2", -6, 5, 1, 12),
    ("2f7/2", -4, 3, 2, 8), ("1h9/2", 5, 5, 1, 10),
    ("1i13/2", -7, 6, 1, 14), ("3p3/2", -2, 1, 3, 4),
    ("2f5/2", 3, 3, 2, 6), ("3p1/2", 1, 1, 3, 2),
]


def _spectroscopic_letter(angular: int) -> str:
    letters = "spdfghiklmnoqrtuvwxyz"
    return letters[angular] if angular < len(letters) else f"l{angular}"


def _spherical_shells(target_capacity: int) -> list[tuple[str, int, int, int, int]]:
    shells = list(SHELLS)
    capacity = sum(shell[4] for shell in shells)
    seen = {(shell[2], shell[3], shell[1]) for shell in shells}
    major = 0
    while capacity < target_capacity:
        for angular in range(major, -1, -2):
            principal = (major - angular) // 2 + 1
            for kappa in ([-(angular + 1), angular] if angular else [-1]):
                key = (angular, principal, kappa)
                if key in seen:
                    continue
                seen.add(key)
                twice_j = 2 * abs(kappa) - 1
                degeneracy = 2 * abs(kappa)
                shells.append(
                    (
                        f"{principal}{_spectroscopic_letter(angular)}{twice_j}/2",
                        kappa,
                        angular,
                        principal,
                        degeneracy,
                    )
                )
                capacity += degeneracy
        major += 1
    return shells


def _generalized_laguerre(order: int, alpha: float, x: np.ndarray) -> np.ndarray:
    if order == 0:
        return np.ones_like(x)
    previous = np.ones_like(x)
    current = 1.0 + alpha - x
    for k in range(2, order + 1):
        previous, current = current, (
            (2 * k - 1 + alpha - x) * current - (k - 1 + alpha) * previous
        ) / k
    return current


def independent_rhf_orbital_data(
    z: int,
    n: int,
    r: np.ndarray,
    params: RHFParameters,
    *,
    extra_capacity: int = 12,
    kappa_limits: tuple[int, int] | None = None,
    capacities: dict[str, int] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Construct analytic occupied factors without an RMF/RHF eigenproblem.

    ``extra_capacity`` sets how many states beyond the particle number are
    represented (canonical space of the RHFB solver).  ``kappa_limits=(a, b)``
    keeps only shells with ``-a <= kappa <= b``.
    """

    records = []
    a = z + n
    hbar_omega = 41.0 * a ** (-1.0 / 3.0)
    for species, number, sign in (("n", n, 1.0), ("p", z, -1.0)):
        mass = params.mass_n if species == "n" else params.mass_p
        oscillator_length = math.sqrt(params.hbar_c**2 / (mass * hbar_omega))
        remaining = int(number)
        represented = 0
        target_capacity = int(number) + int(extra_capacity)
        if capacities is not None and capacities.get(species) is not None:
            target_capacity = int(capacities[species])
            if target_capacity < number:
                raise ValueError(f"{species} capacity {target_capacity} is below {number}")
        for name, kappa, angular, principal, degeneracy in _spherical_shells(
            target_capacity
        ):
            if represented >= target_capacity and remaining <= 0:
                break
            if kappa_limits is not None and not (
                -kappa_limits[0] <= kappa <= kappa_limits[1]
            ):
                if remaining > 0:
                    raise ValueError(f"kappa limits exclude occupied shell {name}")
                continue
            occupation = min(max(remaining, 0), degeneracy) / degeneracy
            q = (r / oscillator_length) ** 2
            radial = _generalized_laguerre(principal - 1, angular + 0.5, q)
            large = r ** (angular + 1) * np.exp(-0.5 * q) * radial
            large /= np.sqrt(np.trapezoid(large * large, r))
            records.append(
                (species, sign, name, kappa, angular, principal, occupation, degeneracy, large)
            )
            remaining -= min(max(remaining, 0), degeneracy)
            represented += degeneracy
        if remaining:
            raise ValueError(f"shell table does not cover {species}={number}")

    v0, av = -71.28, 11.1175
    radius_plus = 0.5 * (1.2334 + 1.2496) * a ** (1.0 / 3.0)
    radius_minus = 0.5 * (1.1443 + 1.1400) * a ** (1.0 / 3.0)
    sigma_plus = v0 / (1.0 + np.exp((r - radius_plus) / (0.5 * (0.6150 + 0.6124))))
    sigma_minus = -v0 * av / (
        1.0 + np.exp((r - radius_minus) / (0.5 * (0.6476 + 0.6469)))
    )
    charge_radius = 1.2496 * a ** (1.0 / 3.0)
    initial_coulomb = np.where(
        r < charge_radius,
        (1.0 / params.alpha_inv)
        * z
        * (3.0 / (2.0 * charge_radius) - r * r / (2.0 * charge_radius**3)),
        (1.0 / params.alpha_inv) * z / r,
    )
    epsilon_initial = -0.5 * hbar_omega
    initial_large = np.stack([record[8] for record in records])
    initial_small = []
    for record, large in zip(records, initial_large):
        coulomb = initial_coulomb if record[0] == "p" else 0.0
        mass = params.mass_n if record[0] == "n" else params.mass_p
        vms = sigma_minus / params.hbar_c + coulomb - 2.0 * mass / params.hbar_c
        derivative = np.gradient(large, r, edge_order=2)
        small = (derivative + record[3] * large / r) / (
            epsilon_initial / params.hbar_c - vms
        )
        norm = np.sqrt(np.trapezoid(large * large + small * small, r))
        large /= norm
        initial_small.append(small / norm)

    data = {
        "G": initial_large,
        "F": np.stack(initial_small),
        "kappa": np.asarray([record[3] for record in records], dtype=np.int32),
        "principal": np.asarray([record[5] for record in records], dtype=np.int32),
        "occupancy": np.asarray([record[6] for record in records]),
        "degeneracy": np.asarray([record[7] for record in records]),
        "species_sign": np.asarray([record[1] for record in records]),
    }
    metadata = {
        "species": np.asarray([record[0] for record in records]),
        "it": np.asarray([1 if record[0] == "n" else 2 for record in records]),
        "index": np.arange(1, len(records) + 1, dtype=np.int32),
        "name": np.asarray([record[2] for record in records]),
        "l": np.asarray([record[4] for record in records], dtype=np.int32),
        "n": np.asarray([record[5] for record in records], dtype=np.int32),
    }
    return data, metadata
