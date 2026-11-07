"""Readers for S-RHFB-C (Fortran) output files.

Only files written by the reference program are read here; nothing is solved.
The ``POT/`` files come from the read-only ``FieldDump`` module added to the
field-dump copy of S-RHFB-C (converged local potentials and Fock kernels).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ORBITAL_L = {letter: l for l, letter in enumerate("spdfghijklmno")}


def kappa_from_block_label(label: str) -> int:
    """``N-s.1`` / ``N-h11`` (species, l letter, 2j) -> Dirac kappa."""

    match = re.match(r"^\s*[NP]-([a-z])\.?(\d+)\s*$", label)
    if match is None:
        raise ValueError(f"unrecognized S-RHFB-C block label {label!r}")
    l = _ORBITAL_L[match.group(1)]
    twice_j = int(match.group(2))
    return -(l + 1) if twice_j == 2 * l + 1 else l


def kappa_from_state_label(label: str) -> tuple[int, int]:
    """``N.1s.1/2`` / ``N10s.1/2`` -> (principal, kappa)."""

    match = re.match(r"^\s*[NP]\.?\s*(\d+)([a-z])\.?(\d+)/2\s*$", label)
    if match is None:
        raise ValueError(f"unrecognized S-RHFB-C state label {label!r}")
    principal = int(match.group(1))
    l = _ORBITAL_L[match.group(2)]
    twice_j = int(match.group(3))
    kappa = -(l + 1) if twice_j == 2 * l + 1 else l
    return principal, kappa


@dataclass
class CanonicalBlock:
    kappa: int
    labels: list[str]
    energies: np.ndarray      # canonical single-particle energies (MeV)
    occupations: np.ndarray   # v^2
    upper: np.ndarray         # G(r), shape (npt, nstates)
    lower: np.ndarray         # F(r), shape (npt, nstates)


def _read_wac(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        header = lines[i].strip()
        if not re.match(r"^[NP]-", header):
            i += 1
            continue
        ee = [float(v) for v in lines[i + 1].split()[1:]]
        vv = [float(v) for v in lines[i + 2].split()[1:]]
        labels = re.findall(r"[NP][.\d]\S*", lines[i + 4])
        rows = []
        j = i + 5
        while j < len(lines) and lines[j].strip():
            rows.append([float(v) for v in lines[j].split()])
            j += 1
        data = np.asarray(rows)
        blocks.append({
            "label": header, "ee": np.asarray(ee), "vv": np.asarray(vv),
            "states": labels, "r": data[:, 0], "values": data[:, 1:],
        })
        i = j
    return blocks


def read_canonical_states(run_dir: Path, nucleus: str, species: str,
                          txtfor: str = "PKA1") -> list[CanonicalBlock]:
    """Canonical-basis spinors (WAC/) of one species (``N`` or ``P``)."""

    upper = _read_wac(run_dir / "WAC" / f"{nucleus}.WAV_G-{species}.{txtfor}")
    lower = _read_wac(run_dir / "WAC" / f"{nucleus}.WAV_F-{species}.{txtfor}")
    blocks = []
    for g, f in zip(upper, lower):
        if g["label"] != f["label"]:
            raise ValueError("G and F canonical files are not aligned")
        blocks.append(CanonicalBlock(
            kappa=kappa_from_block_label(g["label"]),
            labels=g["states"], energies=g["ee"], occupations=g["vv"],
            upper=g["values"], lower=f["values"],
        ))
    return blocks


def read_local_potentials(run_dir: Path, nucleus: str, species: str,
                          txtfor: str = "PKA1") -> dict[str, np.ndarray]:
    data = np.loadtxt(run_dir / "POT" / f"{nucleus}-local_{species}.{txtfor}")
    names = ("r", "sigma_plus", "sigma_minus", "sigma_tensor", "scalar",
             "vector", "vector_rho", "vector_coulomb", "rearrangement")
    return {name: data[:, k] for k, name in enumerate(names)}


def read_fock_kernels(run_dir: Path, nucleus: str, species: str) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Fock kernels ``{kappa: (XG, XF, YG, YF)}`` in MeV (column-major dump)."""

    path = run_dir / "POT" / f"{nucleus}-fock_{species}.bin"
    with path.open("rb") as handle:
        npt, nb = np.fromfile(handle, np.int32, 2)
        r = np.fromfile(handle, np.float64, npt)
        kernels = {}
        for _ in range(nb):
            kappa = int(np.fromfile(handle, np.int32, 1)[0])
            stacked = np.fromfile(handle, np.float64, 4 * npt * npt)
            # Fortran column-major (i = first index) -> [component, i, j]
            kernels[kappa] = stacked.reshape(4, npt, npt).transpose(0, 2, 1).copy()
    return r, kernels


def read_parsed(run_dir: Path) -> dict:
    return json.loads((run_dir / "parsed.json").read_text())
