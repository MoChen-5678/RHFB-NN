from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MODEL_INDEX = {
    "PKA1": 0,
    "PKO1": 1,
    "PKO2": 2,
    "PKO3": 3,
    "DD-ME1": 4,
    "DDME1": 4,
    "DD-ME2": 5,
    "DDME2": 5,
    "PKDD": 6,
    "TW99": 7,
    "DD-LZ1": 8,
    "DDLZ1": 8,
}

RMF_MODEL_INDEX = {4, 5, 6, 7, 8}


def model_index(name_or_index: str | int) -> int:
    if isinstance(name_or_index, int):
        idx = name_or_index
    else:
        text = str(name_or_index).strip()
        idx = int(text) if text.isdigit() else MODEL_INDEX[text.upper()]
    if idx not in MODEL_INDEX.values():
        raise ValueError(f"unknown model index: {name_or_index}")
    return idx


def require_rmf_model(name_or_index: str | int) -> int:
    idx = model_index(name_or_index)
    if idx not in RMF_MODEL_INDEX:
        raise ValueError("DPL-RMF v1 only supports IE=1 RMF models: DD-ME1, DD-ME2, PKDD, TW99, DD-LZ1")
    return idx


@dataclass
class Energy:
    e_total: float
    e_per_A: float
    e_total_no_com: float
    e_per_A_no_com: float
    e_total_with_com: float
    e_per_A_with_com: float
    e_cm: float
    e_kin: float
    e_dir: float
    e_exc: float
    e_rearr: float
    rms_n_no_com: float
    rms_p_no_com: float
    rms_matter_no_com: float
    charge_radius_no_com: float
    rms_n_with_com: float
    rms_p_with_com: float
    rms_matter_with_com: float
    charge_radius_with_com: float


def resolve_core_dir(core_dir: str | Path | None = None) -> Path:
    """Locate Core-1204 without depending on the caller's working directory."""
    candidates: list[Path] = []
    if core_dir is not None:
        candidates.append(Path(core_dir).expanduser())
    env_dir = os.environ.get("DPL_RHF_CORE_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    package_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            package_root.parent / "Core-1204",
            package_root / "Core-1204",
            Path.cwd().parent / "Core-1204",
            Path.cwd() / "Core-1204",
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "rhf_ctypes.py").is_file() and (resolved / "librhf.so").is_file():
            return resolved
    searched = "\n  - ".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(
        "Core-1204 runtime was not found. Set --core-dir or DPL_RHF_CORE_DIR. "
        f"Searched:\n  - {searched}"
    )


class RHFCore:
    """ctypes facade around ../Core-1204/librhf.so."""

    def __init__(self, core_dir: str | Path | None = None):
        self.core_dir = resolve_core_dir(core_dir)
        module_path = self.core_dir / "rhf_ctypes.py"
        library_path = self.core_dir / "librhf.so"
        if not module_path.exists():
            raise FileNotFoundError(f"missing Core ctypes wrapper: {module_path}")
        if not library_path.exists():
            raise FileNotFoundError(f"missing Core shared library: {library_path}; rebuild Core-1204 first")

        spec = importlib.util.spec_from_file_location("core1204_rhf_ctypes", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.lib = module
        self.npt = 0
        self.dr = 0.0
        self.r = np.array([], dtype=np.float64)

    def init(self, model: str | int, z: int, n: int, a: int | None = None) -> None:
        idx = model_index(model)
        mass = int(a if a is not None else z + n)
        self.lib.ddrhf_init(idx, int(z), int(n), mass)
        self.r, self.npt, self.dr = self.lib.ddrhf_extract_grid()

    def step(self) -> float:
        return float(self.lib.ddrhf_step())

    def energy(self) -> Energy:
        if hasattr(self.lib, "ddrhf_get_observables"):
            data = self.lib.ddrhf_get_observables()
            data["e_total"] = data["e_total_no_com"]
            data["e_per_A"] = data["e_per_A_no_com"]
        else:
            raw = self.lib.ddrhf_get_energy()
            data = {
                **raw,
                "e_total_no_com": raw["e_total"],
                "e_per_A_no_com": raw["e_per_A"],
                "e_total_with_com": raw["e_total"],
                "e_per_A_with_com": raw["e_per_A"],
                "e_cm": 0.0,
                "rms_n_no_com": 0.0,
                "rms_p_no_com": 0.0,
                "rms_matter_no_com": 0.0,
                "charge_radius_no_com": 0.0,
                "rms_n_with_com": 0.0,
                "rms_p_with_com": 0.0,
                "rms_matter_with_com": 0.0,
                "charge_radius_with_com": 0.0,
            }
        return Energy(**{k: float(v) for k, v in data.items()})

    def local_potentials(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v[: self.npt], dtype=np.float64).copy()
                for k, v in self.lib.ddrhf_extract_local_potentials().items()}

    def local_stack(self) -> np.ndarray:
        pot = self.local_potentials()
        return np.stack([pot["vps_n"], pot["vms_n"], pot["vps_p"], pot["vms_p"]])

    def set_local_stack(self, stack: np.ndarray) -> None:
        arr = np.asarray(stack, dtype=np.float64)
        if arr.shape != (4, self.npt):
            raise ValueError(f"expected local potential stack shape (4, {self.npt}), got {arr.shape}")
        zero = np.zeros(self.npt, dtype=np.float64)
        status = self.lib.ddrhf_set_local_potentials(arr[0], arr[1], zero, arr[2], arr[3], zero)
        if status != 0:
            raise RuntimeError(f"ddrhf_set_local_potentials failed with status {status}")

    def solve_fixed_potential(self) -> None:
        status = self.lib.ddrhf_solve_fixed_potential()
        if status != 0:
            raise RuntimeError(f"ddrhf_solve_fixed_potential failed with status {status}")

    def rebuild_rmf_potentials(self) -> None:
        status = self.lib.ddrhf_rebuild_rmf_potentials()
        if status != 0:
            raise RuntimeError(f"ddrhf_rebuild_rmf_potentials failed with status {status}")

    def fields(self) -> dict[str, np.ndarray]:
        sigma, omega, rho, coul = self.lib.ddrhf_extract_fields()
        return {
            "sigma": np.asarray(sigma[: self.npt], dtype=np.float64).copy(),
            "omega": np.asarray(omega[: self.npt], dtype=np.float64).copy(),
            "rho": np.asarray(rho[: self.npt], dtype=np.float64).copy(),
            "coul": np.asarray(coul[: self.npt], dtype=np.float64).copy(),
        }

    def densities(self) -> dict[str, np.ndarray]:
        rho_s, rho_b, rho_b3 = self.lib.ddrhf_extract_densities()
        return {
            "rho_s": np.asarray(rho_s[: self.npt], dtype=np.float64).copy(),
            "rho_b": np.asarray(rho_b[: self.npt], dtype=np.float64).copy(),
            "rho_b3": np.asarray(rho_b3[: self.npt], dtype=np.float64).copy(),
        }

    def hamiltonian_gradient(self) -> np.ndarray:
        if not hasattr(self.lib, "ddrhf_extract_hamiltonian_gradient"):
            raise RuntimeError("Core wrapper does not expose ddrhf_extract_hamiltonian_gradient; rebuild Core-1204")
        grad = self.lib.ddrhf_extract_hamiltonian_gradient()
        return np.stack([
            np.asarray(grad["vps_n"][: self.npt], dtype=np.float64).copy(),
            np.asarray(grad["vms_n"][: self.npt], dtype=np.float64).copy(),
            np.asarray(grad["vps_p"][: self.npt], dtype=np.float64).copy(),
            np.asarray(grad["vms_p"][: self.npt], dtype=np.float64).copy(),
        ])

    def wavefunctions(self) -> dict[str, np.ndarray]:
        records: list[dict[str, Any]] = []
        for it, species in ((1, "n"), (2, "p")):
            n_orbits = int(self.lib.ddrhf_get_norbits(it))
            for i0 in range(1, n_orbits + 1):
                g, f, name, kappa, l_val, n_val, energy, occ, deg, ok = self.lib.ddrhf_extract_wavefunction(i0, it)
                if int(ok) != 1:
                    continue
                records.append(
                    {
                        "species": species,
                        "it": it,
                        "index": i0,
                        "name": name,
                        "kappa": kappa,
                        "l": l_val,
                        "n": n_val,
                        "energy": energy,
                        "occupancy": occ,
                        "degeneracy": deg,
                        "G": np.asarray(g[: self.npt], dtype=np.float64).copy(),
                        "F": np.asarray(f[: self.npt], dtype=np.float64).copy(),
                    }
                )

        if not records:
            return {
                "r": self.r.copy(),
                "G": np.zeros((0, self.npt), dtype=np.float64),
                "F": np.zeros((0, self.npt), dtype=np.float64),
                "species": np.array([], dtype="U1"),
                "it": np.array([], dtype=np.int32),
                "index": np.array([], dtype=np.int32),
                "name": np.array([], dtype="U16"),
                "kappa": np.array([], dtype=np.int32),
                "l": np.array([], dtype=np.int32),
                "n": np.array([], dtype=np.int32),
                "energy": np.array([], dtype=np.float64),
                "occupancy": np.array([], dtype=np.float64),
                "degeneracy": np.array([], dtype=np.int32),
            }

        return {
            "r": self.r.copy(),
            "G": np.stack([row["G"] for row in records]),
            "F": np.stack([row["F"] for row in records]),
            "species": np.array([row["species"] for row in records], dtype="U1"),
            "it": np.array([row["it"] for row in records], dtype=np.int32),
            "index": np.array([row["index"] for row in records], dtype=np.int32),
            "name": np.array([row["name"] for row in records], dtype="U16"),
            "kappa": np.array([row["kappa"] for row in records], dtype=np.int32),
            "l": np.array([row["l"] for row in records], dtype=np.int32),
            "n": np.array([row["n"] for row in records], dtype=np.int32),
            "energy": np.array([row["energy"] for row in records], dtype=np.float64),
            "occupancy": np.array([row["occupancy"] for row in records], dtype=np.float64),
            "degeneracy": np.array([row["degeneracy"] for row in records], dtype=np.int32),
        }


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
