from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class BlockedRHFBConfig:
    z: int
    n: int
    radius_fm: float = 20.0
    spacing_fm: float = 0.1
    kappa_max: int = 5
    active_n: int | None = None
    active_p: int | None = None
    block_n: str = "auto"
    block_p: str = "auto"
    epochs: int = 300
    seed: int = 20260930
    model: str = "PKA1"
    pairing: str = "D1S"
    pairing_scale: float = 1.0
    candidate_shape: str = "sphere"
    candidate_seed_amplitude: float = 0.5
    chunk_size: int = 128
    pair_chunk: int = 4
    lmax: int | None = None
    dtype: str = "complex128"
    refresh_interval: int = 0

    def __post_init__(self) -> None:
        if self.z < 0 or self.n < 0 or self.z + self.n == 0:
            raise ValueError("Z/N must be nonnegative with A > 0")
        if not math.isfinite(self.radius_fm) or not math.isfinite(self.spacing_fm):
            raise ValueError("grid parameters must be finite")
        if self.spacing_fm <= 0 or self.radius_fm / self.spacing_fm < 10:
            raise ValueError("grid requires at least ten radial intervals")
        intervals = self.radius_fm / self.spacing_fm
        if abs(intervals - round(intervals)) > 1.0e-9:
            raise ValueError("radius must be an integer multiple of radial spacing")
        if self.kappa_max < 1 or self.epochs < 1:
            raise ValueError("kappa_max and epochs must be positive")
        for count, number in ((self.active_n, self.n), (self.active_p, self.z)):
            if count is not None and count < max(1, number):
                raise ValueError("active particle space cannot be smaller than particle number")
        for block, number in ((self.block_n, self.n), (self.block_p, self.z)):
            if number % 2 and block == "none":
                raise ValueError("odd particle number requires an explicitly blocked reference")
            if not number % 2 and block not in {"auto", "none"}:
                raise ValueError("one-quasiparticle blocking would change even number parity")
        if self.model != "PKA1" or self.pairing != "D1S" or self.pairing_scale != 1.0:
            raise ValueError("this version freezes the PKA1 + unscaled D1S nn/pp prescription")
        if self.candidate_shape not in {"sphere", "prolate", "oblate", "triaxial"}:
            raise ValueError("candidate_shape must be sphere, prolate, oblate, or triaxial")
        if not math.isfinite(self.candidate_seed_amplitude) or self.candidate_seed_amplitude < 0:
            raise ValueError("candidate_seed_amplitude must be finite and nonnegative")
        if self.chunk_size < 1 or self.pair_chunk < 1:
            raise ValueError("spatial and active-pair chunk sizes must be positive")
        if self.lmax is not None and self.lmax < 2 * self.kappa_max:
            raise ValueError("lmax must include full angular-product support: lmax>=2*kappa_max")
        if self.dtype != "complex128" or self.refresh_interval != 0:
            raise ValueError("blocked RHFB requires complex128 and zero subspace refreshes")

    @property
    def a(self) -> int:
        return self.z + self.n

    @property
    def number_parities(self) -> dict[str, int]:
        return {"n": self.n % 2, "p": self.z % 2}

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


PHYSICS_PRESCRIPTION = {
    "particle_hole": "PKA1",
    "particle_particle": "Gogny D1S finite-range two-Gaussian central part",
    "pairing_scale": 1.0,
    "pairing_species": ["nn", "pp"],
    "pn_pairing": False,
    "particle_number_projection": False,
    "equal_filling": False,
    "spatial_symmetry": "none imposed",
    "time_reversal": "allowed to break",
    "dtype": "complex128",
    "coupling_density": "sqrt(j_mu j^mu), metric +---",
    "rest_mass_convention": "h is measured in MeV with rest mass subtracted; W=-M",
}
