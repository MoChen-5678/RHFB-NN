from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator

import numpy as np
import torch
from scipy.special import eval_genlaguerre
try:
    from scipy.special import sph_harm_y as _spherical_harmonic

    _SPH_HARM_Y = True
except ImportError:  # SciPy < 1.15
    from scipy.special import sph_harm as _spherical_harmonic

    _SPH_HARM_Y = False

from dpl_rhf.rhf.dirac import RHFRadialDiracOperator
from dpl_rhf.rhf.initialization import SHELLS, _spherical_shells
from dpl_rhf.rhf.parameters import pka1_parameters


_DEFAULT_RHFB_PARAMETERS = pka1_parameters()


def active_size(
    particle_number: int,
    *,
    minimum: int = 40,
    empty_states: int = 32,
) -> int:
    """Initial active-space target, before rounding up to a complete shell."""

    if particle_number < 0:
        raise ValueError("particle_number must be non-negative")
    if minimum < 1 or empty_states < 0:
        raise ValueError("minimum must be positive and empty_states non-negative")
    return max(int(minimum), int(particle_number) + int(empty_states))


def _shells_to_capacity(target: int) -> list[tuple[str, int, int, int, int]]:
    shells = list(SHELLS)
    if sum(item[4] for item in shells) < target:
        shells = _spherical_shells(target)
    selected: list[tuple[str, int, int, int, int]] = []
    capacity = 0
    for shell in shells:
        selected.append(shell)
        capacity += int(shell[4])
        if capacity >= target:
            break
    return selected


@dataclass(frozen=True)
class AngularChannel:
    """One spinor spherical-harmonic channel; ``m2`` stores twice m."""

    kappa: int
    m2: int
    offset: int


class AngularRadialBasis:
    """Complex m-scheme radial Dirac grid with no density symmetry imposed.

    The coefficient space contains both radial Dirac components for every
    interior radial point and every ``(kappa, m)`` channel with
    ``0 < |kappa| <= kappa_max``. Coefficients use the uniform-radial-mesh
    metric (their Euclidean norm is the physical radial norm). ``sample``
    maps them to physical four-component coordinate spinors (including the
    ``1/r`` conversion from reduced radial amplitudes); the returned spatial
    weights carry ``r^2 dr dOmega``.

    ``kappa_max=K`` includes complete spinor multiplets through upper or lower
    orbital angular momentum K-1, together with the opposite-component
    partner channels needed by the Dirac operator.
    """

    def __init__(
        self,
        *,
        radius_fm: float = 20.0,
        spacing_fm: float = 0.1,
        kappa_max: int = 8,
        n_theta: int | None = None,
        n_phi: int | None = None,
        derivative_order: int = 7,
        mass_n_mev: float | None = None,
        mass_p_mev: float | None = None,
        hbar_c_mev_fm: float | None = None,
    ) -> None:
        if radius_fm <= 0.0 or spacing_fm <= 0.0:
            raise ValueError("radius_fm and spacing_fm must be positive")
        if kappa_max < 1:
            raise ValueError("kappa_max must be at least one")
        if derivative_order not in {4, 5, 6, 7}:
            raise ValueError("derivative_order must be one of 4, 5, 6, or 7")
        intervals = int(round(radius_fm / spacing_fm))
        if intervals < 7 or not math.isclose(
            intervals * spacing_fm, radius_fm, rel_tol=1.0e-10, abs_tol=1.0e-10
        ):
            raise ValueError("radius_fm must be an integer multiple of spacing_fm")
        self.radius_fm = float(radius_fm)
        self.spacing_fm = float(spacing_fm)
        self.kappa_max = int(kappa_max)
        self.derivative_order = int(derivative_order)
        self.mass_n_mev = float(
            _DEFAULT_RHFB_PARAMETERS.mass_n if mass_n_mev is None else mass_n_mev
        )
        self.mass_p_mev = float(
            _DEFAULT_RHFB_PARAMETERS.mass_p if mass_p_mev is None else mass_p_mev
        )
        self.hbar_c_mev_fm = float(
            _DEFAULT_RHFB_PARAMETERS.hbar_c
            if hbar_c_mev_fm is None
            else hbar_c_mev_fm
        )

        self.r = np.arange(intervals + 1, dtype=np.float64) * self.spacing_fm
        self.r[0] = 0.001
        self.x = self.r[1:-1].copy()
        self.n_radial = int(self.x.size)
        self.n_theta = int(n_theta or (2 * self.kappa_max + 1))
        self.n_phi = int(n_phi or (4 * self.kappa_max + 2))
        if self.n_theta < 2 * self.kappa_max + 1:
            raise ValueError("n_theta is too small for the requested angular cutoff")
        if self.n_phi < 4 * self.kappa_max + 2:
            raise ValueError("n_phi is too small for the requested angular cutoff")

        mu, w_mu = np.polynomial.legendre.leggauss(self.n_theta)
        phi = (2.0 * np.pi / self.n_phi) * np.arange(self.n_phi)
        # Flatten with phi varying fastest; w_mu integrates d(cos(theta)).
        self.theta = np.repeat(np.arccos(mu), self.n_phi)
        self.phi = np.tile(phi, self.n_theta)
        self.angular_weights = np.repeat(w_mu, self.n_phi) * (2.0 * np.pi / self.n_phi)
        self.n_angular_points = int(self.theta.size)
        self.positions = self._make_positions()
        self.weights = (
            self.spacing_fm
            * self.x[:, None] ** 2
            * self.angular_weights[None, :]
        ).reshape(-1)

        channels: list[AngularChannel] = []
        offset = 0
        for kappa in range(-self.kappa_max, self.kappa_max + 1):
            if kappa == 0:
                continue
            twice_j = 2 * abs(kappa) - 1
            for m2 in range(-twice_j, twice_j + 1, 2):
                channels.append(AngularChannel(kappa, m2, offset))
                offset += 2 * self.n_radial
        self.channels = tuple(channels)
        self.channel_index = {
            (channel.kappa, channel.m2): index
            for index, channel in enumerate(self.channels)
        }
        self.dimension = offset
        self._spherical_harmonic_cache: dict[tuple[int, int], np.ndarray] = {}
        self._harmonics = np.stack(
            [
                self._spinor_harmonic(channel.kappa, channel.m2)
                for channel in self.channels
            ],
            axis=0,
        )
        self._kinetic_cache: dict[int, torch.Tensor] = {}

    @property
    def angular_state_count(self) -> int:
        return len(self.channels)

    @property
    def point_count(self) -> int:
        return self.n_radial * self.n_angular_points

    def active_size(
        self,
        particle_number: int,
        *,
        minimum: int = 40,
        empty_states: int = 32,
    ) -> int:
        """Return a full-multiplet size at or above the default target."""

        target = active_size(
            particle_number, minimum=minimum, empty_states=empty_states
        )
        shells = _shells_to_capacity(target)
        capacity = sum(int(shell[4]) for shell in shells)
        for _, kappa, _, _, _ in shells:
            if abs(int(kappa)) > self.kappa_max:
                raise ValueError(
                    f"kappa_max={self.kappa_max} cannot cover active target {target}; "
                    "increase kappa_max"
                )
        return capacity

    def memory_estimate(
        self,
        active_orbitals: int,
        *,
        complex_bytes: int = 16,
        real_bytes: int = 8,
        sample_chunk_size: int = 1024,
    ) -> dict[str, int]:
        """Estimate factor, full sample, and chunk storage, excluding autograd."""

        if active_orbitals < 1:
            raise ValueError("active_orbitals must be positive")
        if sample_chunk_size < 1:
            raise ValueError("sample_chunk_size must be positive")
        sampled_chunk_points = min(self.point_count, int(sample_chunk_size))
        return {
            "factor_bytes": self.dimension * active_orbitals * complex_bytes,
            "sampled_full_grid_bytes": self.point_count * 4 * active_orbitals * complex_bytes,
            "sampled_chunk_bytes": sampled_chunk_points * 4 * active_orbitals * complex_bytes,
            "point_geometry_bytes": self.point_count * (3 * real_bytes + real_bytes),
            "basis_dimension": self.dimension,
            "point_count": self.point_count,
            "sample_chunk_points": sampled_chunk_points,
        }

    def _make_positions(self) -> np.ndarray:
        sin_theta = np.sin(self.theta)
        angular_xyz = np.stack(
            [
                sin_theta * np.cos(self.phi),
                sin_theta * np.sin(self.phi),
                np.cos(self.theta),
            ],
            axis=1,
        )
        return (self.x[:, None, None] * angular_xyz[None, :, :]).reshape(-1, 3)

    def _spinor_harmonic(self, kappa: int, m2: int) -> np.ndarray:
        """Return Condon-Shortley ``Omega_(kappa,m)`` at angular nodes."""

        j = abs(kappa) - 0.5
        m = 0.5 * m2
        if kappa < 0:
            ell = -kappa - 1
            up = math.sqrt(max(0.0, (j + m) / (2.0 * j)))
            down = math.sqrt(max(0.0, (j - m) / (2.0 * j)))
        else:
            ell = kappa
            denominator = 2.0 * j + 2.0
            up = -math.sqrt(max(0.0, (j + 1.0 - m) / denominator))
            down = math.sqrt(max(0.0, (j + 1.0 + m) / denominator))

        ml_up = int(round(m - 0.5))
        ml_down = int(round(m + 0.5))
        values = np.zeros((self.n_angular_points, 2), dtype=np.complex128)
        if up != 0.0 and abs(ml_up) <= ell:
            values[:, 0] = up * self._spherical_harmonic(ell, ml_up)
        if down != 0.0 and abs(ml_down) <= ell:
            values[:, 1] = down * self._spherical_harmonic(ell, ml_down)
        return values

    def _spherical_harmonic(self, ell: int, m: int) -> np.ndarray:
        key = (ell, m)
        cached = self._spherical_harmonic_cache.get(key)
        if cached is None:
            if _SPH_HARM_Y:
                cached = _spherical_harmonic(ell, m, self.theta, self.phi)
            else:
                # Legacy SciPy argument order is (m, l, azimuth, polar).
                cached = _spherical_harmonic(m, ell, self.phi, self.theta)
            self._spherical_harmonic_cache[key] = cached
        return cached

    def _radial_operator(self, kappa: int, device: torch.device) -> torch.Tensor:
        cached = self._kinetic_cache.get(kappa)
        if cached is None or cached.device != device:
            r = torch.as_tensor(self.r, dtype=torch.float64, device=device)
            operator = RHFRadialDiracOperator(
                r,
                mass_n_mev=self.mass_n_mev,
                mass_p_mev=self.mass_p_mev,
                hbar_c_mev_fm=self.hbar_c_mev_fm,
                derivative_order=self.derivative_order,
            )
            cached = operator.matrix_fm_inverse(1.0, kappa) * self.hbar_c_mev_fm
            self._kinetic_cache[kappa] = cached
        return cached

    def apply_kinetic(
        self,
        coefficients: torch.Tensor,
        *,
        species: str = "n",
    ) -> torch.Tensor:
        """Apply the Hermitian free Dirac operator in the weighted grid metric.

        ``coefficients`` may be a vector or a ``(D, M)`` factor matrix. The
        result is in MeV. Angular channels are independent only for the free
        operator; density and interaction matrices may mix them freely.
        """

        if species not in {"n", "p"}:
            raise ValueError("species must be 'n' or 'p'")
        if coefficients.ndim not in {1, 2} or coefficients.shape[0] != self.dimension:
            raise ValueError("coefficients must have shape (D,) or (D,M)")
        if not coefficients.is_complex():
            raise ValueError("m-scheme coefficients must use a complex dtype")
        output = torch.zeros_like(coefficients)
        mass = self.mass_n_mev if species == "n" else self.mass_p_mev
        operators: dict[int, torch.Tensor] = {}
        for channel in self.channels:
            operator = operators.get(channel.kappa)
            if operator is None:
                operator = self._radial_operator(
                    channel.kappa, coefficients.device
                ).clone()
                operator[self.n_radial :, self.n_radial :] = torch.diag(
                    torch.full(
                        (self.n_radial,),
                        -2.0 * mass,
                        dtype=operator.dtype,
                        device=operator.device,
                    )
                )
                operators[channel.kappa] = operator
            start = channel.offset
            stop = start + 2 * self.n_radial
            output[start:stop] = operator.to(coefficients.dtype) @ coefficients[start:stop]
        return output

    def build_initial_q(
        self,
        particle_number: int,
        *,
        mass_mev: float | None = None,
        oscillator_length_fm: float | None = None,
        initialization: str = "oscillator",
        species: str = "n",
        nucleus_a: int | None = None,
        proton_number: int = 0,
        minimum: int = 40,
        empty_states: int = 32,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.complex128,
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        """Build a non-eigensolved oscillator occupied-plus-empty Q factor.

        Complete spectroscopic ``kappa`` multiplets are added until the
        default active target is covered. ``initialization="oscillator"``
        uses analytic harmonic-oscillator radial functions;
        ``initialization="woods-saxon"`` uses the same analytic upper
        functions with a Woods-Saxon/Coulomb Dirac small-component estimate.
        Reduced QR orthonormalizes the result in the weighted radial metric.
        """

        if particle_number < 0:
            raise ValueError("particle_number must be non-negative")
        if species not in {"n", "p"}:
            raise ValueError("species must be 'n' or 'p'")
        if initialization not in {"oscillator", "woods-saxon"}:
            raise ValueError("initialization must be 'oscillator' or 'woods-saxon'")
        if proton_number < 0:
            raise ValueError("proton_number must be non-negative")
        target = self.active_size(
            particle_number, minimum=minimum, empty_states=empty_states
        )
        selected = _shells_to_capacity(target)
        records: list[tuple[str, int, int, int, int]] = []
        for name, kappa, ell, principal, degeneracy in selected:
            if abs(int(kappa)) > self.kappa_max:
                raise ValueError(
                    f"kappa_max={self.kappa_max} cannot cover active target {target}; "
                    "increase kappa_max"
                )
            twice_j = 2 * abs(int(kappa)) - 1
            for m2 in range(-twice_j, twice_j + 1, 2):
                records.append((name, int(kappa), int(ell), int(principal), m2))
        if len(records) != target:
            raise RuntimeError("shell capacity bookkeeping did not match active_size")

        default_mass = self.mass_n_mev if species == "n" else self.mass_p_mev
        mass = float(mass_mev if mass_mev is not None else default_mass)
        if mass <= 0.0:
            raise ValueError("mass_mev must be positive")
        mass_number = int(nucleus_a if nucleus_a is not None else max(particle_number, 1))
        if mass_number < 1:
            raise ValueError("nucleus_a must be positive")
        if oscillator_length_fm is None:
            omega = 41.0 * mass_number ** (-1.0 / 3.0)
            oscillator_length_fm = math.sqrt(
                self.hbar_c_mev_fm**2 / (mass * omega)
            )
        if oscillator_length_fm <= 0.0:
            raise ValueError("oscillator_length_fm must be positive")

        radial = torch.as_tensor(self.x, dtype=torch.float64, device=device)
        radial_metric_scale = math.sqrt(self.spacing_fm)
        woods_saxon_denominator: torch.Tensor | None = None
        if initialization == "woods-saxon":
            a13 = mass_number ** (1.0 / 3.0)
            r_minus = 0.5 * (1.1443 + 1.1400) * a13
            sigma_minus = 71.28 * 11.1175 / (
                1.0 + torch.exp((radial - r_minus) / 0.64725)
            )
            coulomb = torch.zeros_like(radial)
            if species == "p" and proton_number:
                charge_radius = 1.2496 * a13
                inside = radial < charge_radius
                coulomb[inside] = (proton_number / 137.035999084) * (
                    3.0 / (2.0 * charge_radius)
                    - radial[inside].square() / (2.0 * charge_radius**3)
                )
                coulomb[~inside] = (proton_number / 137.035999084) / radial[~inside]
            v_minus = (
                sigma_minus / self.hbar_c_mev_fm
                + coulomb
                - 2.0 * mass / self.hbar_c_mev_fm
            )
            epsilon = -0.5 * 41.0 * mass_number ** (-1.0 / 3.0)
            woods_saxon_denominator = epsilon / self.hbar_c_mev_fm - v_minus
        columns: list[torch.Tensor] = []
        names: list[str] = []
        for name, kappa, ell, principal, m2 in records:
            radial_index = principal - 1
            q = (radial / oscillator_length_fm).square()
            upper = radial.pow(ell + 1) * torch.exp(-0.5 * q) * torch.as_tensor(
                eval_genlaguerre(radial_index, ell + 0.5, q.cpu().numpy()),
                dtype=torch.float64,
                device=device,
            )
            upper = upper / torch.sqrt(self.spacing_fm * upper.square().sum())
            op = self._radial_operator(kappa, torch.device(device))
            b2 = op[self.n_radial :, : self.n_radial]
            derivative_coupling = b2 @ upper
            if woods_saxon_denominator is None:
                lower = derivative_coupling / (2.0 * mass)
            else:
                lower = (
                    derivative_coupling
                    / self.hbar_c_mev_fm
                    / woods_saxon_denominator
                )
            state = torch.cat([upper, lower])
            state = state / torch.linalg.vector_norm(state) * radial_metric_scale

            channel_idx = self.channel_index.get((kappa, m2))
            if channel_idx is None:
                raise RuntimeError("active shell channel is absent from basis")
            channel = self.channels[channel_idx]
            col = torch.zeros(self.dimension, dtype=torch.complex128, device=device)
            col[channel.offset : channel.offset + 2 * self.n_radial] = state.to(
                torch.complex128
            )
            columns.append(col)
            names.append(f"{name}:kappa={kappa}:m2={m2}")

        candidate = torch.stack(columns, dim=1)
        # The oscillator ordering is close to orthogonal. Reduced QR handles
        # finite-box tails while preserving exact norm and full channel space.
        q_factor, _ = torch.linalg.qr(candidate, mode="reduced")
        return q_factor.to(dtype=dtype), tuple(names)

    def _sample_range(
        self,
        coefficients: torch.Tensor,
        start: int,
        stop: int,
    ) -> torch.Tensor:
        device = coefficients.device
        dtype = coefficients.dtype
        angular_count = self.n_angular_points
        point_ids = torch.arange(start, stop, device=device)
        radial_ids = torch.div(point_ids, angular_count, rounding_mode="floor")
        angular_ids = point_ids.remainder(angular_count)
        result = torch.zeros(
            (stop - start, 4, coefficients.shape[1]),
            dtype=dtype,
            device=device,
        )
        radii = torch.as_tensor(self.x, dtype=torch.float64, device=device)[radial_ids]
        radial_scale = self.spacing_fm**-0.5 / radii
        for index, channel in enumerate(self.channels):
            g = coefficients[
                channel.offset + radial_ids, :
            ] * radial_scale[:, None]
            f = coefficients[
                channel.offset + self.n_radial + radial_ids, :
            ] * radial_scale[:, None]
            harmonic_upper = torch.as_tensor(
                self._harmonics[index], dtype=dtype, device=device
            )[angular_ids]
            harmonic_lower = torch.as_tensor(
                self._harmonics[self.channel_index[(-channel.kappa, channel.m2)]],
                dtype=dtype,
                device=device,
            )[angular_ids]
            result[:, 0:2, :] += g[:, None, :] * harmonic_upper[:, :, None]
            result[:, 2:4, :] += (1j * f[:, None, :]) * harmonic_lower[:, :, None]
        return result

    def project_sample_range(self, values: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        """Apply the volume-weighted adjoint of a coordinate sampling slice.

        Converts physical four-spinor values back to the primitive coefficient
        metric. This is an adjoint, not a fit or a subspace refresh. Summing
        disjoint slices exactly reproduces the full projection; no dense
        coordinate-to-primitive matrix is formed.
        """
        if (values.ndim != 3 or values.shape[:2] != (stop-start, 4)
                or values.dtype != torch.complex128):
            raise ValueError("values must have shape (slice points,4,columns) and complex128 dtype")
        if not 0 <= start <= stop <= self.point_count:
            raise ValueError("invalid coordinate slice")
        ids = torch.arange(start, stop, device=values.device)
        radial_ids = ids//self.n_angular_points
        angular_ids = ids%self.n_angular_points
        radii = torch.as_tensor(self.x, device=values.device, dtype=torch.float64)[radial_ids]
        weights = torch.as_tensor(self.weights[start:stop], device=values.device, dtype=torch.float64)
        weighted = values*(weights*self.spacing_fm**-.5/radii)[:, None, None]
        result = values.new_zeros((self.dimension, values.shape[2]))
        for index, channel in enumerate(self.channels):
            upper = torch.as_tensor(self._harmonics[index], device=values.device, dtype=values.dtype)[angular_ids]
            lower_index = self.channel_index[(-channel.kappa, channel.m2)]
            lower = torch.as_tensor(self._harmonics[lower_index], device=values.device, dtype=values.dtype)[angular_ids]
            g = torch.einsum('ps,psc->pc', upper.conj(), weighted[:, :2])
            f = -1j*torch.einsum('ps,psc->pc', lower.conj(), weighted[:, 2:])
            result = result.index_add(0, channel.offset+radial_ids, g)
            result = result.index_add(0, channel.offset+self.n_radial+radial_ids, f)
        return result

    def sample(
        self,
        coefficients: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return full unweighted ``(P,4,M)`` spinors, weights, and positions.

        This convenience method is intended for small validation bases. Use
        ``iter_samples`` for production contractions.
        """

        self._validate_coefficients(coefficients)
        spinors = self._sample_range(coefficients, 0, self.point_count)
        weights = torch.as_tensor(
            self.weights, dtype=torch.float64, device=coefficients.device
        )
        positions = torch.as_tensor(
            self.positions, dtype=torch.float64, device=coefficients.device
        )
        return spinors, weights, positions

    def iter_samples(
        self,
        coefficients: torch.Tensor,
        *,
        chunk_size: int = 1024,
    ) -> Iterator[tuple[slice, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield bounded coordinate chunks for matrix-free contractions."""

        self._validate_coefficients(coefficients)
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        for start in range(0, self.point_count, chunk_size):
            stop = min(start + chunk_size, self.point_count)
            weights = torch.as_tensor(
                self.weights[start:stop],
                dtype=torch.float64,
                device=coefficients.device,
            )
            positions = torch.as_tensor(
                self.positions[start:stop],
                dtype=torch.float64,
                device=coefficients.device,
            )
            yield (
                slice(start, stop),
                self._sample_range(coefficients, start, stop),
                weights,
                positions,
            )

    def _validate_coefficients(self, coefficients: torch.Tensor) -> None:
        if coefficients.ndim != 2 or coefficients.shape[0] != self.dimension:
            raise ValueError("coefficients must have shape (basis.dimension, M)")
        if not coefficients.is_complex():
            raise ValueError("m-scheme coefficients must use a complex dtype")
