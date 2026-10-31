"""Branch-aware full-Nambu references and saddle diagnostics.

This module intentionally does not call an eigensolver on the training path.
The tiny dense branch selector is an independent validation oracle only.
It encodes an important relativistic distinction: the no-sea reference is
not, in general, the projector onto all negative eigenvalues of the BdG
Hamiltonian.  In the unpaired limit, a negative-Dirac-energy particle state
is empty and its Nambu hole partner is occupied, even though that hole lies
on the positive BdG-energy branch.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .manifold import (
    GeneralizedDensity,
    generalized_density_matrix,
    slater_reference,
    transform_reference,
)


@dataclass(frozen=True)
class NoSeaBranchReference:
    """Pure normal-ordered reference and its selected Nambu branches.

    ``positive_dirac`` labels physical particle orbitals relative to a
    declared no-sea cutoff in the supplied one-body canonical basis.  The
    basis is metadata/initialization, not a training-time refreshed basis.
    ``selected_is_particle[i]`` says that the range of R contains the upper
    Nambu vector for orbital i; otherwise it contains the lower (hole) vector.
    """

    density: GeneralizedDensity
    frame: torch.Tensor
    dirac_energies: torch.Tensor
    positive_dirac: torch.Tensor
    occupied_particles: torch.Tensor
    selected_is_particle: torch.Tensor
    selected_bdg_energies: torch.Tensor
    chemical_potential: float
    dirac_cutoff: float

    @property
    def dimension(self) -> int:
        return int(self.dirac_energies.numel())

    @property
    def particle_number(self) -> float:
        return float(torch.sum(self.occupied_particles).detach())


def no_sea_branch_reference(
    dirac_energies: torch.Tensor,
    occupations: torch.Tensor,
    *,
    dirac_cutoff: float,
    chemical_potential: float,
    blocked_index: int | None = None,
    blocking_operation: str = "toggle",
) -> NoSeaBranchReference:
    """Build the exact unpaired no-sea Nambu branch reference without eigs.

    ``dirac_energies`` are mass-subtracted eigenvalues in a supplied
    initialization basis.  Orbitals below ``dirac_cutoff`` are Dirac-sea
    particle states and may not be occupied as nucleons.  Blocking toggles
    an allowed positive-energy orbital, so it supports both odd addition and
    odd-hole references while preserving number parity exactly.

    The returned R is pure (idempotent) with ρ=diag(n), κ=0.  For every empty
    orbital, including all Dirac-sea orbitals, R selects its Nambu hole
    vector.  Its BdG eigenvalue is ``mu-e``: negative for ordinary empty
    particle states and positive for no-sea Dirac-hole branches.
    """

    e = torch.as_tensor(dirac_energies)
    occ = torch.as_tensor(occupations, device=e.device)
    if e.dtype != torch.float64 or e.ndim != 1 or e.numel() == 0:
        raise TypeError("dirac_energies must be a nonempty float64 vector")
    if occ.dtype != torch.float64 or occ.shape != e.shape:
        raise TypeError("occupations must be a float64 vector matching dirac_energies")
    if not bool(torch.isfinite(e).all()) or not bool(torch.isfinite(occ).all()):
        raise ValueError("branch energies and occupations must be finite")
    if not torch.isfinite(torch.tensor([dirac_cutoff, chemical_potential], dtype=torch.float64)).all():
        raise ValueError("cutoff and chemical potential must be finite")
    if not bool(torch.all((occ == 0.0) | (occ == 1.0))):
        raise ValueError("branch reference occupations must be exactly zero or one")
    positive = e > float(dirac_cutoff)
    if not bool(torch.all(occ[~positive] == 0.0)):
        raise ValueError("no-sea reference cannot occupy negative-Dirac particle orbitals")
    if bool(torch.any(torch.abs(e - float(dirac_cutoff)) <= 1.0e-12)):
        raise ValueError("a Dirac orbital lies on the no-sea cutoff")
    if blocked_index is not None:
        if not 0 <= int(blocked_index) < e.numel():
            raise IndexError("blocked_index lies outside the Dirac basis")
        if not bool(positive[int(blocked_index)]):
            raise ValueError("blocking a negative-Dirac orbital violates the no-sea branch")
    density = slater_reference(
        occ, blocked_index=blocked_index, blocking_operation=blocking_operation
    )
    final_occ = density.rho.diagonal().real.to(torch.float64)
    d = int(e.numel())
    frame = torch.zeros((2 * d, d), dtype=torch.complex128, device=e.device)
    columns = torch.arange(d, device=e.device)
    particle_rows = columns[final_occ.bool()]
    hole_rows = columns[~final_occ.bool()] + d
    frame[particle_rows, particle_rows] = 1.0
    frame[hole_rows, columns[~final_occ.bool()]] = 1.0
    selected_is_particle = final_occ.bool()
    selected_bdg_energies = torch.where(
        selected_is_particle,
        e - float(chemical_potential),
        float(chemical_potential) - e,
    )
    return NoSeaBranchReference(
        density=density,
        frame=frame,
        dirac_energies=e.detach().clone(),
        positive_dirac=positive.detach().clone(),
        occupied_particles=final_occ.detach().clone(),
        selected_is_particle=selected_is_particle.detach().clone(),
        selected_bdg_energies=selected_bdg_energies.detach().clone(),
        chemical_potential=float(chemical_potential),
        dirac_cutoff=float(dirac_cutoff),
    )


def nambu_hamiltonian(
    h: torch.Tensor,
    delta: torch.Tensor,
    *,
    chemical_potential: float | torch.Tensor,
) -> torch.Tensor:
    """Assemble Hermitian BdG matrix in the package's [ρ,κ;−κ*,1−ρ*] convention."""

    h = torch.as_tensor(h)
    delta = torch.as_tensor(delta)
    if h.dtype != torch.complex128 or h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise TypeError("h must be a square complex128 matrix")
    if delta.dtype != torch.complex128 or delta.shape != h.shape:
        raise TypeError("delta must be complex128 and match h")
    if not torch.allclose(h, h.mH, atol=1.0e-10, rtol=0.0):
        raise ValueError("h must be Hermitian")
    if not torch.allclose(delta, -delta.mT, atol=1.0e-10, rtol=0.0):
        raise ValueError("delta must be antisymmetric")
    mu = torch.as_tensor(chemical_potential, dtype=torch.float64, device=h.device)
    if mu.numel() != 1 or not bool(torch.isfinite(mu)):
        raise ValueError("chemical_potential must be one finite scalar")
    eye = torch.eye(h.shape[0], dtype=h.dtype, device=h.device)
    top = torch.cat((h - mu * eye, delta), dim=1)
    bottom = torch.cat((-delta.conj(), -h.conj() + mu * eye), dim=1)
    return torch.cat((top, bottom), dim=0)


def particle_hole_conjugate(vectors: torch.Tensor) -> torch.Tensor:
    """Apply C=τx K to one or more Nambu vectors."""

    vectors = torch.as_tensor(vectors)
    if vectors.dtype != torch.complex128 or vectors.shape[0] % 2:
        raise TypeError("vectors must have even leading dimension and complex128 dtype")
    d = vectors.shape[0] // 2
    return torch.cat((vectors[d:].conj(), vectors[:d].conj()), dim=0)


def select_ph_branches_reference_only(
    h_nambu: torch.Tensor,
    reference_frame: torch.Tensor,
    *,
    minimum_overlap_margin: float = 1.0e-8,
    particle_hole_tolerance: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Dense independent oracle that tracks one branch from every PH pair.

    This is strictly a test/reference operation.  It uses one full Nambu
    diagonalization and selects, from each ±E pair, whichever member overlaps
    the supplied no-sea reference frame.  It refuses zero-energy crossings,
    ambiguous branch assignments, or loss of the PH-complement identity.
    """

    h = torch.as_tensor(h_nambu)
    frame = torch.as_tensor(reference_frame)
    if h.dtype != torch.complex128 or h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise TypeError("h_nambu must be a square complex128 matrix")
    n = h.shape[0]
    if n % 2 or frame.dtype != torch.complex128 or frame.shape != (n, n // 2):
        raise ValueError("reference_frame must be complex128 with shape (2D,D)")
    if not torch.allclose(h, h.mH, atol=1.0e-9, rtol=0.0):
        raise ValueError("Nambu Hamiltonian must be Hermitian")
    values, vectors = torch.linalg.eigh(h)
    d = n // 2
    scale = max(1.0, float(torch.max(torch.abs(values))))
    if float(torch.min(torch.abs(values))) <= 1.0e-10 * scale:
        raise RuntimeError("zero-energy quasiparticle crossing prevents branch assignment")
    positive = torch.nonzero(values > 0.0, as_tuple=False).flatten()
    negative = torch.nonzero(values < 0.0, as_tuple=False).flatten()
    if positive.numel() != d or negative.numel() != d:
        raise RuntimeError("BdG spectrum does not split into D positive and D negative branches")
    remaining_negative = set(int(index) for index in negative.tolist())
    pairs: list[tuple[int, int]] = []
    for p_index in positive.tolist():
        if not remaining_negative:
            raise RuntimeError("failed to pair every BdG branch")
        ph_partner = particle_hole_conjugate(vectors[:, p_index])
        n_index = max(
            remaining_negative,
            key=lambda index: float(torch.abs(torch.vdot(vectors[:, index], ph_partner))),
        )
        remaining_negative.remove(n_index)
        if abs(float(values[n_index] + values[p_index])) > particle_hole_tolerance * scale:
            raise RuntimeError("BdG spectrum violates particle-hole ±E pairing")
        pairs.append((int(n_index), int(p_index)))
    selected: list[torch.Tensor] = []
    selected_energies: list[float] = []
    margins: list[float] = []
    for n_index, p_index in pairs:
        neg, pos = vectors[:, n_index], vectors[:, p_index]
        neg_score = torch.sum(torch.abs(frame.mH @ neg).square())
        pos_score = torch.sum(torch.abs(frame.mH @ pos).square())
        margin = float(torch.abs(neg_score - pos_score))
        if margin <= minimum_overlap_margin:
            raise RuntimeError("no-sea branch tracking is ambiguous for a ±E pair")
        chosen = neg if float(neg_score) > float(pos_score) else pos
        selected.append(chosen)
        selected_energies.append(float(values[n_index] if chosen is neg else values[p_index]))
        margins.append(margin)
    tracked = torch.stack(selected, dim=1)
    density = tracked @ tracked.mH
    density_11, density_12 = density[:d, :d], density[:d, d:]
    density_21, density_22 = density[d:, :d], density[d:, d:]
    ph_density = torch.cat(
        [
            torch.cat((density_22.conj(), density_21.conj()), dim=1),
            torch.cat((density_12.conj(), density_11.conj()), dim=1),
        ],
        dim=0,
    )
    ph_error = float(torch.linalg.matrix_norm(ph_density - (torch.eye(n, dtype=h.dtype, device=h.device) - density)))
    if ph_error > particle_hole_tolerance * max(1.0, float(torch.linalg.matrix_norm(density))):
        raise RuntimeError("tracked branch set is not a PH-complementary pure generalized density")
    return tracked, {
        "particle_hole_complement_residual": ph_error,
        "minimum_overlap_margin": min(margins, default=0.0),
        "selected_positive_bdg_branch_count": float(sum(value > 0 for value in selected_energies)),
        "reference_only_eigensolves": 1.0,
    }


@dataclass(frozen=True)
class BranchStationarity:
    loss: torch.Tensor
    commutator_residual: torch.Tensor
    particle_number_residual: torch.Tensor
    branch_overlap: torch.Tensor
    sea_particle_occupation: torch.Tensor


def branch_stationarity_loss(
    h_nambu: torch.Tensor,
    density: GeneralizedDensity,
    branch: NoSeaBranchReference,
    *,
    target_number: float,
    stationarity_weight: float = 1.0,
    number_weight: float = 1.0,
    branch_weight: float = 1.0e-3,
) -> BranchStationarity:
    """No-eigensolver training loss for branch-followed HFB stationarity.

    The stationarity term is a full-space [H,R] residual (not an energy
    minimization eigenvalue-fill). The overlap term keeps optimization in the
    connected no-sea branch chart. It is a continuation guard and must be
    driven to zero/removed at final verification; it is not a substitute for
    the variational Euler-Lagrange equation.
    """

    h = torch.as_tensor(h_nambu)
    r = density.matrix
    d = branch.dimension
    if h.shape != (2 * d, 2 * d) or r.shape != h.shape:
        raise ValueError("Nambu Hamiltonian/density dimensions do not match branch")
    if not torch.allclose(r @ r, r, atol=1.0e-8, rtol=0.0):
        raise ValueError("branch stationarity requires a pure generalized density")
    commutator = h @ r - r @ h
    residual = torch.linalg.matrix_norm(commutator) / (
        torch.linalg.matrix_norm(h).clamp_min(1.0e-30)
        * torch.linalg.matrix_norm(r).clamp_min(1.0e-30)
    )
    number = torch.trace(density.rho).real
    number_residual = number - float(target_number)
    overlap = torch.linalg.matrix_norm(branch.frame.mH @ density.matrix @ branch.frame).square().real / d
    sea_mask = (~branch.positive_dirac).to(dtype=torch.float64, device=h.device)
    sea_occupation = torch.sum(sea_mask * density.rho.diagonal().real)
    branch_defect = torch.relu(1.0 - overlap)
    loss = (
        float(stationarity_weight) * residual.square()
        + float(number_weight) * number_residual.square()
        + float(branch_weight) * branch_defect.square()
    )
    return BranchStationarity(loss, residual, number_residual, overlap, sea_occupation)


def transform_branch_reference(
    branch: NoSeaBranchReference,
    a: torch.Tensor,
    b: torch.Tensor,
) -> GeneralizedDensity:
    """Canonical no-refresh chart anchored to the explicitly selected branch."""

    return transform_reference(branch.density, a, b)
