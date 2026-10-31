"""Independent fixed-field whole-Dirac vs strict normal-P+ RHFB reference.

All exact spectral operations in this module are reference-only. It does not
iterate fields, train a network, or claim a self-consistent nuclear solution.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .basis import AngularRadialBasis
from .branch import nambu_hamiltonian, particle_hole_conjugate
from .certified_gf_gap import (CertifiedGFGapError, certify_basis_gf_interval,
                               validate_spectrum_for_gf_interval)
from .gaussian_pairing import GaussianPairingAction
from .parity_diagnostic import number_parity_from_density
from .result import atomic_json


@dataclass
class FixedFieldBranchComparison:
    report: dict
    whole_rho: torch.Tensor
    whole_chi: torch.Tensor
    projected_rho: torch.Tensor
    projected_chi: torch.Tensor


class UnsafeDiracSectorSplit(RuntimeError):
    """Raised when the finite normal spectrum does not support sector labels."""

    def __init__(self, message: str, evidence: dict):
        super().__init__(message)
        self.evidence = evidence


def compute_saved_full_d1s_pairing(input_npz: str | Path, metadata_json: str | Path,
                                   output_npz: str | Path) -> dict:
    """Compute full primitive D1S Delta matrices from saved pure seed factors."""
    input_path, meta_path, out_path = map(Path, (input_npz, metadata_json, output_npz))
    meta = json.loads(meta_path.read_text())
    torch.set_num_threads(2)
    basis = AngularRadialBasis(
        radius_fm=meta["radius_fm"], spacing_fm=meta["spacing_fm"],
        kappa_max=meta["kappa_max"], n_theta=meta["n_theta"],
        n_phi=meta["n_phi"], derivative_order=meta["derivative_order"],
    )
    if basis.dimension != int(meta["primitive_dimension"]):
        raise ValueError("saved seed and basis primitive dimensions disagree")
    arrays: dict[str, np.ndarray] = {}
    measurements = {}
    with np.load(input_path) as z:
        for species in ("n", "p"):
            pair_started = time.monotonic()
            q = torch.from_numpy(z[f"{species}_Q"].copy())
            rho = torch.from_numpy(z[f"{species}_rho"].copy())
            chi = torch.from_numpy(z[f"{species}_chi"].copy())
            if q.shape != (basis.dimension, basis.dimension):
                raise ValueError("seed Q must span the full saved primitive space")
            action = GaussianPairingAction(
                basis, q, chi, species=species, chunk_size=4,
                active_chunk_size=4, lmax=meta["lmax"], checkpoint_blocks=False,
            )
            with torch.no_grad():
                delta = action.apply(torch.eye(basis.dimension, dtype=torch.complex128))
                gap_raw = q.mH @ delta @ q.conj()
                gap = 0.5 * (gap_raw - gap_raw.T)
                energy = 0.5 * torch.sum(chi.conj() * gap).real
            antisymmetry = float((delta + delta.T).abs().max())
            if antisymmetry > 1e-9:
                raise RuntimeError(f"full D1S pairing matrix lost antisymmetry: {antisymmetry}")
            arrays[f"{species}_delta"] = delta.cpu().numpy()
            arrays[f"{species}_active_gap"] = gap.cpu().numpy()
            measurements[species] = {
                "chi_frobenius": float(torch.linalg.matrix_norm(chi)),
                "full_delta_frobenius_mev": float(torch.linalg.matrix_norm(delta)),
                "full_delta_antisymmetry_max_mev": antisymmetry,
                "active_gap_frobenius_mev": float(torch.linalg.matrix_norm(gap)),
                "pairing_energy_mev": float(energy),
                "elapsed_pair_action_seconds": time.monotonic() - pair_started,
            }
    np.savez_compressed(out_path, **arrays)
    return {
        "input_state_npz": str(input_path),
        "input_state_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "pairing_source_sha256": hashlib.sha256(Path(__file__).with_name("gaussian_pairing.py").read_bytes()).hexdigest(),
        "saved_metadata": meta,
        "fortran_used": False,
        "optimizer_updates": 0,
        "species": measurements,
        "output_delta_npz": str(out_path.resolve()),
    }


def _density_from_frame(frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r = frame @ frame.mH
    d = r.shape[0] // 2
    rho = r[:d, :d]
    chi = r[:d, d:]
    return r, rho, chi


def _frame_from_positive_modes(positive: torch.Tensor, sea_weight: torch.Tensor,
                               *, toggles: tuple[int, ...] = (),
                               minimum_sector_purity: float = 0.90) -> torch.Tensor:
    """Build a no-sea annihilator frame, then optionally block one particle qp.

    Positive-energy Nambu eigenvectors are labeled by their normal Dirac-sea
    weight. A particle qp contributes its PH partner to R; a sea qp contributes
    itself. This implements the particle/sea annihilator distinction in DIRHB
    Eqs. (46)-(48), rather than choosing a side independently by overlap.
    """
    nambu_dim, count = positive.shape
    d = nambu_dim // 2
    if count != d or sea_weight.shape != (d,):
        raise ValueError("positive Nambu modes and sector weights must both have D columns")
    if bool(((sea_weight > 1. - minimum_sector_purity) &
             (sea_weight < minimum_sector_purity)).any()):
        raise RuntimeError("positive BdG modes are mixed across Dirac sectors; no-sea labels refused")
    is_sea = sea_weight >= minimum_sector_purity
    columns = [positive[:, i] if bool(is_sea[i]) else
               particle_hole_conjugate(positive[:, i]) for i in range(d)]
    frame = torch.stack(columns, dim=1)
    toggle_counts = {mode: toggles.count(mode) % 2 for mode in set(toggles)}
    for blocked in (mode for mode, count in toggle_counts.items() if count):
        particle_modes = torch.nonzero(~is_sea, as_tuple=False).flatten()
        if blocked not in particle_modes.tolist():
            raise RuntimeError("requested blocked mode is not a particle-sector quasiparticle")
        frame[:, blocked] = positive[:, blocked]
    return frame


def _sector_labeled_vacuum(h: torch.Tensor, delta: torch.Tensor, mu: float,
                           pminus: torch.Tensor, *, steps: int,
                           toggle_positive_indices: tuple[int, ...] = (),
                           minimum_sector_purity: float = 0.90) -> tuple[torch.Tensor, dict]:
    """Continue positive BdG modes while keeping no-sea labels physical.

    Overlap assignment transports qp identity only. Occupation is determined
    by the no-sea sector prescription and a single explicit blocked mode.
    """
    if steps < 1:
        raise ValueError("continuation steps must be positive")
    d = h.shape[0]
    eye = torch.eye(d, dtype=h.dtype, device=h.device)
    sector = torch.block_diag(pminus, pminus.conj())
    previous = None
    path = []
    final_frame = None
    blocked_overlaps = []
    for index in range(steps + 1):
        scale = index / steps
        hd = nambu_hamiltonian(h, scale * delta, chemical_potential=mu)
        energies, vectors = torch.linalg.eigh(hd)
        gap = float(energies.abs().min())
        if gap <= 1e-10 * max(1., float(energies.abs().max())):
            raise RuntimeError("zero-energy quasiparticle crossing prevents branch assignment")
        pos_ids = torch.nonzero(energies > 0., as_tuple=False).flatten()
        neg_ids = torch.nonzero(energies < 0., as_tuple=False).flatten()
        if len(pos_ids) != d or len(neg_ids) != d:
            raise RuntimeError("BdG spectrum does not split into D positive and D negative modes")
        pos = vectors[:, pos_ids]
        # Fix a PH-consistent negative complement from each positive mode.
        ph = particle_hole_conjugate(pos)
        ph_errors = torch.linalg.vector_norm(hd @ ph + ph * energies[pos_ids].unsqueeze(0), dim=0)
        ph_error = float(ph_errors.max())
        if ph_error > 1e-7 * max(1., float(torch.linalg.matrix_norm(hd))):
            raise RuntimeError("positive modes do not have matching PH partners")
        assignment_min = 1.
        if previous is not None:
            # Match positive-mode identities by overlap only; this does not
            # choose the physical occupation side of any PH pair.
            overlap = previous.mH @ pos
            rows, cols = __import__("scipy.optimize", fromlist=["linear_sum_assignment"]).linear_sum_assignment(
                (-overlap.abs().square().detach().cpu().numpy()))
            order = np.empty(d, dtype=np.int64)
            order[rows] = cols
            pos = pos[:, torch.as_tensor(order, dtype=torch.long, device=h.device)]
            assignment_min = float(overlap[torch.arange(d, device=h.device),
                torch.as_tensor(order, dtype=torch.long, device=h.device)].abs().square().min())
            if assignment_min < 0.05:
                raise RuntimeError("positive qp identity continuation overlap is too small")
        weights = torch.einsum("ij,ji->i", pos.mH, sector @ pos).real.clamp(0., 1.)
        frame = _frame_from_positive_modes(pos, weights, toggles=toggle_positive_indices,
                                           minimum_sector_purity=minimum_sector_purity)
        # Verify each selected column has exactly its PH complement unselected.
        complement = particle_hole_conjugate(frame)
        ph_complement = float(torch.linalg.matrix_norm(frame.mH @ complement))
        if ph_complement > 1e-7:
            raise RuntimeError("selected no-sea frame is not orthogonal to its PH complement")
        if toggle_positive_indices:
            blocked_overlaps.append(min(float(torch.abs(torch.vdot(pos[:, mode],
                frame[:, mode])).square()) for mode in toggle_positive_indices))
        path.append({
            "pairing_scale": scale,
            "minimum_abs_quasiparticle_energy_mev": gap,
            "minimum_particle_sector_weight": float(weights[weights < 0.5].max()) if bool((weights < .5).any()) else None,
            "minimum_sea_sector_weight": float(weights[weights >= .5].min()) if bool((weights >= .5).any()) else None,
            "minimum_positive_mode_identity_overlap_squared": assignment_min,
            "particle_hole_complement_residual": ph_complement,
            "positive_negative_spectrum_pairing_residual_mev": ph_error,
        })
        previous, final_frame = pos, frame
    return final_frame, {"continuation": path,
        "minimum_quasiparticle_gap_mev": min(x["minimum_abs_quasiparticle_energy_mev"] for x in path),
        "minimum_positive_mode_identity_overlap_squared": min(x["minimum_positive_mode_identity_overlap_squared"] for x in path),
        "minimum_blocked_mode_overlap_squared": min(blocked_overlaps) if blocked_overlaps else None}


def _split_normal_dirac_spectrum(h: torch.Tensor, mass_mev: float,
                                 *, minimum_cutoff_distance_mev: float = 200.0,
                                 minimum_band_separation_mev: float = 1000.0):
    """Separate shifted Dirac sectors only if a demonstrated finite-band gap exists.

    The convention follows h_D-M, whose positive continuum is near zero and
    negative continuum near -2M. No absolute-energy cutoff is used.
    """
    values, vectors = torch.linalg.eigh(h)
    cutoff = -float(mass_mev)
    distance = (values - cutoff).abs()
    sea = values < cutoff
    particle = values > cutoff
    if not bool(sea.any()) or not bool(particle.any()):
        raise RuntimeError("normal spectrum does not contain both Dirac sectors")
    edge_sea = float(values[sea].max())
    edge_particle = float(values[particle].min())
    minimum_distance = float(distance.min())
    band_separation = edge_particle - edge_sea
    evidence = {
        "shifted_dirac_cutoff_mev": cutoff,
        "maximum_negative_sector_energy_mev": edge_sea,
        "minimum_particle_sector_energy_mev": edge_particle,
        "minimum_distance_to_cutoff_mev": minimum_distance,
        "sea_to_particle_band_separation_mev": band_separation,
        "minimum_cutoff_distance_required_mev": minimum_cutoff_distance_mev,
        "minimum_band_separation_required_mev": minimum_band_separation_mev,
        "threshold_origin": "diagnostic-selected fail-closed engineering margins; not a theorem or universal cutoff from the cited papers",
        "safe_sector_split": minimum_distance >= minimum_cutoff_distance_mev
            and band_separation >= minimum_band_separation_mev,
        "negative_sector_dimension": int(sea.sum()),
        "particle_sector_dimension": int(particle.sum()),
        "eigenvalues_mev": values.detach().cpu().tolist(),
    }
    if not evidence["safe_sector_split"]:
        raise UnsafeDiracSectorSplit(
            "shifted Dirac particle/sea band gap fails declared safety margin: "
            f"min_distance={minimum_distance:.6g} MeV "
            f"(required {minimum_cutoff_distance_mev:.6g}), "
            f"band_separation={band_separation:.6g} MeV "
            f"(required {minimum_band_separation_mev:.6g}), "
            f"cutoff={cutoff:.6g} MeV, edges=({edge_sea:.6g},{edge_particle:.6g}) MeV",
            evidence)
    return values, vectors, sea, particle, evidence


def compare_fixed_fields(h: torch.Tensor, delta: torch.Tensor, *, mass_mev: float,
                         target_number: int = 3, continuation_steps: int = 4,
                         chemical_potential_mev: float | None = None,
                         blocked_basis_vector: torch.Tensor | None = None,
                         minimum_cutoff_distance_mev: float = 200.,
                         minimum_band_separation_mev: float = 1000.,
                         sector_policy: str = "fixed_W",
                         basis: AngularRadialBasis | None = None,
                         species: str | None = None) -> FixedFieldBranchComparison:
    """Compare fixed fields to strict normal-P+ using an explicit sector policy.

    ``fixed_W`` is the unchanged historical W=-M gate. The opt-in
    ``certified_gf_gap`` policy chooses W from the actual Hermitian h's G/F
    Gershgorin bounds and the supplied basis row partition. It is a conditional
    finite-matrix reference, not a production no-sea label certification.
    """
    h, delta = torch.as_tensor(h), torch.as_tensor(delta)
    if (h.dtype != torch.complex128 or h.ndim != 2 or h.shape[0] != h.shape[1]
            or delta.dtype != torch.complex128 or delta.shape != h.shape):
        raise TypeError("h and delta must be matching square complex128 matrices")
    if not bool(torch.isfinite(h).all()) or not bool(torch.isfinite(delta).all()):
        raise ValueError("fixed fields contain NaN or infinity")
    if not torch.allclose(h, h.mH, atol=1e-8, rtol=0.):
        raise ValueError("fixed normal field must be Hermitian")
    if not torch.allclose(delta, -delta.mT, atol=1e-8, rtol=0.):
        raise ValueError("fixed pairing field must be antisymmetric")

    if sector_policy == "fixed_W":
        values, vectors, sea_mask, particle_mask, split = _split_normal_dirac_spectrum(
            h, mass_mev, minimum_cutoff_distance_mev=minimum_cutoff_distance_mev,
            minimum_band_separation_mev=minimum_band_separation_mev)
        post_spectrum_gap_check = None
    elif sector_policy == "certified_gf_gap":
        if basis is None:
            raise ValueError("certified_gf_gap policy requires the actual AngularRadialBasis")
        if basis.dimension != h.shape[0]:
            raise ValueError("certified_gf_gap basis dimension does not match h")
        # The G/F Gershgorin proof chooses W without looking at eigenvalues.
        if species not in {"n", "p"}:
            raise ValueError("certified_gf_gap policy requires species='n' or 'p' for mass provenance")
        split = certify_basis_gf_interval(h, basis, mass_mev=mass_mev, species=species)
        h_raw = h
        h = 0.5 * (h_raw + h_raw.mH)
        split["reference_h_symmetrization_max_abs_mev"] = float((h-h_raw).abs().max())
        split["reference_h_symmetrization_frobenius_mev"] = float(torch.linalg.matrix_norm(h-h_raw))
        split["eigensystem_and_bdg_use_same_symmetrized_h"] = True
        values, vectors = torch.linalg.eigh(h)
        post_spectrum_gap_check = validate_spectrum_for_gf_interval(values, split)
        w = float(split["midpoint_W_mev"])
        sea_mask = values < w
        particle_mask = values > w
        if bool((values == w).any()):
            raise CertifiedGFGapError("normal spectrum contains an eigenvalue exactly at certified W", split)
        split = {**split, **post_spectrum_gap_check,
                 "safe_sector_split": True,
                 "sector_assignment_policy": "cutoff-defined lower/upper eigenspaces around G/F-certified W; mathematical G/F gap, physical Dirac branch labels remain conditional",
                 "legacy_W_minus_M_thresholds_used": False,
                 "legacy_thresholds_mev": {"minimum_cutoff_distance": minimum_cutoff_distance_mev,
                                            "minimum_band_separation": minimum_band_separation_mev}}
    else:
        raise ValueError("sector_policy must be 'fixed_W' or 'certified_gf_gap'")
    particle_indices = torch.nonzero(particle_mask, as_tuple=False).flatten()
    odd = target_number % 2 == 1
    if target_number < 0 or int(particle_indices.numel()) < target_number:
        raise RuntimeError("too few positive-HF modes for requested particle number")
    if not odd and int(particle_indices.numel()) <= target_number:
        raise RuntimeError("even HF reference needs an unoccupied mode above the requested filling")
    blocked_spectral_index = None
    blocked_basis_overlap = None
    even_core_count = target_number - 1 if odd else target_number
    if odd and blocked_basis_vector is not None:
        block = torch.as_tensor(blocked_basis_vector, dtype=h.dtype, device=h.device)
        if block.shape != (h.shape[0],) or float(torch.linalg.vector_norm(block)) <= 0.:
            raise ValueError("blocked basis identity must be one nonzero primitive vector")
        block = block / torch.linalg.vector_norm(block)
        scores = (vectors[:, particle_indices].mH @ block).abs().square()
        block_rank = int(scores.argmax())
        blocked_spectral_index = int(particle_indices[block_rank])
        blocked_basis_overlap = float(scores[block_rank])
    if blocked_spectral_index is None and odd:
        blocked_spectral_index = int(particle_indices[even_core_count])
    blocked_operation = None
    if odd and blocked_spectral_index is not None:
        block_energy = float(values[blocked_spectral_index])
        sorted_particle = [int(i) for i in particle_indices.tolist()]
        sorted_particle.sort(key=lambda i: float(values[i]))
        rank = sorted_particle.index(blocked_spectral_index)
        if chemical_potential_mev is not None:
            supplied_mu = float(chemical_potential_mev)
            if abs(block_energy - supplied_mu) <= 1e-10:
                raise RuntimeError("blocked normal mode lies at μ and has no unambiguous particle/hole qp label")
            blocked_operation = "particle" if block_energy > supplied_mu else "hole"
            actual_count = sum(float(values[i]) < supplied_mu for i in particle_indices.tolist())
            # The N±1 count relation is an HF-limit construction rule only.
            # At a caller-supplied final HFB μ, pairing may shift the mean
            # number away from this unpaired count, so record but do not gate.
            occupied_edge = (float(values[sorted_particle[actual_count - 1]]) if actual_count
                             else float(values[sorted_particle[0]]) - 1.)
            empty_edge = (float(values[sorted_particle[actual_count]])
                          if actual_count < len(sorted_particle) else None)
        else:
            # Prefer particle addition when this is a valid N-1 core. If the
            # specified orbital lies inside that core, use the equally valid
            # N+1 even reference and block a hole instead.
            add_core = [i for i in sorted_particle if i != blocked_spectral_index][:target_number - 1]
            can_add = (len(add_core) == target_number - 1 and
                       (not add_core or block_energy > max(float(values[i]) for i in add_core)))
            if can_add:
                blocked_operation = "particle"
                available = [i for i in sorted_particle if i not in add_core]
                occupied_edge = (float(values[add_core[-1]]) if add_core else
                                 min(float(values[i]) for i in particle_indices.tolist()) - 1.)
                empty_edge = min(float(values[i]) for i in available)
            else:
                hole_core = sorted_particle[:target_number + 1]
                if blocked_spectral_index not in hole_core or len(hole_core) < target_number + 1:
                    raise RuntimeError("designated block fits neither an N-1 particle nor N+1 hole reference")
                if len(sorted_particle) <= target_number + 1:
                    raise RuntimeError(
                        "HF-limit hole-reference bracket needs an unoccupied positive-energy mode above N+1; "
                        "the supplied finite spectrum has none")
                blocked_operation = "hole"
                occupied_edge = float(values[hole_core[-1]])
                empty_edge = float(values[sorted_particle[target_number + 1]])
    else:
        occupied_edge = float(values[particle_indices[target_number - 1]])
        empty_edge = float(values[particle_indices[target_number]])
    bracket_mu = (0.5 * (occupied_edge + empty_edge)
                  if empty_edge is not None else None)
    if chemical_potential_mev is None and bracket_mu is None:
        raise RuntimeError("normal HF reference has no finite unoccupied-side bracket")
    mu = bracket_mu if chemical_potential_mev is None else float(chemical_potential_mev)
    if not np.isfinite(mu):
        raise ValueError("chemical potential must be finite")
    if chemical_potential_mev is None and not occupied_edge < mu < empty_edge:
        raise RuntimeError("normal HF reference has no open chemical-potential interval")
    normal_hf_count_at_mu = sum(float(values[i]) < mu for i in particle_indices.tolist())
    expected_hf_reference_count = (target_number - 1 if blocked_operation == "particle" else
                                   target_number + 1 if blocked_operation == "hole" else target_number)
    hf_reference_count_match = normal_hf_count_at_mu == expected_hf_reference_count
    mu_inside_number_gap = (occupied_edge < mu if empty_edge is None else
                            occupied_edge < mu < empty_edge)

    # Identify the positive particle qp connected to the selected normal mode.
    def positive_index_for_normal_mode(hmat, mumat, normal_vec, sector_projector, operation="particle"):
        k0 = nambu_hamiltonian(hmat, torch.zeros_like(hmat), chemical_potential=mumat)
        e0, v0 = torch.linalg.eigh(k0)
        ids = torch.nonzero(e0 > 0., as_tuple=False).flatten()
        modes = v0[:, ids]
        seaop = torch.block_diag(sector_projector, sector_projector.conj())
        sw = torch.einsum("ij,ji->i", modes.mH, seaop @ modes).real
        eligible = torch.nonzero(sw < .1, as_tuple=False).flatten()
        if eligible.numel() == 0:
            raise RuntimeError("no positive-energy particle qp available for blocking")
        if operation == "particle":
            qp_identity = torch.cat((normal_vec, torch.zeros_like(normal_vec)))
        else:
            qp_identity = torch.cat((torch.zeros_like(normal_vec), normal_vec.conj()))
        eligible_scores = (modes[:, eligible].mH @ qp_identity).abs().square()
        local = int(eligible_scores.argmax())
        if float(eligible_scores[local]) < .5:
            raise RuntimeError("normal blocked orbital has no unambiguous positive particle qp label")
        return int(eligible[local]), float(eligible_scores[local]), e0, modes, sw

    pminus = vectors[:, sea_mask] @ vectors[:, sea_mask].mH
    blocked_positive_index = None
    if odd:
        blocked_positive_index, blocked_qp_overlap, e0full, v0full, swfull = positive_index_for_normal_mode(
            h, mu, vectors[:, blocked_spectral_index], pminus, blocked_operation)
    else:
        blocked_qp_overlap = None
        e0full, vfull = torch.linalg.eigh(nambu_hamiltonian(h, torch.zeros_like(h), chemical_potential=mu))
        v0full = vfull[:, torch.nonzero(e0full > 0.).flatten()]
        swfull = torch.einsum("ij,ji->i", v0full.mH,
            torch.block_diag(pminus, pminus.conj()) @ v0full).real
    # If the no-sea quasiparticle vacuum is odd, flip its lowest particle qp
    # to define an even reference; only then flip the named blocked qp.
    base_full = _frame_from_positive_modes(v0full, swfull)
    _, base_rho, base_chi = _density_from_frame(base_full)
    base_parity = number_parity_from_density(base_rho.detach().cpu().numpy(), base_chi.detach().cpu().numpy())
    positive_ids = torch.nonzero(e0full > 0.).flatten()
    positive_full = v0full
    sea_weights = torch.einsum("ij,ji->i", positive_full.mH,
        torch.block_diag(pminus, pminus.conj()) @ positive_full).real
    particle_qps = torch.nonzero(sea_weights < .1).flatten()
    comp_index = None
    if base_parity["number_parity"] < 0:
        if particle_qps.numel() == 0:
            raise RuntimeError("odd no-sea vacuum has no particle qp for even-sector compensation")
        comp_index = int(particle_qps[torch.argmin(e0full[positive_ids[particle_qps]].abs())])
    whole_even_frame, whole_even_path = _sector_labeled_vacuum(
        h, delta, mu, pminus, steps=continuation_steps,
        toggle_positive_indices=(() if comp_index is None else (comp_index,)))
    _, even_rho, even_chi = _density_from_frame(whole_even_frame)
    even_parity = number_parity_from_density(even_rho.detach().cpu().numpy(), even_chi.detach().cpu().numpy())
    if even_parity["number_parity"] != 1:
        raise RuntimeError("constructed whole-Dirac unblocked reference is not even parity")
    whole_toggles = (() if comp_index is None else (comp_index,))
    if odd:
        whole_toggles += (blocked_positive_index,)
    whole_frame, whole_path = _sector_labeled_vacuum(
        h, delta, mu, pminus, steps=continuation_steps,
        toggle_positive_indices=whole_toggles)
    whole_r, whole_rho, whole_chi = _density_from_frame(whole_frame)
    whole_h = nambu_hamiltonian(h, delta, chemical_potential=mu)
    whole_parity = number_parity_from_density(whole_rho.detach().cpu().numpy(), whole_chi.detach().cpu().numpy())
    expected_parity = -1 if odd else 1
    if whole_parity["number_parity"] != expected_parity:
        raise RuntimeError(f"whole-Dirac selected branch has parity {whole_parity['number_parity']}, expected {expected_parity}")

    # The candidate's strict normal-P+ operation projects both particle legs
    # of Delta, then solves only the positive-HF subspace BdG problem.
    uplus = vectors[:, particle_indices]
    hp = uplus.mH @ h @ uplus
    dp = uplus.mH @ delta @ uplus.conj()
    hp_values = values[particle_indices]
    plus_occupied_edge = float(hp_values[target_number - 1])
    plus_empty_edge = float(hp_values[target_number])
    pplus_zero = torch.zeros((hp.shape[0], hp.shape[0]), dtype=h.dtype, device=h.device)
    plus_blocked_index = None
    plus_comp_index = None
    if odd:
        normal_block = uplus.mH @ vectors[:, blocked_spectral_index]
        plus_blocked_index, _, e0plus, v0plus, swplus = positive_index_for_normal_mode(
            hp, mu, normal_block / torch.linalg.vector_norm(normal_block), pplus_zero, blocked_operation)
        plus_pos = v0plus
    else:
        e0plus, v0plus = torch.linalg.eigh(nambu_hamiltonian(hp, torch.zeros_like(hp), chemical_potential=mu))
        swplus = None
        plus_pos_ids = torch.nonzero(e0plus > 0.).flatten()
        plus_pos = v0plus[:, plus_pos_ids]
    plus_pos_ids = torch.nonzero(e0plus > 0.).flatten()
    plus_positive_energies = e0plus[plus_pos_ids]
    if odd:
        plus_comp_index = None
    plus_base = _frame_from_positive_modes(plus_pos, torch.zeros(plus_pos.shape[1], dtype=torch.float64, device=h.device))
    _, plus_base_rho, plus_base_chi = _density_from_frame(plus_base)
    plus_base_parity = number_parity_from_density(plus_base_rho.detach().cpu().numpy(), plus_base_chi.detach().cpu().numpy())
    if plus_base_parity["number_parity"] < 0:
        plus_comp_index = int(torch.argmin(plus_positive_energies.abs()))
    plus_even, plus_even_path = _sector_labeled_vacuum(
        hp, dp, mu, pplus_zero, steps=continuation_steps,
        toggle_positive_indices=(() if plus_comp_index is None else (plus_comp_index,)))
    _, plus_even_rho, plus_even_chi = _density_from_frame(plus_even)
    plus_even_parity = number_parity_from_density(plus_even_rho.detach().cpu().numpy(), plus_even_chi.detach().cpu().numpy())
    if plus_even_parity["number_parity"] != 1:
        raise RuntimeError("constructed projected unblocked reference is not even parity")
    plus_toggles = (() if plus_comp_index is None else (plus_comp_index,))
    if odd:
        plus_toggles += (plus_blocked_index,)
    plus_frame, plus_path = _sector_labeled_vacuum(
        hp, dp, mu, pplus_zero, steps=continuation_steps,
        toggle_positive_indices=plus_toggles)
    plus_r, plus_rho, plus_chi = _density_from_frame(plus_frame)
    plus_parity = number_parity_from_density(plus_rho.detach().cpu().numpy(), plus_chi.detach().cpu().numpy())
    if plus_parity["number_parity"] != expected_parity:
        raise RuntimeError(f"projected selected branch has parity {plus_parity['number_parity']}, expected {expected_parity}")
    nplus = hp.shape[0]
    wplus = torch.block_diag(uplus, uplus.conj())
    projected_r = wplus @ plus_r @ wplus.mH
    d = h.shape[0]
    projected_rho = uplus @ plus_rho @ uplus.mH
    projected_chi = uplus @ plus_chi @ uplus.T

    pplus = uplus @ uplus.mH
    pminus = vectors[:, sea_mask] @ vectors[:, sea_mask].mH
    # The strict P+ model has no sea particles, but its full-space canonical
    # generalized density retains the empty-particle sea-hole vacuum block.
    projected_r = projected_r + torch.block_diag(
        torch.zeros_like(pminus), pminus.conj())

    whole_residual = whole_h @ whole_r - whole_r @ whole_h
    projected_residual = whole_h @ projected_r - projected_r @ whole_h
    projected_inside = nambu_hamiltonian(hp, dp, chemical_potential=mu)
    projected_inside_residual = projected_inside @ plus_r - plus_r @ projected_inside
    projected_parity = number_parity_from_density(
        projected_rho.detach().cpu().numpy(), projected_chi.detach().cpu().numpy())

    whole_rho_block = whole_r[:d, :d]
    whole_chi_block = whole_r[:d, d:]
    canonical_whole_r = torch.cat((torch.cat((whole_rho_block, whole_chi_block), dim=1),
        torch.cat((-whole_chi_block.conj(), torch.eye(d, dtype=h.dtype, device=h.device) -
                   whole_rho_block.conj()), dim=1)), dim=0)
    whole_comm_frob = float(torch.linalg.matrix_norm(whole_residual))
    projected_inside_comm_frob = float(torch.linalg.matrix_norm(projected_inside_residual))
    whole_purity_frob = float(torch.linalg.matrix_norm(whole_r @ whole_r - whole_r))
    whole_canonical_frob = float(torch.linalg.matrix_norm(whole_r-canonical_whole_r))
    whole_chi_skew_max = float((whole_chi_block+whole_chi_block.mT).abs().max())
    strict_plus_purity_frob = float(torch.linalg.matrix_norm(plus_r @ plus_r-plus_r))
    strict_plus_chi_skew_max = float((plus_chi+plus_chi.mT).abs().max())
    whole_number_error = abs(float(whole_parity["particle_number"])-target_number)
    projected_number_error = abs(float(projected_parity["particle_number"])-target_number)

    projected_rho_full = projected_r[:d, :d]
    projected_chi_full = projected_r[:d, d:]
    canonical_projected_r = torch.cat((torch.cat((projected_rho_full, projected_chi_full), dim=1),
        torch.cat((-projected_chi_full.conj(), torch.eye(d, dtype=h.dtype, device=h.device) -
                   projected_rho_full.conj()), dim=1)), dim=0)
    projected_canonical_residual = float(torch.linalg.matrix_norm(projected_r - canonical_projected_r))
    if projected_canonical_residual > 1e-8 or abs(float(torch.trace(projected_r).real) - d) > 1e-8:
        raise RuntimeError("embedded projected density omitted canonical sea-hole vacuum or lost rank")
    cross_pm = pplus @ delta @ pminus.conj()
    cross_mp = pminus @ delta @ pplus.conj()
    whole_gap = whole_path["minimum_quasiparticle_gap_mev"]
    projected_gap = plus_path["minimum_quasiparticle_gap_mev"]
    report = {
        "status": "fixed_field_reference_completed",
        "scope": "same frozen h and full D1S Delta; exact finite-matrix branch comparison only, no field iteration or self-consistent ground-state claim",
        "sector_policy": sector_policy,
        "conditional_reference_limit": ("The G/F-certified W split is a mathematical finite-matrix/homotopy reference only; it does not itself certify nuclear physical Dirac-sea or positive-particle labels." if sector_policy == "certified_gf_gap" else "historical W=-M cutoff policy"),
        "normal_dirac_split": split,
        "fixed_field_matrix_sha256": {
            "h_used_for_reference": hashlib.sha256(h.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            "delta": hashlib.sha256(delta.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        },
        "chemical_potential_mev": mu,
        "normal_number_gap_bracket_mev": [occupied_edge, empty_edge],
        "chemical_potential_inside_input_number_gap": mu_inside_number_gap,
        "unpaired_normal_occupation_count_at_mu": normal_hf_count_at_mu,
        "expected_hf_limit_reference_count_for_selected_blocking": expected_hf_reference_count,
        "hf_limit_reference_count_matches": hf_reference_count_match,
        "hf_limit_count_gate_applies_to_supplied_mu": False,
        "chemical_potential_source": "normal-spectrum midpoint" if chemical_potential_mev is None else "caller-supplied; common to both reference branches",
        "target_number_for_branch_reference": target_number,
        "even_unblocked_reference_parity": even_parity["number_parity"],
        "even_unblocked_reference_particle_number_at_full_delta": even_parity["particle_number"],
        "even_reference_energy_minimum_claim": False,
        "even_reference_scope": "parity-corrected continuation of the Δ=0 no-sea qp vacuum; not proven lowest at full Δ",
        "positive_particle_vacuum_parity_before_even_compensation": base_parity["number_parity"],
        "vacuum_compensation_positive_qp_index": comp_index,
        "blocked_mode_reference": {
            "spectral_index": blocked_spectral_index,
            "overlap_with_seed_basis_vector": blocked_basis_overlap,
            "overlap_with_positive_particle_qp_at_delta_zero": blocked_qp_overlap,
            "operation": blocked_operation,
            "source": "maximal overlap with supplied saved-seed blocked basis vector" if blocked_basis_vector is not None else "lowest positive-HF mode at odd occupation frontier",
        },
        "whole_dirac_branch": {
            **whole_path,
            "particle_number": whole_parity["particle_number"],
            "number_parity": whole_parity["number_parity"],
            "normalized_pfaffian": whole_parity["normalized_pfaffian"],
            "purity_residual": whole_parity["purity_error"],
            "full_field_commutator_frobenius_mev": whole_comm_frob,
            "full_field_commutator_relative": float(torch.linalg.matrix_norm(whole_residual) /
                (torch.linalg.matrix_norm(whole_h) * torch.linalg.matrix_norm(whole_r)).clamp_min(1e-30)),
            "density_projector_trace": float(torch.trace(whole_r).real),
            "target_number_absolute_error": whole_number_error,
            "generalized_density_idempotency_frobenius": whole_purity_frob,
            "canonical_block_form_residual_frobenius": whole_canonical_frob,
            "chi_skew_max_abs": whole_chi_skew_max,
        },
        "strict_normal_positive_projected_bdg": {
            **plus_path,
            "particle_number": projected_parity["particle_number"],
            "number_parity": projected_parity["number_parity"],
            "normalized_pfaffian": projected_parity["normalized_pfaffian"],
            "purity_residual": projected_parity["purity_error"],
            "projected_space_commutator_frobenius_mev": float(torch.linalg.matrix_norm(projected_inside_residual)),
            "projected_space_commutator_gate_quantity_frobenius_mev": projected_inside_comm_frob,
            "same_full_field_commutator_frobenius_mev": float(torch.linalg.matrix_norm(projected_residual)),
            "same_full_field_commutator_relative": float(torch.linalg.matrix_norm(projected_residual) /
                (torch.linalg.matrix_norm(whole_h) * torch.linalg.matrix_norm(projected_r)).clamp_min(1e-30)),
            "density_projector_trace": float(torch.trace(projected_r).real),
            "sea_empty_particle_hole_trace": float(torch.trace(pminus.conj()).real),
            "embedded_canonical_density_residual": projected_canonical_residual,
            "target_number_absolute_error": projected_number_error,
            "active_generalized_density_idempotency_frobenius": strict_plus_purity_frob,
            "active_chi_skew_max_abs": strict_plus_chi_skew_max,
            "active_generalized_density_trace": float(torch.trace(plus_r).real),
        },
        "density_differences_frobenius": {
            "normal_rho": float(torch.linalg.matrix_norm(whole_rho - projected_rho)),
            "anomalous_chi": float(torch.linalg.matrix_norm(whole_chi - projected_chi)),
            "generalized_density": float(torch.linalg.matrix_norm(whole_r - projected_r)),
        },
        "full_d1s_pairing_cross_blocks_frobenius_mev": {
            "positive_output_from_negative_input": float(torch.linalg.matrix_norm(cross_pm)),
            "negative_output_from_positive_input": float(torch.linalg.matrix_norm(cross_mp)),
            "combined_frobenius": float(torch.sqrt(torch.linalg.matrix_norm(cross_pm).square() +
                torch.linalg.matrix_norm(cross_mp).square())),
            "relative_to_full_delta": float(torch.sqrt(torch.linalg.matrix_norm(cross_pm).square() +
                torch.linalg.matrix_norm(cross_mp).square()) /
                torch.linalg.matrix_norm(delta).clamp_min(1e-30)),
        },
        "active_projected_pairing_gap_frobenius_mev": float(torch.linalg.matrix_norm(dp)),
        "fixed_normal_field_frobenius_mev": float(torch.linalg.matrix_norm(h)),
        "fixed_full_pairing_field_frobenius_mev": float(torch.linalg.matrix_norm(delta)),
        "normal_h_spectrum_mev": values.detach().cpu().tolist(),
        "nambu_dimension_whole": 2 * d,
        "nambu_dimension_projected": 2 * nplus,
        "reference_spectral_api_call_counts": {
            "normal_h_eigh": 1,
            "whole_bdg_eigh_total_observed_by_call_graph": 2 * (continuation_steps + 1) + 1,
            "whole_bdg_eigvalsh": 0,
            "projected_bdg_eigh_total_observed_by_call_graph": 2 * (continuation_steps + 1) + 1,
            "projected_bdg_eigvalsh": 0,
            "training_spectral_calls": 0,
        },
        "notes": [
            "Whole-Dirac vacuum uses particle/sea annihilator labels by normal Dirac-sector weights, continuation transports identity only, and a separate explicit blocked-mode toggle creates odd parity.",
            "Strict projected BdG uses the same interacting normal h to form P+ and projects both legs of the full D1S Delta.",
            "Both outputs diagonalize the same fixed field. Neither operation updates h, Delta, meson fields, density, or the NN parameters.",
        ],
    }
    if sector_policy == "certified_gf_gap":
        comm_limit = 1.0e-7
        purity_limit = 1.0e-10
        canonical_limit = 1.0e-10
        skew_limit = 1.0e-10
        number_limit = 1.0e-6
        hard_gates = {
            "whole_full_h_commutator_le_1e-7_mev": whole_comm_frob <= comm_limit,
            "strict_projected_own_hpp_commutator_le_1e-7_mev": projected_inside_comm_frob <= comm_limit,
            "whole_projector_purity_le_1e-10": whole_purity_frob <= purity_limit,
            "strict_projector_purity_le_1e-10": strict_plus_purity_frob <= purity_limit,
            "whole_canonical_block_form_le_1e-10": whole_canonical_frob <= canonical_limit,
            "strict_embedded_canonical_form_le_1e-10": projected_canonical_residual <= canonical_limit,
            "whole_chi_skew_le_1e-10": whole_chi_skew_max <= skew_limit,
            "strict_chi_skew_le_1e-10": strict_plus_chi_skew_max <= skew_limit,
            "whole_expected_number_parity": whole_parity["number_parity"] == expected_parity,
            "strict_expected_number_parity": projected_parity["number_parity"] == expected_parity,
            "whole_target_number_within_1e-6": whole_number_error <= number_limit,
            "strict_target_number_within_1e-6": projected_number_error <= number_limit,
        }
        report["conditional_reference_hard_gates"] = {
            "thresholds": {"commutator_frobenius_mev": comm_limit,
                           "projector_purity_frobenius": purity_limit,
                           "canonical_form_frobenius": canonical_limit,
                           "chi_skew_max_abs": skew_limit,
                           "target_number_absolute_error": number_limit},
            "checks": hard_gates,
            "all_passed": all(hard_gates.values()),
            "strict_projected_same_full_field_commutator_is_reported_only": True,
            "strict_projected_same_full_field_commutator_is_not_a_gate": True,
            "number_interpretation": "At fixed reference μ, a branch with |Tr(rho)-target|>1e-6 does not satisfy the number constraint.",
            "chemical_potential_interpretation": "The common μ is only the fixed-h normal HF bracket midpoint (unless caller supplied); it is not a self-consistent NN chemical potential.",
            "physical_solution_accepted": False,
        }
        report["status"] = ("conditional_gf_reference_hard_gates_passed" if all(hard_gates.values())
                             else "conditional_gf_reference_hard_gate_failed")
    return FixedFieldBranchComparison(report, whole_rho, whole_chi,
                                     projected_rho, projected_chi)


def compare_saved_fields(input_npz: str | Path, output_directory: str | Path, *,
                         mass_n_mev: float, mass_p_mev: float,
                         continuation_steps: int = 4,
                         pairing_npz: str | Path | None = None,
                         chemical_potential_n_mev: float | None = None,
                         chemical_potential_p_mev: float | None = None,
                         blocked_basis_mode_index_n: int | None = None,
                         blocked_basis_mode_index_p: int | None = None):
    """Compare fields saved by the bounded PKA1 build and D1S action build."""
    input_path = Path(input_npz).resolve()
    out = Path(output_directory).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    pairing_path = Path(pairing_npz).resolve() if pairing_npz is not None else input_path
    pairing_digest = hashlib.sha256(pairing_path.read_bytes()).hexdigest()
    source_files = ("basis.py", "functional.py", "branch.py", "no_sea_particle_space.py",
                    "gaussian_pairing.py", "pairing.py", "fixed_field_dirac_branch_reference.py")
    source_hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                     for name in source_files}
    with np.load(input_path) as z, np.load(pairing_path) as dz:
        if "metadata_json" in z:
            config = json.loads(str(z["metadata_json"].item()))
        else:
            sidecar = input_path.with_name("metadata.json")
            if not sidecar.exists():
                sidecar = input_path.with_name("seed_state_metadata.json")
            if not sidecar.exists():
                raise ValueError("field NPZ needs an embedded metadata_json or adjacent metadata sidecar")
            config = json.loads(sidecar.read_text())
        results = {}
        arrays = {}
        failures = {}
        unsafe_spectrum_evidence = {}
        for species, mass, chemical_potential in (
            ("n", mass_n_mev, chemical_potential_n_mev),
            ("p", mass_p_mev, chemical_potential_p_mev),
        ):
            try:
                h = torch.from_numpy(z[f"{species}_h"].copy())
                delta = torch.from_numpy(dz[f"{species}_delta"].copy())
                target = int(config.get(f"{species}_branch_target_number",
                                        3 if species == "n" else 4))
                requested_block = (blocked_basis_mode_index_n if species == "n"
                                   else blocked_basis_mode_index_p)
                if requested_block is None:
                    requested_block = config.get(f"{species}_blocked_mode_index")
                block_vector = None
                q_key = f"{species}_Q"
                if target % 2 and requested_block is not None and q_key in z:
                    q_input = torch.from_numpy(z[q_key].copy())
                    block_vector = q_input[:, int(requested_block)]
                result = compare_fixed_fields(h, delta, mass_mev=mass,
                    target_number=target, continuation_steps=continuation_steps,
                    chemical_potential_mev=chemical_potential,
                    blocked_basis_vector=block_vector)
                results[species] = result.report
                arrays[species] = result
            except (RuntimeError, ValueError) as exc:
                failures[species] = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, UnsafeDiracSectorSplit):
                    unsafe_spectrum_evidence[species] = exc.evidence
    saved_density_arrays = {}
    for species, result in arrays.items():
        saved_density_arrays.update({
            f"{species}_whole_rho": result.whole_rho.cpu().numpy(),
            f"{species}_whole_chi": result.whole_chi.cpu().numpy(),
            f"{species}_projected_rho": result.projected_rho.cpu().numpy(),
            f"{species}_projected_chi": result.projected_chi.cpu().numpy(),
        })
    np.savez_compressed(out / "branch_density_comparison.npz", **saved_density_arrays)
    report = {
        "status": "completed" if not failures else "partial_reference; unsafe species withheld",
        "input_npz": str(input_path),
        "input_sha256": digest,
        "pairing_npz": str(pairing_path),
        "pairing_npz_sha256": pairing_digest,
        "reference_source_sha256_at_run": source_hashes,
        "configuration": config,
        "source_references": {
            "whole_dirac_qp_branch": "DIRHB arXiv:1403.4039 Eqs. 46-48",
            "D1S_dirac_components": "Geng and Long arXiv:2202.00482 Eq. 20d and Appendix B",
        },
        "fortran_used": False,
        "neural_optimizer_updates": 0,
        "species": results,
        "withheld_species": failures,
        "unsafe_normal_spectrum_evidence": unsafe_spectrum_evidence,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(out / "diagnostic.json", report)
    return report
