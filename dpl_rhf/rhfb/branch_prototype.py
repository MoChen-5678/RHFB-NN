"""Tiny full-Dirac odd-HFB branch-following experiment.

This is a numerical method prototype, not a normal-ordered production RHFB
functional.  It keeps the full anomalous tensor and full primitive Dirac
space, uses unscaled two-Gaussian D1S, and excludes negative-Dirac states
from normal-density functional sources.  The precise mapping between a pure
generalized density and the normal-ordered no-sea sources remains an open
physics issue; sea occupation is therefore reported rather than hidden.

The candidate path optimizes a connected canonical exp(K) chart with SciPy
and PyTorch autograd.  The independent reference path diagonalizes the full
BdG matrix, pairs its particle-hole eigenvectors, and follows one member of
each pair from the no-sea branch.  No eigensolve is used by the candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from scipy.optimize import minimize

from .branch import (
    no_sea_branch_reference,
    nambu_hamiltonian,
    particle_hole_conjugate,
    transform_branch_reference,
)
from .manifold import GeneralizedDensity, generalized_density_matrix
from .manifold import canonical_generator
from .pairing import D1S, pairing_gap


@dataclass(frozen=True)
class FullDiracOddModel:
    h0: torch.Tensor
    source_a: torch.Tensor
    source_b: torch.Tensor
    coupling_a: float
    coupling_b: float
    spinors: torch.Tensor
    positions: torch.Tensor
    weights: torch.Tensor
    target_number: int
    dirac_cutoff: float
    pairing_enabled: bool = True

    @property
    def dimension(self) -> int:
        return int(self.h0.shape[0])

    @property
    def sea_mask(self) -> torch.Tensor:
        return torch.diagonal(self.h0).real <= self.dirac_cutoff

    def normal_energy_field(self, rho: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Density-dependent normal functional and its full-space derivative.

        Sea orbitals remain in h0 and in the BdG space.  They are omitted from
        the density-dependent source contractions, representing the
        conventional no-sea particle-hole prescription for this prototype.
        """
        p = torch.diag((~self.sea_mask).to(torch.float64)).to(torch.complex128)
        rho_source = p @ rho @ p
        xa = torch.trace(self.source_a @ rho_source).real
        xb = torch.trace(self.source_b @ rho_source).real
        energy = (
            torch.trace(self.h0 @ rho).real
            + 0.5 * self.coupling_a * xa.square()
            + (self.coupling_b / 3.0) * xb.pow(3)
        )
        field = self.h0 + p @ (
            self.coupling_a * xa * self.source_a
            + self.coupling_b * xb.square() * self.source_b
        ) @ p
        return energy, field

    def energy_fields(self, rho: torch.Tensor, chi: torch.Tensor):
        normal, h = self.normal_energy_field(rho)
        if self.pairing_enabled:
            pair_energy, delta = pairing_gap(
                self.spinors, chi, self.positions, self.weights,
                species="neutron", chunk_size=8, force=D1S,
            )
        else:
            pair_energy = rho.real.new_zeros(())
            delta = torch.zeros_like(chi)
        return normal + pair_energy, h, delta, normal, pair_energy


def make_full_dirac_model(target_number: int = 3, *, pairing_enabled: bool = True,
                          device: str | torch.device = "cpu") -> FullDiracOddModel:
    """Build a reproducible 8-orbital Dirac space: six particle, two sea."""
    if target_number not in (3, 5):
        raise ValueError("the tiny odd model supports N=3 or N=5")
    device = torch.device(device)
    energies = torch.tensor([0.0, 0.42, 0.9, 1.45, 2.1, 2.8, -12.0, -14.0],
                            dtype=torch.float64, device=device)
    h0 = torch.diag(energies).to(torch.complex128)
    a = torch.zeros((8, 8), dtype=torch.complex128, device=device)
    b = torch.zeros_like(a)
    # Couplings are deliberately non-diagonal and weak enough to preserve a
    # broad particle/sea gap while making the self-consistent field active.
    for i, j, value in ((0, 1, .22), (1, 2, -.17), (3, 4, .13),
                        (0, 6, .045), (2, 7, -.035), (4, 6, .025)):
        a[i, j] = a[j, i] = value
    for i, j, value in ((0, 3, .18), (1, 4, -.14), (2, 5, .11),
                        (0, 6, .025), (2, 7, .02)):
        b[i, j] = b[j, i] = value
    # Eight coordinates x four Dirac components give 32 sampled components;
    # a seeded complex QR provides eight mutually orthonormal full-space
    # spinors, all with support in both upper and lower Dirac components.
    generator = torch.Generator(device="cpu").manual_seed(39173)
    raw = torch.randn((32, 8), dtype=torch.float64, generator=generator) + 1j * torch.randn(
        (32, 8), dtype=torch.float64, generator=generator
    )
    spinors_flat, _ = torch.linalg.qr(raw.to(torch.complex128), mode="reduced")
    spinors = spinors_flat.reshape(8, 4, 8).to(device)
    positions = torch.tensor([[0., 0., 0.], [1.1, .1, 0.], [0., 1.2, .2],
                              [.2, .1, 1.3], [1.1, 1.2, .1], [1.2, .2, 1.4],
                              [.1, 1.3, 1.4], [1.3, 1.1, 1.2]], dtype=torch.float64, device=device)
    weights = torch.ones(8, dtype=torch.float64, device=device)
    return FullDiracOddModel(h0, a, b, 0.24, 0.09, spinors, positions, weights,
                             target_number, -5.0, pairing_enabled)


def _coordinate_basis(d: int, device: torch.device):
    """Real coordinates for all anti-Hermitian A and complex-skew B."""
    a_basis, b_basis = [], []
    for i in range(d):
        m = torch.zeros((d, d), dtype=torch.complex128, device=device)
        m[i, i] = 1j
        a_basis.append(m)
    for i in range(d):
        for j in range(i + 1, d):
            m = torch.zeros((d, d), dtype=torch.complex128, device=device)
            m[i, j], m[j, i] = 1., -1.
            a_basis.append(m)
            m = torch.zeros_like(m)
            m[i, j], m[j, i] = 1j, 1j
            a_basis.append(m)
            m = torch.zeros_like(m)
            m[i, j], m[j, i] = 1., -1.
            b_basis.append(m)
            m = torch.zeros_like(m)
            m[i, j], m[j, i] = 1j, -1j
            b_basis.append(m)
    return torch.stack(a_basis), torch.stack(b_basis)


class CanonicalCoordinateNetwork(nn.Module):
    """SiLU network that emits canonical-generator coordinates.

    Each output corresponds to one independent real coordinate of A or B.
    Fixed features identify its channel, orbital pair, and sea participation;
    trainable offsets retain the full expressive canonical chart in this tiny
    benchmark while the MLP learns a smooth structured correction.
    """
    def __init__(self, d: int, device: torch.device):
        super().__init__()
        features = [(0., i/d, i/d, float(i >= d-2)) for i in range(d)]
        for channel in (1., 2.):
            for i in range(d):
                for j in range(i+1, d):
                    sea = float(i >= d-2 or j >= d-2)
                    features.extend(((channel, i/d, j/d, sea),
                                     (channel, i/d, j/d, sea)))
        # A has d diagonal + 2*C(d,2); B has 2*C(d,2) coordinates.
        expected = d*d + d*(d-1)
        if len(features) != expected:
            raise RuntimeError("canonical coordinate feature count mismatch")
        self.register_buffer("features", torch.tensor(features, dtype=torch.float64, device=device))
        self.mlp = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 1)).to(dtype=torch.float64, device=device)
        self.offset = nn.Parameter(torch.zeros(expected, dtype=torch.float64, device=device))
        with torch.no_grad():
            for layer in self.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=.08)
                    nn.init.zeros_(layer.bias)

    def forward(self) -> torch.Tensor:
        return self.offset + self.mlp(self.features).squeeze(-1)


def _density_from_coordinates(x: torch.Tensor, branch, a_basis, b_basis):
    na = a_basis.shape[0]
    a = torch.einsum("i,ijk->jk", x[:na].to(torch.complex128), a_basis)
    b = torch.einsum("i,ijk->jk", x[na:].to(torch.complex128), b_basis)
    return transform_branch_reference(branch, a, b)


def _branch_overlap(branch, state: GeneralizedDensity) -> torch.Tensor:
    d = branch.dimension
    frame = branch.frame
    return torch.linalg.matrix_norm(frame.mH @ state.matrix @ frame).square().real / d


def _projector_overlap(reference: GeneralizedDensity, state: GeneralizedDensity) -> torch.Tensor:
    """Normalized overlap with an adjacent continuation projector."""
    d = state.rho.shape[0]
    return torch.linalg.matrix_norm(reference.matrix @ state.matrix @ reference.matrix).square().real / d


def _paired_continuation_seed(branch, model):
    """Small cross-Fermi canonical seed prevents the zero-κ SCF fixed point."""
    d, device = model.dimension, model.h0.device
    a = torch.zeros((d, d), dtype=torch.complex128, device=device)
    b = torch.zeros_like(a)
    # Pair the highest initially occupied particle state with its next empty
    # positive-energy partner, and include a smaller positive/sea component.
    i, j = 0, 1
    b[i, j], b[j, i] = .075, -.075
    b[0, d - 2], b[d - 2, 0] = .012, -.012
    u = torch.matrix_exp(canonical_generator(a, b))
    state = GeneralizedDensity(
        (u @ branch.density.matrix @ u.mH)[:d, :d],
        (u @ branch.density.matrix @ u.mH)[:d, d:],
        u @ branch.density.matrix @ u.mH,
        branch.density.fermion_parity,
    )
    return state, u @ branch.frame


def _metrics(model, rho, chi, state, branch, energy, ref_state=None, *, chemical_potential=0.0):
    total, h, delta, normal, pair = model.energy_fields(rho, chi)
    hbdg = nambu_hamiltonian(h, delta, chemical_potential=chemical_potential)
    comm = torch.linalg.matrix_norm(hbdg @ state.matrix - state.matrix @ hbdg)
    comm_matrix = hbdg @ state.matrix - state.matrix @ hbdg
    d = model.dimension
    active = torch.cat((torch.arange(6, device=rho.device),
                        torch.arange(d, d + 6, device=rho.device)))
    sea = torch.cat((torch.arange(6, d, device=rho.device),
                     torch.arange(d + 6, 2*d, device=rho.device)))
    active_block = comm_matrix[active[:, None], active[None, :]]
    sea_cross = torch.cat((comm_matrix[active[:, None], sea[None, :]].reshape(-1),
                           comm_matrix[sea[:, None], active[None, :]].reshape(-1)))
    sea_block = comm_matrix[sea[:, None], sea[None, :]]
    n = torch.trace(rho).real
    sea_n = torch.sum(model.sea_mask.to(torch.float64) * rho.diagonal().real)
    result = {
        "particle_number": float(n.detach()),
        "particle_number_error": float((n - model.target_number).abs().detach()),
        "pairing_tensor_norm": float(torch.linalg.matrix_norm(chi).detach()),
        "pairing_energy_mev": float(pair.detach()),
        "pairing_field_norm_mev": float(torch.linalg.matrix_norm(delta).detach()),
        "normal_energy_mev": float(normal.detach()),
        "energy_mev": float(total.detach()),
        "commutator_frobenius": float(comm.detach()),
        "commutator_rms_mev": float(torch.sqrt(torch.mean(torch.abs(comm_matrix).square())).detach()),
        "commutator_active_rms_mev": float(torch.sqrt(torch.mean(torch.abs(active_block).square())).detach()),
        "commutator_sea_cross_rms_mev": float(torch.sqrt(torch.mean(torch.abs(sea_cross).square())).detach()),
        "commutator_sea_rms_mev": float(torch.sqrt(torch.mean(torch.abs(sea_block).square())).detach()),
        "commutator_relative": float((comm / (torch.linalg.matrix_norm(hbdg) *
                                                     torch.linalg.matrix_norm(state.matrix)).clamp_min(1e-30)).detach()),
        "generalized_idempotency": float(torch.linalg.matrix_norm(state.matrix @ state.matrix - state.matrix).detach()),
        "sea_particle_occupation": float(sea_n.detach()),
        "branch_overlap": float(_branch_overlap(branch, state).detach()),
        "fermion_parity": state.fermion_parity,
        "candidate_reference_frobenius": None,
    }
    if ref_state is not None:
        result["candidate_reference_frobenius"] = float(torch.linalg.matrix_norm(state.matrix - ref_state.matrix).detach())
        result["candidate_reference_rho_frobenius"] = float(torch.linalg.matrix_norm(state.rho - ref_state.rho).detach())
        ref_chi = ref_state.chi.reshape(-1)
        candidate_chi = chi.reshape(-1)
        phase_overlap = torch.vdot(ref_chi, candidate_chi)
        phase = (torch.angle(phase_overlap) if float(torch.abs(phase_overlap).detach()) > 1.0e-14
                 else phase_overlap.real.new_zeros(()))
        aligned_chi = chi * torch.exp(-1j * phase)
        aligned_state = generalized_density_matrix(state.rho, aligned_chi)
        result["candidate_reference_gauge_aligned_frobenius"] = float(
            torch.linalg.matrix_norm(aligned_state - ref_state.matrix).detach()
        )
    return result


def optimize_connected_candidate(model: FullDiracOddModel, *, maxiter: int = 220,
                                  seed: int = 1, pairing_seed: float = .85) -> dict[str, Any]:
    """Find the paired no-sea saddle by energy continuation and residual polish.

    A paired energy warm start enforces the target number and follows only a
    nearby projector with small sea support.  It is followed by an absolute,
    full-space commutator solve.  Neither stage calls an eigensolver.
    """
    d, device = model.dimension, model.h0.device
    occupations = torch.zeros(d, dtype=torch.float64, device=device)
    occupations[:model.target_number] = 1.0
    branch = no_sea_branch_reference(model.h0.diagonal().real, occupations,
                                      dirac_cutoff=model.dirac_cutoff,
                                      chemical_potential=1.0)
    a_basis, b_basis = _coordinate_basis(d, device)
    rng_devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                   if device.type == "cuda" else [])
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(int(seed))
        network = CanonicalCoordinateNetwork(d, device)
    size = sum(parameter.numel() for parameter in network.parameters())
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        network.offset.copy_(torch.as_tensor(rng.normal(0., .005, network.offset.numel()),
                                             dtype=torch.float64, device=device))
        # Anomalous B rotations across the Fermi surface have zero tangent at
        # a Slater projector.  Seed occupied-occupied and empty-empty pair
        # directions instead; leave the odd blocked orbital (index 2) alone.
        b0 = a_basis.shape[0]
        # Pair (0,1) depletes an occupied pair.  The second pair is empty-empty
        # for N=3, or another occupied pair for N=5; neither touches the odd
        # blocked orbital.
        network.offset[b0] = pairing_seed
        second_pair_id = 21 if model.target_number == 3 else 14  # (3,4) or (2,3)
        network.offset[b0 + 2 * second_pair_id] = pairing_seed
    parameters = tuple(network.parameters())
    x = parameters_to_vector(parameters).detach().cpu().numpy().copy()
    calls = 0

    def evaluate_chart(coords):
        nonlocal calls
        calls += 1
        vector_to_parameters(torch.as_tensor(coords, dtype=torch.float64, device=device), parameters)
        state = _density_from_coordinates(network(), branch, a_basis, b_basis)
        total, _h, _delta, _normal, _pair = model.energy_fields(state.rho, state.chi)
        number_error = torch.trace(state.rho).real - model.target_number
        sea_number = torch.sum(model.sea_mask.to(torch.float64) * state.rho.diagonal().real)
        return state, total, number_error, sea_number

    with torch.no_grad():
        initial_state, *_ = evaluate_chart(x)
        if float(torch.linalg.matrix_norm(initial_state.chi)) < 1.0e-4:
            raise RuntimeError("paired seed produced a numerically zero anomalous tensor")

    # Warm start by lowering the valence energy while enforcing exact N.  The
    # 1% sea-support ceiling is a continuation trust bound, not an energy term;
    # it keeps the seed in the valence branch while the saddle is approached.
    sea_support_ceiling = 0.01
    adjacent_overlap_floor = 0.50

    def energy_value_gradient(coords):
        state, total, number_error, _sea = evaluate_chart(coords)
        grad = torch.autograd.grad(total, parameters)[0:]
        flat = parameters_to_vector(grad).detach().cpu().numpy()
        return float(total.detach()), flat.astype(np.float64)

    def number_value_gradient(coords):
        _state, _total, number_error, _sea = evaluate_chart(coords)
        grad = torch.autograd.grad(number_error, parameters)
        flat = parameters_to_vector(grad).detach().cpu().numpy()
        return float(number_error.detach()), flat.astype(np.float64)

    def warm_constraints(coords):
        state, _total, _number_error, sea_number = evaluate_chart(coords)
        adjacent = _projector_overlap(initial_state, state)
        return np.array([float(adjacent.detach()) - adjacent_overlap_floor,
                         sea_support_ceiling - float(sea_number.detach())])

    def warm_constraint_jacobian(coords):
        state, _total, _number_error, sea_number = evaluate_chart(coords)
        adjacent = _projector_overlap(initial_state, state)
        adjacent_grad = parameters_to_vector(torch.autograd.grad(
            adjacent, parameters, retain_graph=True
        )).detach().cpu().numpy()
        sea_grad = parameters_to_vector(torch.autograd.grad(sea_number, parameters)).detach().cpu().numpy()
        return np.stack((adjacent_grad, -sea_grad))

    warm = minimize(
        energy_value_gradient, x, method="SLSQP", jac=True,
        constraints=[
            {"type": "eq", "fun": lambda z: number_value_gradient(z)[0],
             "jac": lambda z: number_value_gradient(z)[1]},
            {"type": "ineq", "fun": warm_constraints, "jac": warm_constraint_jacobian},
        ],
        options={"maxiter": max(150, maxiter // 2), "ftol": 1.0e-11, "disp": False},
    )
    x = warm.x
    with torch.no_grad():
        warm_state, *_ = evaluate_chart(x)

    # Polish the absolute full-space HFB equations.  The seed-to-warm and
    # warm-to-candidate projector overlaps are measured after solving; no
    # fixed bare-HF overlap is used to bias the paired root.
    def stationarity_objective(coords):
        nonlocal calls
        calls += 1
        vector_to_parameters(torch.as_tensor(coords[:-1], dtype=torch.float64, device=device), parameters)
        network.zero_grad(set_to_none=True)
        state = _density_from_coordinates(network(), branch, a_basis, b_basis)
        _total, h, delta, _normal, _pair = model.energy_fields(state.rho, state.chi)
        mu = torch.tensor(float(coords[-1]), dtype=torch.float64, device=device, requires_grad=True)
        hbdg = nambu_hamiltonian(h, delta, chemical_potential=mu)
        comm = hbdg @ state.matrix - state.matrix @ hbdg
        number_error = torch.trace(state.rho).real - model.target_number
        sea_number = torch.sum(model.sea_mask.to(torch.float64) * state.rho.diagonal().real)
        objective = (
            comm.abs().square().sum()
            + 2.0e4 * number_error.square()
            + 1.0e4 * torch.relu(sea_number - 0.1).square()
        )
        objective.backward()
        grad = parameters_to_vector([p.grad for p in parameters]).detach().cpu().numpy()
        return float(objective.detach()), np.concatenate((grad, [float(mu.grad.detach())]))

    x = np.concatenate((x, np.array([1.18], dtype=np.float64)))
    result = minimize(stationarity_objective, x, method="L-BFGS-B", jac=True,
                      options={"maxiter": maxiter, "ftol": 2.0e-15,
                               "gtol": 1.0e-9, "maxls": 40, "maxcor": 20})
    x = result.x
    with torch.no_grad():
        vector_to_parameters(torch.as_tensor(x[:-1], dtype=torch.float64, device=device), parameters)
        state = _density_from_coordinates(network(), branch, a_basis, b_basis)
        total, _, _, _, _ = model.energy_fields(state.rho, state.chi)
        metrics = _metrics(model, state.rho, state.chi, state, branch, total,
                           chemical_potential=float(x[-1]))
    metrics.update({"optimizer_success": bool(result.success), "optimizer_message": str(result.message),
                    "optimizer_iterations": int(result.nit), "objective_evaluations": calls,
                    "training_eigensolves": 0, "target_number": model.target_number,
                    "coordinate_network": "trainable 4-8-SiLU-1 per-coordinate MLP plus residual canonical coordinates",
                    "coordinate_network_parameters": size,
                    "energy_warmup_iterations": int(warm.nit),
                    "energy_warmup_success": bool(warm.success),
                    "energy_warmup_message": str(warm.message),
                    "chemical_potential_mev": float(x[-1]),
                    "sea_support_trust_ceiling": sea_support_ceiling,
                    "adjacent_overlap_floor": adjacent_overlap_floor,
                    "seed_to_warm_projector_overlap": float(_projector_overlap(initial_state, warm_state).detach()),
                    "warm_to_candidate_projector_overlap": float(_projector_overlap(warm_state, state).detach()),
                    "paired_seed_amplitude": float(pairing_seed)})
    return {"metrics": metrics, "state": state, "branch": branch, "coordinates": x[:-1]}


def _select_continued_full_bdg(hbdg: torch.Tensor, frame: torch.Tensor,
                               fermion_parity: int = 0):
    """Independent full diagonalization; select PH partners by continuation overlap."""
    vals, vecs = torch.linalg.eigh(hbdg)
    d = hbdg.shape[0] // 2
    pos = torch.nonzero(vals > 1e-10, as_tuple=False).flatten()
    neg = torch.nonzero(vals < -1e-10, as_tuple=False).flatten()
    if pos.numel() != d or neg.numel() != d:
        raise RuntimeError("full BdG reference has zero/malformed particle-hole branches")
    unused = set(int(i) for i in neg.tolist())
    chosen, margins = [], []
    for ip in pos.tolist():
        ph = particle_hole_conjugate(vecs[:, ip])
        im = max(unused, key=lambda i: float(torch.abs(torch.vdot(vecs[:, i], ph))))
        unused.remove(im)
        if abs(float(vals[im] + vals[ip])) > 2e-8 * max(1., float(vals.abs().max())):
            raise RuntimeError("full BdG eigenvalues failed ±E pairing")
        scores = [torch.linalg.vector_norm(frame.mH @ vecs[:, i]).square() for i in (im, ip)]
        margin = float(torch.abs(scores[0] - scores[1]))
        if margin < 1e-8:
            raise RuntimeError("reference branch continuation became ambiguous")
        take = im if float(scores[0]) > float(scores[1]) else ip
        chosen.append(vecs[:, take]); margins.append(margin)
    frame_new = torch.stack(chosen, dim=1)
    r = frame_new @ frame_new.mH
    rho, chi = r[:d, :d], r[:d, d:]
    state = GeneralizedDensity(rho, chi, generalized_density_matrix(rho, chi), fermion_parity)
    return state, frame_new, min(margins)


def solve_full_bdg_reference(model: FullDiracOddModel, *, maxiter: int = 500,
                             mixing: float = .35) -> dict[str, Any]:
    """Self-consistent independent full-BdG eigensolver reference.

    A scalar chemical-potential bracket is tuned against the branch-selected
    particle number.  The iteration tracks occupied Nambu subspaces from the
    no-sea odd frame, retaining both positive- and negative-energy Dirac
    expansion states in the complete anomalous tensor.
    """
    d, device = model.dimension, model.h0.device
    occ = torch.zeros(d, dtype=torch.float64, device=device); occ[:model.target_number] = 1.
    branch = no_sea_branch_reference(model.h0.diagonal().real, occ,
                                      dirac_cutoff=model.dirac_cutoff,
                                      chemical_potential=1.0)
    state, frame = _paired_continuation_seed(branch, model)
    levels = torch.diagonal(model.h0).real[:6]
    mu = 0.5 * float(levels[model.target_number - 1] + levels[model.target_number])
    final_margin = 0.0
    for iteration in range(maxiter):
        with torch.no_grad():
            _, h, delta, _, _ = model.energy_fields(state.rho, state.chi)
            hbdg = nambu_hamiltonian(h, delta, chemical_potential=mu)
            next_state, next_frame, margin = _select_continued_full_bdg(
                hbdg, frame, fermion_parity=branch.density.fermion_parity
            )
            n = float(torch.trace(next_state.rho).real)
            dr = float(torch.linalg.matrix_norm(next_state.matrix-state.matrix))
            # Damp frame/density while retaining a pure selected projector via
            # updating only the reference chart on alternate iteration steps.
            if abs(n-model.target_number) > 1e-5:
                lo = float(levels[model.target_number - 1]) - 1.0
                hi = float(levels[model.target_number]) + 2.5
                mu = max(lo, min(hi, mu + .035*(model.target_number-n)))
            if dr < 2e-9 and abs(n-model.target_number) < 2e-6:
                state, frame, final_margin = next_state, next_frame, margin
                break
            # Continuation follows each selected eigenframe exactly; a weak
            # density mixing is used only to stabilize fields, followed by a
            # fresh projector on the next diagonalization.
            state, frame, final_margin = next_state, next_frame, margin
    with torch.no_grad():
        total, _h, _delta, _, _ = model.energy_fields(state.rho, state.chi)
        metrics = _metrics(model, state.rho, state.chi, state, branch, total,
                           chemical_potential=mu)
        metrics.update({"minimum_branch_overlap_margin": final_margin,
                        "reference_eigensolves": iteration + 1,
                        "target_number": model.target_number})
    return {"state": state, "branch": branch, "metrics": metrics}


def run_branch_experiment(target_number: int = 3, *, candidate_maxiter: int = 900,
                          reference_maxiter: int = 500) -> dict[str, Any]:
    """Run three NN seeds, an independent full-BdG reference, and HF control."""
    model = make_full_dirac_model(target_number)
    candidate_runs = [optimize_connected_candidate(model, maxiter=candidate_maxiter,
                                                   pairing_seed=seed, seed=i+1)
                      for i, seed in enumerate((.35, .6, .85))]
    reference = solve_full_bdg_reference(model, maxiter=reference_maxiter)
    for run in candidate_runs:
        comparison = _metrics(
            model, run["state"].rho, run["state"].chi, run["state"], run["branch"],
            torch.tensor(run["metrics"]["energy_mev"], dtype=torch.float64, device=model.h0.device),
            reference["state"], chemical_potential=run["metrics"]["chemical_potential_mev"],
        )
        for key, value in comparison.items():
            if key.startswith("candidate_reference_"):
                run["metrics"][key] = value
    # Select the lowest-valence-energy stationary solution that retains odd
    # parity, small no-sea support, and a paired continuation from its seed.
    # The branch-overlap with the bare HF projector is diagnostic only.
    eligible = [run for run in candidate_runs
                if run["metrics"]["particle_number_error"] < 1.e-5
                and run["metrics"]["fermion_parity"] == -1
                and run["metrics"]["sea_particle_occupation"] < .01
                and run["metrics"]["commutator_rms_mev"] < 1.e-4
                and run["metrics"]["warm_to_candidate_projector_overlap"] > .90
                and run["metrics"]["pairing_tensor_norm"] > 1.e-3]
    selected = min(eligible, key=lambda run: run["metrics"]["energy_mev"]) if eligible else None
    lowest_index = min(range(len(candidate_runs)), key=lambda i: candidate_runs[i]["metrics"]["energy_mev"])
    lowest_trial = candidate_runs[lowest_index]
    no_pair = solve_full_bdg_reference(make_full_dirac_model(target_number, pairing_enabled=False),
                                       maxiter=min(reference_maxiter, 30))
    return {"model": {"dirac_dimension": model.dimension, "positive_dirac_states": int((~model.sea_mask).sum()),
                       "negative_dirac_states": int(model.sea_mask.sum()), "target_number": target_number,
                       "pairing_force": "unscaled Gogny D1S two-Gaussian nn", "density_dependent_normal_functional": True,
                       "normal_ordering_status": "prototype convention; full pure-R/source mapping unresolved",
                       "training_eigensolves": 0,
                       "reference_eigensolves": reference["metrics"]["reference_eigensolves"]},
            "candidate_selection": {"status": "converged paired saddle" if selected else
                                    "no seed passed paired-stationarity gate",
                                    "selected_seed": selected["metrics"].get("paired_seed_amplitude") if selected else None,
                                    "selected_energy_mev": selected["metrics"]["energy_mev"] if selected else None,
                                    "lowest_energy_trial_seed_index": lowest_index,
                                    "eligible_seed_count": len(eligible),
                                    "selection_rule": "minimum valence energy among number-, parity-, sea-support-, and absolute-commutator-converged paired candidates"},
            "candidate": selected["metrics"] if selected else None,
            "lowest_energy_candidate_trial": lowest_trial["metrics"],
            "candidate_seed_trials": [run["metrics"] for run in candidate_runs],
            "full_bdg_reference": reference["metrics"],
            "pairing_off_control": no_pair["metrics"]}
