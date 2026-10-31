"""Spherical canonical RHFB backend: RHF limit, field identities, numbers, no sea."""
import math

import torch

from dpl_rhf.backends.base import NucleusCase
from dpl_rhf.models.canonical_hfb_net import CanonicalHFBNet
from dpl_rhf.models.density_matrix_net import DensityMatrixNet
from dpl_rhf.rhf.backend import DensityMatrixRHFBackend
from dpl_rhf.rhf.hfb_backend import SphericalRHFBBackend
from dpl_rhf.training.sea_projector import positive_energy_projector

def _device():
    # Under the protected launcher with --gpu-uuid the selected GPU is cuda:0;
    # otherwise the tests run on the CPU.
    import os
    if os.environ.get("RHFB_CUDA_UUID") and torch.cuda.is_available():
        from dpl_rhf.rhfb.cuda_resources import configure_cuda
        configure_cuda("cuda:0", budget_mib=int(os.environ["RHFB_CUDA_BUDGET_MIB"]))
        return torch.device("cuda:0")
    return torch.device("cpu")


CPU = _device()


def _network(backend, logits=None):
    return CanonicalHFBNet(torch.as_tensor(backend.r, device=backend.device), backend.block_specs,
                           logits if logits is not None else backend.initial_logits()).to(backend.device)


def test_unpaired_sharp_occupations_reproduce_the_rhf_backend():
    case = NucleusCase("PKA1", 8, 8, 16)
    rhf = DensityMatrixRHFBackend(case, device=CPU)
    hfb = SphericalRHFBBackend(case, device=CPU, pairing=False, extra_capacity=8)
    rhf_blocks = DensityMatrixNet(torch.as_tensor(rhf.r, device=CPU), rhf.block_specs).to(CPU)()
    reference = rhf.evaluate(rhf_blocks).energy_total_no_com
    # Occupied canonical columns come first, so QR leaves them identical.
    logits = [torch.where(spec.occupations > 0.5, 40.0, -40.0).to(torch.float64).to(CPU)
              for spec in hfb.block_specs]
    network = _network(hfb, logits)
    result = hfb.evaluate_hfb(network(), network.logits())
    assert result.particle_numbers == {"n": 8.0, "p": 8.0} or (
        abs(result.particle_numbers["n"] - 8) < 1e-10
        and abs(result.particle_numbers["p"] - 8) < 1e-10)
    assert math.isclose(result.energy_total, reference, rel_tol=0, abs_tol=1e-8)
    assert result.diagnostics["pairing_energy_n_mev"] == 0.0


def test_logit_gradient_equals_canonical_field_matrix_elements():
    case = NucleusCase("PKA1", 8, 10, 18)
    backend = SphericalRHFBBackend(case, device=CPU, extra_capacity=8)
    logits = [l.clone().requires_grad_(True) for l in backend.initial_logits()]
    network = _network(backend)
    # Freeze the Fermi shift so the gradient is the bare dE/dt_n.
    shift = {"n": 0.3, "p": -0.2}

    def fixed_state(values):
        state = SphericalRHFBBackend.occupation_state(backend, [v.detach() for v in values])
        for i, spec in enumerate(backend.block_specs):
            name = "n" if spec.species_sign > 0 else "p"
            o, p, do, dp, s = backend._occupation_terms(values[i] + shift[name], i)
            state.occupations[i], state.pair_amplitudes[i] = o, p
            state.d_occupations[i], state.d_pairs[i], state.v_squared[i] = do, dp, s
        state.shifts.update(shift)
        return state

    backend.occupation_state = fixed_state
    result = backend.evaluate_hfb(network(), logits)
    assert abs(result.diagnostics["pairing_energy_n_mev"]) > 1e-3
    gradients = torch.autograd.grad(result.energy_total_tensor, logits)
    table = {(row["species"], row["kappa"], row["principal"]): row for row in result.canonical}
    for i, spec in enumerate(backend.block_specs):
        name = "n" if spec.species_sign > 0 else "p"
        t = logits[i].detach() + shift[name]
        _, _, do, dp, _ = backend._occupation_terms(t, i)
        rows = [table[(name, spec.kappa, principal)] for principal in spec.principals]
        h = torch.tensor([row["h_nn_mev"] for row in rows], device=CPU)
        delta = torch.tensor([row["delta_nn_mev"] for row in rows], device=CPU)
        expected = spec.degeneracy * (h * do + delta * dp)
        torch.testing.assert_close(gradients[i], expected, rtol=2e-6, atol=2e-6)


def test_equal_filling_blocking_gives_exact_odd_number():
    case = NucleusCase("PKA1", 8, 9, 17)
    backend = SphericalRHFBBackend(case, device=CPU, extra_capacity=8, blocking={"n": -3})
    state = backend.occupation_state(backend.initial_logits())
    total = sum(spec.degeneracy * float(o.sum()) for spec, o in
                zip(backend.block_specs, state.occupations) if spec.species_sign > 0)
    assert abs(total - 9.0) < 1e-12
    blocked = [i for i, c in enumerate(backend.blocked_columns) if c is not None]
    assert len(blocked) == 1 and backend.block_specs[blocked[0]].kappa == -3
    o, s = state.occupations[blocked[0]][0], state.v_squared[blocked[0]][0]
    # Reference rule: [(2j) v^2 + u^2]/(2j+1) = (1 + (2j-1) v^2)/(2j+1), 2j+1 = 6.
    assert math.isclose(float(o), (1.0 + 4.0 * float(s)) / 6.0, rel_tol=1e-14)


def test_no_sea_projector_is_idempotent_and_commutes_with_the_dirac_block():
    case = NucleusCase("PKA1", 8, 8, 16)
    backend = SphericalRHFBBackend(case, device=CPU, pairing=False, extra_capacity=4)
    h = next(iter(backend.kinetic_hamiltonians().values()))
    result = positive_energy_projector(h, -backend.params.mass_n)
    p = result.projector
    torch.testing.assert_close(p @ p, p, atol=1e-10, rtol=0)
    commutator = torch.linalg.matrix_norm(p @ h - h @ p) / torch.linalg.matrix_norm(h)
    assert float(commutator) < 1e-10
    assert result.sea_dimension == h.shape[0] // 2
