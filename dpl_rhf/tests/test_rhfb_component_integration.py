import torch

from dpl_rhf.rhfb.backend import evaluate_components
from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.manifold import slater_reference, transform_reference
from dpl_rhf.rhfb.operators import DensityFactors


def test_physical_basis_pairing_currents_and_backward_are_integrated():
    basis = AngularRadialBasis(radius_fm=8., spacing_fm=1., kappa_max=1)
    q, _ = basis.build_initial_q(1, nucleus_a=2, minimum=2, empty_states=0)
    q = q.detach().requires_grad_(True)
    vacuum = slater_reference(torch.zeros(2, dtype=torch.float64))
    a = torch.zeros(2, 2, dtype=torch.complex128)
    b = torch.tensor([[0, .35], [-.35, 0]], dtype=torch.complex128, requires_grad=True)
    pair = transform_reference(vacuum, a, b)
    odd = slater_reference(torch.zeros(2, dtype=torch.float64), blocked_index=0)
    states = {"n": DensityFactors(q, odd.rho, odd.chi),
              "p": DensityFactors(q, pair.rho, pair.chi)}
    result = evaluate_components(basis, states, chunk_size=23)
    assert result.status == "components-only"
    assert set(result.species_currents) == {"n", "p"}
    torch.testing.assert_close(result.currents["j_baryon"],
                               result.species_currents["n"]["baryon_current"]
                               + result.species_currents["p"]["baryon_current"])
    assert torch.allclose(result.particle_numbers["n"], torch.tensor(1., dtype=torch.float64))
    assert result.pairing_gaps["p"].abs().max() > 1e-5
    weights = torch.as_tensor(basis.weights, dtype=torch.float64)
    integral = (result.currents["j_baryon"][:, 0] * weights).sum()
    assert torch.allclose(integral, sum(result.particle_numbers.values()), atol=1e-12)
    assert (result.currents["rho_b_invariant"] >= 0).all()
    (result.kinetic_energy + result.pairing_energy).backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert b.grad is not None and torch.isfinite(b.grad).all()
    assert b.grad.abs().max() > 1e-5
