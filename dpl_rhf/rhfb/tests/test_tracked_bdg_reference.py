import torch

from dpl_rhf.rhfb.branch import nambu_hamiltonian
from dpl_rhf.rhfb.parity_diagnostic import number_parity_from_density
from dpl_rhf.rhfb.tracked_bdg_reference import (
    _monotone_orientation, reference_spectral_counter, select_odd_bdg_projector,
)


def _parity(projector):
    d = projector.shape[0] // 2
    return number_parity_from_density(projector[:d, :d].numpy(),
                                      projector[:d, d:].numpy())


def test_even_paired_vacuum_and_designated_block_stay_odd_under_mu_continuation():
    # Two paired orbitals: blocking the first leaves the second pair anomalous.
    h = torch.diag(torch.tensor([.31, .67, .45, .45], dtype=torch.float64)).to(torch.complex128)
    delta = torch.zeros((4, 4), dtype=torch.complex128)
    delta[0, 1], delta[1, 0] = .19, -.19
    delta[2, 3], delta[3, 2] = .25, -.25
    tag = torch.tensor([1, 0, 0, 0], dtype=torch.complex128)
    states = []
    vacuum_numbers = []
    for mu in (.12, .121):
        bdg = nambu_hamiltonian(h, delta, chemical_potential=mu)
        energies, modes = torch.linalg.eigh(bdg)
        vacuum = modes[:, energies < 0]
        vacuum_projector = vacuum @ vacuum.mH
        vacuum_numbers.append(float(torch.trace(vacuum_projector[:4, :4]).real))
        xi1, xi2 = (.31 + .67) / 2 - mu, .45 - mu
        expected_number = (1 - xi1 / (xi1**2 + .19**2)**.5
                           + 1 - xi2 / (xi2**2 + .25**2)**.5)
        assert abs(vacuum_numbers[-1] - expected_number) < 1e-10
        state = select_odd_bdg_projector(
            bdg, tag,
            previous_blocked_qp=None if not states else states[-1][4],
            previous_quasiframe=None if not states else states[-1][0],
        )
        states.append(state)
        _frame, projector, _rho, _chi, _blocked, metrics = state
        assert _parity(projector)["number_parity"] == -1
        assert torch.linalg.matrix_norm(_chi) > 1e-5
        assert metrics["vacuum_parity"]["number_parity"] == 1
        assert metrics["vacuum_positive_branches_occupied"] == 0
        assert metrics["even_reference_compensation_pair"] is None
        assert metrics["computed_selected_parity"]["number_parity"] == -1
    assert states[1][5]["blocked_qp_overlap"] > .99
    assert vacuum_numbers[1] > vacuum_numbers[0]


def test_odd_time_odd_vacuum_is_compensated_before_designated_blocking():
    # A time-odd splitting of a reduced partner pair leaves one level below μ,
    # so the all-negative Nambu projector is odd. The lowest-energy qp first
    # defines the even reference; block orbital 1 only afterward, so the final
    # pure generalized density remains odd.
    h = torch.diag(torch.tensor([-.20, .70], dtype=torch.float64)).to(torch.complex128)
    delta = torch.zeros((2, 2), dtype=torch.complex128)
    tag = torch.tensor([0, 1], dtype=torch.complex128)
    _frame, projector, _rho, _chi, _blocked, metrics = select_odd_bdg_projector(
        nambu_hamiltonian(h, delta, chemical_potential=0.), tag,
    )
    assert metrics["vacuum_parity"]["number_parity"] == -1
    assert metrics["even_reference_compensation_pair"] == 0
    assert metrics["even_reference_positive_branches_occupied"] == 1
    assert metrics["computed_selected_parity"]["number_parity"] == -1
    assert _parity(projector)["number_parity"] == -1


def test_number_bracket_validation_accepts_either_monotone_orientation_only():
    assert _monotone_orientation([(0., 3.2), (1., 2.8), (2., 2.1)]) == -1
    assert _monotone_orientation([(0., 2.1), (1., 2.8), (2., 3.2)]) == 1
    try:
        _monotone_orientation([(0., 2.1), (1., 3.2), (2., 2.8)])
    except RuntimeError as exc:
        assert "not monotone" in str(exc)
    else:
        raise AssertionError("a branch reversal was accepted as a valid μ bracket")


def test_reference_eigensolver_counter_records_one_probe_call():
    with reference_spectral_counter() as counts:
        torch.linalg.eigh(torch.eye(1, dtype=torch.complex128))
    assert counts == {"torch.linalg.eigh": 1,
                      "torch.linalg.eigvalsh": 0,
                      "torch.linalg.eig": 0}
