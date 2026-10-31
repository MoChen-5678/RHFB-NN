"""Independent Fock-space checks for center_of_mass.py's Wick contractions."""
from __future__ import annotations

import numpy as np
import pytest

from dpl_rhf.rhfb.center_of_mass import evaluate_center_of_mass_correction


def _annihilation_matrix(n_modes: int, mode: int) -> np.ndarray:
    dimension = 1 << n_modes
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    for state in range(dimension):
        if (state >> mode) & 1:
            sign = -1 if (state & ((1 << mode) - 1)).bit_count() % 2 else 1
            result[state ^ (1 << mode), state] = sign
    return result


def _creation_matrix(n_modes: int, mode: int) -> np.ndarray:
    return _annihilation_matrix(n_modes, mode).conj().T


def _densities_from_state(state: np.ndarray, annihilators: list[np.ndarray]):
    m = len(annihilators)
    rho = np.empty((m, m), dtype=np.complex128)
    chi = np.empty((m, m), dtype=np.complex128)
    creators = [a.conj().T for a in annihilators]
    for a in range(m):
        for b in range(m):
            rho[a, b] = np.vdot(state, creators[b] @ annihilators[a] @ state)
            chi[a, b] = np.vdot(state, annihilators[b] @ annihilators[a] @ state)
    return rho, chi


def _momentum_operator(momentum: np.ndarray, annihilators: list[np.ndarray]) -> np.ndarray:
    result = np.zeros_like(annihilators[0])
    creators = [a.conj().T for a in annihilators]
    for a in range(len(annihilators)):
        for b in range(len(annihilators)):
            result += momentum[a, b] * creators[a] @ annihilators[b]
    return result


def _make_state(n_modes: int, occupied_superpositions: list[tuple[tuple[int, complex], ...]]):
    state = np.zeros(1 << n_modes, dtype=np.complex128)
    state[0] = 1.0
    creators = [_creation_matrix(n_modes, i) for i in range(n_modes)]
    for orbital in occupied_superpositions:
        create = sum((amp * creators[index] for index, amp in orbital),
                     np.zeros_like(creators[0]))
        state = create @ state
    norm = np.linalg.norm(state)
    if norm == 0:
        raise AssertionError("test Fock state construction produced the zero vector")
    return state / norm


def _p_stack(p: np.ndarray) -> np.ndarray:
    return np.stack((p, np.zeros_like(p), np.zeros_like(p)))


def test_bcs_anomalous_contraction_matches_explicit_fock_operator():
    n_modes = 4
    annihilators = [_annihilation_matrix(n_modes, i) for i in range(n_modes)]
    # |BCS> = u|0> + v c†_0 c†_1|0>; proton modes stay empty.
    u, v = np.sqrt(0.63), np.sqrt(0.37)
    vacuum = np.zeros(1 << n_modes, dtype=np.complex128)
    vacuum[0] = u
    pair = annihilators[0].conj().T @ annihilators[1].conj().T
    vacuum += v * (pair @ np.eye(1 << n_modes, dtype=np.complex128)[:, 0])
    rho_n, chi_n = _densities_from_state(vacuum, annihilators[:2])
    rho_p, chi_p = _densities_from_state(vacuum, annihilators[2:])

    p = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    p2 = p @ p
    p_total = _momentum_operator(np.block([[p, np.zeros_like(p)],
                                           [np.zeros_like(p), np.zeros_like(p)]]), annihilators)
    exact = float(np.vdot(vacuum, p_total @ p_total @ vacuum).real)
    result = evaluate_center_of_mass_correction(
        {"n": (rho_n, chi_n), "p": (rho_p, chi_p)}, _p_stack(p), p2,
        mass_number=2, nucleon_mass_mev=939.0, momentum_units="MeV")
    assert np.linalg.norm(chi_n) > 0.0
    assert result["momentum_squared_expectation"] == pytest.approx(exact, abs=1e-12)
    assert result["anomalous_wick_signed"] != 0.0


def test_blocked_odd_hfb_vacuum_includes_unpaired_orbital():
    n_modes = 6
    annihilators = [_annihilation_matrix(n_modes, i) for i in range(n_modes)]
    u, v = np.sqrt(0.7), np.sqrt(0.3)
    state = np.zeros(1 << n_modes, dtype=np.complex128)
    state[0] = u
    state += v * (annihilators[0].conj().T @ annihilators[1].conj().T
                  @ np.eye(1 << n_modes, dtype=np.complex128)[:, 0])
    state = annihilators[2].conj().T @ state  # blocked odd neutron orbital
    state /= np.linalg.norm(state)
    rho_n, chi_n = _densities_from_state(state, annihilators[:3])
    rho_p, chi_p = _densities_from_state(state, annihilators[3:])

    p_n = np.zeros((3, 3), dtype=np.complex128)
    p_n[:2, :2] = [[0.0, -1.0j], [1.0j, 0.0]]
    p_n[2, 2] = 0.4
    p_p = np.zeros_like(p_n)
    p_stack = np.stack((p_n, np.zeros_like(p_n), np.zeros_like(p_n)))
    p_total = _momentum_operator(np.block([[p_n, np.zeros_like(p_n)],
                                           [np.zeros_like(p_n), p_p]]), annihilators)
    exact = float(np.vdot(state, p_total @ p_total @ state).real)
    result = evaluate_center_of_mass_correction(
        {"n": (rho_n, chi_n), "p": (rho_p, chi_p)}, p_stack,
        p_n.conj().T @ p_n,
        mass_number=3, nucleon_mass_mev=939.0, momentum_units="MeV")
    assert np.trace(rho_n).real == pytest.approx(1.0 + 2.0 * v**2)
    assert np.linalg.norm(chi_n) > 0.0
    assert result["momentum_squared_expectation"] == pytest.approx(exact, abs=1e-12)


def test_time_odd_complex_slater_state_keeps_neutron_proton_mean_cross_term():
    n_modes = 4
    annihilators = [_annihilation_matrix(n_modes, i) for i in range(n_modes)]
    orbital = ((0, 1.0 / np.sqrt(2.0)), (1, 1.0j / np.sqrt(2.0)))
    state = _make_state(n_modes, [orbital, ((2, 1.0 / np.sqrt(2.0)),
                                            (3, 1.0j / np.sqrt(2.0)))])
    rho_n, chi_n = _densities_from_state(state, annihilators[:2])
    rho_p, chi_p = _densities_from_state(state, annihilators[2:])
    p = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    p_full = np.block([[p, np.zeros_like(p)], [np.zeros_like(p), p]])
    p_total = _momentum_operator(p_full, annihilators)
    exact = float(np.vdot(state, p_total @ p_total @ state).real)
    result = evaluate_center_of_mass_correction(
        {"n": (rho_n, chi_n), "p": (rho_p, chi_p)}, _p_stack(p), p @ p,
        mass_number=2, nucleon_mass_mev=939.0, momentum_units="MeV")
    assert np.allclose(chi_n, 0.0) and np.allclose(chi_p, 0.0)
    component = result["cartesian_components"][0]
    assert component["mean_momentum_by_species"]["n"]["real"] == pytest.approx(1.0)
    assert component["mean_momentum_by_species"]["p"]["real"] == pytest.approx(1.0)
    assert component["mean_momentum_square_total"] == pytest.approx(4.0)
    assert result["momentum_squared_expectation"] == pytest.approx(exact, abs=1e-12)


def test_independent_p_squared_matrix_is_used_as_supplied():
    rho = np.diag([1.0, 0.0]).astype(np.complex128)
    chi = np.zeros((2, 2), dtype=np.complex128)
    p = np.zeros((3, 2, 2), dtype=np.complex128)
    independent_p2 = np.diag([7.0, 11.0]).astype(np.complex128)
    result = evaluate_center_of_mass_correction(
        {"n": (rho, chi), "p": (np.zeros_like(rho), chi)}, p, independent_p2,
        mass_number=1, nucleon_mass_mev=939.0, momentum_units="MeV")
    assert result["one_body_p2"] == pytest.approx(7.0)
    assert result["momentum_squared_expectation"] == pytest.approx(7.0)
    assert result["energy_correction_mev"] == pytest.approx(-7.0 / 1878.0)


def test_species_projected_momentum_operators_allow_different_active_dimensions():
    # Neutron and proton frames need not share the same active-space dimension
    # or projected operators. Keep p and p^2 independently supplied per species.
    rho_n = np.diag([1.0, 0.0]).astype(np.complex128)
    rho_p = np.diag([0.0, 1.0, 0.0]).astype(np.complex128)
    chi_n = np.zeros((2, 2), dtype=np.complex128)
    chi_p = np.zeros((3, 3), dtype=np.complex128)
    p_n = np.zeros((3, 2, 2), dtype=np.complex128)
    p_p = np.zeros((3, 3, 3), dtype=np.complex128)
    p_n[0, 0, 0] = 1.0
    p_p[0, 1, 1] = 2.0
    p2_n = np.diag([4.0, 0.0]).astype(np.complex128)
    p2_p = np.diag([0.0, 9.0, 0.0]).astype(np.complex128)

    result = evaluate_center_of_mass_correction(
        {"n": (rho_n, chi_n), "p": (rho_p, chi_p)},
        {"n": p_n, "p": p_p}, {"n": p2_n, "p": p2_p},
        mass_number=2, nucleon_mass_mev=1.0, momentum_units="MeV")

    assert result["one_body_p2_by_species"] == {"n": 4.0, "p": 9.0}
    assert result["one_body_p2"] == pytest.approx(13.0)
    assert result["mean_momentum_square_total"] == pytest.approx(9.0)
    assert result["normal_exchange_signed"] == pytest.approx(-5.0)
    assert result["momentum_squared_expectation"] == pytest.approx(17.0)
    component = result["cartesian_components"][0]
    assert component["mean_momentum_by_species"]["n"]["real"] == pytest.approx(1.0)
    assert component["mean_momentum_by_species"]["p"]["real"] == pytest.approx(2.0)


def test_inverse_femtometer_units_convert_with_hbar_c_squared():
    rho = np.diag([1.0, 0.0]).astype(np.complex128)
    chi = np.zeros((2, 2), dtype=np.complex128)
    p = np.zeros((3, 2, 2), dtype=np.complex128)
    p2 = np.diag([1.0, 3.0]).astype(np.complex128)
    result = evaluate_center_of_mass_correction(
        {"n": (rho, chi), "p": (np.zeros_like(rho), chi)}, p, p2,
        mass_number=1, nucleon_mass_mev=2.0, momentum_units="fm^-1",
        hbar_c_mev_fm=10.0)
    assert result["momentum_squared_units"] == "fm^-2"
    assert result["energy_correction_mev"] == pytest.approx(-25.0)
