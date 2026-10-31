"""Isolated small-space checks for the conditional mixed Nambu branch.

These tests intentionally do not certify an O17/continuum branch.  The
eigensolver is used only as an independent tiny-matrix oracle, outside the
operator implementation.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dpl_rhf.rhfb.bdg_branch_filter import BandBoundSource, BandEnclosureCertificate
from dpl_rhf.rhfb.full_branch_stationarity import nambu_hamiltonian
from dpl_rhf.rhfb.mixed_nucleon_sea_branch import (
    EvidenceScope,
    MixedNambuBandEvidence,
    NoSeaBranchEvidenceError,
    construct_mixed_nucleon_sea_density,
    hamiltonian_sha256,
    mixed_branch_fixed_point_loss,
    MixedNucleonSeaBranchPolicy,
    BlockedModeEvidence,
    block_mixed_nucleon_sea_density,
)


def _cert(*, pmin=1.0, pmax=1.0, smin=20.0, smax=20.0,
          zero=1.0, maxabs=20.0):
    return BandEnclosureCertificate(
        particle_abs_lower=pmin, particle_abs_upper=pmax,
        sea_abs_lower=smin, sea_abs_upper=smax,
        particle_annihilator_energy_sign=1,
        sea_annihilator_energy_sign=-1,
        zero_gap_lower=zero, max_abs_upper=maxabs,
        finite_space_complete=True, hermitian_verified=True,
        particle_hole_verified=True, particle_band_complete=True,
        sea_band_complete=True, no_unclassified_modes=True,
        outside_training_spectral_teacher=True,
        source=BandBoundSource.ANALYTIC_TOY, source_id="analytic-toy-bounds-v1",
    )


def _exact_evidence(h, cert, divider=5.0):
    return MixedNambuBandEvidence(
        divider_mev=divider, certificate=cert,
        scope=EvidenceScope.EXACT_HAMILTONIAN, proof_id="toy-exact-matrix",
        hamiltonian_sha256=hamiltonian_sha256(h),
    )


def _oracle(h, divider):
    """Independent test-only spectral formula; never called by branch code."""
    values, vectors = np.linalg.eigh(h.detach().cpu().numpy())
    weights = 0.5 * (1.0 - np.sign(values) * np.sign(divider**2 - values**2))
    return (vectors * weights) @ vectors.conj().T


def test_delta_zero_matches_independent_oracle_and_operator_action(monkeypatch):
    h0 = torch.diag(torch.tensor([1.0, 20.0], dtype=torch.complex128))
    delta = torch.zeros_like(h0)
    h = nambu_hamiltonian(h0, delta, 0.0)
    cert = _cert()
    evidence = _exact_evidence(h, cert)
    expected = _oracle(h, evidence.divider_mev)

    def no_eigh(*_args, **_kwargs):
        raise AssertionError("training/operator path must not diagonalize")

    monkeypatch.setattr(torch.linalg, "eigh", no_eigh)
    branch = construct_mixed_nucleon_sea_density(
        h, evidence=evidence, quadrature_order=321, log_step=0.2,
    )
    torch.testing.assert_close(
        branch.generalized_density,
        torch.as_tensor(expected, dtype=torch.complex128), atol=2e-7, rtol=2e-7,
    )
    torch.testing.assert_close(
        branch.generalized_density.diagonal().real,
        torch.tensor([0., 1., 1., 0.], dtype=torch.float64),
        atol=1.0e-10, rtol=1.0e-10,
    )
    probe = torch.tensor([1 + 2j, 2 - 1j, -1j, 3 + 0j], dtype=torch.complex128)
    torch.testing.assert_close(branch.operator_action(probe), branch.generalized_density @ probe)
    assert branch.physical_no_sea_certified is False
    assert branch.physical_identity_certified is False


def test_paired_branch_has_full_h_response_in_fixed_point_loss():
    h0 = torch.diag(torch.tensor([1.0, 20.0], dtype=torch.complex128))
    delta = torch.zeros((2, 2), dtype=torch.complex128)
    delta[0, 1] = 0.08
    delta[1, 0] = -0.08
    h = nambu_hamiltonian(h0, delta, 0.0).detach().requires_grad_(True)
    # Weyl bounds around the unpaired ±1 and ±20 levels for this small toy
    # coupling; this is an explicit analytic toy family, not a nuclear bound.
    cert = _cert(pmin=0.8, pmax=1.2, smin=19.8, smax=20.2,
                 zero=0.8, maxabs=20.3)
    evidence = MixedNambuBandEvidence(
        divider_mev=5.0, certificate=cert,
        scope=EvidenceScope.SELFCONSISTENT_FAMILY,
        proof_id="analytic-toy-perturbation-family-v1",
        field_family_id="h0-diagonal-plus-small-antisymmetric-pairing",
    )
    # A distinct candidate guarantees a nonzero residual: this is a loss,
    # not a pre-training acceptance gate.
    candidate = torch.zeros((4, 4), dtype=torch.complex128)
    result = mixed_branch_fixed_point_loss(h, candidate, evidence=evidence,
                                           quadrature_order=321, log_step=0.2)
    assert float(result.loss.detach()) > 0.0
    grad, = torch.autograd.grad(result.loss, h)
    assert torch.isfinite(grad).all()
    assert float(torch.linalg.vector_norm(grad).detach()) > 0.0
    oracle = _oracle(h, evidence.divider_mev)
    torch.testing.assert_close(
        result.branch_density.generalized_density,
        torch.as_tensor(oracle, dtype=torch.complex128), atol=3e-6, rtol=3e-6,
    )


def test_high_positive_continuum_overlap_has_no_legal_divider():
    h0 = torch.diag(torch.tensor([1.0, 25.0, -20.0], dtype=torch.complex128))
    h = nambu_hamiltonian(h0, torch.zeros_like(h0), 0.0)
    cert = _cert(pmin=1.0, pmax=25.0, smin=20.0, smax=25.0,
                 zero=1.0, maxabs=25.0)
    evidence = _exact_evidence(h, cert, divider=22.5)
    with pytest.raises(ValueError, match="strictly between complete particle and sea bands"):
        construct_mixed_nucleon_sea_density(h, evidence=evidence)


def test_missing_evidence_hash_mismatch_and_payload_fingerprint_fail_closed():
    h0 = torch.diag(torch.tensor([1.0, 20.0], dtype=torch.complex128))
    h = nambu_hamiltonian(h0, torch.zeros_like(h0), 0.0)
    with pytest.raises(NoSeaBranchEvidenceError, match="refusing to guess Ec"):
        construct_mixed_nucleon_sea_density(h, evidence=None)
    evidence = _exact_evidence(h, _cert())
    restored = MixedNambuBandEvidence.from_payload(evidence.to_payload())
    assert restored.fingerprint() == evidence.fingerprint()
    h_changed = h.clone()
    h_changed[0, 0] += 0.01
    with pytest.raises(NoSeaBranchEvidenceError, match="different matrix"):
        construct_mixed_nucleon_sea_density(h_changed, evidence=evidence)
    with pytest.raises(ValueError, match="cannot certify Hamiltonian quasiparticle identity"):
        replace(evidence, physical_identity_certified=True).validate_for(h)


def test_trainer_policy_consumes_full_delta_and_actual_differentiable_mu():
    from dpl_rhf.rhfb.operators import DensityFactors

    d = 2
    q = torch.eye(d, dtype=torch.complex128)
    # Two occupied primitive states give a genuine even pure candidate.
    rho = torch.eye(d, dtype=torch.complex128)
    chi = torch.zeros((d, d), dtype=torch.complex128)
    states = {s: DensityFactors(q, rho, chi) for s in ("n", "p")}
    h0 = torch.diag(torch.tensor([1.0, -20.0], dtype=torch.complex128))
    mus = {s: torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
           for s in ("n", "p")}
    h_fields = {s: h0.clone().requires_grad_(True) for s in ("n", "p")}
    delta = torch.zeros_like(h0)
    delta[0, 1] = 0.08
    delta[1, 0] = -0.08

    class DeltaAction:
        def apply(self, vectors):
            return delta @ vectors

    evaluation = SimpleNamespace(
        normal_fields=h_fields,
        delta_actions={s: DeltaAction() for s in ("n", "p")},
    )
    evidence = {}
    for s in ("n", "p"):
        evidence[s] = MixedNambuBandEvidence(
            divider_mev=5.0,
            certificate=_cert(pmin=0.8, pmax=1.2, smin=19.8, smax=20.2,
                              zero=0.8, maxabs=20.3),
            scope=EvidenceScope.SELFCONSISTENT_FAMILY,
            proof_id=f"analytic-even-toy-family-{s}",
            field_family_id="diagonal-h0-small-pairing-family",
        )
    policy = MixedNucleonSeaBranchPolicy(evidence, blocked_species=(),
                                         quadrature_order=321, log_step=0.2)
    loss, diagnostics = policy.evaluate(states, evaluation, mus)
    assert loss.ndim == 0
    assert diagnostics["physical_no_sea_certified"] is False
    assert set(diagnostics["species"]) == {"n", "p"}
    metadata = policy.checkpoint_metadata()
    assert set(metadata["evidence_fingerprints"]) == {"n", "p"}
    assert metadata["physical_identity_certified"] is False
    grad_n_mu, grad_p_mu = torch.autograd.grad(loss, (mus["n"], mus["p"]),
                                                allow_unused=True)
    assert grad_n_mu is not None and grad_p_mu is not None
    assert float(torch.stack((grad_n_mu.abs(), grad_p_mu.abs())).norm()) > 1.0e-12
    assert diagnostics["species"]["n"]["number_parity"] == 1


def test_odd_block_swap_is_pure_ph_symmetric_and_flips_parity():
    # One valence positive mode and one negative-energy sea basis mode.
    h0 = torch.diag(torch.tensor([1.0, -20.0], dtype=torch.complex128))
    h = nambu_hamiltonian(h0, torch.zeros_like(h0), 0.0)
    base = construct_mixed_nucleon_sea_density(
        h, evidence=_exact_evidence(h, _cert()), quadrature_order=321, log_step=0.2,
    )
    exact = torch.as_tensor(_oracle(h, 5.0), dtype=torch.complex128)
    torch.testing.assert_close(base.generalized_density, exact, atol=2e-7, rtol=2e-7)
    assert float(base.rho.diagonal().real.sum()) == pytest.approx(0.0, abs=2e-7)

    mode = torch.tensor([1., 0., 0., 0.], dtype=torch.complex128)
    blocked_witness = BlockedModeEvidence(
        mode=mode, scope=EvidenceScope.EXACT_HAMILTONIAN,
        proof_id="analytic-positive-valence-mode",
        hamiltonian_sha256=hamiltonian_sha256(h),
    )
    odd = block_mixed_nucleon_sea_density(base, blocked_witness)
    odd_oracle = exact + torch.outer(mode, mode.conj()) - torch.outer(
        torch.tensor([0., 0., 1., 0.], dtype=torch.complex128),
        torch.tensor([0., 0., 1., 0.], dtype=torch.complex128).conj(),
    )
    torch.testing.assert_close(odd.generalized_density, odd_oracle, atol=2e-7, rtol=2e-7)
    torch.testing.assert_close(odd.generalized_density @ odd.generalized_density,
                               odd.generalized_density, atol=2e-7, rtol=2e-7)
    assert float(odd.rho.diagonal().real.sum()) == pytest.approx(1.0, abs=2e-7)
    assert odd.construction.diagnostics.purity_error < 2e-7
    assert odd.construction.diagnostics.particle_hole_error < 2e-7
    assert odd.physical_identity_certified is False

    # A sea mode is occupied by the base selector, so it cannot masquerade as
    # the unoccupied valence qp that is toggled in an odd solution.
    sea_positive = torch.tensor([0., 0., 0., 1.], dtype=torch.complex128)
    invalid = BlockedModeEvidence(
        mode=sea_positive, scope=EvidenceScope.EXACT_HAMILTONIAN,
        proof_id="wrong-sea-mode", hamiltonian_sha256=hamiltonian_sha256(h),
    )
    with pytest.raises(NoSeaBranchEvidenceError, match="unoccupied plane"):
        block_mixed_nucleon_sea_density(base, invalid)


def test_odd_policy_requires_explicit_blocked_mode_source():
    h0 = torch.diag(torch.tensor([1.0, -20.0], dtype=torch.complex128))
    h = nambu_hamiltonian(h0, torch.zeros_like(h0), 0.0)
    evidence = {s: _exact_evidence(h, _cert()) for s in ("n", "p")}
    with pytest.raises(ValueError, match="each odd species requires exactly one"):
        MixedNucleonSeaBranchPolicy(evidence, blocked_species=("n",))


def test_policy_rejects_unblocked_selector_with_wrong_base_parity():
    from dpl_rhf.rhfb.operators import DensityFactors

    d = 2
    q = torch.eye(d, dtype=torch.complex128)
    rho_even = torch.eye(d, dtype=torch.complex128)
    chi = torch.zeros_like(rho_even)
    states = {s: DensityFactors(q, rho_even, chi) for s in ("n", "p")}
    # This unblocked selector has one occupied physical particle state, so
    # its Pfaffian parity is odd despite the requested even branch metadata.
    h0 = torch.diag(torch.tensor([1.0, 20.0], dtype=torch.complex128))
    h_fields = {s: h0.clone().requires_grad_(True) for s in ("n", "p")}
    delta = torch.zeros_like(h0)

    class DeltaAction:
        def apply(self, vectors):
            return delta @ vectors

    evaluation = SimpleNamespace(
        normal_fields=h_fields,
        delta_actions={s: DeltaAction() for s in ("n", "p")},
    )
    h_mu = nambu_hamiltonian(h0, delta, 0.0)
    evidence = {s: _exact_evidence(h_mu, _cert()) for s in ("n", "p")}
    policy = MixedNucleonSeaBranchPolicy(evidence, blocked_species=(),
                                         quadrature_order=321, log_step=0.2)
    mus = {s: torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
           for s in ("n", "p")}
    with pytest.raises(NoSeaBranchEvidenceError, match="target parity -1 disagrees"):
        policy.evaluate(states, evaluation, mus)


def test_odd_policy_calls_graph_connected_mode_provider_on_current_field():
    from dpl_rhf.rhfb.operators import DensityFactors

    d = 2
    q = torch.eye(d, dtype=torch.complex128)
    rho_even = torch.zeros((d, d), dtype=torch.complex128)
    chi = torch.zeros_like(rho_even)
    states = {s: DensityFactors(q, rho_even, chi) for s in ("n", "p")}
    h0 = torch.diag(torch.tensor([1.0, -20.0], dtype=torch.complex128))
    h_fields = {s: h0.clone().requires_grad_(True) for s in ("n", "p")}
    delta = torch.zeros_like(h0)

    class DeltaAction:
        def apply(self, vectors):
            return delta @ vectors

    class ModeProvider:
        calls = 0

        def checkpoint_metadata(self):
            return {"provider": "analytic-diagonal-toy-mode-v1"}

        def __call__(self, species, h_mu, base_density):
            del species, base_density
            self.calls += 1
            mode = torch.tensor([1., 0., 0., 0.], dtype=torch.complex128)
            mode = mode + 0.0 * h_mu[0, 0]
            return BlockedModeEvidence(
                mode=mode, scope=EvidenceScope.SELFCONSISTENT_FAMILY,
                proof_id="analytic-diagonal-mode-family-v1",
                field_family_id="diagonal-h0-pairing-zero-family",
            )

    eval_data = SimpleNamespace(
        normal_fields=h_fields,
        delta_actions={s: DeltaAction() for s in ("n", "p")},
    )
    h_mu = nambu_hamiltonian(h0, delta, 0.0)
    family_evidence = MixedNambuBandEvidence(
        divider_mev=5.0, certificate=_cert(),
        scope=EvidenceScope.SELFCONSISTENT_FAMILY,
        proof_id="analytic-diagonal-band-family-v1",
        field_family_id="diagonal-h0-pairing-zero-family",
    )
    evidence = {"n": family_evidence, "p": _exact_evidence(h_mu, _cert())}
    provider = ModeProvider()
    policy = MixedNucleonSeaBranchPolicy(
        evidence, blocked_species=("n",), blocked_mode_sources={"n": provider},
        quadrature_order=321, log_step=0.2,
    )
    mus = {s: torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
           for s in ("n", "p")}
    loss, diagnostics = policy.evaluate(states, eval_data, mus)
    assert provider.calls == 1
    assert diagnostics["species"]["n"]["blocked_mode_fingerprint"]
    assert policy.checkpoint_metadata()["blocked_mode_family_proof_machine_verified"] is False
    grad, = torch.autograd.grad(loss, h_fields["n"])
    assert torch.isfinite(grad).all()


def test_blocked_mode_independent_grad_leaf_is_rejected():
    h0 = torch.diag(torch.tensor([1.0, -20.0], dtype=torch.complex128))
    h = nambu_hamiltonian(h0, torch.zeros_like(h0), 0.0).requires_grad_(True)
    family_evidence = MixedNambuBandEvidence(
        divider_mev=5.0, certificate=_cert(),
        scope=EvidenceScope.SELFCONSISTENT_FAMILY,
        proof_id="analytic-diagonal-band-family-v1",
        field_family_id="diagonal-h0-pairing-zero-family",
    )
    branch = construct_mixed_nucleon_sea_density(
        h, evidence=family_evidence, quadrature_order=321, log_step=0.2,
    )
    # This vector is trainable in isolation but is not a function of H_mu.
    mode = torch.tensor([1., 0., 0., 0.], dtype=torch.complex128, requires_grad=True)
    disconnected = BlockedModeEvidence(
        mode=mode, scope=EvidenceScope.SELFCONSISTENT_FAMILY,
        proof_id="independent-leaf-is-not-a-mode-response",
        field_family_id="diagonal-h0-pairing-zero-family",
    )
    with pytest.raises(NoSeaBranchEvidenceError, match="disconnected from current H_mu"):
        block_mixed_nucleon_sea_density(branch, disconnected)
