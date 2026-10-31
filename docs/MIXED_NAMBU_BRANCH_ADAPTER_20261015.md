# Conditional mixed Nambu branch adapter

`dpl_rhf.rhfb.mixed_nucleon_sea_branch` now exposes a fail-closed finite-space
branch map. Call `construct_mixed_nucleon_sea_density(H_mu, evidence=...)` on
the *current* full Nambu Hamiltonian. `H_mu` must use the package convention
`[[h-mu, Delta],[-Delta*, -h*+mu]]`; `nambu_hamiltonian` in
`full_branch_stationarity.py` builds it and checks Hermiticity and pairing
antisymmetry. The returned object contains the complete `generalized_density`,
`rho`, and `chi`, and offers `operator_action(probe)` and
`commutator_action(probe)` for operator diagnostics. Each call validates the
current matrix against the supplied certificate. Exact-H evidence is byte-hash
bound; a field-family certificate must identify an externally established
uniform enclosure. The evidence payload is JSON serializable and its stable
SHA-256 is available as `evidence.fingerprint()` for checkpoint metadata.

For a finite represented space with a proved particle-band/sea-band gap around
an explicit divider `Ec`, the constructed map is

`R = (I - sign(H_mu) sign(Ec^2 I - H_mu^2))/2`.

All components are full Nambu matrices, and the rational sign uses shifted
linear solves. `mixed_branch_fixed_point_loss(H_mu, R_nn, evidence=...)`
computes `mean(|R_nn-R(H_mu)|^2)` without detaching `H_mu`; gradients therefore
include the `H_mu -> R(H_mu)` response. This can supply a residual objective
to a fixed-point trainer. A nonzero residual is not a trial rejection rule.
The adapter accepts at most primitive dimension 96 (full Nambu dimension
192), as a small-space prototype limit, not as a claim that a dense full
nuclear calculation is bounded-memory. The training/operator path calls no
eigensolver. The tests use `numpy.linalg.eigh` only as an isolated independent
small-matrix oracle.

Odd branches require the policy constructor to name `blocked_species`
explicitly. Each such species must also supply either exact-H
`BlockedModeEvidence` (fixed-matrix tests only) or a field-varying provider
called on every current `H_mu`. The provider must return a current,
normalized, positive-energy eigenmode `b`; validation checks its eigenmode
residual and `R_base b=0`, `R_base Cb=Cb`. The odd density is
`R_odd=R_base+bb†-(Cb)(Cb)†`, which keeps the full mixed qp sum and swaps the
occupation of the mode with its particle-hole partner. Hermiticity, purity,
particle-hole symmetry, and pairing antisymmetry are checked after the swap.
An occupied-plane transport tag alone is not an instantaneous qp eigenmode
witness. A family provider is invoked for each field evaluation and its
identity is recorded in checkpoint metadata, but the code cannot machine-
verify its claimed family proof or certify quasiparticle identity. Missing
odd-mode evidence is an error; it never silently falls back to the even
unblocked selector.

When `H_mu` is trainable, the adapter also probes autograd connectivity from
the supplied mode output back to that exact `H_mu` tensor with
`autograd.grad(..., allow_unused=True, retain_graph=True)`. A connected zero
response is accepted; a disconnected trainable leaf is rejected. This only
checks graph connectivity, not the mathematical correctness of the provider's
mode derivative or its field-family proof.

After construction, the policy also audits the final target parity from its
pure-density Pfaffian (without diagonalization). It requires even parity for
an unblocked selector and odd parity after one explicit block swap, matching
the declared `blocked_species`. A selector whose parity changed across a
quasiparticle crossing is rejected rather than relabeled.

The physical motivation is conditional and narrower than a production no-sea
claim. DIRHB's RHB equations sum both the particle `E_p>0` and sea `E_a<0`
quasiparticle sectors in both normal and pairing densities (Eqs. 46–48,
pp. 12–13 of [DIRHB, arXiv:1403.4039](https://arxiv.org/abs/1403.4039)). The
authors state that omitting the small `V_na` contribution leaves the pairing
density only approximately antisymmetric. Long et al. define the no-sea energy
functional without negative-energy single-particle contributions while using
positive and negative DWS basis states for completeness (Eqs. 7, 32–34,
pp. 2–5 of [arXiv:0812.1103](https://arxiv.org/abs/0812.1103)); basis inclusion
does not imply sea occupation. Geng and Long show sensitivity to the negative
DWS expansion cutoff for 156Sm (pp. 9–11 of
[arXiv:2202.00482](https://arxiv.org/abs/2202.00482)), which likewise does not
establish a universal quasiparticle divider.

The mixed-sign formula follows algebraically if the *complete represented*
particle and sea BdG bands are separated by the supplied `Ec` and the
particle-hole/Hermitian assumptions hold. That does not prove those bands are
separable for a self-consistent nuclear field family. The positive continuum
can extend to energies overlapping the absolute-energy sea band in a finite
window, and no uniform O17 enclosure has been established here. Therefore the
adapter deliberately chooses no `Ec`, supplies no witness, and reports both
`physical_no_sea_certified=False` and `physical_identity_certified=False`.
Missing or stale evidence raises an exception; the caller should reject that
trial and retain the last accepted state. Do not substitute `P+`, a fitted
window, a local spectral threshold, or a single-field spectrum as a family
proof. This adapter does not certify production continuum RHFB or solve the
remaining physical branch-identification problem.
