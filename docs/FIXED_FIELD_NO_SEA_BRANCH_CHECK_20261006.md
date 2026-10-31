# Fixed-field Dirac-sector check — 2026-10-06

## Result

This run did **not** complete the whole-Dirac versus strict (P_+(h)) BdG
comparison. The predeclared shifted-spectrum separation test rejected both
species before assigning quasiparticle-sector labels. We did not replace that
test with an absolute-value cutoff. The saved density, full PKA1 normal fields,
and full D1S pairing matrices remain available for the independent frozen-field
NN residual calculation.

The input is the (R=8\,\mathrm{fm}, h=0.8\,\mathrm{fm}, \kappa_{\max}=1,
l_{\max}=2) bounded probe, with 72 primitive columns. Its neutron seed has
(N=3), number parity (-1), nonzero (|\chi\|_F=0.9797959), and saved
blocked basis mode 19; the proton seed has (N=4), parity (+1), and zero
pairing. These are fixed-field probe inputs, not an O17 ground state. The
primitive full D1S neutron field has (|\Delta\|_F=19.41661\) MeV and
antisymmetry error (1.78\times10^{-15}\) MeV. No optimizer update or field
iteration was performed.

The shifted normal-Hamiltonian split used cutoff (E=-M=-938.9\) MeV, with
declared minimum cutoff distance 200 MeV and minimum particle/sea band gap
1000 MeV. The 200/1000 MeV margins are conservative, fail-closed thresholds
chosen for this diagnostic. They are **not** a theorem or universal cutoff in
DIRHB or any cited paper. DIRHB describes a typical roughly 1200 MeV gap in
its RHB qp spectrum as useful for distinguishing Dirac-sea branches; that
observation does not prescribe our finite-matrix normal-spectrum margin. The
exact finite-matrix results were:

| Species | Nearest distance to cutoff | Highest sea eigenvalue | Lowest particle eigenvalue | Edge separation | Decision |
|---|---:|---:|---:|---:|---|
| neutron | 255.753 MeV | −1533.944 MeV | −683.147 MeV | 850.797 MeV | withheld: edge gap below 1000 MeV |
| proton | 139.208 MeV | −1498.889 MeV | −799.692 MeV | 699.197 MeV | withheld: both margins fail |

Accordingly, the report records no whole-Dirac branch energies, spectra,
densities, or same-field residuals. The attempted calls took 0.010 s because
they stopped immediately after the inexpensive normal (h) eigenspectrum.
This is a safe refusal result, not a failed HFB solve.

## Physical scope

DIRHB writes the relativistic no-sea RHB quasiparticle sums in Eqs. (46)–(48):
positive-energy particle quasiparticles and negative-energy Dirac-sea
quasiparticles are the annihilator branches. The sums include both particle
and sea contributions to the normal and anomalous densities; dropping the
sea terms is only an approximation to the anomalous density. See Nikšić et al.,
[arXiv:1403.4039, Sec. 2.3, Eqs. (46)–(48), PDF pp. 11–13](https://arxiv.org/pdf/1403.4039).
The full Dirac-space D1S pairing prescription has independent support in
Geng & Long, [arXiv:2202.00482, Eq. (20d), PDF p. 5, and Appendix B, Eqs. (B5)–(B10), printed p. 26](https://arxiv.org/pdf/2202.00482).

Those references support the particle/sea quasiparticle prescription and the
full Dirac-component pairing force. They do not identify it with the candidate
model's strict two-sided instantaneous support
\(\rho=P_+\rho P_+,\;\chi=P_+\chi P_+^T\), where (P_+) is rebuilt from the
interacting normal (h\). The latter is a distinct projected model. If its
active generalized density is embedded in the full primitive space, the
canonical full (R\) also contains the empty-particle sea-hole block
\(\operatorname{diag}(0,P_-^*)\); omitting it gives the wrong trace and breaks
the particle-hole complement. The comparison implementation now includes
that block and checks the embedded canonical density before measuring the
same-full-field commutator.

For a valid whole-space reference, positive BdG modes are labeled by their
normal sea-projector weight. A particle-sector positive mode contributes its
PH partner to the vacuum projector; a sea-sector positive mode contributes
itself. Continuation overlap transports qp identity only. An impure sector
label, a zero-energy crossing, or a failed actual-Pfaffian parity check
withholds the reference. For an odd target, the construction first verifies
an even unblocked reference (compensating an odd quadratic vacuum by toggling
its lowest-|E| particle qp only when needed), then toggles the requested
blocked qp and independently checks \(\Pi=-1\). Even though that
reference construction and the missing sea-hole embedding are now covered by
small API tests, the O(72) spectrum above does not pass the sector-gap gate.

### Odd blocking and parity scope

Odd blocking is not restricted to particle addition. In the unpaired limit,
for a normal eigenmode of energy \(\epsilon\), the BdG diagonal entries are
\(\xi=\epsilon-\mu\) and \(-\xi\). When \(\epsilon>\mu\), the positive qp is
particle-like and toggling its selected column adds one particle. When
\(\epsilon<\mu\), the positive qp is hole-like and the same
creator/annihilator exchange removes one particle from the even reference.
For any supplied final \(\mu\), the sign of \(\epsilon-\mu\) identifies the
HF-limit qp as particle-like (addition) or hole-like (removal); the code does
not require the unpaired normal occupation count to equal \(N-1\) or \(N+1\).
It records that count and whether it matches the corresponding HF-limit
reference family. This matters because a same-final-\(\mu\), paired HFB
reference is not generally constrained to have an integer \(N\pm1\) count in
the unpaired \(\Delta=0\) spectrum. At \(\epsilon=\mu\) it refuses an
ambiguous zero-energy label. When \(\mu\) is generated internally, the code
uses an \(N-1\) even-core bracket for particle addition, or an \(N+1\) even-core
bracket for hole removal when the designated mode cannot be included in the
addition family. That HF-limit hole bracket requires at least one additional
unoccupied particle mode; otherwise it raises a specific RuntimeError rather
than indexing past the finite spectrum.
This is the creator/annihilator exchange of one chosen qp in Li et al.,
[arXiv:1203.1363, Eqs. (1)–(5), PDF pp. 1–2](https://arxiv.org/pdf/1203.1363).

The even-parity reference used in continuation is not certified as the
lowest-energy even branch at full \(\Delta\). If the \(\Delta=0\) no-sea
quadratic vacuum is odd, the implementation toggles its lowest-|E| particle
qp to correct parity, then follows that fixed choice as \(\Delta\) increases.
It does not minimize energy or enumerate even branches at \(\Delta>0\).
Kasuya & Yoshida, [arXiv:2005.03276, Eqs. (21)–(23), PDF p. 7](https://arxiv.org/pdf/2005.03276),
relate an arbitrary pure HFB state's parity to the Pfaffian and signed qp
energies. Their simplified `sgn pf(H_X)` result identifies the lowest state
only when it is actually the lowest-energy state; under time reversal and
without zero modes their Eq. (41) gives even parity for that lowest-energy
solution (PDF p. 11). The code's independently computed Pfaffian is therefore
only a parity invariant and does not certify energetic ordering.

## Reproduction artifacts

- Full seed factors and normal fields: `outputs/validation/cached_normal_action_20261006/seed_fields.npz`
- Pure seed state: `outputs/validation/cached_normal_action_20261006/seed_state_input.npz`
- Full primitive D1S Δ: `outputs/validation/cached_normal_action_20261006/full_d1s_pairing_fields.npz`
- Field, seed, and source metadata are the adjacent JSON files in that directory.
- The independent split refusal, thresholds, and source hashes are in
  `outputs/validation/fixed_field_dirac_vs_projected_20261006_post_hole_review/diagnostic.json`.
- The branch-construction implementation is
  `dpl_rhf/rhfb/fixed_field_dirac_branch_reference.py`; its synthetic API tests
  are `dpl_rhf/rhfb/tests/test_fixed_field_dirac_branch_reference.py`.

The full input-state SHA256 is
`b8e0235617f33081228133c4fdcc48b606c8ba3f643ee65d00ddc72a0444f8ce`; the
normal-field NPZ SHA256 is
`d6a309cf5f982a29132f9d568d243993f9a4ec181bb5bf0b3e94479bf7bf2d2c`; and the
pairing-field NPZ SHA256 is
`f99f13c8cabe2658fb1ef4ef66c9ac4be508b1982bc8ee297c33f89638c28296`.
No Fortran was used.
