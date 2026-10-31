# Fixed-density normal-field action cache (6 October 2026)

The reference convention is the full particle-hole derivative, not an orbital
eigenvalue prescription. Geng and Long define the generalized density in
their Eq. (21) and the generalized Hamiltonian in Eq. (22), with
`h = ∂E/∂ρ` and `Δ = ∂E/∂κ` (arXiv:2202.00482, PDF p. 5). Their Eqs. (20a–c)
separate kinetic, particle-hole Hartree, and particle-hole Fock energy; the
Gogny replacement in Eq. (20d) is the separate particle-particle prescription.
The paper also states that variation only with respect to the V component
does not recover the full RHFB equation because U and V are related by the
Bogoliubov unitarity constraint. This supports retaining a generalized-density
variation for the complete normal field. The paper does not validate this
repository's PKA1 contractions or its numerical derivatives.

Primary source: [Geng & Long, *Relativistic Hartree-Fock-Bogoliubov model for
axially deformed nuclei*, arXiv:2202.00482](https://arxiv.org/abs/2202.00482).
The underlying PKA1 vertices are specified separately in Long et al.,
[arXiv:0706.3497](https://arxiv.org/abs/0706.3497).

`dpl_rhf.rhfb.cached_normal_action.CachedNormalAction` stores exact measured
columns `hU` from the existing full-energy `AutogradNormalAction`, together
with an orthonormal input basis `U`. For subsequent vectors `v`, it returns
`(hU)(U†v)` only when `v` lies in `span(U)` to the configured tolerance. It
rejects vectors outside that span. A primitive identity basis therefore gives
a complete dense finite-space action; an active frame plus probes gives an
exact cache only on that declared subspace. The class does not alter the EDF,
drop rearrangement, invert occupations, diagonalize, or approximate the
functional. It saves energy/gradient reevaluations after its columns have
been constructed; it does not make initial column construction free.
The stored action is tied to the fixed density and basis used during
construction. If `Q`, `rho`, the basis, couplings, or energy callback changes,
the cache must be rebuilt. It is a diagnostic/frozen-field object, not a
truncated self-consistent training response. The two-species helper
`evaluate_two_species_normal_columns` evaluates independent neutron and proton
Hermitian tangents in one complete PH energy pass per probe block; it is used
to construct both fields efficiently for the fixed seed.

The bounded branch-seed run uses R=8 fm, h=0.8 fm, κmax=1, lmax=2 and 72
primitive radial-spinor modes. It specifies an N=3 odd neutron density and an
N=4 even proton density, then asks the complete current particle-hole EDF for
all primitive columns. Exact state arrays, fields, masses, configuration,
source hashes, and elapsed time are recorded in
`outputs/validation/cached_normal_action_20261006/seed_fields.npz` and
`metadata.json`. This is a frozen-field reference input for follow-on pairing
and branch comparisons, not a converged nucleus or a validation of physical
readiness.

The exact full-field run took 18.17 s in the evaluator (20.70 s wall time) with
peak resident memory 1.55 GiB. Nine 8-column tangent batches generated both
72×72 fields. The largest Hermiticity defect was 2.42×10⁻¹³ MeV. A support
internal Kramers-pair occupation-transfer direction gave total analytic
`dE/dε = -6.78625832864 MeV` and central FD `-6.78625833075 MeV` at
`ε=10⁻⁵`, an absolute difference of 2.11×10⁻⁹ MeV. Five resolved PH channels
had maximum relative FD error 4.33×10⁻⁹; near-zero terms are judged by absolute
error (maximum 1.78×10⁻¹⁰ MeV), avoiding meaningless relative errors at zero.
This deliberately time-reversal-paired seed and its diagonal Kramers transfer
leave several parity/time-odd mixed-channel directional derivatives at zero;
those zero entries are recorded, not counted as nonzero independent checks.
The separate matrix-free oracle matched an external-to-support probe in each
species to 1.2×10⁻¹⁵ MeV. Full per-channel and threshold records are in
`outputs/validation/cached_normal_action_20261006/closure_diagnostics.json`.

The O17 stationarity artifact at
`outputs/validation/O17_stationarity_20261006_h08_v3/` stores an active-frame
field `hQ` for 14 columns per species in a 432-dimensional primitive space.
That gives an exact reusable cache on `span(Q)`; it says nothing about omitted
directions. A selected external probe is therefore evaluated against the full
energy oracle separately to measure the cost and verify the cache contract.
The O17 test remains a diagnostic of the saved neural density, not a
self-consistent no-sea or converged-space result.
For the real O17 saved density, the diagnostic adds one new external probe to
the previously measured 14-column `hQ` and caches exactly `span(Q, probe)`.
The newly evaluated external `hV` took 25.28 s; cache linear-combination
agreement is 2.25×10⁻¹⁴ MeV. The artifact is
`outputs/validation/cached_normal_action_20261006/O17_active_plus_external_cache.npz`.
Only this 15-column domain is certified by that cache; it does not supply the
other O17 primitive directions.
