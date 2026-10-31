# Scope of the time-reversal-averaged one-body reference

`time_reversal_averaged_odd_one_body_reference()` is intentionally named and
typed as a one-body reference. Starting from one pure odd blocked HFB density,
it computes the normal and anomalous contractions of that state and its
time-reversed partner, then averages them at weights 1/2 and 1/2:

\[
  \rho_T=J\rho^*J^\dagger,\quad \chi_T=J\chi^*J^T,\qquad
  \rho_{\rm avg}=\tfrac12(\rho+\rho_T),\quad
  \chi_{\rm avg}=\tfrac12(\chi+\chi_T).
\]

The return value is not `GeneralizedDensity`, has no generalized-density
matrix or sharp-parity field, and says `wick_energy_closed=False`. It is for
one-body comparisons/diagnostics only. Do not send it to the pure-state energy,
stationarity, or variational paths. The original pure blocked component
remains the appropriate input for those paths.

This object is **not** the standard equal-filling approximation (EFA). In
standard HFB-EFA, the quasiparticle statistical occupancies are 1/2 for both
the blocked quasiparticle and its Kramers partner, and 0 for the other modes.
The statistical density operator therefore includes the zero-, one-, and
two-quasiparticle sectors. Its one-body contractions can be written as the
average over the two blocked odd states, but its two-body energy is not the
average energy of only those odd states. The EFA ensemble itself does not have
a sharp fermion-number parity. An averaged `rho,chi` pair alone does not
specify which statistical ensemble produced it, so this module does not label
that pair with a parity or treat it as energy-closed.

The distinction has a minimal Fock-space counterexample. For two modes with
basis `|0>, |1>, |2>, |12>`, the odd-only mixture
`(|1><1|+|2><2|)/2` and the standard EFA four-state mixture with each basis
state weighted 1/4 have identical one-body occupations `rho=diag(1/2,1/2)`
and zero anomalous density. But the odd-only mixture has
`<n1 n2>=0`, while the four-state EFA mixture has `<n1 n2>=1/4`, which is
the Wick value from those one-body contractions. Thus the one-body average
cannot recover the odd-only ensemble's two-body energy.

## Primary sources and source comparison

- S. Perez-Martin and L. M. Robledo, *Microscopic justification of the Equal
  Filling approximation*, Phys. Rev. C 78, 014304 (2008),
  [arXiv:0805.4318](https://arxiv.org/abs/0805.4318),
  [full text](https://arxiv.org/html/0805.4318). Eqs. (15)–(18) give the
  generalized statistical density and half occupancies for the blocked
  quasiparticle/Kramers pair; Eqs. (20)–(28) derive the density operator and
  the average over zero-, one-, and two-quasiparticle states. Eqs. (31)–(32)
  show that the one-body normal/anomalous contractions equal the average of
  the two blocked odd states. The text around Eqs. (28)–(32) cautions that
  this does not make the energy the average of the two odd-state energies.
- L. M. Robledo, R. Bernard, and G. F. Bertsch, *Pairing gaps in the
  Hartree-Fock-Bogoliubov theory with the Gogny D1S interaction*, Phys. Rev. C
  86, 064313 (2012), [publisher DOI](https://doi.org/10.1103/PhysRevC.86.064313),
  [author-hosted paper](https://archive.int.washington.edu/users/bertsch/articles/354.pdf).
  Page 1 describes the filling approximation's time-odd-density omission and
  statistical time-reversed-partner interpretation; the paper distinguishes
  it from full blocking with time-odd fields.
- The supplied read-only radial sources have blocked multiplicity changes in
  `/home/ubuntu/rhf/dpl-param/reference_fortran/user-rhfb-20261015/`:
  `RHFB-BI/Density.f90` lines 54–69 and 83–104;
  `S-RHFB-C/Density.f90` lines 53–70 and 86–107; and
  `S-RHFB-C/Gogny.f90` lines 318–337. They use spherical/radial degeneracy
  factors (`mu-1`, and `mu-2` in the pairing accumulation); these are not a
  general deformed-m pure-blocking prescription.

The existing D1S kernel's W/B/H/M central combinations, antisymmetrization,
and `Delta = (1/2) Vbar chi` convention were already implemented and were not
changed here. This addition does not alter the project's pure-state odd
blocking model or claim support for standard EFA energy minimization.
