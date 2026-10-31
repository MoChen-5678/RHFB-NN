# Isolated spherical radial (p^2) matrix-element helper

`dpl_rhf/rhfb/cm_spherical_p2.py` implements only the one-body (p^2) radial matrix element for a single spherical Dirac angular channel. The source is Geng and Long, [arXiv:2202.00482](https://arxiv.org/pdf/2202.00482), Appendix C, Eq. (C2), PDF p. 27: the upper and lower reduced radial components each receive (-d^2/dr^2+l(l+1)/r^2), with their own orbital angular momenta. The integration measure is (dr) for reduced (G,F), and the helper multiplies by ((hbar c)^2) to return MeV².

The caller supplies the radial quadrature, second-derivative matrix, and boundary convention. The helper checks Hermiticity and rejects a mismatch instead of repairing it by symmetrization. Its unit test compares the matrix against analytic derivatives and independent Gaussian integration for polynomial reduced radial functions, plus an asymmetric-operator rejection check.

This addresses only the direct one-body part (mathrm{Tr}(ho p^2)) in the microscopic correction. It does not construct the rank-one (p_mu) matrix elements needed by the normal-exchange or anomalous Wick contractions, and it cannot be applied to the current ADF/m-scheme O17 space without a separately derived angular transform, metric, and boundary validation. It is not imported by the energy functional or training code. Existing `center_of_mass.py` remains the algebraic evaluator for the full matrix Wick expression when caller-supplied (p_mu) and independently supplied (p^2) are available.

The pre-change sources are archived under [`cm_prechange_snapshot_20261015`](cm_prechange_snapshot_20261015/); no existing helper or runtime file was modified.
