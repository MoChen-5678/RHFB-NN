# No-sea branch and field-dependent sign response

## Source statements

- DIRHB (Nikšić et al., arXiv:1403.4039), Sec. 3.3, Eqs. (46)–(48), pp. 12–13: its even-system quasiparticle vacuum selects positive-energy quasiparticle solutions above the Dirac sea and negative-energy quasiparticle solutions in the sea. Both sectors enter the normal density and pairing tensor. The text says the sea coefficients are negligible due to the large Dirac gap and that dropping the sea terms leaves the pairing tensor only approximately antisymmetric. This is a statement about the quasiparticle sums in that RHB prescription, not an instruction to remove negative-energy Dirac basis vectors.
- Long et al. (arXiv:0812.1103), Sec. II and DWS expansion, Eqs. (7), (32)–(34), pp. 2–5: the no-sea energy functional neglects contributions from negative-energy single-particle states, while the DWS basis used to expand the RHFB equation contains both positive- and negative-energy states (dimensions (N_F+N_D)). The basis states supply expansion completeness; that fact alone does not imply that negative-energy states are occupied in the nuclear density.
- Geng & Long (arXiv:2202.00482), Sec. III, pp. 9–11: varying the negative DWS cutoff changes some PKA1 results, especially for (^{156})Sm, and the authors discuss those states as expansion terms needed for completeness. Their cutoff study supports a basis-convergence role; it does not define a universal occupied-sea filter or a universal (W).

Primary sources: [DIRHB](https://arxiv.org/pdf/1403.4039), [Long et al.](https://arxiv.org/pdf/0812.1103), [Geng & Long](https://arxiv.org/pdf/2202.00482).

## Mathematical result implemented here

For a finite Hermitian matrix (h), a *caller-declared* shift (W), and a spectral gap around (W), the positive spectral projector is

\[
Q_+(h-W)=\tfrac12[I+\operatorname{sign}(h-W)],\qquad
\operatorname{sign}(A)=\frac{2}{\pi}\int_0^\infty A(A^2+t^2 I)^{-1}\,dt.
\]

Differentiating this integral (or its finite rational shifted-solve approximation) gives the response (D Q_+[h-W](\delta h)). A field-dependent (h[\rho,\chi]\) therefore contributes an extra chain-rule term to a variational derivative whenever the allowed space itself is defined by (Q_+(h-W)). Existing `no_sea_variation.external_projector_response_source` supplies the cotangent with respect to the projector, but by itself it does not apply (D Q_+/D h) or close the self-consistent field response.

The isolated `no_sea_sign_response.py` module now supplies that missing *finite-matrix response primitive*: it evaluates a declared positive projector with `RationalSignOperator` shifted linear solves, and its source contraction stays in the autograd graph. A two-level analytic test checks the response against a closed-form derivative. No eigensolver is used on this path.

## Scope and unresolved physics

The new code proves only differentiation of a finite rational matrix-sign approximation under the caller's spectral-enclosure assumption. The enclosure is not verified by the routine; the shift (W) is mandatory input and has no default; quadrature error is not a nuclear-model uncertainty. This does not establish that a moving positive spectrum of (h-W) is the intended or unique nuclear no-sea prescription, select (W), certify that the gap survives self-consistency, or implement the full implicit EDF Hessian/field closure. It is not imported by the NN trainer or O17 driver. The mixed-sign DIRHB-window prototype remains conditional on its separate complete-band assumptions and is not upgraded by this work.
