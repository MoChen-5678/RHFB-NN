# PKA1 covariant rearrangement and time-reversal closure

This note records an independent small-space closure of the density-dependent PKA1 vertices in the experimental Cartesian RHFB functional. It does not change the production energy implementation or establish physical readiness of the solver.

## Covariant VDD chain rule

Use the (+---) metric and the total neutron-plus-proton baryon current

\[
j^\mu(x)=\sum_{s=n,p}\bar\psi_s(x)\gamma^\mu\psi_s(x),\qquad
\rho_B(x)=\sqrt{j_\mu j^\mu}
 =\sqrt{(j^0)^2-\boldsymbol j^2}.
\]

At a future-timelike point,

\[
\delta\rho_B=\frac{j_\mu\delta j^\mu}{\rho_B}
=\frac{j^0\delta j^0-\boldsymbol j\cdot\delta\boldsymbol j}{\rho_B}.
\]

For every density-dependent vertex (g_i(\rho_B)), define the endpoint sensitivity

\[
D_i(x)\equiv\frac{\delta E_{\rm PH}}{\delta g_i(x)},
\qquad
\delta E_{\rm PH}=\sum_i\int d^3x\,D_i(x)\,\delta g_i(x).
\]

Each (D_i) is formed by differentiating both occurrences of the vertex in the two-body kernel. For example, for one symmetric Fock channel written as

\[
E_F=C\sum_{ab}\int d^3x d^3y\;g(x)L_{ab}(x)K(x,y)g(y)R_{ba}(y),
\]

the two endpoint contributions are

\[
D_F(x)=C\sum_{ab}\left[L_{ab}(x)\,K[gR_{ba}](x)
+R_{ba}(x)\,K^\dagger[gL_{ab}](x)\right].
\]

The second endpoint uses the adjoint kernel and the reversed active-state indices. It is not safe to identify bra and ket indices implicitly when the active density matrix is complex. For derivative channels let (G=\boldsymbol\nabla), (G^\dagger) be its physical-volume adjoint, and (D_m) the meson propagator. With (L=G^\dagger[gJ_L]), (R=G^\dagger[gJ_R]), the corresponding endpoint structure is

\[
D_F(x)=C\left[J_L(x)\cdot G D_m R(x)
+J_R(x)\cdot G D_m L(x)\right],
\]

again preserving the reversed state-pair ordering. The ordered (\rho\)-VT and (\rho\)-TV terms each have their own (g_\rho) and (f_\rho) endpoints. The direct mixed term contracts the vector-current component (j^\mu) with the tensor source carrying the same Lorentz index (\mu). A local compensation energy quadratic in a vertex contributes (D_i=2E_i(x)/g_i(x)).

The scalar rearrangement source and its four-vector field therefore are

\[
S_R(x)=\sum_i g_i'(\rho_B(x))D_i(x),\qquad
\Sigma_R^\mu(x)=S_R(x)\frac{j^\mu(x)}{\rho_B(x)}.
\]

In a Dirac Hamiltonian the local rearrangement operator is

\[
h_R(x)=\gamma^0\gamma_\mu\Sigma_R^\mu
=\frac{S_R(x)}{\rho_B(x)}
\left[j^0(x)I_4-\sum_{i=1}^3j^i(x)\alpha_i\right],
\qquad \alpha_i=\gamma^0\gamma^i.
\]

Thus the contravariant spatial field has (\Sigma_R^i=S_Rj^i/\rho_B), while its contribution to the Hamiltonian is the conventional negative (\boldsymbol\alpha\cdot\boldsymbol\Sigma_R\).

For the actual PKA1 implementation the effective density-dependent vertices are (g_\sigma,g_\omega,g_\rho,f_\rho,f_\pi), with the (\rho\)-tensor strength represented by the model's (g_\rho f_\rho/(2M)) conversion and its own PKA1 density exponent. Their derivatives are included in the endpoint reference. The density-independent kinetic, Coulomb, and Gaussian D1S pairing strengths have no extra PKA1 vertex rearrangement source in this functional.

## Relation to the literature

Fuchs, Lenske, and Wolter define the invariant VDD density from the square of the baryon four-current and derive the resulting vector rearrangement self-energy in Eqs. (16), (26), and (27), printed pp. 3047–3048 (PDF pp. 5–6). Their Eq. (27) has the covariant (u^\mu=j^\mu/\rho_B) structure and, in their Hartree example, combines the scalar and vector meson vertex derivatives. The paper explicitly says that (j_\mu j^\mu), rather than only (j^0\), is required for covariance. This is the established VDD result; the sum over all PKA1 Fock, derivative, mixed, and contact endpoint sensitivities above is the chain-rule extension for the energy functional used here, not a formula claimed verbatim from that Hartree paper. [Fuchs et al., *Phys. Rev. C* 52, 3043 (1995), full text](https://harvest.aps.org/v2/journals/articles/10.1103/PhysRevC.52.3043/fulltext).

The local manuscript [RHF-e.pdf](/home/ubuntu/rhf/DFT-RHF-NoRefresh/variants/blocked-rhfb/RHF-e.pdf) gives the general no-sea Hamiltonian and (\sigma\), (\omega\), (\rho\)-V, (\rho\)-T, ordered (\rho\)-VT, (\pi\)-PV, and Coulomb vertices in Eqs. (1.24)–(1.25), PDF p. 6 (printed p. 4). Its density-dependent rational and exponential vertices and the statement that their variation generates rearrangement are Eqs. (1.41)–(1.44), PDF p. 8 (printed p. 6). It is a manuscript source. The published O-RHF formalism independently states the two-body RHF energy form and channel set in Sec. II.A, Eqs. (1)–(4), and identifies the rearrangement contribution from density-dependent couplings: Peng, Geng, and Long, *Chin. Phys. C* 49, 074107 (2025), [DOI page](https://doi.org/10.1088/1674-1137/adc11e).

## Time reversal and observed closure

In the Dirac representation use the antiunitary one-nucleon operation (T=U_TK), (U_T=i\gamma^1\gamma^3). In the sampled spinor frame,

\[
\psi_T=U_T\psi^*,\quad
\rho_T=U_T\rho^*U_T^\dagger,\quad
\chi_T=U_T\chi^*U_T^T.
\]

In the transformed active frame this is implemented as (q_T=\operatorname{project}(U_Tq^*)), (\rho_{\rm active,T}=\rho_{\rm active}^*), and (\chi_{\rm active,T}=\chi_{\rm active}^*). The tested bilinear parities are

\[
\rho_s,\ j^0,\ A^0,\ T^{0i}:\;+1;
\qquad
j^i,\ A^i,\ T^{ij}:\;-1.
\]

The invariant (\rho_B), every particle-hole channel energy, and the complete normal-plus-anomalous energy are unchanged. In particular the mixed vector-tensor contractions remain even because their time components pair even with even and their spatial components pair odd with odd; the (\rho\)-tensor and pion pseudo-vector quadratic kernels are also even. The vector rearrangement current transforms as (\Sigma_R^0\to\Sigma_R^0\), \(\boldsymbol\Sigma_R\to-\boldsymbol\Sigma_R\), so the Dirac Hamiltonian is time-reversal covariant.

The explicit reference implementation is [analytic_rearrangement.py](/home/ubuntu/rhf/DFT-RHF-NoRefresh/variants/blocked-rhfb/dpl_rhf/rhfb/analytic_rearrangement.py), and its independent parameter-direction, two signed nonuniform endpoint-field directions, Dirac-operator closure, and time-reversal checks are in [test_analytic_rearrangement.py](/home/ubuntu/rhf/DFT-RHF-NoRefresh/variants/blocked-rhfb/dpl_rhf/rhfb/tests/test_analytic_rearrangement.py). Numerical evidence is in [analytic_rearrangement_time_reversal_20261006.xml](/home/ubuntu/rhf/DFT-RHF-NoRefresh/variants/blocked-rhfb/outputs/validation/analytic_rearrangement_time_reversal_20261006.xml). The tests use a (\kappa_{\max}=1), (l_{\max}=2) small space as a closure oracle; they do not establish angular-cutoff convergence or solver readiness.

The saved five-test run passed with no failures. Across the uniform effective-vertex parameter directions, the largest endpoint-source finite-difference error was (5.78\times10^{-13}) absolute ((8.05\times10^{-12}) relative). Each of the \(\rho\)-tensor direct, Fock, and compensation-contact derivatives also matched its own energy-term difference. Two nonuniform signed perturbations for each effective vertex used epsilon=1e-3 with atol=2e-11 and rtol=2e-7; the maximum observed absolute/relative errors were 2.73e-14 and 4.74e-10. The local Dirac rearrangement expectation and invariant-current source agreed within 1.70e-21. With epsilon=1e-3, the independent rearrangement-energy direction differed by 3.05e-14 (relative 5.64e-8); its tolerance was atol=2e-11 and rtol=2e-5. Every particle-hole and full energy term changed by at most 3.34e-16 under time reversal. A compact machine-readable summary is [analytic_rearrangement_time_reversal_20261006.json](/home/ubuntu/rhf/DFT-RHF-NoRefresh/variants/blocked-rhfb/outputs/validation/analytic_rearrangement_time_reversal_20261006.json).
