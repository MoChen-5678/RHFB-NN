# Independent PKA1 time-odd derivative-vertex check

This note reports a bounded algebra check of the production direct derivative channels on a complex, time-reversal-breaking normal density. It checks their Lorentz signs and two-endpoint derivative placement against explicit coordinate-pair Gamma-matrix sums. It does not certify the full PKA1 functional or solver.

## Source and convention

The original PKA1 paper starts from the covariant rho-meson Lagrangian in Eq. (1), derives its field equation and static Hamiltonian in Eqs. (2)–(5), and writes the effective V, T, and mixed VT vertices in Eqs. (8a)–(8c), arXiv PDF pp. 3–4. Eq. (8b) has two spatial derivatives and a contracted Lorentz tensor index; Eq. (8c) explicitly contains both ordered vector-at-one-end/tensor-at-the-other terms. The authors state in Eq. (5) that the meson energy transfer is neglected, i.e. retardation is omitted. This is a static-interaction approximation, not a restriction to time-even baryon densities: the vertices retain the four-vector and Lorentz-tensor bilinears. The cited paper does not report an odd blocked calculation validating those time-odd channels numerically. [Long et al., *Phys. Rev. C* 76, 034314 (2007), original PKA1 paper](https://arxiv.org/pdf/0706.3497).

The 2025 RHF-e manuscript makes derivative placement explicit in Eq. (1.25), PDF p. 6 (printed p. 4): rho-T, rho-VT, and pion-PV carry spatial derivatives, and the accompanying note says the differential operator acts only on the propagator. In particular, Eq. (1.25d) gives the pion-PV product with an overall minus sign and a spatial axial bilinear. [RHF-e manuscript, local copy](/home/ubuntu/rhf/DFT-RHF-NoRefresh/variants/blocked-rhfb/RHF-e.pdf).

This implementation uses metric (+---), Dirac-representation gamma matrices, and Hamiltonian-space operators O=gamma^0 Gamma, since sampled spinors enter as q-dagger O q. Define

    J^mu(x) = sum_s t_s Tr[rho_s q_s-dagger(x) gamma^0 gamma^mu q_s(x)]
    T^(i mu)(x) = sum_s t_s Tr[rho_s q_s-dagger(x) gamma^0 sigma^(i mu) q_s(x)]
    A^i(x) = sum_s t_s Tr[rho_s q_s-dagger(x) gamma^0 gamma_5 gamma^i q_s(x)]

with t_n=+1, t_p=-1, sigma^(mu nu)=i[gamma^mu,gamma^nu]/2, and i=1,2,3. The explicit reference forms the raw endpoint kernel derivatives K_ij(x,y)=partial_(x_i) partial_(y_j) D_rho(x,y), K^y_i(x,y)=partial_(y_i)D_rho(x,y), and K^x_i(x,y)=partial_(x_i)D_rho(x,y), then contracts them directly before any divergence or direct-energy helper is called.

Writing the density-dependent vertices at the two endpoints as g_T(x), g_rho(x), and g_pi(x)=f_pi(x) hbar-c/m_pi, the direct checks are

    E_rhoT^H = (hbar-c/2) sum_mu g^(mu mu) int dx dy g_T(x) T^(i mu)(x)
                 K_ij(x,y) g_T(y) T^(j mu)(y)

    C_VT = hbar-c sum_mu g^(mu mu) int dx dy g_rho(x) J^mu(x)
                 K^y_i(x,y) g_T(y) T^(i mu)(y)
    E_rhoVT^H = E_rhoTV^H = C_VT/2

    E_piPV^H = -(hbar-c/2) int dx dy g_pi(x) A^i(x)
                  partial_(x_i)partial_(y_j)D_pi(x,y) g_pi(y) A^j(y)

The equality of the two mixed Hartree orientations follows from exchanging endpoints of the symmetric static propagator. The minus sign in the pion energy follows Eq. (1.25d). Because only the propagator is differentiated, the coupling arrays remain endpoint multipliers; no derivative of g_i(x) is inserted into these bare vertices.

## Bounded result

The independent construction is in [general_current_vertex_reference.py](/home/ubuntu/rhf/DFT-RHF-NoRefresh/variants/blocked-rhfb/dpl_rhf/rhfb/general_current_vertex_reference.py); one bounded test is in [test_general_current_vertex_reference.py](/home/ubuntu/rhf/DFT-RHF-NoRefresh/variants/blocked-rhfb/dpl_rhf/rhfb/tests/test_general_current_vertex_reference.py). It uses kappa_max=1, lmax=2, a five-point radial stencil, 144 coordinate points, complex two-state neutron/proton factors, and Hermitian active densities restricted to the complete kappa=-1 s1/2 multiplet. This preserves a definite Dirac parity while the complex active density breaks time reversal. It is a small algebra stress input, not a physical nucleus. Its maximum spatial baryon current is 0.06877 in sampled units, maximum spatial axial current is 1.03618, and minimum invariant baryon density is positive (0.02594 fm^-3). Thus this gives a concrete parity-conserving case with nonzero time-odd currents and pion axial source.

| Production term | Explicit endpoint (MeV) | Production (MeV) | Absolute difference (MeV) |
|---|---:|---:|---:|
| rho-T direct, sum over all four Lorentz components | -3.97217406211692 | -3.97217406211692 | 0 |
| rho-VT direct | 0.03915699002450059 | 0.03915699002450060 | 6.94e-18 |
| rho-TV direct | 0.03915699002450059 | 0.03915699002450059 | 0 |
| pion-PV direct | -2.53921674437774 | -2.53921674437774 | 4.44e-16 |
| rho-T Fock | 27.17625527700950 | 27.17625527700950 | 0 |
| rho-VT Fock | -0.0273571531524728 | -0.0273571531524728 | 3.47e-18 |
| rho-TV Fock | -0.0273571531524728 | -0.0273571531524728 | 1.04e-17 |
| pion-PV Fock | 19.04968884228187 | 19.04968884228187 | 0 |

The independent rho-T direct Lorentz components (mu=0,1,2,3) are +0.03027742, -0.72364227, -1.82479153, and -1.45401768 MeV. All three spatial contributions are nonzero and enter with (+---) metric signs. The mixed direct orientations agree component-by-component to roundoff, including nonzero spatial-vector contributions. Machine-readable values and pytest XML are saved in [JSON](/home/ubuntu/rhf/DFT-RHF-NoRefresh/variants/blocked-rhfb/outputs/validation/pka1_current_vertex_explicit_gamma_20261006.json) and [XML](/home/ubuntu/rhf/DFT-RHF-NoRefresh/variants/blocked-rhfb/outputs/validation/pka1_current_vertex_explicit_gamma_20261006.xml). The one bounded test passed.

## Scope and limits

For the exchange check, the normal one-body density in coordinate-spinor space is R_s(x,y)=Q_s(x) rho_s Q_s(y)^dagger. With B_st(x)=Q_s(x)^dagger O_x Q_t(x), Wick's exchange contraction is Tr_active[rho_s B_st(x) rho_t B_ts(y)]; in components it is sum_ab [rho_s B_st(x) rho_t]_ab [B_ts(y)]_ba. Thus the reverse endpoint is B_ts with its indices b,a; one must not take an extra elementwise conjugate or transpose after forming that reversed pair. The implementation constructs it directly as q_t^dagger O_y q_s, while the left transition is (Q_s rho_s)^dagger O_x (Q_t rho_t). This is valid for the complex Hermitian test density, not just real orbitals.

The charge-basis matrix elements of tau_1 dot tau_2 are 1 for ordered nn and pp pairs, and 2 for each ordered np and pn exchange pair. The reference enumerates all four ordered species pairs with these coefficients. VT means vector at x, tensor at y with a single spatial derivative at y; TV means tensor at x, vector at y with a single derivative at x. Each ordered channel has its own -1/2 Wick prefactor and Lorentz metric sign. The 3x mixed discrepancy from an earlier run was a reference-code loop nesting error: its mixed spatial-axis sum was repeated once per rho-T spatial axis. The failure is preserved in [the threefold-failure JSON](/home/ubuntu/rhf/DFT-RHF-NoRefresh/variants/blocked-rhfb/outputs/validation/pka1_current_vertex_fock_threefold_reference_failure_20261006.json) and its pre-correction source snapshot; it was not evidence of a production factor error. The initial endpoint-transpose failure is separately retained in [its diagnostic JSON](/home/ubuntu/rhf/DFT-RHF-NoRefresh/variants/blocked-rhfb/outputs/validation/pka1_current_vertex_fock_initial_failure_20261006.json). The corrected raw traces are stored by ordered species pair and by Lorentz/spatial indices in the canonical JSON artifact.

The agreement independently checks direct rho-T Lorentz metric signs, the equal split of the two ordered direct rho-VT/rho-TV endpoints, the negative pion-PV direct sign with a nonzero spatial axial source, and the discrete endpoint algebra for all four listed Fock terms. The explicit exchange contractions begin from endpoint Gamma matrices and raw differentiated kernel blocks, rather than production current construction, source divergence, contact expression, or production energy contraction. Production and reference do share the projected Green kernel, so this is not an independent propagator validation.

The raw projected Green kernel is shared with production; this test therefore does not validate its continuum approximation, coincident/distributional limit, or angular convergence. The reference's discrete derivative kernel is assembled independently from the production weak-source contractions, but both use the same declared projected Green operator and spatial derivative matrices. Nor does this establish that the original even-even PKA1 fit validated a blocked time-odd extension. The original paper's covariant Lagrangian permits these bilinears in the static approximation; applying that general vertex to a time-reversal-breaking density is a theoretical extension, not a published odd-PKA1 numerical benchmark. No conclusion about physical readiness or the O17 result follows from this small test.
