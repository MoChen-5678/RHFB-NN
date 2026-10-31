# Symmetric product-integration prototype for the bare l=0 Yukawa kernel

## Literature basis and boundaries

The physical kernel is fixed by the RHFB paper, not by this prototype: Geng and Long state that after neglecting retardation the meson propagator is `D_phi(r-r')=exp[-m_phi |r-r'|]/(4*pi*|r-r'|)`, with `D_A=1/(4*pi*|r-r'|)` for the photon (Eq. 9, PDF p. 4). With fm coordinates the inverse range is `mu=m_phi/(hbar*c)`. [Geng and Long, arXiv:2202.00482](https://arxiv.org/pdf/2202.00482).

The RHF-e manuscript expands this kernel in spherical harmonics and modified-Bessel radial multipoles (Chapter 3.1, Eqs. 3.15-3.16). Its numerical section later documents six-point source interpolation and composite Simpson integration, with folded radial weights in Eqs. 3.400-3.407 (printed pp. 57-58). That published algorithm is a one-coordinate source interpolation/integration prescription. It does not specify the symmetric P1 cell-pair matrix used here. [RHF-e manuscript, local copy](/home/ubuntu/rhf/DFT-RHF-NoRefresh/variants/blocked-rhfb/RHF-e.pdf).

Pöschl, Vretenar, and Ring describe arbitrary-order finite elements on an adaptive radial mesh for coordinate-space Dirac-Hartree-Bogoliubov equations coupled to local Klein-Gordon meson-field equations (Secs. 2-3). This establishes finite elements as a published radial discretization approach in spherical RHB. Their method solves local field equations and does not prescribe product quadrature for the nonlocal RHF Fock energy considered here. [Pöschl, Vretenar, and Ring, arXiv:nucl-th/9706031](https://arxiv.org/pdf/nucl-th/9706031).

## Prototype and equivalence conditions

For a spherical density, angular integration of the paper's three-dimensional propagator gives the bare l=0 interaction form

    I_mu = 4*pi int_0^R dr int_0^R dr'
           f(r) f(r') sinh(mu*r_<) exp(-mu*r_>) / mu,
    f(r)=r*rho(r),

with the kernel factor replaced by `r_<` at `mu=0`. Thus the prototype changes only the numerical representation of the same fixed-box, constant-coupling continuum bilinear. It includes no density-dependent vertex, fitted rescaling, contact subtraction, or derivative interaction.

It represents `f(r)=r*rho(r)` by continuous piecewise-linear finite-element hats at radial nodes from 0 to R. Pair integrals are evaluated cell-by-cell with order-8 Gauss quadrature. Off-diagonal cell blocks are entered with their transpose; diagonal cell squares are split into both ordered triangles with a Duffy map. This makes the discrete matrix A symmetric and gives `I=c^T A c` for nodal coefficients c. The Hartree energy is `E=I/2`, so its coefficient derivative is `dE/dc=A c`. The source is sampled from the original analytic Gaussian and is not renormalized. The finite-element source is an approximation to the same continuum source, with convergence as h is reduced.

The code lives in [symmetric_yukawa_product_integration.py](/home/ubuntu/rhf/DFT-RHF-NoRefresh/variants/blocked-rhfb/dpl_rhf/rhfb/symmetric_yukawa_product_integration.py); [its test](/home/ubuntu/rhf/DFT-RHF-NoRefresh/variants/blocked-rhfb/dpl_rhf/rhfb/tests/test_symmetric_yukawa_product_integration.py) compares against both the existing Green application and the independent Fourier/finite-box radial references. It also checks matrix symmetry and the directional derivative of the product-integration energy.

## Bounded comparison

For the fixed `N=1`, `b=2 fm`, `R=12 fm` source, the P1 source norm remains within `3e-15` of N on all tested meshes without rescaling. Against the same finite-box Gaussian reference, the relative errors are:

| Channel | Original Green h=.4 | P1 h=.4 | Original h=.2 | P1 h=.2 | Original h=.1 | P1 h=.1 |
|---|---:|---:|---:|---:|---:|---:|
| sigma | 9.003% | 1.844% | 2.268% | 0.4660% | 0.5682% | 0.1168% |
| omega | 21.20% | 1.914% | 5.442% | 0.4839% | 1.370% | 0.1213% |
| pion | 1.366% | 1.399% | 0.3403% | 0.3528% | 0.08500% | 0.08838% |
| Coulomb | 0.3353% | 0.6607% | 0.08346% | 0.1663% | 0.02084% | 0.04164% |

For sigma the error drops by a factor of about 4.87, and for omega by about 11.1-11.3 at each tested h. This shows the symmetric P1 product integral can substantially reduce the observed short-range error for this smooth Gaussian. It is not uniformly better: pion is about 2-4% worse in relative error, and Coulomb about twice as large. The P1 approximation changes the discretization error coefficient, while both sequences converge toward the same continuum integral. Do not infer a universal improvement from this case.

The assembled matrices have exactly zero measured asymmetry in double precision for this run. Directional finite differences of `E=0.5 c^T A c` agree with `dE/dc=A c` with relative errors at most `4.6e-10`. The complete per-channel/mesh values, norms, improvement ratios, matrix asymmetry, and derivative residuals are in the [JSON artifact](/home/ubuntu/rhf/DFT-RHF-NoRefresh/variants/blocked-rhfb/outputs/validation/green_symmetric_product_integration_20261006.json), with the [pytest XML](/home/ubuntu/rhf/DFT-RHF-NoRefresh/variants/blocked-rhfb/outputs/validation/green_symmetric_product_integration_20261006.xml). The module and test source snapshots are stored under `outputs/validation/source_snapshots/`.

This prototype is limited to a spherical l=0 bare kernel and an analytic smooth source. It does not establish nonspherical multipole accuracy, density-dependent rearrangement consistency, time-odd channels, derivative Green/Hessian accuracy, coincident/contact normalization, or self-consistent solver behavior. It is independent research code only; neither `spatial.py` nor the production functional was modified.
