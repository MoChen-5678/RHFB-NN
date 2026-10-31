# Radial ADF stencil check

The reference is Y. Wang, X. Cao, J. Hu, and Y. Zhang, “Solving the relativistic Hartree-Bogoliubov equation with the finite-difference method,” *Chinese Physics C* **49**, 014106 (2025), DOI: [10.1088/1674-1137/ad806f](https://doi.org/10.1088/1674-1137/ad806f). The local copy is `/home/ubuntu/rhf/DFT-RHF-NoRefresh/张颖-Wang_2025_Chinese_Phys._C_49_014106.pdf`.

On PDF page 4, Eqs. (13) and (14), the article gives the five-point forward and backward ADF stencils, each fourth-order accurate:

\[
 f'(r_i)_{\rm fwd} = \frac{-25f_i+48f_{i+1}-36f_{i+2}+16f_{i+3}-3f_{i+4}}{12h}+O(h^4),
\]
\[
 f'(r_i)_{\rm bwd} = \frac{25f_i-48f_{i-1}+36f_{i-2}-16f_{i-3}+3f_{i-4}}{12h}+O(h^4).
\]

PDF page 4 also explains why the same one-sided stencil cannot simply be assigned to both radial Dirac components: the resulting Hamiltonian may be non-Hermitian. The article’s parity prescription is described on PDF page 9 (article page 014106-8): use the forward ADF for odd reduced-radial extensions and the backward ADF for even ones. Since the large and small radial components have opposite extension parity, their derivative assignments alternate. The article says this prescription guarantees Hermiticity. In this variant, the backward derivative matrix is constructed as the negative transpose of the forward matrix; tests verify the printed Eq. (14) coefficients in interior rows and verify Hermiticity of the assembled radial Dirac matrix.

The current implementation supports both this five-point stencil (`order=5`) and a seven-point, sixth-order forward extension (`order=7`, the default). The article explicitly prints the five-point formulas; its results section uses 5PADF. It does not print a seven-point formula. The seven-point option is checked independently by its polynomial moments and by smooth-function convergence, and remains unchanged by this audit.

## Boundary rows remain unverified against the paper

The article states on PDF page 4 that the radial functions vanish at `r=0` and outside the box (`r > R_box`). It does not print the complete finite-difference boundary rows or specify whether negative-radius ghost values are parity-reflected when a backward stencil reaches the origin. The production forward matrix drops entries beyond the last interior point, corresponding to the stated zero exterior values, and defines the backward matrix globally by `-D_forward.T`. At the origin it does not add parity-reflected negative-radius ghost points. Therefore the interior stencil, parity assignment, and Hermiticity are verified, but equivalence of the production boundary rows—especially the origin row—to the paper’s matrix construction is not established here.

The focused tests in `dpl_rhf/tests/test_zhang_adf.py` compare `order=5` directly with the printed coefficients, verify that `-D_forward.T` has the backward Eq. (14) stencil away from boundaries, check fourth- and sixth-order interior convergence on `exp(r)`, and confirm the upper-component parity switch and Dirac-matrix Hermiticity for representative `s_{1/2}` and `p_{1/2}` channels. These tests address the radial spatial derivative only; the separate centered finite difference in the PRL test is a parameter-gradient check and is unchanged.
