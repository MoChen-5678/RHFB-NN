# Bare meson propagator normalization check

This is a bounded comparison of the projected, nondifferentiated Green operator against an independent spherical Gaussian source. It checks the bare Yukawa/Coulomb kernel normalization and radial mesh quadrature. It does not validate any density-dependent vertex, derivative vertex, or contact compensation.

## What the papers state

Geng and Long's RHFB paper starts from the static meson-exchange Hamiltonian and states that retardation is neglected. Its Eq. (9), PDF p. 4, gives the coordinate-space kernel

    D_phi(r-r') = exp[-m_phi |r-r'|] / (4 pi |r-r'|)
    D_A(r-r')   = 1 / (4 pi |r-r'|).

The paper uses natural units. In this code's fm coordinates, the inverse range is therefore `mu = m_phi / (hbar*c)` in fm^-1. The photon is the `mu=0` Coulomb limit. [Geng and Long, arXiv:2202.00482, Eq. (9)](https://arxiv.org/pdf/2202.00482).

The RHF-e manuscript's Chapter 3.1, Eqs. (3.15)-(3.16), PDF p. 23 (printed p. 21), expands that same Yukawa kernel in spherical harmonics and modified-Bessel radial multipoles. The l=0 production radial factor is the corresponding angular projection. The comparison below tests it against the three-dimensional kernel above; the Gaussian reference does not call the production radial propagator. The five-point metadata is retained to match the requested basis setup, but no derivative is applied here. [RHF-e manuscript, local copy](/home/ubuntu/rhf/DFT-RHF-NoRefresh/variants/blocked-rhfb/RHF-e.pdf).

## Independent Gaussian reference

Take a fixed, normalized spherical source

    rho(r) = N / (pi^(3/2) b^3) * exp(-r^2/b^2),
    rho_tilde(q) = N * exp(-b^2 q^2/4).

Fourier transforming the paper's `D_mu(r)=exp(-mu*r)/(4*pi*r)` gives `D_tilde_mu(q)=1/(q^2+mu^2)`. Thus the unhalved bare interaction integral, derived here rather than quoted from either paper, is

    I_mu = integral d^3r d^3r' rho(r) D_mu(r-r') rho(r')
         = N^2/(2*pi^2) integral_0^infinity dq q^2 exp(-a q^2)/(q^2+mu^2),
    a = b^2/2.

Evaluating this integral gives

    I_mu = N^2/(2*pi^2) * [sqrt(pi)/(2 sqrt(a))
                            - pi*mu/2 * erfcx(mu*sqrt(a))],  mu > 0,
    I_0  = N^2/(2*pi^2) * sqrt(pi)/(2 sqrt(a)).

The separate SciPy one-dimensional Fourier quadrature evaluates the integral expression directly. A second independent radial reference analytically integrates the angles, truncates both source coordinates at the same fixed box radius R, and numerically integrates the remaining radial variables. This separates the negligible Gaussian box tail from the mesh quadrature error. For comparison with a functional Hartree term, the corresponding interaction energy is `I_mu/2`; no coupling constants are included.

## Bounded grid result

The source uses `N=1`, `b=2 fm`, and a fixed `R=12 fm` box. The Green angular cutoff is `lmax=0`; the spherical source is exactly represented by this channel. Radial spacings are `0.8, 0.4, 0.2, 0.1, 0.05 fm`, and the comparison uses the PKA1 sigma, omega, pion masses plus the massless Coulomb kernel. The basis carries `derivative_order=5` metadata, but the bare Green action does not use its derivative matrices.

Across all four channels, the closed Fourier formula and independent Fourier quadrature agree within `1.2e-17 fm^-1`. The independent finite-box radial values differ from the infinite-space formula by less than `3e-14` relative at `R=12 fm`, so the table reports mesh error against the finite-box reference:

The sampled source norm `sum_p w_p rho_p` stays within `8e-15` of `N=1` on every grid. Thus the large coarse sigma/omega interaction errors are not caused by loss of total source normalization; they arise from under-resolving the short-range convolution.

| Channel | Relative error at h=0.8 fm | h=0.2 fm | h=0.05 fm |
|---|---:|---:|---:|
| sigma | 34.88% | 2.268% | 0.1421% |
| omega | 76.98% | 5.442% | 0.3430% |
| pion | 5.542% | 0.3403% | 0.02125% |
| Coulomb | 1.366% | 0.08346% | 0.005209% |

Halving h reduces the observed integration error by approximately four in this bounded smooth-source case, consistent with the interior radial rectangle/trapezoid sampling dominating the bare-kernel integral. This is a numerical inference from this test, not a continuum-order claim for derivative/contact kernels. At the coarsest spacing, especially for short-range sigma and omega kernels, the projected Green action is substantially under-resolved. Fine-grid agreement does not certify accuracy for localized, nonspherical, density-dependent sources.

The full values by channel and spacing, radial point counts, Fourier check, finite-box check, and error definitions are recorded in the [JSON artifact](/home/ubuntu/rhf/DFT-RHF-NoRefresh/variants/blocked-rhfb/outputs/validation/green_gaussian_propagator_20261006.json), with test output in [XML](/home/ubuntu/rhf/DFT-RHF-NoRefresh/variants/blocked-rhfb/outputs/validation/green_gaussian_propagator_20261006.xml). The independent formula is in [green_propagator_reference.py](/home/ubuntu/rhf/DFT-RHF-NoRefresh/variants/blocked-rhfb/dpl_rhf/rhfb/green_propagator_reference.py), and the comparison is in [test_green_propagator_reference.py](/home/ubuntu/rhf/DFT-RHF-NoRefresh/variants/blocked-rhfb/dpl_rhf/rhfb/tests/test_green_propagator_reference.py). Source snapshots are kept under `outputs/validation/source_snapshots/`.

This is a bare, constant-coupling, l=0 source check. It does not establish contact/distributional normalization, derivative-kernel accuracy, rho-T or VT/TV closure, pion-PV derivative closure, or angular convergence beyond l=0. No production Green implementation was modified.
