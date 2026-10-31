# PKO2 RHF formula and implementation audit

## Scope

This audit checks the isolated spherical PKO2 density-matrix RHF solver against
`RHF-e.pdf` and the physical formulas used by `Core-1204`.  Core wavefunctions
and energies are used only by the read-only validation script.  They do not
enter the neural-network loss.

PKO2 has nonzero sigma-scalar, omega-vector, rho-vector and electromagnetic
vertices.  Its pion pseudovector and rho-tensor couplings are exactly zero.
Consequently, omitting pion and rho-tensor terms is correct for PKO2, but would
not be correct for PKO1, PKO3 or PKA1.  The command-line interface rejects those
parameter sets until their complete functionals are implemented.

## State and density indices

The complete occupied single-particle label is

```text
a = (q_a, kappa_a, n_a, m_a),    q_a in {neutron, proton}.
```

Spherical symmetry removes the explicit magnetic-substate loop and supplies
the factor `2j_a+1`.  It does not remove `n_a` or `kappa_a`.  For each fixed
species and kappa block the code constructs

```text
rho_kappa^(mu,nu)(r,r')
  = (2j_kappa+1) sum_n v_(n,kappa)^2
    u_(n,kappa)^mu(r) u_(n,kappa)^nu(r').
```

The implementation order is therefore:

1. retain every occupied `(n,kappa,q)` neural orbital and occupation;
2. sum `n` inside a fixed `(kappa,q)` nonlocal-density block;
3. form its four Dirac components `GG`, `GF`, `FG` and `FF`;
4. for each target `kappa_a`, sum every source block `kappa_b` in each Fock
   channel;
5. apply the resulting target-block operator separately to every `n_a` column.

This is the same nesting as `Core-1204/Density.f90:89-101` followed by
`Core-1204/Meanfield.f90:92-160`.  The angular propagator depends on
`(kappa_a,kappa_b)` and not on `n`; the `n` dependence has already been summed
into the source density.  `NonlocalDensityBlock.principals`, `occupations` and
`state_matrix()` keep that sum inspectable instead of discarding its labels.

## Hartree and Fock density relationship

There are exactly four local PKO2 Hartree source densities:

```text
rho_s_n, rho_s_p, rho_v_n, rho_v_p.
```

They are not separately predicted quantities.  They are diagonal contractions
of the same nonlocal density used by Fock exchange,

```text
rho_s_q(r) = sum_kappa [rho_kappa^GG(r,r)-rho_kappa^FF(r,r)]/(4 pi r^2),
rho_v_q(r) = sum_kappa [rho_kappa^GG(r,r)+rho_kappa^FF(r,r)]/(4 pi r^2).
```

Thus Hartree reuses the Fock density without replacing exchange by a local
approximation.  The four full nonlocal Fock components remain independent.

## Channel checklist

| Item | Formula retained | Result |
| --- | --- | --- |
| sigma Hartree/Fock | scalar signs `(+,-) x (+,-)` | checked |
| omega Hartree/Fock | time and spatial vector contractions | checked |
| rho Hartree/Fock | time and spatial contractions; same-isospin factor 1, unlike factor 2 | checked |
| Coulomb Hartree/Fock | proton-only vertex `e^2=4 pi alpha`; time and spatial parts | checked |
| density dependence | `g_sigma(rho_b)`, `g_omega(rho_b)`, `g_rho(rho_b)` remain in autograd graph | checked |
| direct rearrangement | derivative of all Hartree density-dependent vertices | checked |
| exchange rearrangement | derivative of complete `E_fock` with respect to baryon density | checked |
| radial propagator | Yukawa/Coulomb multipoles with six-point half-mesh folding | checked against Core |
| angular factors | standard Clebsch-Gordan squares and parity selection | checked against Core C3J/C3L/C3S |
| variational Hamiltonian | `h_kappa=(2j+1)^(-1) dE/dP_kappa` | checked by finite difference |
| no-sea constraint | fixed positive-energy subspace plus exact QR orthonormalization | checked |
| center-of-mass correction | deliberately absent from both compared energies | checked |

The network uses SiLU throughout.  Particle number, Hermiticity, occupations,
orthonormality and no-sea membership are construction constraints, not soft
penalties.  The projected Euler-Lagrange residual
`(I-QQ^T)hQ` is included in the loss.

## Omission found and repaired

The previous spatial-vector Fock kernel used

```text
L_max = |kappa_a| + |kappa_b| - 1,
```

which is the scalar/time-component bound.  Spatial vector terms couple upper
and lower orbital angular momenta, so their strict bound is

```text
L_max_space = max(l_a^+ + l_b^-, l_a^- + l_b^+).
```

For example, `kappa_a=-2`, `kappa_b=+1` requires the `L=3` contribution; the
old bound stopped at `L=2`.  `PKO2AngularKernels.maximum_space_multipole()` now
uses the cross-component bound.  A regression test explicitly reconstructs
the `L=1` and `L=3` terms.

No extra `(2L+1)` factor was added.  A direct call to the Core shared library
showed that Core stores `C3J=|CG|^2/(2L+1)` and later multiplies by `(2L+1)`.
The two factors cancel, matching the standard Clebsch-Gordan square used by
the Python kernel.

## Same-density closure test

`scripts/audit_rhf_same_density.py` imports the converged occupied orbitals
from Core and evaluates the Python functional on exactly that density.  This
separates errors in the functional from errors in neural optimization.

| Nucleus | Quantity | Core (MeV) | Python (MeV) | Python-Core (MeV) |
| --- | ---: | ---: | ---: | ---: |
| O-16 | direct | -405.45703284 | -405.45945927 | -0.00242642 |
| O-16 | Fock | 77.80709648 | 77.80714060 | +0.00004413 |
| Ca-40 | direct | -1079.40059315 | -1079.39316390 | +0.00742925 |
| Ca-40 | Fock | 224.57736213 | 224.57748104 | +0.00011891 |

Ca-40 includes multiple principal states within a kappa block, so this also
tests the required `sum_n` before `sum_kappa_b`.  Before repairing the spatial
vector multipole bound, the O-16 same-density Fock difference was
`-2.6381 MeV`; afterward it is `+4.4e-5 MeV`.

The remaining rearrangement discrepancy is concentrated at the first interior
mesh point: about `1.26e-3 fm^-1` for O-16 and `2.55e-3 fm^-1` for Ca-40.  It is
an origin-boundary discrete-adjoint difference between Core's explicit source
folding and the energy derivative.  The training Hamiltonian uses the exact
autograd derivative of the discretized total energy, so its variational energy
and gradient remain internally consistent.  This boundary discrepancy is
reported rather than hidden.

## Neural 500-epoch regression

Command:

```bash
python -m dpl_rhf.cli.rhf_train --model PKO2 --z 8 --n 8 --a 16 \
  --epochs 500 --device cuda \
  --out outputs/rhf/PKO2/O16_fock_multipole_fix_500
```

| Metric | Neural variational minimum | Core no-CoM | Relative difference |
| --- | ---: | ---: | ---: |
| total energy (MeV) | -116.50568034 | -116.19235238 | -0.2697% |
| energy/A (MeV) | -7.28160502 | -7.26202202 | -0.2697% |
| neutron rms (fm) | 2.57153356 | 2.57086268 | +0.0261% |
| proton rms (fm) | 2.60163293 | 2.59488748 | +0.2600% |
| matter rms (fm) | 2.58662703 | 2.58290302 | +0.1442% |
| charge radius (fm) | 2.72004446 | 2.71359338 | +0.2377% |

The energy decomposition at the variational minimum is:

| Channel (MeV) | Neural | Core | Neural-Core |
| --- | ---: | ---: | ---: |
| kinetic | 213.927207 | 214.505572 | -0.578365 |
| direct total | -411.383716 | -412.481558 | +1.097842 |
| sigma Fock | 353.803509 | 355.287527 | -1.484018 |
| omega Fock | -232.379805 | -233.085660 | +0.705855 |
| rho-vector Fock | -37.483542 | -37.429230 | -0.054312 |
| Coulomb Fock | -2.989333 | -2.989002 | -0.000331 |
| Fock total | 80.950829 | 81.783634 | -0.832805 |

The same-density closure above shows that these finite trained-density channel
differences are optimization/density differences, not missing exchange
formulas.

Epoch 251 is the point numerically closest to the Core finite-grid SCF energy:
`E/A=-7.26199280 MeV`, a difference of `2.92e-5 MeV/A`.  Epoch 500 is the
correct variational-best checkpoint because the configured objective continues
to decrease.  These are different selection criteria and are reported
separately.

## Reproducibility checks

```bash
python -m pytest -q dpl_rhf/tests
python scripts/audit_rhf_same_density.py --z 8 --n 8 --a 16
python scripts/audit_rhf_same_density.py --z 20 --n 20 --a 40
```

Generated validation artifacts are in
`outputs/rhf/PKO2/O16_fock_multipole_fix_500/`, including the complete Fock
Hamiltonian matrices, local fields, density profiles, comparison JSON and
diagnostic plot.
