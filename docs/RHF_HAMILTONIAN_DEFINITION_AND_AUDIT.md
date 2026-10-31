# RHF Hamiltonian definition and acceptance audit

## 1. Definition from the nuclear-physics reference

The production PKA1/PKO2 operator follows `RHF-e.pdf`, Eqs. (3.384)-(3.394).
For every target state `a=(n,kappa,m,q)`, the radial equation is

```text
integral dr' h_a(r,r') psi_a(r') = epsilon_a psi_a(r)
h_a = h_kin,a + h_D,q + h_E,a
```

The rest-mass-shifted kinetic matrix is

```text
h_kin = [[0,                 -D_backward + kappa/r],
         [D_forward+kappa/r, -2 M_q/(hbar*c)       ]].
```

`D_forward` is the seven-point asymmetric derivative and
`D_backward=-D_forward^T`. This is the discrete adjoint pair corresponding to
the two differential blocks in Eqs. (3.384), (3.390), and (3.393).

The local direct operator is assembled explicitly, not inferred from an RMF
Hamiltonian and not reconstructed from a quadrature gradient:

```text
h_D,q(r) = [[Sigma_plus,q(r), Sigma_T,q(r)],
            [Sigma_T,q(r),    Sigma_minus,q(r)]].
```

For PKA1, `Sigma_plus`, `Sigma_minus`, and `Sigma_T` contain the sigma,
omega, rho-vector, Coulomb, rho-tensor, both ordered rho-vector-tensor direct
terms, and the full density-dependent rearrangement potential. The latter is
the sum of direct and Fock rearrangement contractions from all five
density-dependent vertices: `g_sigma`, `g_omega`, `g_rho`, `f_pi`, and
`g_rhoT`.

The exchange operator is the nonlocal matrix of Eq. (3.394):

```text
h_E,a(r,r') = [[V_++, V_+-],
               [V_-+, V_--]].
```

It contains sigma-S, omega-V time/space, rho-V time/space, rho-T time/space,
both rho-VT orderings, pion-PV finite-range plus residual contact terms, and
Coulomb time/space exchange. It depends on target `kappa_a` and sums occupied
source states first over `n_b` inside each `kappa_b`, then over `kappa_b`.

## 2. Independent RHF implementation

The production path is contained in `dpl_rhf/rhf/`:

| Object | Implementation |
| --- | --- |
| PKA1/PKO2 constants and vertices | `parameters.py` |
| local RHF fields and direct energies | `direct.py:RHFDirectFunctional` |
| local/nonlocal densities | `density.py` |
| finite-range and contact kernels | `kernels.py` |
| complete RHF energy functional | `functional.py:SphericalRHFFunctional` |
| `h_kin`, explicit `h_D`, nonlocal `h_E` | `backend.py` |

No class from `dpl_rhf/functionals/pkdd_action.py` or any other RMF functional
is imported by this runtime. The direct RHF equations are implemented locally
because shared code previously allowed RMF-specific assumptions to enter the
PKA1 Hamiltonian.

The local sources are diagonal contractions of the same one-body density
matrix used by Fock exchange. This is a physical identity, not an RMF
approximation:

```text
R_(q,kappa)^(mu,mu')(r,r')
  = (2j+1) sum_n v_(n,kappa,q)^2
    u_(n,kappa,q)^mu(r) u_(n,kappa,q)^mu'(r')
```

## 3. Rearrangement treatment

Density dependence cannot be represented by a generic scalar correction. The
code evaluates the direct rearrangement channel by channel and evaluates the
Fock rearrangement using the outer-coordinate contractions corresponding to
`Core-1204/Meanfield.f90`, lines 236-289. In compact notation,

```text
Sigma_R^E = [dgsig Csig + dgome Come + dgrho (Crho + Cvt)
             + dfpio Cpio + dgrhoT (Ctt + Ctv)]
            / [4 pi r^2 rho_sat].
```

The full Fock functional derivative contains this local term. The backend
subtracts it once from that derivative because the same `Sigma_R^E` has
already been inserted once into `h_D`. This prevents both omission and double
counting.

## 4. Network variational constraint

The SiLU network produces occupied factors `Q`. QR enforces orthonormality and
the density matrix is `P=Q f Q^T`. Every forward pass evaluates the complete
RHF functional and embeds the projected Euler-Lagrange equation

```text
(I - Q Q^T) h_RHF[P] Q = 0
```

in the loss. No Dirac equation is solved, no Hamiltonian is diagonalized, and
no subspace is refreshed during training. Core-1204 is used only by separate
post-training validation and audit commands.

## 5. PKA1 O16 same-density audit

`scripts/audit_rhf_hamiltonian.py` inserts converged Core occupied states into
the Python functional as a read-only closure test. It does not train on Core
data. The current result is:

| Quantity | Python | Core-1204 | Absolute difference |
| --- | ---: | ---: | ---: |
| `E/A`, no CoM (MeV) | -7.3489426711 | -7.3489466818 | 0.0000040108 |
| kinetic energy (MeV) | 190.87813810 | 190.87818466 | 0.00004656 |
| direct energy (MeV) | -248.99369902 | -248.99383943 | 0.00014041 |
| Fock energy (MeV) | -59.46752182 | -59.46749214 | 0.00002967 |
| occupied-level maximum error (MeV) | - | - | 0.00004667 |
| local-potential maximum error (MeV) | - | - | 0.00004883 |
| total rearrangement maximum error (`fm^-1`) | - | - | 0.00000000508 |
| explicit `h_D` assembly error (MeV) | - | - | 0 |

All initialization, refresh, and training Dirac-solve counters are zero. The
coordinate-space stationarity residual is `0.17936 MeV`; unlike the energy and
Rayleigh closures, it compares the seven-point matrix representation against
Core's shooting representation and is therefore tracked separately.

## 6. Mandatory acceptance gates

The audit exits nonzero if any of these checks fail:

| Check | Default tolerance |
| --- | ---: |
| `E/A` | `1e-4 MeV` |
| kinetic/direct/Fock component | `1e-3 MeV` |
| occupied single-particle level | `1e-3 MeV` |
| local potential | `1e-3 MeV` |
| direct field | `1e-6 fm^-1` |
| total rearrangement | `1e-7 fm^-1` |
| explicit `h_D` matrix assembly | `1e-10 MeV` |

Run the audit with:

```bash
conda activate torch_env
python scripts/audit_pka1_parameters.py \
  --core-dir ../Core-1204 \
  --out outputs/audits/PKA1_parameter_and_coupling_closure.json
python scripts/audit_rhf_hamiltonian.py \
  --device cuda --core-dir ../Core-1204 \
  --out outputs/audits/PKA1_O16_rhf_native_full_hamiltonian_closure.json
```

Training is not accepted merely because it reaches a low energy. A checkpoint
must also satisfy the RHF projected-stationarity, branch, particle-number,
orthogonality, Hermiticity, and channel-completeness constraints.
