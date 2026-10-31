# Strict PKO2/PKA1 density-matrix RHF implementation

## Status and isolation

The validated RMF solver remains available through every existing
`dpl-rmf-*` command.  RHF is implemented in the separate `dpl_rhf/rhf/`
package and uses separate `dpl-rhf-*` commands and output directories.

PKO2 and PKA1 are implemented as separate complete RHF parameterizations.
PKO2 is not a channel-truncated approximation: that parameter set itself has
`f_pi=0` and `f_rho=0`.
Consequently its complete finite-nucleus functional contains:

- sigma-scalar Hartree and Fock terms;
- omega-vector Hartree, time-like Fock and space-like Fock terms;
- rho-vector Hartree, time-like Fock and space-like Fock terms;
- direct and exchange electromagnetic terms;
- the density dependence of every meson vertex and the resulting variational
  rearrangement contribution.

There is no local-density replacement for exchange, no fitted Fock potential,
no low-rank propagator and no Core-1204 quantity in the training loss.

PKA1 is also complete.  In addition to the common PKO2 channel structure it
contains density-dependent pion pseudovector and rho-tensor vertices.  The
implementation retains pion finite-range and contact subtraction terms, pure
rho-tensor time/space terms, both rho vector-tensor vertex orderings, the
rho-tensor Hartree source, and the finite-range/zero-range compensation terms.
The formula-by-formula channel inventory, first-forward gradient audit and
PKA1 O16/Ca40 same-density closure are recorded in
`PKA1_COMPLETE_CHANNEL_AUDIT.md`.

## Formula-to-code mapping

The reference is `RHF-e.pdf`, especially Chapters 1 and 3.

| Physics object | Reference | Implementation |
| --- | --- | --- |
| PKO2/PKA1 density-dependent vertices | Eqs. (1.41)-(1.44) | `parameters.py`, `direct.py:RHFDirectFunctional.couplings` |
| Yukawa/Coulomb multipoles | Eqs. (3.15)-(3.16) | `kernels.py:SphericalPropagators` |
| Six-point source interpolation | Eqs. (3.400)-(3.402) | `_fold_half_mesh` |
| Split extended-Simpson integration | Eqs. (3.397)-(3.409) | `folded_yukawa`, `folded_coulomb` |
| Nonlocal `GG/GF/FG/FF` density | Eq. (3.94) | `density.py:nonlocal_density_from_block` |
| Local scalar/vector/tensor density | Eqs. (3.5), (3.57) | `density.py:local_source_densities` |
| sigma exchange | Eqs. (3.324)-(3.328) | `functional.py:_scalar_exchange` |
| omega exchange, including space part | Eqs. (3.329)-(3.333) | `functional.py:_vector_exchange` |
| rho-vector exchange and isospin factor | Eqs. (2.12), (3.334)-(3.337) | `_isospin_factor`, `_vector_exchange` |
| Full integral-differential RHF operator | Eqs. (3.384)-(3.394) | `backend.py:evaluate` |
| Variational principle | Eqs. (3.299)-(3.305) | total-energy and projected-stationarity loss |

The local density is the diagonal contraction of the same one-body density
matrix used by exchange.  For each species and spherical `kappa` block,

```text
P = Q n Q^T
R^{mu,mu'}(r,r') = (2j+1) P^{mu,mu'}(r,r') / h
```

`Q` is the neural output after QR. This enforces Hermiticity, occupation
spectrum, neutron/proton number, and orthonormality before the energy is
evaluated. The no-sea branch is enforced variationally; no positive-energy
subspace projection or refresh is performed in this version.

The complete occupied-state index is `a=(n_a,kappa_a,m_a,q_a)`.  Principal
quantum numbers and occupations remain attached to every column of `Q`.  After
the spherical magnetic-substate sum, Core-1204 `Density.f90` and this program
construct

```text
rho_(kappa,q)^(mu,mu')(r,r')
  = (2j_kappa+1) sum_n v_(n,kappa,q)^2
    u_(n,kappa,q)^mu(r) u_(n,kappa,q)^mu'(r').
```

Thus the order is: sum occupied `n` inside each fixed `(kappa,q)` block, then
sum source blocks `kappa_b` when constructing the Fock field for target
`kappa_a`.  The target operator is applied separately to every `n_a` in that
target block.  The meson angular kernel itself depends on
`(kappa_a,kappa_b)`, not on `n`.  The four PKO2 Hartree
sources `rho_s_n`, `rho_s_p`, `rho_v_n`, and `rho_v_p` are diagonal
contractions of these same Fock density blocks.  Hartree and Fock do not keep
independent copies of the physical density.

The scalar/time and spatial-vector angular reductions do not have the same
upper multipole bound.  The spatial terms couple upper and lower orbital
angular momenta and therefore use

```text
L_max_space = max(l_a^+ + l_b^-, l_a^- + l_b^+).
```

This retains, for example, the required `L=3` term for
`(kappa_a,kappa_b)=(-2,+1)`.  The detailed formula audit and same-density Core
closure results are recorded in `RHF_FORMULA_AUDIT_20260809.md`.

## Variation and rearrangement

The production RHF solver has its own local direct functional and complete
nonlocal exchange functional under `dpl_rhf/rhf/`. It does not import the RMF
functional and does not call `eigh` during initialization or training. A SiLU
network generates occupied density factors, QR imposes the hard algebraic
constraints, and the complete RHF functional supplies the variational
Hamiltonian. Core is not used for initialization or loss and is reserved for
final validation.

The energy graph is

```text
network -> Q -> P -> local and nonlocal densities
        -> density-dependent vertices
        -> RHF direct fields and all exchange energies
        -> total RHF energy
```

The complete block Hamiltonian follows Eqs. (3.392)-(3.394):

```text
h_a = h_kin,a + h_D,q + h_E,a.
```

`h_D` is assembled explicitly from `Sigma_plus`, `Sigma_minus`, and `Sigma_T`.
`h_E` is the nonlocal Fock functional derivative after removing the local
Fock rearrangement already included once in `h_D`. All PKA1 vertex derivatives
remain active; rearrangement is neither copied from SCF nor treated as an
optional loss. The projected Euler-Lagrange residual is

```text
(I - Q Q^T) h Q,
```

and is embedded in the loss together with the total energy and inverse-
Hamiltonian branch constraint. The training loop does not diagonalize `h`,
does not solve the Dirac equation, and forbids subspace refresh. The required
invariants are `initialization_block_diagonalizations=0`,
`refresh_block_diagonalizations=0`, and `training_dirac_solves=0`.

Both RMF and RHF impose the density boundary without an epoch-wise orbital
solve.  For each occupied `(q,kappa,n)` density column, the shifted energy
`epsilon=E_total-M` fixes
`lambda=sqrt[-epsilon(epsilon+2M)]/(hbar*c)` and the exact lower/upper tail
ratio.  The code continues all four nonlocal `GG/GF/FG/FF` bilinears at the
outer endpoint and evaluates analytic exterior number/radius moments.  At the
origin each block uses its own upper/lower angular powers.  Shell blocks and
their neural embeddings expand with requested spherical N/Z; fractional last
shell occupancy is equal filling. Pairing, blocking and deformation remain
outside the present model space.

For PKA1, the first forward pass additionally differentiates every required
direct and Fock channel separately. Training stops before the first optimizer
step if any textbook channel is absent, disconnected, or non-finite.

## Commands

Install the package in the existing environment:

```bash
conda activate torch_env
python -m pip install -e .
```

Train PKO2 for oxygen-16:

```bash
dpl-rhf-train --model PKO2 --z 8 --n 8 --a 16 \
  --epochs 500 --device cuda --out outputs/rhf/PKO2/O16
```

For a heavy nucleus on a host with limited RAM, saved autograd tensors can be
offloaded to local disk:

```bash
dpl-rhf-train --model PKA1 --z 50 --n 82 --a 132 \
  --epochs 500 --device cuda --resume \
  --graph-offload disk \
  --graph-offload-dir outputs/.pka1_graph_cache \
  --graph-offload-min-mib 0.25 \
  --out outputs/rhf/PKA1/Sn132
```

This is a storage backend, not a physical or mathematical approximation.  It
keeps the existing second-order variational training path, including
`create_graph=True`, the complete differentiable RHF functional and the
projected stationarity term.  Only tensors that autograd would otherwise keep
in RAM are serialized; they are reconstructed on the original device during
backward.  The per-epoch cache is deleted in `finally`, and stale graph
directories left by an interrupted process are removed when the next disk
offload run starts.  The optimization log reports the bytes written and read
for each epoch.  Because one backward pass may read a saved tensor more than
once, read volume can exceed the cache's peak size.

Disk offload is used for nuclei with `A >= 48`; smaller systems retain tensors
normally.  Use a local NVMe filesystem.  For sequential Sn132/Pb208 training,
keep at least 500 GiB free; concurrent heavy-nucleus runs require the sum of
their cache peaks.  Check current use with:

```bash
du -sh outputs/.pka1_graph_cache
df -h .
tail -f outputs/pkdd_pka1_five_nuclei_refresh4_500/PKA1/Sn132/train.log
```

Evaluate and plot a checkpoint:

```bash
dpl-rhf-evaluate outputs/rhf/PKO2/O16/best_checkpoint.pt \
  --device cuda --out outputs/rhf/PKO2/O16/evaluation
dpl-rhf-plot outputs/rhf/PKO2/O16/evaluation
```

Compare against a read-only Core reference JSON:

```bash
dpl-rhf-compare-core outputs/rhf/PKO2/O16/summary.json core_pko2_o16.json
```

## PKA1 oxygen-16 Hamiltonian validation

Before neural training, converged Core occupied states are passed through the
independent Python RHF functional as a read-only closure test. Current maximum
errors are `4.01e-6 MeV` in `E/A`, `1.41e-4 MeV` among kinetic/direct/Fock
energy components, `4.67e-5 MeV` in occupied levels, `4.88e-5 MeV` in local
potentials, and `5.08e-9 fm^-1` in total rearrangement. The explicit `h_D`
matrix closes identically. See `RHF_HAMILTONIAN_DEFINITION_AND_AUDIT.md` and
`outputs/audits/PKA1_O16_rhf_native_full_hamiltonian_closure.json`.

## Output interpretation

- `summary.json`: no-center-of-mass total energy, every direct/exchange channel,
  radii, constraints and run configuration.
- `local_observables.npz`: local densities and local RHF direct fields.
- `nonlocal_hamiltonians.npz`: complete total and Fock `398 x 398` matrices for
  each species/`kappa` block.
- `history.csv`: energy and projected RHF residual at every epoch.
- `best_checkpoint.pt`: neural parameters and reproducible run configuration.

Fock exchange is nonlocal, so it is deliberately not reported as a unique
local `S(r)` or `V(r)`. The local plots show RHF direct fields. The Fock output
is the block matrix required by the integral-differential RHF equation.
`local_observables.npz` also contains the direct, Fock and total local
rearrangement contributions.  The non-rearrangement Fock operator remains in
`nonlocal_hamiltonians.npz`.

## Validation status

On exact occupied Core densities, the repaired Python PKO2 functional differs
from Core Fock energies by `4.4e-5 MeV` for O-16 and `1.2e-4 MeV` for Ca-40.
The Ca-40 test includes multiple principal states in a fixed kappa block.  A
500-epoch O-16 neural run gives `E/A=-7.281605 MeV`, compared with the no-CoM
Core result `-7.262022 MeV`; all reported radii differ by less than `0.27%`.

## Current boundary

PKO1/PKO3 are not exposed yet.  Pairing, deformation and form-factor Chapter 4
extensions are also outside the present spherical closed-shell release.
