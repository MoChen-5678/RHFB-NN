# Saved O17 neural-state physical review

Date: 2026-10-06  
Scope: read-only interpretation of the already saved Python/PyTorch O17 experiments and their source snapshots. No training was rerun and no Fortran was used. The model target remains full PKA1 plus D1S; none of the cited nonrelativistic work is treated as a replacement model.

## Findings at a glance

The saved state has the requested odd-neutron/even-proton number parity, and the block seed is the O17 valence configuration `1d5/2`, `κ=-3`, `m=+5/2`. The spinor basis includes complete `m` multiplets for each included `κ`, both radial Dirac components, and the expected `Ωκ`/`Ω−κ` spinor harmonics. Its correction envelope uses the correct leading reduced-radial powers `r^(l+1)` for both components. Thus the available evidence does **not** show that the expansion omitted small components or all states needed to represent the stated block.

The saved 20-update `h=0.8 fm` intrinsic state has substantial spatial-parity mixing in its anomalous density: its odd-parity component is 41% (neutrons) and 37% (protons) of the total Frobenius norm. In the separate 2-update `h=0.2 fm` output, the corresponding fractions are below 0.06%. This is not by itself a global defect in the unrestricted calculation: the stated variational scope permits reflection/parity breaking. It does establish a gap for a definite-parity `5/2+` O17 baseline: no parity-preserving initial-state requirement or independent spatial-parity check was applied. Do not impose a global parity restriction that would remove legitimate reflection-breaking candidates.

Do not impose equality of separately trained `m`-seeded outputs as a direct acceptance rule. Rotational invariance requires equal energy for a state and the fully rotated copy of that same state under an exactly rotation-covariant functional and complete numerical representation. It does not make a chosen intrinsic blocked density spherical, nor does time reversal alone guarantee degeneracy after exact blocking. The saved runs contain no independently converged `m` branches or angular-momentum projection, so their `m` degeneracy has not been tested.

## What the papers imply for O17 blocking

The experimental ground state of O17 is `Iπ=5/2+`; mean-field descriptions identify its valence neutron with `1d5/2`. The `d` orbital has even single-particle parity, and the six `m` values are magnetic components of the same `j=5/2` multiplet. This supports the saved block label as a physically motivated configuration, but it does not select `m=+5/2` as uniquely physical. [The O17 spherical-nucleus study](https://csnsdoc.ihep.ac.cn/article/doi/10.1088/1674-1137/ac2f94) identifies O17 as a measured spherical nucleus with one neutron outside the doubly closed core and uses `Iπ=5/2+` and the `1d5/2` valence orbital (Sec. III.A; see also Fig. 1).

In exact blocking, the odd state is constructed by changing the occupation of one quasiparticle; it is not the equal-filling approximation (EFA). Kasuya and Yoshida explain the parity-changing quasiparticle/hole-column swap and show that odd systems intrinsically break time reversal and induce time-odd densities and fields (PTEP 2021, Secs. 1 and 3, especially Eqs. (22), (30)–(32)). In their spherical discussion, the `(2j+1)`-fold degeneracy lets a constraint select a desired `Ω`; the selected projection labels the intrinsic blocked solution. [Primary article](https://academic.oup.com/ptep/article/2021/1/013D01/5981141), [arXiv record](https://arxiv.org/abs/2005.03276). This paper develops the general HFB blocking/time-reversal logic in a nonrelativistic EDF, so only that symmetry logic is used here.

The relativistic analogue also cautions against demanding `+m` and `−m` degeneracy from the *same fixed blocked Hamiltonian*: in odd relativistic RHB, blocking breaks time reversal, removes the Kramers relation between opposite projections, and requires the two branches to be treated separately. The paper implements blocking by exchanging the blocked quasiparticle's `(U,V)` and `(V*,U*)` amplitudes and reversing its quasiparticle energy (2024, Sec. 2). [Primary relativistic RHB paper on time-odd fields](https://scoap3-prod-backend.s3.cern.ch/media/files/86329/10.1016/j.physletb.2024.138792.pdf). It uses PC-PK1 and is cited only for the relativistic symmetry/blocking principle, not as a PKA1 benchmark.

There are two different comparisons:

1. **Time-reversal partner at a fixed blocked field.** Exact blocking breaks time reversal, so the `+m` and `−m` eigenvalues need not be degenerate in one fixed blocked Hamiltonian. But the time reverse of a *whole self-consistent solution* also reverses its time-odd densities and currents; for a time-reversal-invariant total functional, that paired solution has the same scalar energy. EFA restores time reversal by averaging the two blocked partners, but that is a different approximation and omits their intrinsic time-odd polarization.
2. **Rotated copies under a scalar rotationally invariant functional.** For a fully rotation-covariant model, `E[U(R)Rho U(R)†] = E[Rho]`. A rotation of a self-consistent blocked solution therefore has the same energy. This does not imply that separately seeded states with different `|m|` are related by a rotation: a rigid rotation preserves the magnitude of the intrinsic angular-momentum expectation. An intrinsic `m`-oriented solution may have non-spherical density and current; the laboratory `I=5/2` state requires symmetry restoration or a rotational average for scalar one-body densities. Lewin and Paul state the HFB rotation action and invariance explicitly in Sec. 3.3.2, Eq. (3.22), pp. 65–66. Their article uses a simpler interaction, so this is a general symmetry statement, not a PKA1/D1S calculation. [Primary mathematical HFB reference](https://www.numdam.org/item/10.1051/m2an/2013094.pdf).

Thus the energy test, if later performed, should rotate a converged full density (including its current and anomalous density) and evaluate the same functional, or compare independently converged branches only after controlling their variational branch and finite-grid symmetry errors. Merely replacing the initial block label with another `m` and observing different unconverged network endpoints would not falsify rotational invariance.

## Saved data and source checks

Primary dataset: `outputs/demo/O17_neural_experiment_20261005_20updates_v2/`. The second saved run is `outputs/demo/O17_neural_experiment_20261005_h02/`.

| Evidence | Saved value or source fact | Interpretation |
|---|---:|---|
| Neutron block seed | `κ=-3`, `m2=5` (`m=+5/2`), `1d5/2`, particle block, active index 13 | Correctly names the O17 `5/2+` valence configuration at initialization. The label is not proof that the final unrestricted state remains an eigenstate of `Jz` or `J²`. |
| Number parity, main run | neutron `−1`, proton `+1`; normalized Pfaffians `−0.9999999999999978`, `+0.9999999999999953` | Independent saved-density parity check confirms odd N and even Z sectors. It does not certify the physical functional or orbital assignment. See [`O17_number_parity_20261006.json`](../outputs/validation/O17_number_parity_20261006.json). |
| Time-odd indicator | integrated neutron spatial-current L1 = `0.2064` in the main run | Consistent with a time-reversal-breaking exact blocked state. It is not a closure check on the PKA1 time-odd fields. |
| Intrinsic deformation | neutron `Q20≈−1.64765 fm²` in the main run; near zero for protons | Not by itself unphysical: a fixed magnetic substate can have an oriented intrinsic density even when the observed O17 nucleus is spherical. The run is unconverged and no angular-momentum projection was done. |
| Basis channels, snapshot `basis.py` | all `κ=±1,±2,±3`, each with all allowed `m2`; upper and lower radial coefficients for each; harmonic lower component uses `i f Ω_{−κ,m}` | Complete magnetic multiplets and both Dirac components are present through the stated angular cutoff. The `κ=−3,m2=5` state has upper `l=2` and lower `l=3`, as required for a Dirac `d5/2` spinor. |
| Origin/boundary envelope, snapshot `network.py` | each component correction multiplied by `(r/R)^(l+1)(1−r/R)` | Correct leading regularity for reduced radial amplitudes and vanishing at the box endpoint in the continuous envelope. The mesh omits the origin and endpoint, and unrestricted per-grid `raw_coordinates` mean smooth higher-order radial regularity is not certified. |

The primitive dimension is 432 at `h=0.8 fm` and 1872 at `h=0.2 fm`, with only 14 active orbitals per species in both runs. The active initial shell sequence fills `1s1/2`, `1p3/2`, `1p1/2`, then all six `1d5/2` magnetic substates; index 13 is the final `m=+5/2` member. The variational ansatz then allows dense active-space canonical mixing and unrestricted complex primitive factors, so the run configuration's block label is the chosen *reference* block, not a post-training angular-momentum measurement.

### Verified parity violation in saved pairing tensor

For the implemented spinor harmonics, the parity eigenvalue of a `κ` channel is the upper orbital parity `(-1)^{l_upper}`; the lower component has the opposite orbital `l`, and its intrinsic Dirac parity sign makes the whole spinor carry that same channel parity. Constructing the primitive anomalous density as `χ_prim = Q χ_active Qᵀ`, its parity-even condition for the `5/2+` state is

`P χ_prim Pᵀ = χ_prim`,

where `P` is diagonal in the primitive `κ,m` channels. Directly applying this check to the saved `density_factors.npz` gives:

The reported diagnostic ratio is

`d = ||χ_prim − Pχ_primPᵀ||F / ||χ_prim||F = 2 ||χ_odd||F / ||χ_prim||F`.

Thus `d` itself is **twice** the odd-component fraction; it is not the percentage of the anomalous density that is odd. The exact saved-array results are:

| Saved run/species | `||χ_prim||F` | `||χ_prim − Pχ_primPᵀ||F` | `||χ_odd||F` | odd component / total |
|---|---:|---:|---:|---:|
| 20 updates, `h=0.8 fm`, n | 0.0034661281 | 0.0028397682 | 0.0014198841 | 0.40965 |
| 20 updates, `h=0.8 fm`, p | 0.0031534358 | 0.0023320437 | 0.0011660218 | 0.36976 |
| 2 updates, `h=0.2 fm`, n | 0.0037931373 | 0.0000045099 | 0.0000022550 | 0.0005945 |
| 2 updates, `h=0.2 fm`, p | 0.0038220652 | 0.0000035442 | 0.0000017721 | 0.0004637 |

These values are reproducible with [`check_spatial_parity.py`](../outputs/validation/O17_spatial_parity_20261006/check_spatial_parity.py); the script reads only the saved NPZ arrays. Full precision and input hashes are in [`spatial_parity_diagnostic.json`](../outputs/validation/O17_spatial_parity_20261006/spatial_parity_diagnostic.json).

The violation is computed directly from the saved density, not inferred by comparing implementations. The active canonical transformation and primitive factors are not projected onto spatial parity, so the unrestricted state can acquire cross-parity pairing. The normal density's parity-odd Frobenius fraction in the 20-update `h=0.8 fm` state is much smaller (`5.5e-4` for neutrons and `5.9e-4` for protons, defined as the cross-parity block norm divided by `||QρQ†||F`). The 2-update `h=0.2 fm` anomalous density is nearly parity even. This difference cautions against describing parity mixing as a universal ansatz defect or attributing it solely to the initialization.

For interpreting a definite-parity `5/2+` O17 baseline, the `h=0.8 fm` output cannot be treated as a parity eigenstate; the `h=0.2 fm` output passes this particular check to about `6e-4` in odd-component fraction. Neither observation certifies a physical O17 solution. Correct number parity does not imply definite spatial parity: the Pfaffian diagnostic checks `(-1)^N`, while the condition above checks spatial parity covariance of the pair tensor. Because the scope is unrestricted, parity should be recorded and compared across candidate solutions, not imposed as a global hard constraint.

## Numerical error is separately established

These network outputs are not converged physical energies. The main run used `R=8 fm`, `h=0.8 fm`, 20 optimizer updates, `|κ|≤3`, 14 active states/species; the `h=0.2 fm` output used the same box and active size but only two updates and a different primitive dimension. Their experimental functional totals (`+281.7745` and `−139.6781 MeV` initially) must not be interpreted as a physical resolution-convergence pair.

The independent saved fixed-orbital grid diagnostic is cleaner: before any optimizer update, it evaluated the same analytic O17 initial orbitals on `h=0.8,0.4,0.2 fm` grids. It reports a coarse-to-fine total-energy change of `421.45 MeV` at both 8 and 12 fm boxes, dominated by sigma/omega Yukawa terms; the `h=0.4` to `0.2 fm` change is still `85.18 MeV`. A fixed-source radial Yukawa probe reports relative integration errors at `h=0.8 fm` of `78.6%` for omega, `76.3%` for rho, and `36.4%` for sigma; at `h=0.2 fm`, these fall to about `5.5%`, `5.3%`, and `2.3%`, respectively. These are quadrature/integration errors, not a statement about the physical validity of the functional. See [`grid_diagnostic.json`](../outputs/validation/O17_grid_diagnostic_20261005/grid_diagnostic.json), [`grid_diagnostic.csv`](../outputs/validation/O17_grid_diagnostic_20261005/grid_diagnostic.csv), and its [scope audit](../outputs/validation/O17_grid_diagnostic_20261005/physics_literature_audit.md).

Additional non-acceptance facts are already recorded in each run: the functional readiness gate was `ready=false`; no Bogoliubov or Dirac eigenproblems were used in training; the interacting no-sea branch constraint and full stationary residual were not certified; the reported rearrangement-energy row is a zero bookkeeping placeholder; and the result was explicitly marked `experimental_uncertified`. These are independent limitations and are not inferred from the symmetry analysis.

## Open questions, not established failures

- The saved result does not test whether complete rotations of one converged density produce the same scalar energy. There is no rotated-density comparison and no converged PKA1 odd-RHFB stationary solution.
- The endpoint `m=+5/2` label has not been verified on the transported blocked Nambu vector after training; no final `Jz`, `J²`, or projected-spin diagnostic is saved.
- The angular grid (`nθ=7`, `nφ=14`) is the minimum chosen for the `|κ|≤3` basis and can integrate basis harmonics adequately, but it has not been shown converged for nonlinear density-dependent PKA1 contractions. This is a numerical convergence concern, not proof of a missing angular channel.
- The radial envelope enforces the correct leading powers on the network correction, but neither saved data file establishes continuum smoothness at `r=0` or convergence of the finite-difference boundary treatment. At `h=0.8 fm`, the first interior sample is already 0.8 fm from the origin.
- A definite-parity O17 baseline has no parity-preserving initial-state requirement or independent formal acceptance gate. Reflection-breaking solutions remain within the unrestricted variational scope, so any future check should report the symmetry sector rather than globally exclude those candidates.

## Sources

1. H. Kasuya and K. Yoshida, “Hartree–Fock–Bogoliubov theory for odd-mass nuclei with a time-odd constraint,” PTEP 2021, 013D01, especially Secs. 1, 3 and Eqs. (22), (30)–(32): [journal](https://academic.oup.com/ptep/article/2021/1/013D01/5981141), [arXiv](https://arxiv.org/abs/2005.03276).
2. “Nuclear magnetism in the deformed halo nucleus 31Ne,” Phys. Lett. B 855 (2024) 138792, Sec. 2: [publisher article PDF](https://scoap3-prod-backend.s3.cern.ch/media/files/86329/10.1016/j.physletb.2024.138792.pdf).
3. M. Lewin and S. Paul, “A numerical perspective on Hartree–Fock–Bogoliubov theory,” M2AN 48 (2014) 53–86, Sec. 3.3.2, Eq. (3.22), pp. 65–66: [journal PDF](https://www.numdam.org/item/10.1051/m2an/2013094.pdf).
4. “Comparative studies on nuclear elastic magnetic form factors between the relativistic and non-relativistic mean-field approaches,” Sec. III.A and Fig. 1: [journal article](https://csnsdoc.ihep.ac.cn/article/doi/10.1088/1674-1137/ac2f94).
