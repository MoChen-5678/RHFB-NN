# Physics review: odd-particle blocking and mode identity

**Scope.** Read-only review of the blocked-RHFB training source and primary HFB blocking literature. This note does not report new numerical results and does not establish that any trained state is a ground state.

## Finding

The current `minimum_block_overlap=0.90` check is an optimizer safeguard tied to the initial parameterization. It is stronger than number-parity conservation and is not, by itself, a physical criterion for preserving an odd-particle configuration. Because the code checks every trial mode against the *initial* mode, a legitimate continuous deformation that substantially rotates the blocked direction can be rejected even while it remains on a continuous branch. Conversely, simply replacing the initial reference with the last accepted mode makes the check local and permits cumulative drift, but does not by itself define a gauge-independent physical identity or prove that the tracked branch is the lowest-energy odd state.

Recommended hierarchy:

1. Treat the target number parity as a hard invariant. It distinguishes odd from even HFB vacua, but does not label which odd configuration is occupied.
2. Define the configuration to be followed with a physical tag where one exists: conserved quantum numbers and/or a specified single-particle orbital. Select the current quasiparticle by overlap with that tag, including particle-like and hole-like components as appropriate. This is the established maximum-overlap blocking use case.
3. For a direct-optimization model without current quasiparticle eigenvectors, use an explicitly declared transported blocked direction as a *continuation convention*. Compare consecutive accepted states after aligning the Bogoliubov frame gauge, and report the overlap and ambiguity. Keep the initial tag for audit, but do not use a large initial-overlap floor as a hidden constraint on all later shapes.
4. Treat a state as a ground-state candidate only after independent candidate starts/configurations have been converged and their energies compared under the same functional, basis, symmetries, and constraints. A falling loss or a successful identity check is not a ground-state certificate.

## What the source currently does

In `dpl_rhf/rhfb/selfconsistent_neural_training.py`, `_branch_metrics` first checks number parity from the density against target parity. For odd targets it additionally requires a model with a `blocked_index`; it obtains `mode` from `forward_with_blocked_mode()` and rejects a trial if

\[
  \left|m_0^\dagger m_{\rm trial}\right| < \texttt{minimum\_block\_overlap},
\]

where `m_0` is captured once at initialization and the default cutoff is 0.90. `blocking_overlap` in `manifold.py` takes the magnitude of the vector inner product, so it removes only an overall complex phase. The `initial_modes` reference is not refreshed after accepted updates.

In `dpl_rhf/rhfb/network.py`, `forward_with_blocked_mode()` forms the generalized-density transform `T`, picks the particle or hole member based on the diagonal reference occupation, and returns the corresponding column lifted by `diag(Q,Q*)`:

\[
  m = \begin{pmatrix}Q & 0\\0&Q^*\end{pmatrix} T e_i.
\]

This is an explicit transported column of the chosen parameterization. It is not selected by diagonalizing the current HFB Hamiltonian and then identifying an eigen-quasiparticle. The construction is useful as a deterministic continuation label, but interpreting its raw column overlap as a fully physical identity requires fixing the residual frame freedom described below.

## Literature: source statements

### Number parity is invariant under generalized Thouless motion

Robledo and Bertsch state that a Bogoliubov transformation has even or odd number parity, that a generalized Thouless transformation does not change that parity, and that an odd-parity starting state therefore remains in the odd-particle-number sector (Sec. III, especially pp. 3–4 of the article). They further say their gradient method searches the odd-parity transformation space and can find its global minimum **if that space is adequately sampled**. The qualification matters: parity preservation does not identify a particular orbital, and the formal global-minimum statement is not a guarantee for a restricted ansatz or a finite optimizer run. [Robledo & Bertsch, *Application of the gradient method to Hartree-Fock-Bogoliubov theory*, PRC 84, 014312 (2011), Sec. III, Eqs. (12)–(14) and pp. 3–4](https://arxiv.org/abs/1104.5453) ([full text](https://arxiv.org/html/1104.5453v1)).

### Odd states use explicit blocking and may have distinct local minima

For Gogny D1S, Robledo, Bernard, and Bertsch generate trial odd states by swapping the `U` and `V` components of a column from converged neighboring even-even solutions. They identify the axial label `K=±j_z` from the block of that swap, run the gradient solver from a set of starts (oblate, prolate, spherical), obtain many local minima, and select the lowest among those found. They explicitly say this protocol does not guarantee the global minimum, though they judged deeper missed minima unlikely for their examples (Sec. II, p. 2). This supports candidate-seeded searches and energy comparison; it does not support imposing one universal large overlap to the initial column as the definition of the physical ground state. [Robledo, Bernard & Bertsch, *Pairing gaps in the Hartree-Fock-Bogoliubov theory with the Gogny D1S interaction*, PRC 86, 064313 (2012), Sec. II, p. 2](https://doi.org/10.1103/PhysRevC.86.064313) ([author PDF](https://archive.int.washington.edu/users/bertsch/articles/354.pdf)).

### Maximum-overlap blocking tracks a tagged orbital through self-consistency

Schunck et al. describe the odd-nucleus state as a one-quasiparticle excitation and give the blocked density and pairing tensor as the even-vacuum terms plus/minus the selected quasiparticle's `U,V` contributions (Eq. 15). Their implementation first identifies a single-particle target by symmetry block and orbital identity, then, on each iteration, associates that target with a quasiparticle using the largest overlap with its upper component `U` and the time-reversed lower component `V̄` near the Fermi level (Sec. II.D, p. 4). In their discussion, the largest-overlap choice is how the code tracks the requested configuration, rather than an absolute overlap floor to a frozen initial quasiparticle vector. [Schunck et al., *One-quasiparticle States in the Nuclear Energy Density Functional Theory*, PRC 81, 024316 (2010), Sec. II.D, p. 4, Eq. (15)](https://arxiv.org/abs/0910.2164) ([full text](https://arxiv.org/html/0910.2164)).

## Interpretation for this model

**Physics distinction.** Parity is a property of the Bogoliubov vacuum. A selected column is additional structure used to say which odd configuration is being followed. The overlap test is phase invariant, but it does not remove general rotations among quasiparticle/frame directions that leave the generalized density unchanged. In the code's notation, `R = T R0 T†` is unchanged under `T -> T G` whenever `G` commutes with `R0`; columns within a degenerate occupied or empty subspace can rotate while the represented density stays fixed. The extracted single column `T e_i` can therefore change under a parameterization gauge motion. An overlap loss caused only by that rotation is not evidence of a physical branch switch.

**Initial anchor.** Comparing every step to `m0` can prohibit a valid path to a deformed minimum when the blocked direction rotates continuously. This is a real limitation of a fixed-reference gate, independent of whether the starting reference is physically sensible. It can also overconstrain a state where the intended odd configuration is defined by a conserved label but its primitive-space wavefunction polarizes substantially.

**Previous accepted state.** Replacing `m0` with the last accepted mode is a reasonable local-continuation heuristic and removes the artificial requirement that the final state remain close to the initial vector. It still has three limits: (a) local overlap is parameterization/gauge dependent unless the full frame is aligned first; (b) many small changes can accumulate into a different configuration; and (c) local continuity can faithfully preserve a metastable starting branch rather than find the lowest odd minimum. Thus it solves the fixed-anchor restriction, but not identity or ground-state selection by itself.

**What counts as a branch change.** A physical branch change should be based on a change in the tagged configuration after gauge alignment, loss/change of a required conserved label, or a clear switch in the maximum-overlap candidate. A raw selected-column overlap drop alone is insufficient. If the top two candidate overlaps are nearly tied, identity is ambiguous and should be reported as unresolved rather than silently assigned. If no independently defined tag or candidate set exists in the model, call the quantity “transported-column continuity,” not a proof of physical quasiparticle identity.

## Reviewable code-level proposal (not implemented)

- Preserve the existing parity guard as the hard odd/even check.
- Rename the current initial-mode quantity and history field so it is explicit that it measures transported-column overlap, if retained. Avoid presenting it as an independent physical identity check.
- Add a branch-reference policy for trial acceptance: keep an immutable initial tag for diagnostics, and compare a trial to the last accepted state after a documented Bogoliubov-frame alignment. Use the line search to reject discontinuous trial steps, while allowing cumulative rotation over accepted steps. Record both the local overlap and initial-to-current overlap.
- Where a current quasiparticle basis is available, implement a separate tagged-candidate rule based on maximum overlap with a chosen single-particle reference and the `U` / time-reversed `V` components in the appropriate symmetry block, with a uniqueness margin for the winning candidate. Do not conflate this candidate selection with the density parity check.
- For ground-state claims, use multiple physically distinct seeds (for example candidate blocked orbitals and allowed shape starts), converge each, and compare the final energies. Record convergence/stationarity, particle-number residuals, parity, identity ambiguity, and candidate energy ordering. A lower stationarity loss alone is not evidence of a lower physical energy or a global minimum.

## Suggested failure/reporting conditions

1. **Parity failure:** density parity differs from target. Reject the trial immediately.
2. **Tag failure:** a required symmetry label changes, no candidate matches the tag, or candidate overlaps are ambiguous within a stated margin. Reject or mark the branch unresolved; do not infer a branch switch from column overlap alone.
3. **Local continuation failure:** after gauge alignment, consecutive accepted-step overlap is below the declared continuation floor. Reject that step or reduce its step size; retain the last accepted branch.
4. **Search limitation:** no stationarity/convergence criterion is met, or the multi-start candidate set was not explored. Report the result as an unconverged/stationary candidate, not as a ground state.
5. **Ground-state comparison:** report “lowest among the converged candidates searched” with their energies and labels. Promote to a ground-state claim only with an adequately justified search domain.

## References

1. L. M. Robledo and G. F. Bertsch, “Application of the gradient method to Hartree-Fock-Bogoliubov theory,” *Phys. Rev. C* **84**, 014312 (2011), [arXiv:1104.5453](https://arxiv.org/abs/1104.5453), [full text](https://arxiv.org/html/1104.5453v1).
2. L. M. Robledo, R. Bernard, and G. F. Bertsch, “Pairing gaps in the Hartree-Fock-Bogoliubov theory with the Gogny D1S interaction,” *Phys. Rev. C* **86**, 064313 (2012), [DOI](https://doi.org/10.1103/PhysRevC.86.064313), [author PDF](https://archive.int.washington.edu/users/bertsch/articles/354.pdf).
3. N. Schunck et al., “One-quasiparticle States in the Nuclear Energy Density Functional Theory,” *Phys. Rev. C* **81**, 024316 (2010), [arXiv:0910.2164](https://arxiv.org/abs/0910.2164), [full text](https://arxiv.org/html/0910.2164).
