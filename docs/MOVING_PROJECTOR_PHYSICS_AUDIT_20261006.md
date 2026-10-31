# Moving positive-projector variation audit — 2026-10-06

## Finding

The residual `[H_a,R_a]=0` is the correct first-order equation for canonical variations **at fixed active projector**. It is not, in general, the Euler equation of the composed energy obtained by embedding `rho=P rho_a P`, `chi=P chi_a P^T` with a projector that moves as `P=P_+(h[rho])`. Automatic differentiation through `P` while evaluating this residual differentiates the residual map; it does not turn the residual into the derivative of that composed energy.

There is an important cancellation. For an exact spectral projector of the same normal field `h=δE_ph/δrho`, with `rho=P rho P`, the normal-energy contribution from projector motion is zero at first order. The pairing contribution need not be zero: it contracts the cross-space blocks of the full pairing field with the projector response. The reviewed equations do not supply a generic identity that removes this term. A projected pairing field would remove this particular contraction by assumption, but would define a different pairing functional unless that projection is part of the chosen model.

This is a scope finding, not a claim that every computed state has a numerically nonzero omitted term. For a particular state it can vanish through a zero projector response, a zero pair tensor, zero cross-block pairing field, or an accidental contraction. Such a state-specific cancellation must be calculated.

## Literature basis

Robledo and Bertsch derive the generalized Thouless variation `W(Z)=W(0) exp(i Z)` and

\[
\delta E=\frac{i}{2}\operatorname{Tr}_2([\mathbb R,\mathbb H]\mathbb Z)+O(\mathbb Z^2),
\]

with `H` formed from the fields at the same generalized density. Their full-blocking form replaces `R` by the blocked generalized density, or equivalently swaps the blocked quasiparticle columns and transforms `H` and `Z` (Eqs. 18–21 in the arXiv PDF, printed pp. 9–10). This supports the commutator as the fixed-space full-HFB first variation and as the fixed-`P` active-space variation. It does not derive a density-dependent moving `P` constraint. [Robledo & Bertsch, arXiv:1205.4443](https://arxiv.org/abs/1205.4443)

Geng and Long define the RHFB energy as the expectation of the Hamiltonian in the Bogoliubov state (Eq. 19); their normal terms and pairing term are given in Eqs. 20a–20d. They then vary with respect to the generalized density and identify `h=∂E/∂rho` and `Delta=∂E/∂kappa` (Eqs. 21–22, PDF p. 5). For Gogny D1S they replace the pp-channel vertex with `gamma^0` at both endpoints (Eq. 20d discussion, PDF p. 5). Their no-sea quantization uses positive-energy Dirac single-particle states (Eqs. 10–11, PDF p. 4); separately, they state that both positive- and negative-energy DWS solutions belong in the `U,V` expansion for completeness, and explicitly warn not to confuse this basis completeness with neglecting the Dirac sea in densities/currents (PDF p. 9). Thus “no sea” is not identified with projecting every density onto the instantaneous positive spectrum of the current `h`. [Geng & Long, arXiv:2202.00482](https://arxiv.org/abs/2202.00482)

The DIRHB review writes the standard RHB densities and pairing tensor as quasiparticle sums. Its no-sea prescription chooses the vacuum conditions `alpha_p|Phi>=0` for nuclear-sector `E_p>0` and `alpha_a|Phi>=0` for Dirac-sea `E_a<0`; Eqs. 46–48 (PDF pp. 12–13) then give both sectors' contributions to `rho` and `kappa`. The authors say the sea `V_a` terms are often neglected because of the large gap, but warn that dropping them makes the pairing tensor only approximately antisymmetric. This is a quasiparticle-annihilation-vacuum prescription and a controlled approximation, not the same algebraic constraint as `rho=P rho P` and `kappa=P kappa P^T`. [Nikšić et al., arXiv:1403.4039](https://arxiv.org/abs/1403.4039)

## Projector-response derivation

Let `P` be an exact, gapped spectral projector of a Hermitian normal field `h`, and let `rho_a` and `kappa_a` be active-space densities. Suppress the frame-coordinate gauge and write the embedded densities as

\[
\rho=P\rho_aP,\qquad \kappa=P\kappa_aP^T.
\]

For a variation that changes `P` while holding the active coordinates fixed,

\[
\delta_P\rho=(\delta P)\rho_aP+P\rho_a(\delta P),\qquad
\delta_P\kappa=(\delta P)\kappa_aP^T+P\kappa_a(\delta P)^T.
\]

For the spectral projector, in an eigenbasis `h|i>=epsilon_i|i>` with `p_i` equal to 1 or 0,

\[
(D P_h[X])_{ij}=
\begin{cases}
\dfrac{p_i-p_j}{\epsilon_i-\epsilon_j}X_{ij},&p_i\ne p_j,\\
0,&p_i=p_j.
\end{cases}
\]

In particular, `P δP P=0` and `(1-P)δP(1-P)=0`: projector motion is purely between the selected and excluded subspaces. If the same self-consistent normal field defines `P`, then `[h,P]=0`. With `rho=P rho P`,

\[
\begin{aligned}
\delta_P E_{\rm ph}
 &=\operatorname{Tr}\!\left[h\{(\delta P)\rho P+P\rho(\delta P)\}\right]\\
 &=\operatorname{Tr}[(\rho h+h\rho)\delta P]=0.
\end{aligned}
\]

The last equality follows because `rho h+h rho` is block diagonal in `P ⊕ (1-P)`, whereas `δP` is off diagonal. This cancellation is exact for a gapped exact projector and the matching normal gradient. It is not a general license to omit every response term.

Choose the pairing-gradient normalization `δE_pp=Re Tr(Delta† δkappa)`; any convention-dependent overall factor can instead be absorbed into `Delta`. Then

\[
\delta_P E_{\rm pp}=\operatorname{ReTr}\!\left[\Delta^\dagger\{(\delta P)\kappa_aP^T+P\kappa_a(\delta P)^T\}\right].
\]

Only the cross blocks of `Delta` between `P` and `1-P` can contract with this variation. They are not forced to vanish by `[h,P]=0`, by `P δP P=0`, or by active-space stationarity. Since `P=P_+(h[rho])`, this term pulls back through both maps:

\[
\delta P=D P_h[D h_{\rm ph}[\delta\rho]],
\]

and, when the variation is self-consistent, `delta rho` itself contains the `delta P` terms above. In a scalar-source representation `h=h_0+g x S`, `x=Tr(S rho)`, the response is `δP=D P_h[g S δx]`; the field-closure row `x=Tr(S rho)` makes this dependence explicit. An exact response implementation must solve or differentiate this implicit linear relation, including its projector gap and rank assumptions.

The full composed-energy derivative is therefore the fixed-projector active derivative plus the response contraction above. In active coordinates, `[H_a,R_a]=0` sets only the first term to zero. It does not test the `P`-response term. A full-space RHFB commutator, if satisfied in the correct blocked quasiparticle vacuum, is a stronger condition and tests all canonical directions; it is not implied by the compressed commutator.

This distinction also separates a fixed-point equation from an energy KKT system. The current root conditions are the active commutator, `eta-Tr(S rho)=0`, and `N-N0=0`; `mu` enters the number-constrained BdG field. The `eta` row enforces field closure, but it is not a multiplier equation for the derivative of a composed energy. If `eta` is instead promoted to an independent variational variable with closure treated as an equality constraint, a Lagrangian multiplier `lambda_eta` is required: variation in the active coordinates contains `-lambda_eta D Tr(S rho)` and variation in `eta` contains `partial_eta E_comp + lambda_eta=0`. Neither relation is present in the current residual. Likewise, automatic differentiation used to form a Jacobian of these root equations computes `D r / D theta`; it does not supply the omitted energy gradient or an implicit multiplier.

## Minimal analytic counterexample

Use four particle states ordered `(p1,p2,s1,s2)`, with two selected states and two excluded states. At the reference point take

\[
h=\operatorname{diag}(e,e,-e,-e),\quad P=\operatorname{diag}(1,1,0,0),\quad
(\kappa_a)_{12}=k,\quad(\kappa_a)_{21}=-k.
\]

Let `delta h_31=delta h_13*=q`; the exact divided-difference formula gives `delta P_31=q/(2e)` and its Hermitian conjugate. Direct substitution yields

\[
(\delta_P\kappa)_{32}=\frac{qk}{2e},\qquad
(\delta_P\kappa)_{23}=-\frac{qk}{2e}.
\]

Allow a valid antisymmetric pairing gradient with `Delta_32=d`, `Delta_23=-d`. Then

\[
\delta_P E_{\rm pp}=2\operatorname{Re}\!\left(d^*\frac{qk}{2e}\right),
\]

which is nonzero for generic `d`, `q`, and `k`. Yet the normal response is zero by the cancellation above. `q` can arise from a normal density-dependent field: for example, `h[rho]=h_0+g Tr(S rho)S` with a nonzero `S_31`, and an allowed active HFB tangent with `Tr(S delta rho_a) != 0`; for small `g`, `q=g S_31 Tr(S delta rho_a)+O(g^2)`. A decoupled occupied blocked spectator can be appended without changing this subblock, so the response is also present in the odd blocked sector. The example only establishes generic non-equivalence; it does not assert a particular numerical value for the current D1S state.

The four-state response block can be chosen to be a pure Gaussian state, not an arbitrary anomalous matrix: take `rho_a=0.2 I_2`, `kappa_a=0.4 J`, `J_12=1`, `J_21=-1`. Then `rho-rho^2=kappa kappa†=0.16 I_2` and `rho kappa=kappa rho*`. Its active BdG block can already be stationary with `h_+-mu=0.3 I_2` and `Delta_a=-0.4 J`: the generalized Hamiltonian obeys `H_a^2=0.25 I` and `R_a=0.5(I-H_a/0.5)`, hence `[H_a,R_a]=0`. Add a second active pair with `rho_b=0.8 I_2`, `kappa_b=0.4 J`, `h_b-mu=-0.3 I_2`, and `Delta_b=-0.4 J`, plus a decoupled occupied spectator. The two paired blocks have `Tr rho_a + Tr rho_b=2`; with the blocked spectator the state has odd parity and `N=3`. Both active BdG blocks commute with their pure generalized densities.

There is also a number-conserving active tangent. Parameterize each pair by `v=sin(theta)`, `u=cos(theta)`, and vary `delta theta_a=t`, `delta theta_b=-t`. Since both have `uv=0.4`, their number derivatives `4uv delta theta` cancel. A normal source that samples the first pair but not the second has `delta x_active=1.6t`. To include the implicit field closure, use the explicit Hermitian source `S=diag(1,1,0,0,0,0)` on the active pair plus `S_61=S_16=1` between the selected `p1` state and excluded state `s1`, with `h=h0+g x S`, `x=Tr(S rho)`, and `g=10/41`. Choose `h0=h-g x0 S` at the reference state. The p1–s1 normal gap is four, so `delta P_61=q/4`; the projector density response contributes `delta x_P=2 rho_11 delta P_61=0.1q`. The closed linear response is therefore `q=g(1.6+0.1q)`, whose solution is `q=0.4`; it explicitly uses `delta rho_total=delta rho_active+delta P rho+rho delta P`. For `e=2`, `k=0.4`, and `d=0.3`, this gives `delta P_61=0.1`, `delta kappa_62=0.04`, `delta kappa_26=-0.04`, `Tr(h delta_P rho)=0`, and `Re Tr(Delta† delta_P kappa)=0.024`. The full six-state pure generalized density has `N=3`; the independent Pfaffian `number_parity_from_density` diagnostic returns odd parity. The active BdG blocks commute before the variation, so the nonzero result is solely the omitted projector-response derivative. This is a deliberately constructed finite Gaussian model, not a D1S or nuclear calculation. An independent Python matrix check, including purity, Pfaffian odd parity, integer number, active commutator, number-conserving tangent, implicit source closure, and zero eigen-API calls, is [here](../dpl_rhf/rhfb/tests/test_moving_projector_variational_counterexample.py).

## Mapping to the current candidate

In `projected_rhfb_fixedpoint.py`, `state_from_coordinates` constructs `h_aux=h0+g eta S`, makes `P` through rational-sign and fixed-seed QR, embeds `rho_a,chi_a` as `B rho_a B†, B chi_a Bᵀ`, evaluates the full primitive D1S pairing field, and restricts the Nambu Hamiltonian to the active frame. `residual_blocks` then returns the active commutator, `eta-Tr(S rho)`, and number residual. On the field-closure surface, `h_aux` and the toy normal field agree. The normal projector-response cancellation therefore applies there. The pairing field is generated from the full primitive pairing action; the active equation uses only its frame-restricted block. The current residual contains no separately formed `D P_h[D h[delta rho]]` pairing contraction and no multiplier representing that response.

The module `no_sea_particle_space.py` already describes `active_stationarity_residual` as a fixed-projector Euler equation and warns callers with field-dependent projectors to include their response. This audit supplies the normal-term cancellation and the explicit pairing-response boundary for the present implementation.

## Saved O-17 D1S action evidence

A separate post-processing diagnostic now evaluates the full implemented primitive-space action `Delta Q*` for the saved paired O-17 candidate and resolves it in the *free* Dirac positive/negative sectors. It finds nonzero sampled cross-sector action: `||P_- Delta P_+* Q*||_F = 8.56e-4 MeV` for neutrons and `9.42e-4 MeV` for protons; the reverse blocks are `2.77e-6` and `3.09e-6 MeV`. It also finds that 70.2% and 62.7% of the full `Delta Q*` action norms lie outside the saved active `Q` frame. The state has nonzero neutron/proton anomalous densities and independently checked number parity `-1/+1`; the computed active gaps and pair energies reproduce the saved fields to floating-point precision. These are action-on-this-frame measurements with a free-field sector split, not full operator block norms. See the [saved O-17 D1S sector-action report](../outputs/validation/O17_d1s_cross_dirac_pairing_20261006/REPORT.md) and its JSON/NPZ artifacts.

This strengthens the practical motivation for checking the primitive pairing action rather than only its active compression. It does not compute the interacting `P_+(h)`, its response, or an RHFB no-sea branch. It therefore is not evidence that the actual interacting moving-projector response is nonzero, nor evidence against the conventional quasiparticle-vacuum no-sea prescription. The primary literature instead defines no-sea through the quasiparticle branch and its density sums (Nikšić et al., arXiv:1403.4039, Eqs. 46–48), while the D1S RHFB prescription retains the full Dirac component structure (Geng & Long, arXiv:2202.00482, Eq. 20d and Appendix B); the two formulations should not be conflated with the diagnostic's free-projector labels.

## Decision and stop criteria

Do not label an active-commutator root as a stationary point of the energy composed with `P_+(h[rho])` unless the following state-level gate passes:

1. Certify the spectral gap and compare the rational-sign projector derivative with the exact divided-difference derivative on a tiny oracle case.
2. Independently evaluate the full pairing field's `P`/`1-P` cross-block contraction against `D P_h[D h[delta rho]]` over a basis of number-constrained blocked canonical tangents. Report its norm and the maximum absolute directional derivative.
3. Compare centered finite differences of the composed energy, with the projector recomputed on both sides, against the active Thouless derivative plus the explicit projector-response term. Require agreement at a declared numerical tolerance and a response norm below the declared physical stationarity tolerance before treating `[H_a,R_a]=0` alone as sufficient.
4. Separately compare against the chosen standard no-sea blocked-RHFB quasiparticle-vacuum density construction and full canonical stationarity. Verify the blocked quasiparticle branch, number, generalized-density purity, and the selected treatment of sea qp sums.

If the target is standard blocked RHFB/no-sea, use the literature-defined full quasiparticle vacuum, its blocked Thouless variation, and the energy-derived `h,Delta`; do not add `P_+(h[rho])` as a density-support constraint. If the project intentionally selects the moving-projector energy model, keep it explicitly labeled as that model and add the projector-response derivative (and any required constrained-variation multiplier equations). If the response is intentionally zeroed by projecting the pairing interaction, document that as a changed functional and validate it separately against the desired D1S prescription. A small AD residual or a successful differentiable projector implementation is not evidence of variational equivalence.

## Code pointers

- [`projected_rhfb_fixedpoint.py`](../dpl_rhf/rhfb/projected_rhfb_fixedpoint.py): projector construction, embedding, fields, and active residual.
- [`no_sea_particle_space.py`](../dpl_rhf/rhfb/no_sea_particle_space.py): active fixed-projector commutator and response warning.
- [`pairing.py`](../dpl_rhf/rhfb/pairing.py): primitive all-J D1S energy and full pairing field action.
- [Existing RHFB/no-sea prescription review](RHFB_NO_SEA_PRESCRIPTION_REVIEW_20261006.md): broader target selection and validation boundaries.
