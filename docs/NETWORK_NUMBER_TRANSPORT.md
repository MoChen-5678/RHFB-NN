# Number retraction in the neural canonical frame

`network_number_transport.apply_mean_number_correction(model, target)` applies
the existing finite-space canonical number retraction to the actual active
U/V frame returned by `model.forward_with_bogoliubov_frame()`. If correction
converges, it computes

\[
  C_\Delta=W_{\rm old}^\dagger W_{\rm corrected},\qquad
  C_{\rm total}\leftarrow C_{\rm total}C_\Delta.
\]

The model stores `C_total` in the registered `canonical_number_transport`
buffer. Given the fixed analytic Slater reference frame `W_ref` and learned
network transform `T`, its next normal forward uses

\[
  T_{\rm eff}=T W_{\rm ref}C_{\rm total}W_{\rm ref}^\dagger,
  \qquad W_{\rm next}=T_{\rm eff}W_{\rm ref}=T W_{\rm ref}C_{\rm total}.
\]

This makes the next `forward()` produce the corrected density itself. It does
not replace only an exported density. The primitive `Q` buffer and all
trainable `Parameter` objects are unchanged; optimizer state and gradients
remain attached to the same parameters. This operation is a canonical state
transport, not a basis/orbital refresh. `apply_mean_number_correction`
verifies that a new normal forward reproduces the corrected frame and keeps
Q fixed. The report includes before/after transport SHA-256 values.

The helper returns without changing the buffer when the current state already
has the target mean number or when retraction fails. A singular `N20` response
is reported as `singular_number_response`; the helper does not seed pairing to
manufacture a response. The current analytic transport requires a diagonal
zero/one Slater reference frame. If that frame is unavailable, it fails
explicitly rather than diagonalizing or refreshing a basis. No quasiparticle
identity is certified by transporting a column tag.

The registered buffer is part of the model `state_dict`; metadata advances to
the corresponding v2 format and records the transport convention. Strict loads
require the buffer, and the loader rejects its absence even with `strict=False`.
An explicit `upgrade_legacy_seed_state(model, legacy_state)` helper can convert
one exact-metadata v1 network seed by adding identity transport and v2
metadata. It rejects every state-key, architecture, or metadata mismatch and
is only for seed conversion; legacy training checkpoints are never silently
resumed with identity transport.
Line-search code must snapshot and restore `canonical_number_transport` along
with trainable weights; use `snapshot_number_transport()` and
`restore_number_transport()`.

## Physics/source basis

- L. M. Robledo and G. F. Bertsch, *Application of the gradient method to
  Hartree-Fock-Bogoliubov theory*, Phys. Rev. C 84, 014312 (2011),
  [arXiv:1104.5453](https://arxiv.org/abs/1104.5453). Eqs. (12)–(14) give
  generalized Thouless U/V updates and linear variations. Eqs. (20)–(22)
  explicitly correct an already constraint-violating starting configuration
  with `Z_deltaq=-M^{-1} deltaq Q20` and apply that correction together with
  any tangent-projected step. Sec. III states the generalized Thouless update
  preserves particle-number parity.
- Supplied read-only RHFB source set:
  `/home/ubuntu/rhf/dpl-param/reference_fortran/user-rhfb-20261015/`.
  `S-RHFB-C/RHFB.f90` lines 43–57 invokes `LambdaG`/`LambdaD` for initial
  chemical potentials; `S-RHFB-C/Dirac.f90` lines 383–419 iterates the
  chemical potential using the density/number error; `RHFB-BI/Broyden/Dirac.f90`
  lines 317–421 contains the corresponding particle-conservation chemical
  potential loop. Those are source observations for the separate lambda
  approach. This network transport instead implements the variational
  generalized-Thouless correction from the cited gradient-method paper.

This is finite-space mean-number correction only. It is not exact
particle-number projection and does not certify a no-sea variational branch,
stationarity, or a minimum.
