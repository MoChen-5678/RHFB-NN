# Draft tests for tile-width-compatible resume

Future tests only. These are deliberately not under `tests/` because the runtime still treats tile width as an exact compatibility field. Add them only if the proposed `_resume_compatibility` change is adopted. Do not run numerical tests except through `scripts/run_rhfb_limited.py`.

## Canonical compatibility contract

Add a small metadata fixture with every key read by `_resume_compatibility`:

```python
def _compat_metadata():
    return {
        "physics": "PKA1 + D1S",
        "production_physics_asserted": True,
        "optimization_target": "stationarity residual",
        "physical_prescription": {"density": "covariant", "pairing": "D1S"},
        "particle_hole_parameters": {"g_sigma": 1.0},
        "pairing_parameters": {"strength": 1.0},
        "normal_functional": {
            "class": "StreamedPrimitiveNormalFunctional",
            "lmax": 2,
            "tile_size": 8,
            "chunk_size": 32,
        },
        "stationarity_evaluator": "production.evaluator",
        "basis": {
            "dimension": 48, "point_count": 108, "radius_fm": 1.4,
            "spacing_fm": 0.2, "kappa_max": 1, "n_theta": 3,
            "n_phi": 6, "derivative_order": 5,
        },
        "number_targets": {"n": 3, "p": 2},
        "models": {
            s: {"architecture": "same", "state_fingerprint": f"state-{s}"}
            for s in ("n", "p")
        },
        "model_reference_fingerprints": {"n": "ref-n", "p": "ref-p"},
        "device": "cpu",
        "options": {
            "normal_tile_size": 8, "normal_chunk_size": 32,
            "residual_chunk_size": 8, "pairing_chunk_size": 4,
            "pairing_active_chunk_size": 2, "lmax": 2,
            "number_weight": 1.0, "learning_rate": 1e-3,
            "number_augmented_lagrangian": False,
            "number_augmented_lagrangian_beta": 10000.0,
            "updates_requested": 1,
            "chemical_potential_initial": {"n": 0.0, "p": 0.0},
        },
    }
```

Then test that changing `normal_functional.tile_size` and `options.normal_tile_size` together preserves the canonical compatibility fingerprint, while changing any protected field changes it:

```python
def test_resume_compatibility_ignores_only_normal_tile_width():
    metadata = _compat_metadata()
    before = _fingerprint(_resume_compatibility(metadata))
    changed_tile = copy.deepcopy(metadata)
    changed_tile["normal_functional"]["tile_size"] = 32
    changed_tile["options"]["normal_tile_size"] = 32
    assert _fingerprint(_resume_compatibility(changed_tile)) == before

    for edit in (
        lambda m: m["basis"].update(radius_fm=1.6),
        lambda m: m["particle_hole_parameters"].update(g_sigma=1.01),
        lambda m: m["normal_functional"].update(lmax=4),
        lambda m: m["normal_functional"].update(chunk_size=64),
        lambda m: m["options"].update(learning_rate=2e-3),
    ):
        incompatible = copy.deepcopy(metadata)
        edit(incompatible)
        assert _fingerprint(_resume_compatibility(incompatible)) != before
```

Also retain a same-dimension basis mismatch case (radius/spacing/grid changed while D and P are held fixed) so compatibility is based on the complete basis metadata, not only matrix sizes.

## End-to-end resume and provenance

Extend `test_joint_training_resume_uses_full_accepted_checkpoint` using `_toy_evaluator`: make a checkpoint at tile size 8, resume from it at 16, then assert (a) the existing model and Adam state restore, (b) resumed progress completes under the requested update count, (c) checkpoint metadata records current tile 16, and (d) a provenance event records source tile 8, target tile 16, source checkpoint, accepted-step number, and that the change is execution-only with possible floating-point path differences. Add a second continuation and assert both tile transitions remain in `resume_provenance_history`.

Make separate resume checkpoints with one protected mismatch each (basis, PKA1 parameter, `lmax`, chunk size, and learning rate) and assert rejection before state restoration. If full production checkpoint construction makes this integration test too expensive, test the canonical fingerprint and retain the existing production checks for provider/basis/parameter/`lmax` agreement; do not substitute a fake production functional for an end-to-end physics-resume test.

