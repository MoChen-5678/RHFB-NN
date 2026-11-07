"""Canonical pairing space, coherent pair-sign start and mapped warm start."""

from __future__ import annotations

import json

import torch

from dpl_rhf.backends.base import NucleusCase
from dpl_rhf.rhf.hfb_backend import SphericalRHFBBackend
from dpl_rhf.rhf import hfb_trainer as T

torch.set_default_dtype(torch.float64)
CPU = torch.device("cpu")


def _backend(z=8, n=10, block=None, **kwargs):
    options = dict(pairing=False, kappa_limits=(3, 2), blocking={"n": block, "p": None},
                   capacities={"n": 20, "p": 14})
    options.update(kwargs)
    return SphericalRHFBBackend(NucleusCase("PKA1", z, n, z + n), device=CPU, **options)


def test_min_states_per_kappa_fills_every_block_without_changing_occupied_shells():
    small = _backend()
    large = _backend(min_states_per_kappa=4)
    per_kappa = {(s.species_sign, s.kappa): len(s.principals) for s in large.block_specs}
    assert set(per_kappa) == {(sign, k) for sign in (1.0, -1.0) for k in (-3, -2, -1, 1, 2)}
    assert min(per_kappa.values()) >= 4
    occupied = lambda b: {(s.species_sign, s.kappa, p): float(o)
                          for s in b.block_specs for p, o in zip(s.principals, s.occupations) if o > 0}
    assert occupied(small) == occupied(large)


def test_pair_signs_start_in_the_coherent_phase_pattern():
    backend = _backend(min_states_per_kappa=2)
    for spec, signs in zip(backend.block_specs, backend.pair_signs):
        assert torch.all(signs == (-1) ** (abs(spec.kappa) - 1))
    legacy = _backend(min_states_per_kappa=2, pair_sign_init="ones")
    assert all(torch.all(signs == 1.0) for signs in legacy.pair_signs)


def test_fermi_shift_conserves_particle_number_with_blocking():
    backend = _backend(n=9, block=-3, min_states_per_kappa=3)
    generator = torch.Generator().manual_seed(3)
    logits = [torch.randn(len(s.principals), generator=generator) * 4 for s in backend.block_specs]
    state = backend.occupation_state(logits)
    for name, target in (("n", 9), ("p", 8)):
        number = sum(float(s.degeneracy * o.sum()) for s, o in zip(backend.block_specs, state.occupations)
                     if (s.species_sign > 0) == (name == "n"))
        assert abs(number - target) < 1e-10


def test_mapped_warm_start_reproduces_matched_orbitals(tmp_path):
    old = _backend()
    old_net = T.build_network(old, T.RHFBTrainingConfig(hidden=8, depth=1))
    with torch.no_grad():
        for head in old_net.output_heads:
            head.weight.normal_(std=0.3)
            head.bias.normal_(std=0.3)
        for logit in old_net.occupation_logits:
            logit.normal_(std=3.0)
    old.pair_signs = [torch.where(torch.rand(len(s)) > 0.5, 1.0, -1.0) for s in old.pair_signs]
    path = tmp_path / "model_final.pt"
    torch.save({"model_state": old_net.state_dict(), "pair_signs": old.pair_signs}, path)
    blocks = [{"species": "n" if s.species_sign > 0 else "p", "kappa": s.kappa,
               "principals": list(s.principals)} for s in old.block_specs]
    (tmp_path / "summary.json").write_text(json.dumps({"canonical_space": {"blocks": blocks}}))

    new = _backend(min_states_per_kappa=4)
    new_net = T.build_network(new, T.RHFBTrainingConfig(hidden=8, depth=1))
    info = T._load_mapped_warm_start(new_net, new, torch.load(path, weights_only=False), path,
                                     logit_clip=1e9)
    assert info["matched_states"] == sum(len(s.principals) for s in old.block_specs)
    old_blocks, new_blocks = old_net(), new_net()
    for io, so in enumerate(old.block_specs):
        i = next(i for i, s in enumerate(new.block_specs) if (s.species_sign, s.kappa) == (so.species_sign, so.kappa))
        k = len(so.principals)
        assert torch.allclose(old_blocks[io]["factors"], new_blocks[i]["factors"][:, :k], atol=1e-14)
        assert torch.allclose(old_net.occupation_logits[io], new_net.occupation_logits[i][:k], atol=1e-12)
        assert torch.equal(old.pair_signs[io], new.pair_signs[i][:k])
