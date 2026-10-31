from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import os
import tempfile

import torch

from . import FORMAT_VERSION
from .config import BlockedRHFBConfig
from .prescription import (PhysicalPrescription, checkpoint_prescription_fields,
                           validate_checkpoint_prescription)


def save_checkpoint(path, config: BlockedRHFBConfig, *, basis_spec: dict,
                    blocked_reference: dict, physics_prescription: PhysicalPrescription,
                    network_state: dict,
                    optimizer_state: dict | None = None,
                    trainer_state: dict | None = None) -> None:
    """Save trusted local state atomically; no legacy RHF state is reinterpreted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "solver": "blocked-rhfb",
        "config": asdict(config),
        "config_fingerprint": config.fingerprint(),
        "basis_spec": basis_spec,
        "blocked_reference": blocked_reference,
        "network_state": network_state,
        "optimizer_state": optimizer_state,
        "trainer_state": trainer_state,
    }
    payload.update(checkpoint_prescription_fields(physics_prescription))
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, config: BlockedRHFBConfig, *, basis_spec: dict,
                    blocked_reference: dict, physics_prescription: PhysicalPrescription,
                    map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("solver") != "blocked-rhfb" or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("checkpoint is not a compatible blocked-RHFB state")
    if payload.get("config_fingerprint") != config.fingerprint():
        raise ValueError("RHFB configuration mismatch; use an explicit initial-state conversion")
    if payload.get("basis_spec") != basis_spec:
        raise ValueError("RHFB primitive/active-space mismatch")
    if payload.get("blocked_reference") != blocked_reference:
        raise ValueError("RHFB blocked-reference mismatch")
    validate_checkpoint_prescription(payload, physics_prescription)
    return payload
