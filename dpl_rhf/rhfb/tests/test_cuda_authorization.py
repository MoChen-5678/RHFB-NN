"""Host GPU authorization precedes CUDA initialization and allocation."""
import json
from types import SimpleNamespace

import pytest
import torch

from dpl_rhf.rhfb import cuda_resources as resources
from dpl_rhf.rhfb.execution_limits import ResourceSafetyError


def _policy(monkeypatch, tmp_path, physical_index):
    selected_uuid = f"GPU-test-physical-{physical_index}"
    monkeypatch.setenv("RHFB_CUDA_UUID", selected_uuid)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", selected_uuid)
    monkeypatch.setenv("RHFB_CUDA_BUDGET_MIB", "2048")
    monkeypatch.setenv("RHFB_CUDA_POLICY_EVIDENCE", str(tmp_path / "policy.json"))
    monkeypatch.setattr(resources, "gpu_inventory", lambda: [
        {"uuid": selected_uuid, "index": physical_index}])


def test_other_physical_gpu_refused_before_cuda_initialization(monkeypatch, tmp_path):
    # 2026-10-27 authorization covers physical GPUs 0 and 1 only.
    _policy(monkeypatch, tmp_path, 2)

    def forbidden(*args, **kwargs):
        raise AssertionError("unauthorized CUDA device was accessed")

    for name in ("is_available", "device_count", "current_device", "set_device"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    with pytest.raises(ResourceSafetyError, match="only physical GPU indices 0 and 1"):
        resources.configure_cuda("cuda:0", budget_mib=2048)
    assert not (tmp_path / "policy.json").exists()


def test_physical_one_maps_to_local_zero_and_keeps_allocator_limit(monkeypatch, tmp_path):
    _policy(monkeypatch, tmp_path, 1)
    allocations = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(
        total_memory=16*1024**3))
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction, index: allocations.append((fraction, index)))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "mock GPU 1")
    evidence = resources.configure_cuda("cuda:0", budget_mib=2048)
    assert evidence["physical_index"] == 1
    assert evidence["device"] == "cuda:0"
    assert evidence["allocator_limit_bytes"] == 1536*1024**2
    assert allocations == [(1536/16384, 0)]
    assert json.loads((tmp_path/"policy.json").read_text()) == evidence


def test_two_authorized_gpus_get_separate_allocator_limits(monkeypatch, tmp_path):
    uuids = "GPU-test-physical-1,GPU-test-physical-0"
    monkeypatch.setenv("RHFB_CUDA_UUID", uuids)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", uuids)
    monkeypatch.setenv("RHFB_CUDA_BUDGET_MIB", "2048")
    monkeypatch.setenv("RHFB_CUDA_POLICY_EVIDENCE", str(tmp_path / "policy.json"))
    monkeypatch.setattr(resources, "gpu_inventory", lambda: [
        {"uuid": "GPU-test-physical-0", "index": 0}, {"uuid": "GPU-test-physical-1", "index": 1}])
    allocations = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(
        total_memory=16*1024**3))
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction, index: allocations.append((fraction, index)))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "mock GPU")
    evidence = resources.configure_cuda("cuda:0", budget_mib=2048)
    assert evidence["physical_index"] == [1, 0]
    assert allocations == [(1536/16384, 0), (1536/16384, 1)]
