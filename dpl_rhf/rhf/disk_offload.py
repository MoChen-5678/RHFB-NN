from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
import threading
from typing import BinaryIO

import numpy as np
import torch


@dataclass(frozen=True)
class _DiskTensor:
    path: Path
    device: torch.device
    dtype: torch.dtype
    shape: tuple[int, ...]
    offset: int
    numel: int


class SavedTensorDiskOffload(AbstractContextManager["SavedTensorDiskOffload"]):
    """Store large autograd-saved CUDA tensors on disk without changing values."""

    def __init__(self, root: str | Path, *, min_bytes: int = 1 << 20) -> None:
        if min_bytes < 0:
            raise ValueError("min_bytes must be non-negative")
        root_path = Path(root).expanduser().resolve()
        root_path.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="rhf-graph-", dir=root_path))
        self.min_bytes = int(min_bytes)
        self.files_written = 0
        self.bytes_written = 0
        self.files_read = 0
        self.bytes_read = 0
        self._counter = 0
        self._lock = threading.Lock()
        self._hooks: AbstractContextManager[object] | None = None
        self._files: dict[torch.dtype, BinaryIO] = {}

    @staticmethod
    def _supported_dtype(dtype: torch.dtype) -> bool:
        return dtype in {
            torch.float64,
            torch.float32,
            torch.float16,
            torch.int64,
            torch.int32,
            torch.int16,
            torch.int8,
            torch.uint8,
        }

    @staticmethod
    def _numpy_dtype(dtype: torch.dtype) -> np.dtype:
        return np.dtype(
            {
                torch.float64: np.float64,
                torch.float32: np.float32,
                torch.float16: np.float16,
                torch.int64: np.int64,
                torch.int32: np.int32,
                torch.int16: np.int16,
                torch.int8: np.int8,
                torch.uint8: np.uint8,
            }[dtype]
        )

    def _file_for(self, dtype: torch.dtype) -> tuple[Path, BinaryIO]:
        handle = self._files.get(dtype)
        path = self.directory / f"saved-{str(dtype).replace('torch.', '')}.bin"
        if handle is None:
            handle = path.open("w+b", buffering=0)
            self._files[dtype] = handle
        return path, handle

    def _pack(self, tensor: torch.Tensor) -> torch.Tensor | _DiskTensor:
        size = tensor.numel() * tensor.element_size()
        if (
            tensor.device.type != "cuda"
            or size < self.min_bytes
            or not self._supported_dtype(tensor.dtype)
        ):
            return tensor.detach()
        with self._lock:
            path, handle = self._file_for(tensor.dtype)
            offset = handle.tell() // tensor.element_size()
            cpu = tensor.detach().to(device="cpu").contiguous()
            cpu.numpy().tofile(handle)
            self._counter += 1
        self.files_written += 1
        self.bytes_written += size
        return _DiskTensor(
            path=path,
            device=tensor.device,
            dtype=tensor.dtype,
            shape=tuple(tensor.shape),
            offset=offset,
            numel=tensor.numel(),
        )

    def _unpack(self, packed: torch.Tensor | _DiskTensor) -> torch.Tensor:
        if isinstance(packed, torch.Tensor):
            return packed
        handle = self._files.get(packed.dtype)
        if handle is not None:
            handle.flush()
        # Map only this tensor's byte range. Mapping [0, offset+numel] for every
        # unpack makes late tensors reserve tens of GiB of redundant virtual
        # address space and fails on low-memory hosts.
        mapped = np.memmap(
            packed.path,
            dtype=self._numpy_dtype(packed.dtype),
            mode="c",
            offset=packed.offset * torch.empty((), dtype=packed.dtype).element_size(),
            shape=packed.shape,
        )
        tensor = torch.from_numpy(mapped).to(packed.device)
        self.files_read += 1
        self.bytes_read += tensor.numel() * tensor.element_size()
        return tensor

    def __enter__(self) -> "SavedTensorDiskOffload":
        self._hooks = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)
        self._hooks.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        if self._hooks is None:
            return None
        return self._hooks.__exit__(exc_type, exc_value, traceback)

    def cleanup(self) -> None:
        for handle in self._files.values():
            handle.close()
        self._files.clear()
        shutil.rmtree(self.directory, ignore_errors=True)


def remove_stale_graph_directories(root: str | Path) -> int:
    """Remove graph caches left by a killed process; never touches checkpoints."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        return 0
    removed = 0
    for path in root_path.glob("rhf-graph-*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    return removed
