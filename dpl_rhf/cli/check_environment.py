from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import torch

from dpl_rhf import __release_name__, __version__
from dpl_rhf.legacy.dpl_rmf_core import resolve_core_dir


def inspect_environment(core_dir: str | None = None) -> tuple[dict, bool]:
    checks: dict[str, object] = {
        "dpl_rmf_version": __version__,
        "release_name": __release_name__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "matplotlib": matplotlib.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gfortran": shutil.which("gfortran"),
    }
    if torch.cuda.is_available():
        checks["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    core_ok = False
    try:
        resolved = resolve_core_dir(core_dir)
        ctypes.CDLL(str(resolved / "librhf.so"))
        checks["core1204_dir"] = str(resolved)
        checks["core1204_library_loadable"] = True
        core_ok = True
    except (FileNotFoundError, OSError) as error:
        checks["core1204_library_loadable"] = False
        checks["core1204_error"] = str(error)
    required_ok = sys.version_info >= (3, 10) and platform.machine() in {"x86_64", "amd64"}
    return checks, bool(required_ok and core_ok)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the DPL-RMF Python, CUDA, and Core-1204 runtime")
    parser.add_argument("--core-dir", default=None)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    report, ok = inspect_environment(args.core_dir)
    if args.require_cuda and not report["cuda_available"]:
        ok = False
    report["passed"] = ok
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
