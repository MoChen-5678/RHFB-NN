from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
import numpy
import scipy
import sympy
import torch
from torch.utils.checkpoint import checkpoint

from dpl_rhf import __version__
from dpl_rhf.rhf.disk_offload import SavedTensorDiskOffload
from dpl_rhf.rhf.parameters import rhf_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a standalone DPL-RHF installation")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    x = torch.linspace(-1.0, 1.0, 32, device=device, requires_grad=True)
    disk_written = 0
    if device.type == "cuda":
        with tempfile.TemporaryDirectory(prefix="dpl-rhf-check-") as temporary:
            offload = SavedTensorDiskOffload(Path(temporary), min_bytes=0)
            try:
                with offload:
                    first = torch.autograd.grad(x.sin().square().sum(), x, create_graph=True)[0]
                    second_loss = first.square().sum()
                second_loss.backward()
                disk_written = offload.bytes_written
            finally:
                offload.cleanup()
    else:
        first = torch.autograd.grad(x.sin().square().sum(), x, create_graph=True)[0]
        first.square().sum().backward()

    parameters = rhf_parameters("PKA1")
    checkpoint_input = torch.linspace(
        -1.0, 1.0, 32, device=device, requires_grad=True
    )
    checkpoint_output = checkpoint(
        lambda values: values.sin().square().sum(),
        checkpoint_input,
        use_reentrant=False,
        preserve_rng_state=False,
    )
    checkpoint_first = torch.autograd.grad(
        checkpoint_output, checkpoint_input, create_graph=True
    )[0]
    checkpoint_first.square().sum().backward()
    report = {
        "dpl_rhf_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "sympy": sympy.__version__,
        "matplotlib": matplotlib.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "float64_second_derivative_finite": bool(torch.isfinite(x.grad).all()),
        "disk_offload_test_bytes": disk_written,
        "recompute_second_derivative_finite": bool(
            torch.isfinite(checkpoint_input.grad).all()
        ),
        "pka1_channels": [
            "sigma_scalar",
            "omega_vector",
            "rho_vector",
            "rho_tensor",
            "rho_vector_tensor",
            "pion_pseudovector",
            "coulomb",
        ],
        "pka1_f_pi": parameters.f_pi,
        "pka1_f_rho_over_g_rho": parameters.f_rho,
    }
    report["passed"] = bool(
        sys.version_info >= (3, 10)
        and report["float64_second_derivative_finite"]
        and report["recompute_second_derivative_finite"]
        and (not args.require_cuda or torch.cuda.is_available())
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
