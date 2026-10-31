"""Explicit GPU-1-only small-space check; run via the protected launcher."""
import torch

from dpl_rhf.rhfb.cuda_resources import configure_cuda
from dpl_rhf.rhfb.network_number_transport import apply_mean_number_correction
from dpl_rhf.rhfb.tests.test_main_physics_integration import networks


def test_cuda_transport_matches_cpu_and_keeps_network_gradients():
    policy=configure_cuda("cuda:0",budget_mib=2048)
    assert policy["physical_index"] == 1
    _,cpu_models=networks()
    _,gpu_models=networks()
    cpu,gpu=cpu_models["n"],gpu_models["n"].to("cuda:0")
    before_cpu=cpu.state_dict()
    gpu.load_state_dict(before_cpu,strict=True)
    cpu_report=apply_mean_number_correction(cpu,2)
    gpu_report=apply_mean_number_correction(gpu,2)
    assert cpu_report.converged and gpu_report.converged
    assert cpu_report.q_unchanged and gpu_report.q_unchanged
    _,cpu_density=cpu()
    _,gpu_density=gpu()
    torch.testing.assert_close(cpu_density.matrix,gpu_density.matrix.cpu(),atol=1.e-10,rtol=0.)

    def gradient(model):
        _,gd=model()
        diagonal=torch.arange(model.active_dimension,dtype=torch.float64,
                              device=gd.rho.device)
        energy=torch.sum(diagonal*torch.diag(gd.rho).real)+.1*gd.chi.real.sum()
        return torch.autograd.grad(energy,model.raw_b)[0]
    grad_cpu,grad_gpu=gradient(cpu),gradient(gpu).cpu()
    assert float(torch.linalg.vector_norm(grad_cpu)) > 0
    torch.testing.assert_close(grad_cpu,grad_gpu,atol=2.e-9,rtol=1.e-9)
