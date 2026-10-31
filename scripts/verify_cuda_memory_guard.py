"""Small protected CUDA allocation/recompute proof, independent of nuclear physics."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dpl_rhf.rhfb.cuda_resources import configure_cuda, cuda_snapshot
from dpl_rhf.rhfb.runtime_resources import resource_snapshot

def main():
    import torch
    from torch.utils.checkpoint import checkpoint
    policy = configure_cuda('cuda:0', int(os.environ['RHFB_CUDA_BUDGET_MIB']))
    # The over-budget request must fail before allocating the requested storage.
    refused = False
    try:
        torch.empty((policy['driver_monitor_budget_bytes'],), dtype=torch.uint8, device='cuda:0')
    except torch.cuda.OutOfMemoryError:
        refused = True
    if not refused:
        raise RuntimeError('CUDA allocation beyond the hard allocator budget was accepted')
    torch.cuda.empty_cache()
    x = torch.linspace(-0.4, 0.6, 256, dtype=torch.float64, device='cuda:0').to(torch.complex128)
    x.requires_grad_()
    counts = {'executions': 0}
    def block(values):
        counts['executions'] += 1
        return (values.sin().abs().square() + values.abs().pow(4)).sum()
    plain = block(x)
    expected, = torch.autograd.grad(plain, x)
    counts['executions'] = 0
    rebuilt = checkpoint(block, x, use_reentrant=False)
    actual, = torch.autograd.grad(rebuilt, x)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    if counts['executions'] < 2:
        raise RuntimeError('backward did not recompute the checkpointed block')
    torch.cuda.synchronize()
    report = {'status': 'passed', 'allocator_over_budget_refused': refused,
              'recompute_gradient_equal': True, 'block_executions': counts['executions'],
              'policy': policy, 'cuda_memory': cuda_snapshot('cuda:0'),
              'host_memory': resource_snapshot(), 'fortran_used': False}
    path = Path(sys.argv[1])
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(report, indent=2)+'\n')
    temporary.replace(path)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
