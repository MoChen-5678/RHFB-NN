# Exact Fock recomputation benchmark

## Scope

The `recompute` mode changes only autograd activation storage. Each complete
Fock channel is evaluated through PyTorch non-reentrant activation
checkpointing. During backward, the original float64 channel formula is run
again from the same nonlocal-density blocks and density-dependent vertices.
No physical channel, matrix element, radial point or variational term is
removed or approximated.

## Exactness checks

PKA1 O16 was evaluated with identical network parameters in the original and
recompute paths. The compared objective contained energy per nucleon and the
squared projected stationarity residual, so its network gradient includes the
required second derivative of the RHF functional.

| check | CPU difference | CUDA difference |
| --- | ---: | ---: |
| total loss | 0 | 0 |
| stationarity residual | 0 | 0 |
| maximum energy-channel difference | 0 | 0 |
| maximum network-gradient difference | 0 | 3.33e-16 |
| network-gradient relative L2 difference | 0 | 8.97e-15 |

These are float64 rounding-level differences. The physical objective and its
second-order parameter gradient are unchanged.

## Performance checks

Hardware and software:

- one Tesla V100-SXM2 32 GB;
- PyTorch 2.5.1+cu121;
- float64 throughout;
- hidden width 64, depth 3, radial basis size 16;
- two complete forwards and one optimizer backward (`epochs=1`);
- no sparse no-sea refresh during the timed interval.

### PKA1 Ca48

| mode | wall time | maximum RSS | graph I/O | peak GPU memory |
| --- | ---: | ---: | ---: | ---: |
| disk | 294.2 s | 1.38 GiB | 28.68 GiB write / 73.74 GiB read | not sampled |
| recompute | 39.5 s | 1.36 GiB | no graph files | 10.42 GiB |

Both modes produced exactly:

```text
epoch 0: loss=-8.32763464, E/A=-8.35917439, stationarity=3.209e-01 MeV
epoch 1: loss=-8.32770542, E/A=-8.35921540, stationarity=3.207e-01 MeV
```

On this storage device, recomputation was about 7.4 times faster.

### PKA1 Sn132 checkpoint probe

One optimizer step from the production epoch-160 checkpoint plus the final
evaluation took 81 seconds and peaked at 22.37 GiB GPU memory. The formal
output directory was not modified by this probe.

```text
epoch 160: loss=-8.28558352, E/A=-8.28828691, stationarity=8.648e-02 MeV
epoch 161: loss=-8.28558878, E/A=-8.28829210, stationarity=8.648e-02 MeV
```

## Operational recommendation

Use `recompute` on a 32 GB V100 for PKA1 through Sn132. Keep the cgroup host
memory ceiling as a fault-containment measure. Pb208's ordinary training step
also fits when the exact initial-state stationarity weight is supplied. The
automatic weight calibration adds two retained gradient traversals and exceeds
32 GB; the identical disk-mode initial state gave
`0.4657901750825708 MeV^-1`, which is used by the persistent Pb208 batch.
because its number of occupied `(q,kappa,n)` blocks is larger; fall back to
`disk` if its measured CUDA peak exceeds the available device memory.
