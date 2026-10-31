# DPL-NN-RHFB: no-refresh neural-network relativistic Hartree-Fock-Bogoliubov (spherical, PKA1 + D1S)

This package is a spherical **relativistic Hartree-Fock-Bogoliubov (RHFB)**
solver. The particle-hole channel is PKA1, with the full Fock terms of σ, ω,
ρ (vector and tensor), π and Coulomb. The pairing channel is Gogny D1S at
finite range (J=0, all four Dirac components). Odd nuclei use the spherical
equal-filling approximation.

- **Canonical (Bloch–Messiah) parameterization.** A neural network
  generates the canonical Dirac spinors. Each canonical level has one logit,
  giving v² = σ(t+c) and uv.
- **Exact particle number.** At every step a Fermi shift c is found by
  bisection so that Tr ρ = N exactly, and its implicit derivative is kept.
- **No Dirac sea.** A positive-energy projector P⁺ of the self-consistent
  Dirac Hamiltonian is built with the Newton–Schulz sign iteration, using
  matrix products only.
- **No diagonalization of any Dirac, HF or HFB matrix, and no basis
  refresh.**
- **Memory-bounded Fock exchange.** The exchange is checkpointed per κ pair
  and can be split across several GPUs.

## Validation (versus the Fortran RHFB program S-RHFB-C)

The reference uses the same PKA1+D1S functional, the same 0.1 fm / 20 fm
mesh and the same equal-filling blocking. All energies are without the
centre-of-mass correction.

| Nucleus | E/A NN (MeV) | E/A Ref (MeV) | Δ (keV/A) | E_pair(n) NN / Ref (MeV) | r_ch NN / Ref (fm) |
|---|---|---|---|---|---|
| ¹⁶O | -7.34572 | -7.34892 | +3.2 | -0.05 / 0.00 | 2.8607 / 2.8588 |
| ¹⁷O (ν1d5/2 blocked) | -7.23310 | -7.23287 | -0.2 | -0.00 / 0.00 | 2.8470 / 2.8458 |
| ¹⁸O | -7.29154 | -7.29958 | +8.0 | -4.52 / -5.21 | 2.8360 / 2.8374 |
| ¹⁹O (ν1d5/2 blocked) | -7.14750 | -7.14642 | -1.1 | -2.42 / -2.95 | 2.8259 / 2.8273 |
| ⁴⁰Ca | -8.33649 | -8.33772 | +1.2 | -0.11 / 0.00 | 3.5478 / 3.5479 |
| ⁴⁸Ca | -8.49344 | -8.47872 | -14.7 | -0.22 / 0.00 | 3.5102 / 3.5087 |
| ¹³²Sn | -8.30409 | -8.29735 | -6.7 | -0.04 / 0.00 | 4.7049 / 4.7063 |
| ²⁰⁸Pb | -7.83902 | -7.83561 | -3.4 | -0.05 / 0.00 | 5.5162 / 5.5154 |
| ²¹⁰Pb | -7.79899 | -7.79777 | -1.2 | -0.99 / -3.26 | 5.5288 / 5.5304 |

Closed-shell nuclei (¹⁶O, ⁴⁰Ca, ⁴⁸Ca, ¹³²Sn, ²⁰⁸Pb) lie 0.8–3.2 keV/A above
the independent coordinate-space Core-1204 RHF. For ⁴⁸Ca, ¹³²Sn and ²⁰⁸Pb the
NN energies are *below* S-RHFB-C. Since the NN energy is a variational upper
bound, S-RHFB-C (basis expansion) has not converged to the minimum for these
nuclei.

For ¹⁸O, λn and the pairing gap agree with the reference. The pairing energies
of ¹⁸O and ²¹⁰Pb are weaker, most likely because of the truncated canonical
space (not yet confirmed). Canonical single-particle energies agree to 7–86 keV
on average; fully occupied levels within one κ block are compared by their
block trace.

Same-prescription Fortran references are bundled in
`outputs/fortran_reference/` for:

- O16–O20;
- Ca40, Ca41, Ca42, Ca44, Ca48;
- Sn132, Pb208, Pb210 (KMX=(8,8)).

## 1. Installation

```bash
tar xzf dpl-nn-rhfb-*.tar.gz && cd dpl-nn-rhfb-*
bash install_env.sh            # creates ./env (torch 2.5.1+cu121, numpy 2.2.6, scipy 1.15.3)
```

`install_env.sh` only touches `./env` and `./.pkgs`. You need an NVIDIA
driver supporting CUDA 12.1 (≥ 530), and conda, mamba, micromamba or
python3.10.

## 2. Resource protection (required reading)

**The RHFB training command refuses to run outside a resource guard.** It
checks that the process is inside a kernel cgroup with a memory limit, or
inside a live `run_rhfb_procguard.py` session. Use the provided scripts;
they always start jobs through the guard. The guard does the following:

- samples the **resident memory of the whole process tree every 0.5 s** and
  **each card's VRAM every 2 s**;
- terminates the whole job on overrun: host memory `HOST_MIB` (default
  8192), any card `GPU_MIB` (default 8192), or timeout;
- sets a **PyTorch allocator hard limit** on every card (`GPU_MIB` minus
  512 MiB);
- uses **only the physical cards listed in `RHFB_AUTHORIZED_GPUS`**; if the
  variable is unset, no GPU is used;
- admits a card only if its free VRAM is at least `GPU_MIB` + 1 GiB;
- takes one lock per card: jobs that share a card are mutually exclusive,
  while different cards can run in parallel.

It is sampling-based, not a kernel limit. **Never kill other users'
processes.**

```bash
export RHFB_AUTHORIZED_GPUS=0,1,2,3     # cards you may use on this server
```

## 3. Acceptance test (before the first run)

```bash
examples/run_tests.sh 0,1        # two-card streamed Fock equivalence + RHFB backend (a few minutes)
FULL=1 examples/run_tests.sh 0,1 # adds the m-scheme pairing oracle (CPU-heavy, 10+ minutes)
```

Pass criterion: everything `passed` and `logs/guard_tests.json` shows
`"status": "completed"`.

## 4. Running

```
examples/run_one.sh NAME Z N EPOCHS CAP_N CAP_P KNEG KPOS GPUS [INIT_NAME] [BLOCK_KAPPA]
```

| Argument | Meaning |
|---|---|
| CAP_N / CAP_P | Shell capacity of the neutron/proton canonical space (canonical-space truncation) |
| KNEG KPOS | κ range: −KNEG ≤ κ ≤ KPOS |
| GPUS | `0` = one card; `0,1` = Fock exchange split across two cards |
| INIT_NAME | Warm start from `runs/INIT_NAME/model_final.pt` (needs exactly the same canonical space) |
| BLOCK_KAPPA | κ of the blocked neutron level for odd N (e.g. −3 = 1d5/2, −4 = 1f7/2) |

Settings already used:

| Nucleus | CAP_N | CAP_P | KNEG KPOS | Note |
|---|---|---|---|---|
| O16–O20 | 40 | 28 | 5 4 | Odd isotopes `-3`; warm start from O16/O18 |
| Ca40–Ca48 | 50 | 40 | 5 4 | Ca41 `-4`; Ca48 warm starts from Ca40 |
| Sn132 | 100 | 58 | 7 6 | Needs h11/2 |
| Pb208 / Pb210 | 148 | 92 | 8 7 | Needs i13/2, j15/2; Pb210 can warm-start from Pb208 |

```bash
examples/run_one.sh O16   8  8   500 40 28 5 4 0
examples/run_one.sh O17   8  9   500 40 28 5 4 0 O16 -3
examples/run_one.sh Pb208 82 126 500 148 92 8 7 0,1
```

Each run trains first and then re-evaluates automatically: the no-sea
projector and pair signs are iterated to self-consistency, and the
**final results** go to `runs_eval/NAME/`.

### Running in the background, safe against dropped connections

```bash
setsid nohup examples/run_one.sh Sn132 50 82 500 100 58 7 6 0,1 > logs/Sn132.out 2>&1 < /dev/null &
# several two-card lanes in parallel (edit the card assignments in examples/run_lanes.sh first):
setsid nohup examples/run_lanes.sh > logs/lanes.out 2>&1 < /dev/null &
```

### Notes

- **Resuming.** Run the same command again; it resumes automatically from
  `runs/NAME/training_checkpoint.pt` (checkpoints every 10 epochs).
- **Warm-start models.** `warm_start/{O16,O18,Ca40}/model_final.pt` are
  bundled. To use them, copy one to `runs/O16/` (and so on) and pass
  `O16` as INIT_NAME.
- **Measured speed** (time per epoch, with 4 lanes running in parallel on
  A100s at the same speed):

  | Nucleus | T4 | A100 |
  |---|---|---|
  | O | 2–4 s | — |
  | Ca | about 9 s | — |
  | Sn132 | — | about 14 s |
  | Pb208, Pb210 | — | about 20 s |

  The computation is CPU-bound. On a heavily loaded machine (check
  `uptime`) it slows down noticeably, and adding more GPUs does not help.
- **Measured per-card VRAM peaks** (two-card streaming, T4):

  | Nucleus | Per-card peak |
  |---|---|
  | ⁴⁰Ca | 2.3 GB |
  | ¹³²Sn | 3.7 GB |
  | ²⁰⁸Pb | 5.1 GB |

  Host memory stays below 2 GB.

## 5. Output

`runs_eval/NAME/` (final, self-consistent) and `runs/NAME/` (training) contain:

| File | Contents |
|---|---|
| `summary.json` | `energy_per_a_no_com_mev`, `energy_terms_mev` (including `E_pair_n/p`), `chemical_potentials_mev`, `pairing_gap_reference_definition_mev` (same definition as the reference program), `radii_fm`, `stationarity`, `no_sea.sea_leakage_self_consistent` (should be ~1e-11), `canonical_levels` (h_νν, Δ_νν, v², and blocked flag for each level), `counts` (eigensolve = 0) |
| `history.csv` | Per-epoch E/A, pairing energy, λ, residuals, branch score, time |
| `profiles.npz` | r, local densities, neutron/proton local pairing density `kappa_local_n/p` |
| `model_final.pt`, `training_checkpoint.pt` | Model (usable for warm starts) / resumable checkpoint |

### Comparing with the reference

```bash
env/bin/python scripts/compare_rhfb_fortran.py --nn-root runs_eval --nuclei O16 O18 Ca40 --out comparison.json
```

## 6. Sending results back

```bash
tar czf results_$(hostname)_$(date +%m%d).tar.gz runs_eval runs/*/history.csv runs/*/model_final.pt logs/
```

## 7. Common problems

| Message | Meaning and fix |
|---|---|
| `requires an enforced cgroup memory limit` | Started without the guard; use `examples/run_one.sh` |
| `No authorized GPU list` | Set `RHFB_AUTHORIZED_GPUS` |
| `unavailable or busy` | Not enough free VRAM on that card |
| `Another guarded RHFB job is active on the requested devices` | One of your own jobs already holds that card |
| `gpu_memory_budget_exceeded` | Use two cards for heavy nuclei (`GPUS=0,1`) |
| `--kappa-limits exclude occupied shell` | The κ range is too small for this nucleus (e.g. Pb needs 8 7) |
| A warm start fails to load | The canonical spaces (CAP_N, CAP_P, KNEG, KPOS) don't match exactly |
