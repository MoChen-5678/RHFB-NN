# Numerical execution after the memory incident

The user's 2026-10-15 hardware authorization permits only physical GPU index 1
for out-of-sandbox numerical work. Resolve that index to its current UUID;
never select physical GPU 0 as a fallback. The launcher exposes the selected
physical GPU as process-local `cuda:0`, which does not mean physical GPU 0.
Retain busy-device admission checks and never terminate another GPU process.

The original program and backups must remain untouched. All development is
inside this variant. The user's no-Fortran and physics-literature requirements
continue to apply.

Every numerical RHFB test, diagnostic, or training job must run through
`scripts/run_rhfb_limited.py`: an actual kernel cgroup memory limit at most
4 GiB, no swap, runtime limit, and one shared numerical-job lock. A failed or
unavailable guard means stop that numerical launch; never fall back to an
unrestricted launch. Read-only source reviews do not need this launcher.

The dense coordinate prototype is quarantined for more than 192 spatial points
or 96 primitive dimensions. Do not remove/override its admission checks or
reduce the requested physical space to make a full-nucleus calculation appear
successful. Full-size evaluation requires a genuinely bounded-memory design
and verification under the same resource launcher first. Do not automatically
restart the interrupted O17 coordinate experiment.

There was no enforced 24-GiB limit in the interrupted experiment. Its JSON
`limits` field was declarative, not a kernel limit. Treat storage estimates
excluding AD tape as partial accounting, never as a peak-memory certificate.

2026-10-26 user decision: the user's Fortran RHFB program (S-RHFB-C) may be
compiled and run only as a post-training reference for the spherical neural
RHFB solver, from the copy in `outputs/fortran_reference/build/` (originals
untouched), through the same guarded launcher. Fortran is never called during
neural training. The copy uses KMX=(5,4) to fit the 4 GiB limit (see
`outputs/fortran_reference/BUILD_NOTES.md`).

2026-10-27 user decision: physical GPUs 0 and 1 are both authorized, so one
numerical job may span both cards (for example the per-kappa-pair streamed
Fock exchange, `--fock-stream-devices cuda:0,cuda:1`). No other index is
allowed. The single numerical-job lock, the 4 GiB host cgroup limit and the
per-device GPU budget (allocator limit and driver monitor on every selected
card) are unchanged; busy-device admission still applies to each card.

2026-10-27 user decision (remote host GPU04, ssh alias `rhf-7789`, work dir
`~/rhfb-nn/`): only physical GPUs 6 and 7 may be used; the GPUs are shared with
another user's job, which must never be touched. The host has cgroup v1 and no
user memory delegation, so numerical jobs there run through
`scripts/run_rhfb_procguard.py` (process-level guard: session resident memory
sampled every 0.5 s, GPU memory every 2 s, termination on overrun/timeout,
host limit <= 8 GiB, per-GPU budget <= 8 GiB with the PyTorch allocator limit,
one GPU job at a time; CPU-only Fortran reference jobs use a separate lock,
one at a time, with host limit <= 16 GiB by the same day's user decision).
This sampling guard is not a kernel limit and is accepted
only on that host; local jobs keep `run_rhfb_limited.py`. The remote Python
environment is `~/rhfb-nn/env` (torch 2.5.1+cu121), separate from the other
projects in that home directory.

2026-10-28 user decision (remote GPU04): all eight GPUs (0-7) may be used when
idle. Jobs run in parallel lanes of two GPUs each; the process guard takes one
lock per selected GPU (jobs sharing a card are mutually exclusive) and still
requires free memory for the budget on every card. Each job keeps host <= 8 GiB
and per-GPU <= 8 GiB with the allocator limit, so four lanes total <= 32 GiB.

2026-10-30 user decision (remote GPU04): the per-GPU budget may be up to
24 GiB (allocator limit and guard monitor), while sharing the cards with
another user's idle job; host memory stays <= 8 GiB per job. Drip-line chains
run as eight one-GPU lanes started with `setsid nohup`.
