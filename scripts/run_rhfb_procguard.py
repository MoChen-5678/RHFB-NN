#!/usr/bin/env python3
"""Process-level guard for hosts without a delegable cgroup v2 memory limit.

Used on hosts with cgroup v1 or without user memory delegation (first on the
remote GPU04 host, 2026-10-27 user decision).  The payload runs in its own session; every 0.5 s the
supervisor sums the resident memory of all processes in that session and, every
2 s, the driver memory of those processes on each selected GPU.  It terminates
the whole session on host-memory overrun, GPU-budget overrun or timeout.  Only
physical GPUs listed in --authorized-gpus (or RHFB_AUTHORIZED_GPUS) may be
selected; GPU use is refused when no list is given; other users' processes on those
GPUs are tolerated, but enough free memory must remain for the budget.  One
guarded job runs at a time (file lock).  This is sampling, not a kernel limit.
"""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
# The operator of each host states which physical GPUs may be used
# (GPU04: 2026-10-27 GPUs 6,7; 2026-10-28 all eight).  No default.
AUTHORIZED_ENV = "RHFB_AUTHORIZED_GPUS"
MAX_HOST_MIB = 8192           # jobs that use GPUs
MAX_CPU_ONLY_HOST_MIB = 16384  # CPU-only reference jobs (Fortran KMX=(8,8)), 2026-10-27
MAX_GPU_MIB = 24576           # per device; 2026-10-30 user decision (GPU04)
PAGE = os.sysconf("SC_PAGE_SIZE")


def write_json(path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def gpu_inventory():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=10)
    return [{"index": int(i), "uuid": u.strip(), "total_mib": int(t), "used_mib": int(m)}
            for i, u, t, m in csv.reader(out.stdout.splitlines())]


def session_pids(sid):
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            fields = stat[stat.rindex(")") + 2:].split()
            if int(fields[3]) == sid:  # field 6 of /proc/pid/stat: session id
                pids.append(int(entry.name))
        except (OSError, ValueError):
            continue
    return pids


def session_rss_bytes(pids):
    total = 0
    for pid in pids:
        try:
            total += int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * PAGE
        except (OSError, ValueError, IndexError):
            continue
    return total


def gpu_bytes_by_uuid(pids, uuids):
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=10)
    usage = {u: 0 for u in uuids}
    wanted = set(pids)
    for gpu_uuid, pid, used in csv.reader(out.stdout.splitlines()):
        gpu_uuid = gpu_uuid.strip()
        if gpu_uuid in usage and int(pid) in wanted:
            usage[gpu_uuid] += int(used) * 1024**2
    return usage


def kill_session(sid):
    for sig, wait in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(sid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and session_pids(sid):
            time.sleep(0.2)
        if not session_pids(sid):
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-mib", type=int, default=MAX_HOST_MIB)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--gpu-indices", default="")
    parser.add_argument("--gpu-memory-mib", type=int, default=MAX_GPU_MIB)
    parser.add_argument("--authorized-gpus", default=os.environ.get(AUTHORIZED_ENV, ""),
                        help="comma list of physical GPU indices the operator allows "
                             f"(default: ${AUTHORIZED_ENV}); required for GPU jobs")
    parser.add_argument("--threads", type=int, default=2,
                        help="OpenMP/BLAS threads of the payload (1..16)")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after --")
    cpu_only = not args.gpu_indices.strip()
    host_max = MAX_CPU_ONLY_HOST_MIB if cpu_only else MAX_HOST_MIB
    if not 256 <= args.memory_mib <= host_max:
        parser.error(f"--memory-mib must be 256..{host_max}")
    if not 256 <= args.gpu_memory_mib <= MAX_GPU_MIB:
        parser.error(f"--gpu-memory-mib must be 256..{MAX_GPU_MIB}")
    if not 1 <= args.threads <= 16:
        parser.error("--threads must be 1..16")
    if not 1 <= args.timeout_seconds <= 14 * 86400:
        parser.error("--timeout-seconds must be 1..14 days")

    locks = ROOT / "outputs/execution"
    locks.mkdir(parents=True, exist_ok=True)
    indices = [int(i) for i in args.gpu_indices.split(",") if i.strip()]
    # One lock per selected GPU: jobs sharing any card are mutually exclusive,
    # jobs on disjoint cards may run concurrently.  At most one CPU-only job.
    lock_names = ["procguard_cpu.lock"] if cpu_only else [f"procguard_gpu{i}.lock" for i in sorted(set(indices))]
    held = []
    for name in lock_names:
        handle = (locks / name).open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another guarded RHFB job is active on the requested devices; concurrent runs are refused.")
        held.append(handle)

    uuids = []
    authorized = tuple(int(i) for i in args.authorized_gpus.split(",") if i.strip())
    if indices:
        if not authorized:
            raise SystemExit("No authorized GPU list (--authorized-gpus or "
                             f"${AUTHORIZED_ENV}); GPU launch refused.")
        if len(set(indices)) != len(indices) or any(i not in authorized for i in indices):
            raise SystemExit(f"Only physical GPUs {authorized} are authorized; launch refused.")
        inventory = {g["index"]: g for g in gpu_inventory()}
        for i in indices:
            gpu = inventory.get(i)
            if gpu is None:
                raise SystemExit(f"physical GPU {i} not found; launch refused.")
            if gpu["total_mib"] - gpu["used_mib"] < args.gpu_memory_mib + 1024:
                raise SystemExit(f"The requested GPU {i} is unavailable or busy (free memory below budget).")
            uuids.append(gpu["uuid"])

    report = args.report.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    evidence_path = report.with_suffix(".guard_evidence.json")
    limit = args.memory_mib * 1024**2
    gpu_limit = args.gpu_memory_mib * 1024**2
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(args.threads)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(uuids)
    if uuids:
        env["RHFB_CUDA_UUID"] = ",".join(uuids)
        env["RHFB_CUDA_BUDGET_MIB"] = str(args.gpu_memory_mib)
        env["RHFB_CUDA_POLICY_EVIDENCE"] = str(report.with_suffix(".cuda.json"))
    env["RHFB_PROCESS_GUARD"] = str(evidence_path)

    record = {"mode": "process_guard", "guard_pid": os.getpid(), "payload_sid": None,
              "memory_max_bytes": limit, "gpu_budget_bytes": gpu_limit,
              "authorized_gpu_indices": indices, "gpu_uuids": uuids,
              "timeout_seconds": args.timeout_seconds, "command": command,
              "started_unix": time.time(), "status": "starting", "termination_reason": None,
              "peak_rss_bytes": 0, "peak_gpu_bytes": {u: 0 for u in uuids},
              "scope": "sampled every 0.5 s (GPU every 2 s); not a kernel limit"}
    terminated = {"flag": False}

    def on_signal(signum, frame):
        terminated["flag"] = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    # The payload waits on a pipe until the evidence names its session, so it
    # can never validate the guard before the guard knows the session id.
    read_fd, write_fd = os.pipe()
    process = subprocess.Popen(
        [sys.executable, "-c",
         "import os,sys; os.read(int(sys.argv[1]),1); os.execvp(sys.argv[2], sys.argv[2:])",
         str(read_fd)] + command,
        cwd=ROOT, env=env, start_new_session=True, pass_fds=(read_fd,))
    os.close(read_fd)
    sid = process.pid
    record.update(payload_sid=sid, status="running")
    write_json(evidence_path, record)
    os.write(write_fd, b"x")
    os.close(write_fd)
    started = time.monotonic()
    last_gpu = 0.0
    while process.poll() is None:
        pids = session_pids(sid)
        rss = session_rss_bytes(pids)
        record["peak_rss_bytes"] = max(record["peak_rss_bytes"], rss)
        if rss >= limit:
            record["termination_reason"] = "host_memory_limit_exceeded"
        if uuids and time.monotonic() - last_gpu >= 2.0:
            last_gpu = time.monotonic()
            try:
                usage = gpu_bytes_by_uuid(pids, uuids)
            except (subprocess.SubprocessError, OSError) as exc:
                record["termination_reason"] = f"gpu_monitor_failed: {exc}"
                usage = {}
            for u, used in usage.items():
                record["peak_gpu_bytes"][u] = max(record["peak_gpu_bytes"][u], used)
                if used >= gpu_limit:
                    record["termination_reason"] = "gpu_memory_budget_exceeded"
        if time.monotonic() - started >= args.timeout_seconds:
            record["termination_reason"] = "timeout"
        if terminated["flag"]:
            record["termination_reason"] = "received_signal"
        if record["termination_reason"]:
            write_json(evidence_path, record)
            kill_session(sid)
            break
        time.sleep(0.5)
    code = process.wait()
    kill_session(sid)  # no orphaned descendants
    record["exit_code"] = code
    record["remaining_pids"] = session_pids(sid)
    record["finished_unix"] = time.time()
    if record["termination_reason"]:
        record["status"] = record["termination_reason"]
    else:
        record["status"] = "completed" if code == 0 else "failed"
    write_json(evidence_path, record)
    write_json(report, record)
    raise SystemExit(0 if record["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
