#!/usr/bin/env python3
"""Run one numerical job in a kernel-enforced, non-swapping user cgroup."""
import argparse
import csv
import fcntl
import json
import os
import signal
import resource
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAX_TIMEOUT_SECONDS = 1800
LONG_RUN_MAX_TIMEOUT_SECONDS = 14 * 24 * 60 * 60


def validate_runtime_arguments(memory_mib, timeout_seconds, gpu_memory_mib, *, long_run=False):
    """Validate launcher bounds; long-run mode relaxes only the wall clock."""
    if not 32 <= memory_mib <= 4096:
        raise ValueError('memory must be 32..4096 MiB (kernel MemoryMax <= 4 GiB)')
    timeout_max = LONG_RUN_MAX_TIMEOUT_SECONDS if long_run else DEFAULT_MAX_TIMEOUT_SECONDS
    if not 1 <= timeout_seconds <= timeout_max:
        if long_run:
            raise ValueError(f'wall time must be 1..{LONG_RUN_MAX_TIMEOUT_SECONDS} seconds (14 days) in --long-run mode')
        raise ValueError(
            f'wall time must be 1..{DEFAULT_MAX_TIMEOUT_SECONDS} seconds; '
            'use --long-run for a bounded maximum of 14 days'
        )
    if not 256 <= gpu_memory_mib <= 8192:
        raise ValueError('GPU monitor budget must be 256..8192 MiB')


def write_report(path, data):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


def cgroup_snapshot(directory):
    result = {}
    for name in ('memory.current', 'memory.peak'):
        path = directory/name
        result[name] = int(path.read_text()) if path.exists() else None
    result['memory.events'] = dict(line.split() for line in
                                  (directory/'memory.events').read_text().splitlines())
    result['memory.events'] = {key: int(value) for key, value in result['memory.events'].items()}
    return result


def driver_memory_bytes(directory, gpu_uuid):
    """Largest per-device driver memory of this job; each device has the full budget."""
    pids = {int(value) for value in (directory/'cgroup.procs').read_text().split()}
    result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
         '--format=csv,noheader,nounits'], capture_output=True, text=True,
        check=True, timeout=3)
    per_device = {value: 0 for value in gpu_uuid.split(',')}
    for uuid_value, pid, memory in csv.reader(result.stdout.splitlines()):
        if uuid_value.strip() in per_device and int(pid) in pids:
            per_device[uuid_value.strip()] += int(memory)*1024**2
    return max(per_device.values())


def stop_child(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=1)


def payload_pids(directory):
    return {int(value) for value in (directory/'cgroup.procs').read_text().split()} - {os.getpid()}


def cleanup_descendants(directory):
    """Only this service's remaining payload processes; never other GPU jobs."""
    initial = payload_pids(directory)
    for sig, seconds in ((signal.SIGTERM, 0.4), (signal.SIGKILL, 0.8)):
        for pid in payload_pids(directory):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic()+seconds
        while payload_pids(directory) and time.monotonic() < deadline:
            time.sleep(0.05)
        if not payload_pids(directory):
            break
    return sorted(initial), sorted(payload_pids(directory))


def collect_evidence(report, evidence_path, unit):
    properties = subprocess.run(
        ['systemctl', '--user', 'show', unit+'.service',
         '--property=LoadState,Result,ExecMainCode,ExecMainStatus,MemoryMax,MemorySwapMax'],
        capture_output=True, text=True, check=False, timeout=5)
    report['systemd_result'] = properties.stdout.strip()
    report['systemd_properties_are_post_exit'] = True
    try:
        report['kernel_limit_evidence'] = json.loads(evidence_path.read_text())
    except (OSError, ValueError) as exc:
        report['evidence_read_error'] = str(exc)


def execution_status(report):
    report['completion_evidence_verified'] = False
    final = report.get('kernel_limit_evidence', {})
    properties = dict(line.split('=', 1) for line in
                      report.get('systemd_result', '').splitlines() if '=' in line)
    start = final.get('resource_start', {})
    end = final.get('resource_final', {})
    start_events = start.get('memory.events', {})
    end_events = end.get('memory.events', {})
    event_keys = {'low', 'high', 'max', 'oom', 'oom_kill'}
    complete_events = (event_keys <= start_events.keys() and event_keys <= end_events.keys()
                       and all(isinstance(end_events[key], int) and isinstance(start_events[key], int)
                               and end_events[key] >= start_events[key] >= 0 for key in event_keys))
    oom_killed = complete_events and end_events['oom_kill'] > start_events['oom_kill']
    if oom_killed or (properties.get('LoadState') == 'loaded' and properties.get('Result') == 'oom-kill'):
        return 'oom_killed'
    readback = final.get('kernel_limits_final', {})
    complete = (final.get('worker_finalized') is True
                and isinstance(final.get('child_actual_exit_code'), int)
                and final.get('descendants_clear') is True
                and final.get('enforced') is True
                and readback.get('enforced') is True
                and isinstance(end.get('memory.current'), int)
                and end['memory.current'] >= 0 and complete_events
                and final.get('memory_swap_max_bytes') == readback.get('memory_swap_max_bytes') == 0
                and 0 < final.get('memory_max_bytes', 0) <= report['memory_max_bytes'] <= 4*1024**3
                and readback.get('memory_max_bytes') == final.get('memory_max_bytes'))
    complete = complete and properties.get('LoadState') in {'loaded', 'not-found'}
    report['completion_evidence_verified'] = bool(complete)
    # memory.peak is optional on older kernels; missing peaks are never zero.
    if not complete:
        return 'incomplete'
    reason = final.get('termination_reason')
    if reason != 'child_exited':
        return reason or 'incomplete'
    if final['child_actual_exit_code'] != 0:
        return 'failed'
    if report.get('gpu_uuid'):
        cuda_policy = final.get('cuda_allocator_policy', {})
        if (cuda_policy.get('uuid') != report['gpu_uuid']
                or cuda_policy.get('allocator_limit_enforced') is not True
                or cuda_policy.get('driver_monitor_budget_bytes') != report['gpu_driver_monitor_budget_mib']*1024**2
                or not 0 < cuda_policy.get('allocator_limit_bytes', 0) < cuda_policy['driver_monitor_budget_bytes']):
            report['cuda_allocator_evidence_missing'] = True
            return 'incomplete'
    if end_events['oom'] > start_events['oom']:
        return 'memory_allocation_failure'
    if report.get('exit_code') != 0:
        return 'failed'
    if properties.get('LoadState') == 'loaded' and properties.get('Result') != 'success':
        return 'failed'
    return 'completed'


def worker(command, evidence_path, policy_path):
    sys.path.insert(0, str(ROOT))
    from dpl_rhf.rhfb.execution_limits import require_bounded_cgroup
    # This distinct lock survives launcher death and exec of the payload.
    execution_lock = (ROOT/'outputs/execution/rhfb_worker.lock').open('a')
    try:
        fcntl.flock(execution_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another numerical worker still holds the execution lock.')
    evidence = require_bounded_cgroup()
    directory = Path(evidence['cgroup'])
    policy = json.loads(policy_path.read_text())
    evidence.update(verified_unix=time.time(), worker_finalized=False,
                    termination_reason=None, child_actual_exit_code=None,
                    resource_start=cgroup_snapshot(directory),
                    sampled_memory_current_peak_bytes=0,
                    sampled_driver_memory_peak_bytes=0)
    write_report(evidence_path, evidence)
    print(json.dumps({'event': 'kernel-memory-limit-verified', **evidence}), flush=True)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    if policy.get('gpu_uuid'):
        os.environ['CUDA_VISIBLE_DEVICES'] = policy['gpu_uuid']
        os.environ['RHFB_CUDA_UUID'] = policy['gpu_uuid']
        os.environ['RHFB_CUDA_BUDGET_MIB'] = str(policy['gpu_memory_mib'])
        os.environ['RHFB_CUDA_POLICY_EVIDENCE'] = str(evidence_path.with_suffix('.cuda.json'))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    process = None
    started = time.monotonic()
    pressure_since = None
    try:
        # The supervisor retains the lock until all payload cleanup completes.
        process = subprocess.Popen(command, start_new_session=True)
        evidence.update(child_pid=process.pid, child_started_unix=time.time())
        write_report(evidence_path, evidence)
        while process.poll() is None:
            snapshot = cgroup_snapshot(directory)
            evidence['resource_latest'] = snapshot
            current = snapshot['memory.current']
            evidence['sampled_memory_current_peak_bytes'] = max(
                evidence['sampled_memory_current_peak_bytes'], current)
            if current >= 3*evidence['memory_max_bytes']//4:
                pressure_since = pressure_since or time.monotonic()
            else:
                pressure_since = None
            if policy.get('gpu_uuid'):
                used = driver_memory_bytes(directory, policy['gpu_uuid'])
                evidence['sampled_driver_memory_peak_bytes'] = max(
                    evidence['sampled_driver_memory_peak_bytes'], used)
                if used >= policy['gpu_memory_mib']*1024**2:
                    evidence['termination_reason'] = 'gpu_memory_budget_exceeded'
            if pressure_since and time.monotonic()-pressure_since >= 10:
                evidence['termination_reason'] = 'sustained_host_memory_pressure'
            if time.monotonic()-started >= policy['timeout_seconds']:
                evidence['termination_reason'] = 'timeout'
            write_report(evidence_path, evidence)
            if evidence['termination_reason']:
                stop_child(process)
                break
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        evidence['child_actual_exit_code'] = process.wait()
        evidence['termination_reason'] = evidence['termination_reason'] or 'child_exited'
    except KeyboardInterrupt:
        evidence['termination_reason'] = 'received_signal'
        write_report(evidence_path, evidence)
        stop_child(process)
        if process is not None:
            evidence['child_actual_exit_code'] = process.poll()
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        evidence['termination_reason'] = 'supervisor_error'
        evidence['supervisor_error'] = str(exc)
        stop_child(process)
        if process is not None:
            evidence['child_actual_exit_code'] = process.poll()
    finally:
        # Finalization certifies cleanup and evidence, not just entry to finally.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if process is not None and process.poll() is None:
                stop_child(process)
            if process is not None:
                evidence['child_actual_exit_code'] = process.poll()
            removed, remaining = cleanup_descendants(directory)
            evidence['descendants_cleaned_pids'] = removed
            evidence['remaining_payload_pids'] = remaining
            evidence['descendants_clear'] = not remaining
            if removed and evidence['termination_reason'] == 'child_exited':
                evidence['termination_reason'] = 'orphaned_descendants'
            evidence['resource_final'] = cgroup_snapshot(directory)
            evidence['kernel_limits_final'] = require_bounded_cgroup()
            if policy.get('gpu_uuid'):
                cuda_path = evidence_path.with_suffix('.cuda.json')
                if cuda_path.exists():
                    evidence['cuda_allocator_policy'] = json.loads(cuda_path.read_text())
            evidence['child_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024
            evidence['worker_finalized'] = (
                isinstance(evidence['child_actual_exit_code'], int) and not remaining)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            evidence['finalization_error'] = str(exc)
            evidence['worker_finalized'] = False
        evidence['finished_unix'] = time.time()
        write_report(evidence_path, evidence)
    good = (evidence['worker_finalized'] and evidence['termination_reason'] == 'child_exited'
            and evidence['child_actual_exit_code'] == 0)
    raise SystemExit(0 if good else 1)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '_worker':
        evidence_path = Path(sys.argv[2])
        policy_path = Path(sys.argv[3])
        command = sys.argv[4:]
        if command and command[0] == '--':
            command = command[1:]
        if not command:
            raise SystemExit('missing worker command')
        worker(command, evidence_path, policy_path)
        return
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-mib', type=int, default=4096)
    parser.add_argument('--timeout-seconds', type=int, default=900)
    parser.add_argument(
        '--long-run',
        action='store_true',
        help='allow a finite timeout up to 14 days; memory/GPU guards remain unchanged',
    )
    parser.add_argument('--report', type=Path)
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--gpu-memory-mib', type=int, default=8192)
    parser.add_argument('--require-junit', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        parser.error('provide a command after --')
    try:
        validate_runtime_arguments(
            args.memory_mib,
            args.timeout_seconds,
            args.gpu_memory_mib,
            long_run=args.long_run,
        )
    except ValueError as exc:
        parser.error(str(exc))
    locks = ROOT/'outputs/execution'
    locks.mkdir(parents=True, exist_ok=True)
    with (locks/'rhfb_numerical.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another bounded RHFB job is active; concurrent runs are refused.')
        # The launcher can die while its systemd service remains alive. Check
        # those services as well as the process lock before admitting a job.
        active = subprocess.run(
            ['systemctl', '--user', 'list-units', '--type=service',
             '--state=activating,active,deactivating', '--no-legend', '--plain',
             'rhfb-guard-*.service'], capture_output=True, text=True, check=False)
        if active.returncode != 0:
            raise SystemExit('Cannot verify the resource manager; numerical launch refused.')
        if active.stdout.strip():
            raise SystemExit('A bounded RHFB service is still active; numerical launch refused.')
        if args.gpu_uuid:
            sys.path.insert(0, str(ROOT))
            from dpl_rhf.rhfb.cuda_resources import gpu_inventory, USER_AUTHORIZED_GPU_INDICES
            requested = args.gpu_uuid.split(',')
            if len(set(requested)) != len(requested) or len(requested) > len(USER_AUTHORIZED_GPU_INDICES):
                raise SystemExit('GPU list must name distinct authorized devices; launch refused.')
            inventory = gpu_inventory()
            processes = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                 '--format=csv,noheader,nounits'], capture_output=True, text=True,
                check=True, timeout=3)
            busy_uuids = {row[0].strip() for row in csv.reader(processes.stdout.splitlines())}
            for requested_uuid in requested:
                selected = next((gpu for gpu in inventory if gpu['uuid'] == requested_uuid), None)
                if selected is not None and selected['index'] not in USER_AUTHORIZED_GPU_INDICES:
                    raise SystemExit('User authorization permits only physical GPU indices 0 and 1; launch refused.')
                if selected is None or selected['memory_used_mib'] > 256 or selected['utilization_percent'] > 5:
                    raise SystemExit('The requested GPU is unavailable or busy; numerical launch refused.')
                if args.gpu_memory_mib >= selected['memory_total_mib']:
                    raise SystemExit('GPU budget must leave device headroom.')
                if requested_uuid in busy_uuids:
                    raise SystemExit('The requested GPU has an existing compute process; launch refused.')
        unit = 'rhfb-guard-'+uuid.uuid4().hex
        report_path = args.report or locks/(unit+'.json')
        report_path = report_path.resolve()
        evidence_path = locks/(unit+'.limits.json')
        policy_path = locks/(unit+'.policy.json')
        write_report(policy_path, {'timeout_seconds': args.timeout_seconds,
                                  'long_run_enabled': args.long_run,
                                  'gpu_uuid': args.gpu_uuid,
                                  'gpu_memory_mib': args.gpu_memory_mib})
        report_path.parent.mkdir(parents=True, exist_ok=True)
        limit = args.memory_mib*1024**2
        report = {'unit': unit+'.service', 'status': 'starting',
                  'memory_max_bytes': limit, 'memory_high_bytes': 3*limit//4,
                  'memory_swap_max_bytes': 0, 'timeout_seconds': args.timeout_seconds,
                  'long_run_enabled': args.long_run,
                  'gpu_uuid': args.gpu_uuid, 'gpu_driver_monitor_budget_mib': args.gpu_memory_mib if args.gpu_uuid else None,
                  'single_job_lock': str(locks/'rhfb_numerical.lock'),
                  'started_unix': time.time()}
        write_report(report_path, report)
        launcher = ['systemd-run', '--user', '--wait', '--pipe', '--quiet',
                    '--unit='+unit, '--property=MemoryMax='+str(limit),
                    '--property=MemoryHigh='+str(3*limit//4),
                    '--property=MemorySwapMax=0', '--property=OOMPolicy=kill',
                    '--property=RuntimeMaxSec='+str(args.timeout_seconds+4)+'s',
                    '--property=TimeoutStopSec=3s', '--property=TasksMax=64',
                    '--property=CPUQuota=200%', '--property=Nice=10',
                    '--working-directory='+str(ROOT),
                    sys.executable, str(Path(__file__).resolve()), '_worker',
                    str(evidence_path), str(policy_path), '--', *command]
        try:
            result = subprocess.run(launcher, check=False)
            report['exit_code'] = result.returncode
            collect_evidence(report, evidence_path, unit)
            report['status'] = execution_status(report)
            if args.require_junit and report['status'] == 'completed':
                import xml.etree.ElementTree as ET
                try:
                    junit = args.require_junit.resolve()
                    if junit.stat().st_mtime < report['started_unix']:
                        raise ValueError('JUnit is stale')
                    cases = list(ET.parse(junit).getroot().iter('testcase'))
                    if not cases or any(case.find('failure') is not None or case.find('error') is not None
                                        or case.find('skipped') is not None
                                        for case in cases):
                        raise ValueError('JUnit does not prove passing tests')
                    report['junit_tests_passed'] = len(cases)
                except (OSError, ValueError, ET.ParseError) as exc:
                    report['status'] = 'test_evidence_missing_or_failed'
                    report['test_evidence_error'] = str(exc)
            if report['status'] != 'completed' and report['exit_code'] == 0:
                report['exit_code'] = 2
            # Keep the transient failure evidence in JSON, then release the unit.
            subprocess.run(['systemctl', '--user', 'reset-failed', unit+'.service'],
                           capture_output=True, check=False)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            report['status'] = 'launch_failed'
            report['error'] = str(exc)
            report['exit_code'] = 2
        except KeyboardInterrupt:
            report['parent_interrupted'] = True
            report['exit_code'] = 130
            try:
                subprocess.run(['systemctl', '--user', 'stop', unit+'.service'],
                               check=False, timeout=8)
                collect_evidence(report, evidence_path, unit)
                report['status'] = execution_status(report)
                if report['status'] in {'completed', 'failed', 'received_signal'}:
                    report['status'] = 'interrupted'
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                report['status'] = 'incomplete'
                report['interrupt_cleanup_error'] = str(exc)
        finally:
            report['finished_unix'] = time.time()
            write_report(report_path, report)
        raise SystemExit(report['exit_code'])


if __name__ == '__main__':
    main()
