"""Exercise external stop and launcher death using only small waiting children."""
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'outputs/validation/local_safe_restart_20261007'

def start(name):
    # Every invocation gets new artifacts; never mistake a prior probe for a
    # newly started service while its launcher is still creating its report.
    identity = name+'_'+uuid.uuid4().hex[:8]
    path = OUTPUT/(identity+'.json')
    log = (OUTPUT/(identity+'.log')).open('w')
    proc = subprocess.Popen([sys.executable, str(ROOT/'scripts/run_rhfb_limited.py'),
                             '--memory-mib', '64', '--timeout-seconds', '25',
                             '--report', str(path), '--', sys.executable,
                             '-c', 'import time; time.sleep(20)'], stdout=log, stderr=log)
    try:
        deadline = time.monotonic()+8
        while time.monotonic() < deadline:
            if path.exists():
                data = json.loads(path.read_text())
                sidecar = ROOT/'outputs/execution'/(data['unit'].removesuffix('.service')+'.limits.json')
                if sidecar.exists() and json.loads(sidecar.read_text()).get('child_pid'):
                    return proc, log, path, data['unit']
            if proc.poll() is not None:
                raise RuntimeError('waiting probe failed to start')
            time.sleep(0.1)
        raise RuntimeError('waiting probe did not start in time')
    except BaseException:
        proc.terminate()
        proc.wait(timeout=8)
        log.close()
        raise

def stop(unit):
    if not unit.startswith('rhfb-guard-') or not unit.endswith('.service'):
        raise RuntimeError('unexpected probe unit identity')
    subprocess.run(['systemctl', '--user', 'stop', unit], check=True)

def main():
    results = []
    proc, log, path, unit = start('external_stop')
    try:
        stop(unit)
        code = proc.wait(timeout=8)
    finally:
        if proc.poll() is None:
            stop(unit)
            proc.wait(timeout=8)
        log.close()
    data = json.loads(path.read_text())
    assert code != 0 and data['status'] != 'completed', data
    results.append({'case': 'external_stop', 'passed': True, 'status': data['status'],
                    'launcher_exit': code,
                    'evidence': data.get('kernel_limit_evidence')})
    proc, log, path, unit = start('launcher_death')
    try:
        proc.kill()
        proc.wait(timeout=8)
        second = subprocess.run([sys.executable, str(ROOT/'scripts/run_rhfb_limited.py'),
                                 '--memory-mib', '64', '--timeout-seconds', '2',
                                 '--', sys.executable, '-c', 'pass'],
                                capture_output=True, text=True, timeout=8)
        assert second.returncode != 0 and 'still active' in second.stderr, second
        results.append({'case': 'launcher_death_concurrent_refused', 'passed': True,
                        'refusal': second.stderr.strip()})
    finally:
        stop(unit)
        log.close()
    active = subprocess.run(['systemctl', '--user', 'list-units', '--type=service',
                             '--state=activating,active,deactivating', '--no-legend',
                             '--plain', 'rhfb-guard-*.service'],
                            check=True, capture_output=True, text=True)
    assert not active.stdout.strip(), active.stdout
    target = OUTPUT/('interruption_tests_'+uuid.uuid4().hex[:8]+'.json')
    target.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
