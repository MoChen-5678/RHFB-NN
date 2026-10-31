"""Read runtime evidence while the protected cgroup still exists."""
from pathlib import Path
import resource

from .execution_limits import require_bounded_cgroup


def resource_snapshot():
    evidence = require_bounded_cgroup()
    directory = Path(evidence['cgroup'])
    for name in ('memory.current', 'memory.peak'):
        try:
            evidence[name.replace('.', '_')+'_bytes'] = int((directory/name).read_text())
        except (OSError, ValueError):
            evidence[name.replace('.', '_')+'_bytes'] = None
    try:
        evidence['memory_events'] = {
            key: int(value) for key, value in
            (line.split() for line in (directory/'memory.events').read_text().splitlines())
        }
    except (OSError, ValueError):
        evidence['memory_events'] = None
    evidence['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    evidence['scope'] = 'current protected service; includes charged file pages and all descendants'
    return evidence
