from __future__ import annotations

import argparse
import json
from pathlib import Path

from dpl_rhf.io.outputs import write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a neural RHF summary with a read-only Core-1204 reference JSON"
    )
    parser.add_argument("result_summary")
    parser.add_argument("core_reference")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result = json.loads(Path(args.result_summary).read_text(encoding="utf-8"))
    core = json.loads(Path(args.core_reference).read_text(encoding="utf-8"))
    aliases = {
        "energy_total_no_com_mev": "e_total_no_com",
        "energy_per_a_no_com_mev": "e_per_A_no_com",
    }
    comparison = {}
    for result_key, core_key in aliases.items():
        actual = float(result[result_key])
        reference = float(core[core_key])
        comparison[result_key] = {
            "rhf_neural": actual,
            "core1204": reference,
            "absolute_error": actual - reference,
            "relative_error_percent": 100.0 * (actual - reference) / abs(reference),
        }
    for key in ("rms_n_no_com", "rms_p_no_com", "rms_matter_no_com", "charge_radius_no_com"):
        if key in core:
            actual = float(result["diagnostics"][key])
            reference = float(core[key])
            comparison[key] = {
                "rhf_neural": actual,
                "core1204": reference,
                "absolute_error": actual - reference,
                "relative_error_percent": 100.0 * (actual - reference) / abs(reference),
            }
    output = Path(args.out) if args.out else Path(args.result_summary).with_name("core_comparison.json")
    write_json(output, {"comparison": comparison})
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
