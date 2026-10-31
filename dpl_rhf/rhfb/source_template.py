"""Read-only parser for the supplied RHFB Fortran input snapshots.

This module parses DATA statements and text input records. It never imports,
compiles, or executes Fortran. DWS cutoffs are retained as basis metadata and
are never converted to NN sea-occupation thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


_PKA1_REQUIRED = {
    "txtfor", "rvs", "amu", "len", "inl", "ie", "amsig", "amome",
    "amrho", "ampio", "gsig", "gome", "grho", "fpio", "asig", "bsig",
    "csig", "dsig", "aome", "bome", "come", "dome", "arho", "apio",
    "grtn", "artn",
}
_D1S_REQUIRED = {"txtgog", "gr", "gw", "gb", "gh", "gm", "w0", "t3", "vg"}


def _pka1_full_ph_channels() -> tuple[str, ...]:
    """Use the current RHF functional's authoritative complete PKA1 registry."""
    from dpl_rhf.rhf.functional import PKA1_DIRECT_CHANNEL_TERMS, PKA1_FOCK_CHANNEL_TERMS

    return tuple(PKA1_DIRECT_CHANNEL_TERMS) + tuple(PKA1_FOCK_CHANNEL_TERMS)


def _number(token: str) -> float:
    return float(token.strip().replace("D", "E").replace("d", "e"))


def _fortran_data_block(text: str, record_name: str, required: set[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    pattern = re.compile(rf"\b{re.escape(record_name)}%([A-Za-z][A-Za-z0-9_]*)\s*/([^/]*)/", re.I)
    for original in text.splitlines():
        line = original.split("!", 1)[0]
        for match in pattern.finditer(line):
            key = match.group(1).lower()
            raw = match.group(2).strip()
            if key in {"txtfor", "txtgog"}:
                parsed: Any = raw.strip().strip("'\"").strip()
            else:
                parsed_values = tuple(_number(part) for part in raw.split(",") if part.strip())
                if not parsed_values:
                    raise ValueError(f"empty DATA value for {record_name}%{key}")
                parsed = parsed_values[0] if len(parsed_values) == 1 else parsed_values
            if key in values:
                raise ValueError(f"duplicate DATA initialization for {record_name}%{key}")
            values[key] = parsed
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"incomplete {record_name} source constants: {missing}")
    return values


def _scalar_data_value(text: str, name: str) -> float:
    pattern = re.compile(rf"\b{re.escape(name)}\s*/([^/]*)/", re.I)
    matches = [match.group(1).strip() for line in text.splitlines()
               for match in pattern.finditer(line.split("!", 1)[0])]
    if len(matches) != 1:
        raise ValueError(f"expected one DATA initialization for {name}, found {len(matches)}")
    values = tuple(_number(token) for token in matches[0].split(",") if token.strip())
    if len(values) != 1 or not math.isfinite(values[0]):
        raise ValueError(f"expected one finite scalar DATA value for {name}")
    return values[0]


def _decode_source_bytes(raw: bytes) -> tuple[str, str]:
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        # Latin-1 maps every byte one-to-one. Physical records are ASCII; this
        # fallback preserves non-UTF8 comments without changing source bytes.
        return raw.decode("latin-1"), "latin-1-fallback"


def _code_records(text: str) -> dict[str, str]:
    records: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("!", 1)[0].strip()
        match = re.match(r"([A-Za-z][A-Za-z0-9_/]*)\s*=\s*(.*?)\s*$", line)
        if match:
            key = re.sub(r"[^a-z0-9]", "", match.group(1).lower())
            records[key] = match.group(2).strip()
    return records


def _parse_numeric_record(raw: str, field: str) -> tuple[float, ...]:
    # Fortran list-directed fields are comma/space separated; reject any
    # nonnumeric tail rather than accepting a partial prefix.
    parts = [part for part in re.split(r"[\s,]+", raw.strip()) if part]
    try:
        values = tuple(_number(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"invalid numeric input record for {field}: {raw!r}") from exc
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite or empty input record for {field}")
    return values


def _parse_integer_record(raw: str, field: str) -> int:
    values = _parse_numeric_record(raw, field)
    if len(values) != 1 or int(values[0]) != values[0]:
        raise ValueError(f"{field} must be one integer, got {raw!r}")
    return int(values[0])


def _same_constant_tree(left: Any, right: Any) -> bool:
    """Compare parsed constants without silently accepting missing fields."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return (set(left) == set(right)
                and all(_same_constant_tree(left[key], right[key]) for key in left))
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(_same_constant_tree(a, b) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and float(left) == float(right)
    return left == right


def _mapping_differences(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[str, ...]:
    keys = sorted(set(left) | set(right))
    return tuple(key for key in keys
                 if key not in left or key not in right
                 or not _same_constant_tree(left[key], right[key]))


def _read_block_counts(input_text: str) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    lines = input_text.splitlines()
    for index, raw in enumerate(lines):
        code = raw.split("!", 1)[0].strip()
        match = re.fullmatch(r"(\d+)\s*,\s*(\d+)", code)
        if not match:
            continue
        counts = (int(match.group(1)), int(match.group(2)))
        active: list[tuple[int, ...]] = []
        for species, count in enumerate(counts):
            following = index + species + 1
            if following >= len(lines):
                raise ValueError("blocking counts are present but index records are missing")
            record = lines[following].split("!", 1)[0].strip()
            if count == 0:
                active.append(())
                continue
            values = _parse_numeric_record(record, "blocking indices")
            # These are signed Dirac kappa labels, not nonnegative array
            # indices: Config.f90 compares kp directly with kbl=+/-j.
            if len(values) < count or any(int(value) != value or value == 0 for value in values[:count]):
                raise ValueError("blocking index record does not match its declared count")
            active.append(tuple(int(value) for value in values[:count]))
        return active[0], active[1]
    return None


@dataclass(frozen=True)
class SourceRunTemplate:
    source_variant: str
    source_directory: str
    input_filename: str
    source_sha256: Mapping[str, str]
    source_encoding: Mapping[str, str]
    parameter_set_index: int | None
    parameter_set_name: str | None
    pka1_constants: Mapping[str, Any]
    pka1_scalar_constants: Mapping[str, float]
    d1s_constants: Mapping[str, Any]
    pairing_force: str | None
    pairing_strength: float | None
    blocking_n: tuple[int, ...] | None
    blocking_p: tuple[int, ...] | None
    xct_positive_mev: float | None
    xct_negative_relative_to_2m_mev: float | None
    xpt_mev: tuple[float, float] | None
    tensor_enabled: bool | None
    effective_exchange_enabled: bool | None
    unsupported_reasons: tuple[str, ...]

    @property
    def full_pka1(self) -> bool:
        return (self.parameter_set_name == "PKA1"
                and self.effective_exchange_enabled is True
                and self.tensor_enabled is True)

    @property
    def source_basis_cutoffs(self) -> dict[str, Any]:
        return {
            "xct_positive_mev": self.xct_positive_mev,
            "xct_negative_relative_to_2m_mev": self.xct_negative_relative_to_2m_mev,
            "xpt_pairing_window_mev": self.xpt_mev,
            "meaning": "DWS single-particle basis selection only; never an NN sea threshold",
            "xpt_meaning": "raw source input values; not interpreted as sea occupancy or proof of an active D1S cutoff",
        }

    def require_nn_startup(
        self,
        *,
        requested_model: str,
        requested_channels: Sequence[str],
        requested_pairing_force: str,
        requested_pairing_strength: float,
    ) -> "NNStartupPhysicalTemplate":
        """Audit a spherical reference against current Python physics.

        This is not an O17 training gate: source N/Z and radial blocking indices
        are reported but do not constrain the unrestricted NN calculation.
        """
        failures = list(self.unsupported_reasons)
        if self.parameter_set_name != "PKA1" or self.parameter_set_index != 0:
            failures.append("source input does not explicitly select PKA1 (pset=0)")
        if not self.full_pka1:
            failures.append("source does not explicitly retain the full PKA1 Fock/tensor channels")
        if requested_model != "PKA1":
            failures.append("requested NN particle-hole model is not PKA1")
        full_model_channels = _pka1_full_ph_channels()
        if (set(requested_channels) != set(full_model_channels)
                or len(requested_channels) != len(full_model_channels)):
            failures.append("requested NN channels are cropped or differ from the authoritative complete PKA1 PH registry")
        if self.pairing_force != "D1S" or requested_pairing_force != "D1S":
            failures.append("both source and NN must select Gogny D1S")
        if self.pairing_strength != 1.0 or requested_pairing_strength != 1.0:
            failures.append("both source and NN D1S strengths must equal exactly 1")
        parameter_mapping = compare_source_to_python(self)
        failures.extend(parameter_mapping.mismatches)
        if failures:
            raise ValueError("source template is not NN-startup compatible: " + "; ".join(failures))
        return NNStartupPhysicalTemplate(
            source_variant=self.source_variant,
            parameter_mapping=parameter_mapping,
            pairing_strength=1.0,
            channels=tuple(requested_channels),
            source_blocking_n=self.blocking_n,
            source_blocking_p=self.blocking_p,
            source_symmetry="spherical_radial_reference",
            source_basis_cutoffs=self.source_basis_cutoffs,
            no_sea_occupation_rule="not inherited from DWS basis cutoffs",
        )


@dataclass(frozen=True)
class NNStartupPhysicalTemplate:
    source_variant: str
    parameter_mapping: "SourceParameterMapping"
    pairing_strength: float
    channels: tuple[str, ...]
    source_blocking_n: tuple[int, ...] | None
    source_blocking_p: tuple[int, ...] | None
    source_symmetry: str
    source_basis_cutoffs: Mapping[str, Any]
    no_sea_occupation_rule: str


@dataclass(frozen=True)
class SourceParameterMapping:
    pka1_source_to_python: Mapping[str, Any]
    pka1_python: Mapping[str, Any]
    d1s_active_source_to_python: Mapping[str, Any]
    d1s_python: Mapping[str, Any]
    pka1_matches: bool
    d1s_active_matches: bool
    mismatches: tuple[str, ...]
    unit_conventions: Mapping[str, str]


def compare_source_to_python(source: SourceRunTemplate) -> SourceParameterMapping:
    """Map the supplied source constants into the actual Python parameter types."""
    from dataclasses import asdict

    from dpl_rhf.rhf.parameters import pka1_parameters
    from dpl_rhf.rhfb.pairing import D1S

    raw = source.pka1_constants
    scalars = source.pka1_scalar_constants
    amu = raw["amu"]
    mapped_pka1 = {
        "model": "PKA1",
        "hbar_c": scalars["hbc"],
        "alpha_inv": scalars["alphi"],
        "rho_sat": raw["rvs"],
        "mass_n": amu[0], "mass_p": amu[1],
        "m_sigma": raw["amsig"], "m_omega": raw["amome"],
        "m_rho": raw["amrho"], "m_pi": raw["ampio"],
        "g_sigma": raw["gsig"], "g_omega": raw["gome"], "g_rho": raw["grho"],
        "a_sigma": raw["asig"], "b_sigma": raw["bsig"],
        "c_sigma": raw["csig"], "d_sigma": raw["dsig"],
        "a_omega": raw["aome"], "b_omega": raw["bome"],
        "c_omega": raw["come"], "d_omega": raw["dome"],
        "a_rho": raw["arho"], "f_pi": raw["fpio"], "a_pi": raw["apio"],
        # Define.f90 grtn is the dimensionless f_rho/g_rho parameter. The
        # functional derives the vertex with g_rho/(M_n/hbar_c+M_p/hbar_c).
        "f_rho": raw["grtn"], "a_rho_tensor": raw["artn"],
    }
    python_pka1 = asdict(pka1_parameters())
    raw_d1s = source.d1s_constants
    mapped_d1s = {
        "ranges_fm": raw_d1s["gr"],
        "w_mev": raw_d1s["gw"],
        "b_mev": raw_d1s["gb"],
        "h_mev": raw_d1s["gh"],
        "m_mev": raw_d1s["gm"],
        "strength": source.pairing_strength,
    }
    python_d1s = asdict(D1S)
    # Gogny's w0/t3 are retained in the source audit, but do not enter this
    # implementation's two-Gaussian central nn/pp pairing force.
    active_python_d1s = {key: python_d1s[key] for key in mapped_d1s}
    pka1_matches = _same_constant_tree(mapped_pka1, python_pka1)
    d1s_matches = _same_constant_tree(mapped_d1s, active_python_d1s)
    mismatches = []
    if not pka1_matches:
        fields = _mapping_differences(mapped_pka1, python_pka1)
        mismatches.append("source-to-Python PKA1 fields differ: " + ", ".join(fields))
    if not d1s_matches:
        fields = _mapping_differences(mapped_d1s, active_python_d1s)
        values = "; ".join(
            f"{key} (source={mapped_d1s.get(key)!r}, Python={active_python_d1s.get(key)!r})"
            for key in fields
        )
        mismatches.append("active central D1S fields differ: " + values)
    return SourceParameterMapping(
        pka1_source_to_python=mapped_pka1,
        pka1_python=python_pka1,
        d1s_active_source_to_python=mapped_d1s,
        d1s_python=active_python_d1s,
        pka1_matches=pka1_matches,
        d1s_active_matches=d1s_matches,
        mismatches=tuple(mismatches),
        unit_conventions={
            "meson_masses_and_nucleon_masses": "MeV",
            "hbar_c": "MeV fm",
            "rho_sat": "fm^-3",
            "PKA1 vertices and density-dependence coefficients": "dimensionless",
            "grtn_to_f_rho": "dimensionless ratio, not converted vertex; vertex=g_rho*f_rho/(M_n/hbar_c+M_p/hbar_c)",
            "D1S ranges_fm": "fm",
            "D1S w_mev/b_mev/h_mev/m_mev": "MeV central-force coefficients",
            "D1S strength": "dimensionless Gogny multiplier",
            "excluded D1S w0/t3": "not used by current Python central pairing implementation",
        },
    )


def parse_source_template(source_directory: str | Path) -> SourceRunTemplate:
    """Parse one supplied source snapshot and its adjacent input file."""
    root = Path(source_directory).resolve()
    if root.name == "Broyden" and root.parent.name == "RHFB-BI":
        variant, code_root, input_name = "RHFB-BI/Broyden", root, "rhfb.dat"
    elif root.name == "RHFB-BI":
        variant, code_root, input_name = "RHFB-BI", root, "rhfb.dat"
    elif root.name == "S-RHFB-C":
        variant, code_root, input_name = "S-RHFB-C", root, "rhf.dat"
    else:
        raise ValueError("source directory must be the supplied RHFB-BI, RHFB-BI/Broyden, or S-RHFB-C snapshot")
    define_path = code_root / "Define.f90"
    gogny_path = code_root / "Gogny.f90"
    inout_path = code_root / "Inout.f90"
    base_path = code_root / "BASE.f90"
    dirac_path = code_root / "Dirac.f90"
    input_path = code_root / input_name
    for path in (define_path, gogny_path, inout_path, base_path, dirac_path, input_path):
        if not path.is_file():
            raise FileNotFoundError(f"required source-template file is absent: {path}")

    source_paths = (define_path, gogny_path, inout_path, base_path, dirac_path, input_path)
    source_bytes = {path.name: path.read_bytes() for path in source_paths}
    decoded = {name: _decode_source_bytes(raw) for name, raw in source_bytes.items()}
    source_text = {name: pair[0] for name, pair in decoded.items()}
    source_encoding = {name: pair[1] for name, pair in decoded.items()}
    define_text = source_text[define_path.name]
    gogny_text = source_text[gogny_path.name]
    input_text = source_text[input_path.name]
    pka1 = _fortran_data_block(define_text, "PKA1", _PKA1_REQUIRED)
    d1s = _fortran_data_block(gogny_text, "D1S", _D1S_REQUIRED)
    scalar_constants = {
        "hbc": _scalar_data_value(define_text, "hbc"),
        "alphi": _scalar_data_value(define_text, "alphi"),
    }
    records = _code_records(input_text)
    issues: list[str] = []

    pset_raw = records.get("pset")
    pset_index = _parse_integer_record(pset_raw, "pset") if pset_raw is not None else None
    pset_name = "PKA1" if pset_index == 0 else (f"pset={pset_index}" if pset_index is not None else None)
    if pset_index is None and variant == "RHFB-BI":
        issues.append("model selection is interactive and absent from rhfb.dat")
    elif pset_index is None:
        issues.append("parameter-set index is absent from the input snapshot")

    lfg_raw = records.get("lfg", records.get("pair"))
    strength_raw = records.get("gnfv0", records.get("v0vg"))
    pairing_force: str | None = None
    pairing_strength: float | None = None
    if lfg_raw is None:
        issues.append("pairing-force selection is absent or interactive")
    elif _parse_integer_record(lfg_raw, "pairing selector") == 1:
        pairing_force = "D1S"
        if strength_raw is None:
            issues.append("Gogny D1S input multiplier is absent or interactive")
        else:
            pairing_strength = _parse_numeric_record(strength_raw, "D1S multiplier")[0]
    elif _parse_integer_record(lfg_raw, "pairing selector") == 0:
        pairing_force = "delta"
        issues.append("zero-range pairing is unsupported by the NN D1S template")
    else:
        issues.append("unrecognized pairing selector")

    if variant == "RHFB-BI":
        effective_exchange = None if pset_index is None else pset_index <= 3
    elif variant == "RHFB-BI/Broyden":
        effective_exchange = None if pset_index is None else pset_index <= 3
    else:
        ie = int(pka1.get("ie", -1)) if pset_index == 0 else None
        effective_exchange = None if ie is None else ie != 0

    tensor_raw = records.get("tensor")
    if tensor_raw is not None:
        tensor_value = _parse_integer_record(tensor_raw, "tensor switch")
        if tensor_value not in (0, 1):
            raise ValueError("tensor switch must be 0 or 1")
        tensor_enabled = tensor_value == 1
    elif variant == "RHFB-BI/Broyden" and effective_exchange:
        # This source has no tensor-off input switch; PKA1 tensor channels are
        # kept by the compiled interaction path.
        tensor_enabled = True
    else:
        tensor_enabled = None
    if pset_index == 0 and effective_exchange is False:
        issues.append("effective interaction mode disables the PKA1 Fock channels")
    if pset_index == 0 and tensor_enabled is False:
        issues.append("source input drops PKA1 tensor channels")

    blocking = _read_block_counts(input_text)
    blocking_n, blocking_p = blocking if blocking is not None else (None, None)
    if blocking is None:
        issues.append("n/p blocking count/index records are absent")

    xct_positive = xct_negative = None
    xpt: tuple[float, float] | None = None
    xct_raw, xpt_raw = records.get("xect"), records.get("xept")
    if xct_raw is not None:
        values = _parse_numeric_record(xct_raw, "XCT")
        if len(values) != 2:
            raise ValueError("XCT must contain positive and negative DWS basis cutoffs")
        xct_positive, xct_negative = values
    if xpt_raw is not None:
        values = _parse_numeric_record(xpt_raw, "XPT")
        if len(values) != 2:
            raise ValueError("XPT must contain neutron and proton pairing windows")
        xpt = (values[0], values[1])

    digests = {name: hashlib.sha256(raw).hexdigest() for name, raw in source_bytes.items()}
    return SourceRunTemplate(
        source_variant=variant,
        source_directory=str(root),
        input_filename=input_name,
        source_sha256=digests,
        source_encoding=source_encoding,
        parameter_set_index=pset_index,
        parameter_set_name=pset_name,
        pka1_constants=pka1,
        pka1_scalar_constants=scalar_constants,
        d1s_constants=d1s,
        pairing_force=pairing_force,
        pairing_strength=pairing_strength,
        blocking_n=blocking_n,
        blocking_p=blocking_p,
        xct_positive_mev=xct_positive,
        xct_negative_relative_to_2m_mev=xct_negative,
        xpt_mev=xpt,
        tensor_enabled=tensor_enabled,
        effective_exchange_enabled=effective_exchange,
        unsupported_reasons=tuple(issues),
    )


def audit_payload(source_directory: str | Path) -> dict[str, Any]:
    template = parse_source_template(source_directory)
    mapping = compare_source_to_python(template)
    return {
        "template": asdict(template),
        "python_parameter_mapping": asdict(mapping),
        "audit_only": True,
        "no_fortran_execution": True,
        "no_sea_occupation_rule": "not inferred from source XCT/XPT values",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only audit of supplied RHFB source inputs and Python parameters")
    parser.add_argument("--source", required=True, help="source directory: RHFB-BI, RHFB-BI/Broyden, or S-RHFB-C")
    parser.add_argument("--output", help="write JSON audit here; default prints JSON to stdout")
    args = parser.parse_args(argv)
    rendered = json.dumps(audit_payload(args.source), indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
