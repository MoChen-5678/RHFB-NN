#!/usr/bin/env python3
"""Build the portable DPL-NN-RHF and DPL-NN-RHFB tarballs (source, guard, examples, README).

Excludes caches, outputs, backups and the Fortran source templates used only by
the 3D prototype tests.  The RHFB package also carries the parsed Fortran
references (results only) and three warm-start models.
"""
import hashlib
import shutil
import tarfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
STAMP = date.today().strftime("%Y%m%d")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "source_templates")


def common(stage: Path, kind: str):
    shutil.copytree(ROOT / "dpl_rhf", stage / "dpl_rhf", ignore=IGNORE)
    for name in ("pyproject.toml", "setup.py"):
        shutil.copy2(ROOT / name, stage / name)
    (stage / "scripts").mkdir()
    shutil.copy2(ROOT / "scripts/run_rhfb_procguard.py", stage / "scripts")
    for name in ("install_env.sh", "requirements.txt"):
        shutil.copy2(ROOT / "packaging/common" / name, stage / name)
    shutil.copy2(ROOT / f"packaging/{kind}/README.md", stage / "README.md")
    shutil.copytree(ROOT / f"packaging/{kind}/examples", stage / "examples")


def rhfb_extras(stage: Path):
    shutil.copy2(ROOT / "scripts/compare_rhfb_fortran.py", stage / "scripts")
    for sub in ("runs", "runs_kmx88_remote"):
        source = ROOT / "outputs/fortran_reference" / sub
        for parsed in sorted(source.glob("*/parsed.json")):
            target = stage / "outputs/fortran_reference" / sub / parsed.parent.name
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(parsed, target / "parsed.json")
    final = ROOT / "outputs/rhfb_spherical/final"
    for nucleus in ("O16", "O18", "Ca40"):
        model = (final / nucleus).resolve() / "model_final.pt"
        if model.exists():
            target = stage / "warm_start" / nucleus
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model, target / "model_final.pt")
            shutil.copy2(model.with_name("summary.json"), target / "summary.json")


def build(kind: str) -> Path:
    name = f"dpl-nn-{kind}-{STAMP}"
    stage = DIST / name
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    common(stage, kind)
    if kind == "rhfb":
        rhfb_extras(stage)
    archive = DIST / f"{name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(stage, arcname=name)
    return archive


def main():
    DIST.mkdir(exist_ok=True)
    lines = []
    for kind in ("rhf", "rhfb"):
        archive = build(kind)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        lines.append(f"{digest}  {archive.name}")
        print(f"{archive}  {archive.stat().st_size/1e6:.1f} MB")
    (DIST / f"SHA256SUMS-{STAMP}").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
