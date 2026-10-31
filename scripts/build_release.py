from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import shutil
import tarfile


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT.parent / "Core-1204"
VERSION = "1.0.0"
NAME = f"DPL-RMF-DensityMatrix-v{VERSION}-linux-x86_64"
FORMAL_SUBPACKAGES = ("backends", "cli", "functionals", "io", "models", "tests", "training")
DOC_FILES = (
    "INSTALLATION.md", "USER_GUIDE.md", "PHYSICS_AND_WORKFLOW.md",
    "EXPERIMENT_DESIGN.md", "RESULTS_PKDD_O16.md", "OUTPUT_SCHEMA.md",
    "DEVELOPER_GUIDE.md", "TROUBLESHOOTING.md", "REFERENCES.md", "index.html",
)
EXAMPLE = ROOT / "outputs" / "pkdd_o16_density_matrix_refresh_cpu_500"


def copy_python_tree(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*.py"):
        if "__pycache__" not in path.parts:
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(release: Path) -> None:
    rows = []
    for path in sorted(p for p in release.rglob("*") if p.is_file() and p.name != "MANIFEST.txt"):
        rows.append(f"{sha256(path)}  {path.stat().st_size:12d}  {path.relative_to(release)}")
    (release / "MANIFEST.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def build(destination: Path) -> tuple[Path, Path]:
    release = destination / NAME
    if release.exists():
        shutil.rmtree(release)
    app = release / "DFT-RHF"; package = app / "dpl_rhf"
    package.mkdir(parents=True)
    for filename in ("__init__.py", "py.typed"):
        shutil.copy2(ROOT / "dpl_rhf" / filename, package / filename)
    for subpackage in FORMAL_SUBPACKAGES:
        copy_python_tree(ROOT / "dpl_rhf" / subpackage, package / subpackage)
    legacy = package / "legacy"; legacy.mkdir()
    for filename in ("__init__.py", "dpl_rmf_core.py"):
        shutil.copy2(ROOT / "dpl_rhf" / "legacy" / filename, legacy / filename)

    for filename in ("README.md", "pyproject.toml", "setup.py", "requirements.txt", "environment.yml", "LICENSES.md"):
        shutil.copy2(ROOT / filename, app / filename)
    scripts = app / "scripts"; scripts.mkdir()
    for filename in ("build_docs.py", "install.sh"):
        shutil.copy2(ROOT / "scripts" / filename, scripts / filename)
    docs = app / "docs"; docs.mkdir()
    for filename in DOC_FILES:
        shutil.copy2(ROOT / "docs" / filename, docs / filename)
    shutil.copytree(ROOT / "docs" / "images", docs / "images")
    shutil.copytree(EXAMPLE, app / "examples" / "PKDD_O16")

    core_target = release / "Core-1204"; core_target.mkdir()
    makefile = (CORE / "Makefile.build").read_text(encoding="utf-8")
    sources = sorted(set(re.findall(r"[A-Za-z0-9_]+\.f90", makefile)))
    for filename in ("Makefile.build", "rhf_ctypes.py", "librhf.so", *sources):
        source = CORE / filename
        if not source.is_file():
            raise FileNotFoundError(f"required Core-1204 file is missing: {source}")
        shutil.copy2(source, core_target / filename)

    install = release / "install.sh"
    install.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "ROOT=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        "cd \"$ROOT/DFT-RHF\"\npython -m pip install . --no-deps --no-build-isolation\n"
        "DPL_RHF_CORE_DIR=\"$ROOT/Core-1204\" python -m dpl_rhf.cli.check_environment\n",
        encoding="utf-8",
    )
    install.chmod(0o755); (scripts / "install.sh").chmod(0o755)
    (release / "check_environment.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "ROOT=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        "cd \"$ROOT/DFT-RHF\"\n"
        "DPL_RHF_CORE_DIR=\"$ROOT/Core-1204\" python -m dpl_rhf.cli.check_environment \"$@\"\n",
        encoding="utf-8",
    )
    (release / "check_environment.sh").chmod(0o755)
    shutil.copy2(ROOT / "LICENSES.md", release / "LICENSES.md")
    write_manifest(release)

    archive = destination / f"{NAME}.tar.gz"
    if archive.exists(): archive.unlink()
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(release, arcname=NAME, recursive=True)
    checksum = destination / f"{archive.name}.sha256"
    checksum.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    return archive, checksum


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the allowlisted standalone DPL-RMF Linux release")
    parser.add_argument("--destination", type=Path, default=ROOT / "dist")
    args = parser.parse_args(); args.destination.mkdir(parents=True, exist_ok=True)
    archive, checksum = build(args.destination)
    print(archive); print(checksum)


if __name__ == "__main__":
    main()
