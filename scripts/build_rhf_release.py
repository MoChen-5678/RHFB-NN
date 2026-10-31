from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.0.0"
NAME = f"DPL-RHF-Standalone-v{VERSION}"
DOCS = (
    "RHF_STANDALONE_GUIDE.md",
    "RHF_RECOMPUTE_BENCHMARK.md",
    "RHF_STRICT_IMPLEMENTATION.md",
    "RHF_FORMULA_AUDIT_20260809.md",
    "PKA1_COMPLETE_CHANNEL_AUDIT.md",
    "PHYSICS_AND_WORKFLOW.md",
    "EXPERIMENT_DESIGN.md",
    "TROUBLESHOOTING.md",
    "REFERENCES.md",
)
SCRIPT_MAP = {
    "setup_rhf_env.sh": "setup_env.sh",
    "activate_rhf_env.sh": "activate_env.sh",
    "run_rhf_release.sh": "run_rhf.sh",
    "run_rhf_background.sh": "run_background.sh",
    "monitor_rhf_release.sh": "monitor_rhf.sh",
    "stop_rhf_background.sh": "stop_rhf.sh",
    "compare_rhf_core_release.sh": "compare_with_core.sh",
    "check_rhf_environment.py": "scripts/check_rhf_environment.py",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def copy_python_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix not in {".py", ".typed"}:
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def write_manifest(release: Path) -> None:
    rows = []
    for path in sorted(item for item in release.rglob("*") if item.is_file() and item.name != "MANIFEST.txt"):
        rows.append(f"{digest(path)}  {path.stat().st_size:12d}  {path.relative_to(release)}")
    (release / "MANIFEST.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def build(destination: Path) -> tuple[Path, Path, Path, Path]:
    release = destination / NAME
    if release.exists():
        shutil.rmtree(release)
    release.mkdir(parents=True)

    copy_python_tree(ROOT / "dpl_rhf", release / "dpl_rhf")
    for source_name, target_name in SCRIPT_MAP.items():
        target = release / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "scripts" / source_name, target)
        if target.suffix == ".sh":
            target.chmod(0o755)

    for filename in ("LICENSES.md",):
        shutil.copy2(ROOT / filename, release / filename)
    shutil.copy2(ROOT / "pyproject-rhf.toml", release / "pyproject.toml")
    shutil.copy2(ROOT / "requirements-rhf.txt", release / "requirements.txt")
    shutil.copy2(ROOT / "environment-rhf.yml", release / "environment.yml")
    shutil.copy2(ROOT / "docs" / "RHF_STANDALONE_GUIDE.md", release / "README.md")

    docs = release / "docs"
    docs.mkdir()
    for filename in DOCS:
        shutil.copy2(ROOT / "docs" / filename, docs / filename)
    shutil.copytree(ROOT / "docs" / "images", docs / "images")
    shutil.copytree(ROOT / "examples" / "rhf", release / "examples" / "rhf")
    for script in (release / "examples" / "rhf").glob("*.sh"):
        script.chmod(0o755)

    (release / "work" / "outputs").mkdir(parents=True)
    (release / "work" / "graph-cache").mkdir(parents=True)
    (release / "work" / "README.md").write_text(
        "Runtime outputs, logs, checkpoints and temporary graph caches are stored here.\n",
        encoding="utf-8",
    )
    write_manifest(release)

    tar_path = destination / f"{NAME}.tar.gz"
    zip_path = destination / f"{NAME}.zip"
    for path in (tar_path, zip_path):
        if path.exists():
            path.unlink()
    with tarfile.open(tar_path, "w:gz") as archive:
        archive.add(release, arcname=NAME)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(release.rglob("*")):
            relative = Path(NAME) / path.relative_to(release)
            if path.is_dir():
                continue
            info = zipfile.ZipInfo.from_file(path, arcname=str(relative))
            with path.open("rb") as handle:
                archive.writestr(info, handle.read(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

    tar_checksum = destination / f"{tar_path.name}.sha256"
    zip_checksum = destination / f"{zip_path.name}.sha256"
    tar_checksum.write_text(f"{digest(tar_path)}  {tar_path.name}\n", encoding="utf-8")
    zip_checksum.write_text(f"{digest(zip_path)}  {zip_path.name}\n", encoding="utf-8")
    return tar_path, tar_checksum, zip_path, zip_checksum


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the standalone DPL-RHF source release")
    parser.add_argument("--destination", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    for path in build(args.destination):
        print(path)


if __name__ == "__main__":
    main()
