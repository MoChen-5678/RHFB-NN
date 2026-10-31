from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re

import numpy as np

from dpl_rhf.backends.base import NucleusCase
from dpl_rhf.io.outputs import write_json
from dpl_rhf.legacy.dpl_rmf_core import RHFCore
from dpl_rhf.rhf.parameters import rhf_parameters


R_COMPONENTS = ("GG", "GF", "FG", "FF")


def _run_core(case: NucleusCase, core_dir: str | None, tolerance: float, limit: int):
    core = RHFCore(core_dir)
    core.init(case.model, case.z, case.n, case.a)
    history = []
    for iteration in range(1, limit + 1):
        residual = core.step()
        history.append({"iteration": iteration, "residual": residual})
        if iteration >= 5 and abs(residual) <= tolerance:
            break
    return core, history


def _dpl_levels(result_dir: Path) -> list[dict[str, float | int | str]]:
    factors = np.load(result_dir / "density_matrix_factors.npz")
    matrices = np.load(result_dir / "nonlocal_hamiltonians.npz")
    rows = []
    block_ids = sorted({key.split("_")[1] for key in factors.files if key.startswith("block_")})
    for block_id in block_ids:
        prefix = f"block_{block_id}"
        species = "n" if float(factors[f"{prefix}_species_sign"]) > 0.0 else "p"
        kappa = int(factors[f"{prefix}_kappa"])
        principals = factors[f"{prefix}_principals"].astype(int)
        occupations = factors[f"{prefix}_occupations"]
        radial = factors[f"{prefix}_factors"]
        matrix = matrices[f"h_total_{species}_kappa_{kappa:+d}"]
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        ranking = np.argsort(np.abs(eigenvectors.T @ radial), axis=0)[::-1]
        used: set[int] = set()
        for state, principal in enumerate(principals):
            selected = next(int(value) for value in ranking[:, state] if int(value) not in used)
            used.add(selected)
            rows.append({
                "species": species,
                "kappa": kappa,
                "principal": int(principal),
                "occupation": float(occupations[state]),
                "dpl_energy_mev": float(eigenvalues[selected]),
            })
    return rows


def _core_levels(waves: dict[str, np.ndarray]) -> dict[tuple[str, int, int], float]:
    levels = {}
    for i in range(len(waves["energy"])):
        if float(waves["occupancy"][i]) <= 1.0e-10:
            continue
        match = re.search(r"\.(\d+)[a-z]", str(waves["name"][i]).lower())
        principal = int(match.group(1)) if match else int(waves["n"][i])
        levels[(str(waves["species"][i]), int(waves["kappa"][i]), principal)] = float(waves["energy"][i])
    return levels


def _dpl_nonlocal_densities(result_dir: Path) -> tuple[np.ndarray, dict[tuple[str, int], np.ndarray]]:
    """Rebuild Eq. (3.94) from the neural occupied density factors."""

    factors = np.load(result_dir / "density_matrix_factors.npz")
    r = np.asarray(factors["r"], dtype=np.float64)
    h = float(r[2] - r[1])
    blocks: dict[tuple[str, int], np.ndarray] = {}
    block_ids = sorted({key.split("_")[1] for key in factors.files if key.startswith("block_")})
    for block_id in block_ids:
        prefix = f"block_{block_id}"
        species = "n" if float(factors[f"{prefix}_species_sign"]) > 0.0 else "p"
        kappa = int(factors[f"{prefix}_kappa"])
        degeneracy = float(factors[f"{prefix}_degeneracy"])
        occupations = np.asarray(factors[f"{prefix}_occupations"], dtype=np.float64)
        radial = np.asarray(factors[f"{prefix}_factors"], dtype=np.float64) / np.sqrt(h)
        split = radial.shape[0] // 2
        g, f = radial[:split], radial[split:]
        blocks[(species, kappa)] = degeneracy * np.stack(
            (
                (g * occupations[None, :]) @ g.T,
                (g * occupations[None, :]) @ f.T,
                (f * occupations[None, :]) @ g.T,
                (f * occupations[None, :]) @ f.T,
            )
        )
    return r[1:-1], blocks


def _core_nonlocal_densities(waves: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[tuple[str, int], np.ndarray]]:
    """Rebuild Core ``denf`` exactly from its final occupied wavefunctions."""

    r = np.asarray(waves["r"], dtype=np.float64)[1:-1]
    blocks: dict[tuple[str, int], np.ndarray] = {}
    keys = sorted(
        {
            (str(waves["species"][i]), int(waves["kappa"][i]))
            for i in range(len(waves["energy"]))
            if float(waves["occupancy"][i]) > 1.0e-10
        }
    )
    for key in keys:
        selected = [
            i
            for i in range(len(waves["energy"]))
            if (str(waves["species"][i]), int(waves["kappa"][i])) == key
            and float(waves["occupancy"][i]) > 1.0e-10
        ]
        g = np.asarray(waves["G"][selected, 1:-1], dtype=np.float64).T
        f = np.asarray(waves["F"][selected, 1:-1], dtype=np.float64).T
        weights = np.asarray(
            [float(waves["occupancy"][i]) * float(waves["degeneracy"][i]) for i in selected],
            dtype=np.float64,
        )
        blocks[key] = np.stack(
            (
                (g * weights[None, :]) @ g.T,
                (g * weights[None, :]) @ f.T,
                (f * weights[None, :]) @ g.T,
                (f * weights[None, :]) @ f.T,
            )
        )
    return r, blocks


def _nonlocal_density_metrics(dpl: np.ndarray, core: np.ndarray) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for index, component in enumerate(R_COMPONENTS):
        delta = dpl[index] - core[index]
        reference_norm = float(np.linalg.norm(core[index]))
        rows.append(
            {
                "component": component,
                "rmse_fm_inverse": float(np.sqrt(np.mean(delta**2))),
                "mae_fm_inverse": float(np.mean(np.abs(delta))),
                "max_abs_fm_inverse": float(np.max(np.abs(delta))),
                "relative_l2_percent": 100.0 * float(np.linalg.norm(delta)) / max(reference_norm, 1.0e-30),
            }
        )
    return rows


def _plot_nonlocal_densities(
    result_dir: Path,
    out: Path,
    waves: dict[str, np.ndarray],
    case: NucleusCase,
) -> dict[str, object]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r_dpl, dpl_blocks = _dpl_nonlocal_densities(result_dir)
    r_core, core_blocks = _core_nonlocal_densities(waves)
    if not np.allclose(r_dpl, r_core, rtol=0.0, atol=1.0e-12):
        raise ValueError("DPL and Core nonlocal-density grids do not match")
    keys = sorted(set(dpl_blocks).intersection(core_blocks))
    if set(dpl_blocks) != set(core_blocks):
        raise ValueError(
            f"DPL/Core occupied (species,kappa) blocks differ: "
            f"DPL={sorted(dpl_blocks)}, Core={sorted(core_blocks)}"
        )

    image_dir = out / "nonlocal_density_R_blocks"
    image_dir.mkdir(parents=True, exist_ok=True)
    archive: dict[str, np.ndarray] = {"r": r_dpl}
    metric_rows: list[dict[str, float | int | str]] = []
    crop = r_dpl <= 12.0
    extent = (float(r_dpl[crop][0]), float(r_dpl[crop][-1])) * 2
    for species, kappa in keys:
        dpl = dpl_blocks[(species, kappa)]
        core = core_blocks[(species, kappa)]
        tag = f"{species}_kappa_{kappa:+d}".replace("+", "p").replace("-", "m")
        archive[f"dpl_{tag}"] = dpl
        archive[f"core_{tag}"] = core
        archive[f"delta_{tag}"] = dpl - core
        rows = _nonlocal_density_metrics(dpl, core)
        for row in rows:
            metric_rows.append({"species": species, "kappa": kappa, **row})

        fig, axes = plt.subplots(4, 3, figsize=(12.0, 13.0), constrained_layout=True)
        for component_index, component in enumerate(R_COMPONENTS):
            dpl_view = dpl[component_index][np.ix_(crop, crop)]
            core_view = core[component_index][np.ix_(crop, crop)]
            delta_view = dpl_view - core_view
            common = max(float(np.max(np.abs(dpl_view))), float(np.max(np.abs(core_view))), 1.0e-30)
            difference = max(float(np.max(np.abs(delta_view))), 1.0e-30)
            for column, (values, title, limit) in enumerate(
                ((dpl_view, "DPL", common), (core_view, "Core-1204", common), (delta_view, "DPL - Core", difference))
            ):
                image = axes[component_index, column].imshow(
                    values,
                    origin="lower",
                    extent=extent,
                    aspect="equal",
                    cmap="coolwarm",
                    vmin=-limit,
                    vmax=limit,
                )
                axes[component_index, column].set_title(f"{component}: {title}")
                axes[component_index, column].set(xlabel="r' (fm)", ylabel="r (fm)")
                fig.colorbar(image, ax=axes[component_index, column], label=r"$R$ (fm$^{-1}$)")
        particle = "neutron" if species == "n" else "proton"
        fig.suptitle(f"{case.model} {case.a}: nonlocal density R, {particle}, kappa={kappa:+d}")
        fig.savefig(image_dir / f"R_{tag}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    np.savez_compressed(out / "nonlocal_density_R_comparison.npz", **archive)
    fields = ["species", "kappa", "component", "rmse_fm_inverse", "mae_fm_inverse", "max_abs_fm_inverse", "relative_l2_percent"]
    with (out / "nonlocal_density_R_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    for axis, component in zip(axes.flat, R_COMPONENTS):
        component_rows = [row for row in metric_rows if row["component"] == component]
        labels = [f"{row['species']} k={int(row['kappa']):+d}" for row in component_rows]
        values = [float(row["relative_l2_percent"]) for row in component_rows]
        axis.bar(np.arange(len(values)), values, color="#3478a8")
        axis.set_title(component)
        axis.set_ylabel("Relative L2 error (%)")
        axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(f"{case.model} {case.a}: DPL vs Core nonlocal-density errors")
    fig.savefig(out / "nonlocal_density_R_error_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    all_dpl = np.concatenate([dpl_blocks[key].reshape(-1) for key in keys])
    all_core = np.concatenate([core_blocks[key].reshape(-1) for key in keys])
    delta = all_dpl - all_core
    return {
        "definition": "R_(kappa,q)^(mu,mu')(r,r')=(2j+1) sum_n v_n^2 u_n^mu(r)u_n^mu'(r')",
        "matched_blocks": len(keys),
        "aggregate_rmse_fm_inverse": float(np.sqrt(np.mean(delta**2))),
        "aggregate_relative_l2_percent": 100.0 * float(np.linalg.norm(delta)) / max(float(np.linalg.norm(all_core)), 1.0e-30),
    }


def _plot_local(result_dir: Path, out: Path, core: RHFCore, case: NucleusCase) -> dict:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = rhf_parameters(case.model)
    local = np.load(result_dir / "local_observables.npz")
    r = np.asarray(local["r"])
    dpl_s = np.asarray(local["component_S_direct"]) * params.hbar_c
    dpl_v0 = np.asarray(local["component_V0_total_local"]) * params.hbar_c
    dpl_v3 = np.asarray(local["component_V3_direct"]) * params.hbar_c
    dpl_vc = np.asarray(local["component_VC_direct"]) * params.hbar_c
    dpl = {"S_n": dpl_s, "S_p": dpl_s, "V_n": dpl_v0 + dpl_v3, "V_p": dpl_v0 - dpl_v3 + dpl_vc}

    stack = core.local_stack()
    scf = {}
    for species, offset, mass in (("n", 0, params.mass_n), ("p", 2, params.mass_p)):
        scf[f"S_{species}"] = 0.5 * params.hbar_c * (stack[offset] - stack[offset + 1]) - mass
        scf[f"V_{species}"] = 0.5 * params.hbar_c * (stack[offset] + stack[offset + 1]) + mass
    np.savez_compressed(out / "local_s_v_comparison.npz", r=r, **{f"dpl_{k}": v for k, v in dpl.items()}, **{f"core_{k}": v for k, v in scf.items()})
    mask = (r >= 0.1) & (r <= 12.0)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharex=True, constrained_layout=True)
    for axis, species in zip(axes, ("n", "p")):
        for source, label, style in ((dpl, "DPL local", "-"), (scf, "Core local", "--")):
            scalar, vector = source[f"S_{species}"], source[f"V_{species}"]
            axis.plot(r[mask], (scalar + vector)[mask], style, lw=2.0, label=f"{label} S+V")
            axis.plot(r[mask], (scalar - vector)[mask], style, lw=1.7, label=f"{label} S-V")
        axis.set_title("Neutron" if species == "n" else "Proton")
        axis.set(xlabel="r (fm)", ylabel="Local potential combination (MeV)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{case.model} Z={case.z} N={case.n}: local direct+rearrangement")
    fig.savefig(out / "s_plus_minus_v_local_dpl_core.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        key: {
            "mae_mev": float(np.mean(np.abs(dpl[key][1:] - scf[key][1:]))),
            "rmse_mev": float(np.sqrt(np.mean((dpl[key][1:] - scf[key][1:]) ** 2))),
        }
        for key in dpl
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Final neural PKA1 RHF versus Core-1204 comparison")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--z", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--a", type=int, required=True)
    parser.add_argument("--core-dir", default=None)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result_dir, out = Path(args.result_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    case = NucleusCase("PKA1", args.z, args.n, args.a)
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    core, history = _run_core(case, args.core_dir, args.tolerance, args.max_iterations)
    energy = core.energy()
    local_metrics = _plot_local(result_dir, out, core, case)
    waves = core.wavefunctions()
    nonlocal_density_metrics = _plot_nonlocal_densities(result_dir, out, waves, case)
    dpl_levels = _dpl_levels(result_dir)
    core_levels = _core_levels(waves)
    matched = []
    for row in dpl_levels:
        key = (str(row["species"]), int(row["kappa"]), int(row["principal"]))
        if key in core_levels and float(row["occupation"]) > 1.0e-10:
            row["core_energy_mev"] = core_levels[key]
            row["delta_mev"] = float(row["dpl_energy_mev"]) - core_levels[key]
            matched.append(row)
    with (out / "single_particle_levels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["species", "kappa", "principal", "occupation", "dpl_energy_mev", "core_energy_mev", "delta_mev"])
        writer.writeheader(); writer.writerows(matched)
    differences = np.asarray([float(row["delta_mev"]) for row in matched])
    comparison = {
        "case": {"model": "PKA1", "z": args.z, "n": args.n, "a": args.a},
        "core_iterations": len(history),
        "core_final_residual": history[-1]["residual"],
        "dpl": {
            "energy_total_no_com_mev": summary["energy_total_no_com_mev"],
            "energy_per_a_no_com_mev": summary["energy_per_a_no_com_mev"],
            **summary["diagnostics"],
            **summary.get("solver_counts", {}),
        },
        "core": energy.__dict__,
        "differences": {
            "energy_per_a_no_com_mev": summary["energy_per_a_no_com_mev"] - energy.e_per_A_no_com,
            "energy_total_no_com_mev": summary["energy_total_no_com_mev"] - energy.e_total_no_com,
            "rms_matter_no_com_fm": summary["diagnostics"]["rms_matter_no_com"] - energy.rms_matter_no_com,
            "charge_radius_no_com_fm": summary["diagnostics"]["charge_radius_no_com"] - energy.charge_radius_no_com,
            "occupied_level_mae_mev": float(np.mean(np.abs(differences))) if differences.size else None,
            "occupied_level_rmse_mev": float(np.sqrt(np.mean(differences**2))) if differences.size else None,
            "local_direct_rearrangement": local_metrics,
            "nonlocal_density_R": nonlocal_density_metrics,
        },
        "warning": "PKA1 Fock is nonlocal. S+/-V compares h_D including all rearrangement; h_E input is compared separately through the four R(r,r') nonlocal-density components for every species/kappa block.",
    }
    write_json(out / "detailed_comparison.json", comparison)
    np.savez_compressed(out / "core1204_scf_profiles.npz", r=core.r, stack=core.local_stack(), **core.fields(), **core.densities())
    print(json.dumps(comparison["differences"], indent=2))


if __name__ == "__main__":
    main()
