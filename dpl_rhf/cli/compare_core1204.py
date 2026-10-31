from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from dpl_rhf.backends.base import NucleusCase
from dpl_rhf.backends.torch_rmf import TorchRMFBackend
from dpl_rhf.legacy.dpl_rmf_core import RHFCore


HBAR_C = 197.328284


def write_vs_comparison(
    out: Path, r: np.ndarray, dpl_stack: np.ndarray, scf_stack: np.ndarray,
    mass_n: float, mass_p: float, title: str,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: dict[str, np.ndarray] = {"r": r}
    for species, offset, mass in (("n", 0, mass_n), ("p", 2, mass_p)):
        for prefix, stack in (("dpl", dpl_stack), ("scf", scf_stack)):
            vps, vms = stack[offset], stack[offset + 1]
            rows[f"{prefix}_V_{species}"] = 0.5 * HBAR_C * (vps + vms) + mass
            rows[f"{prefix}_S_{species}"] = 0.5 * HBAR_C * (vps - vms) - mass
        rows[f"delta_V_{species}"] = rows[f"dpl_V_{species}"] - rows[f"scf_V_{species}"]
        rows[f"delta_S_{species}"] = rows[f"dpl_S_{species}"] - rows[f"scf_S_{species}"]
    np.savez(out / "vs_dpl_scf_data.npz", **rows)

    mask = (r >= 0.1) & (r <= 10.0)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), sharex=True, constrained_layout=True)
    colors = {"V": "#1769aa", "S": "#c43d3d"}
    for column, species in enumerate(("n", "p")):
        for field in ("V", "S"):
            axes[0, column].plot(r[mask], rows[f"dpl_{field}_{species}"][mask], color=colors[field], lw=2.1, label=f"DPL {field}")
            axes[0, column].plot(r[mask], rows[f"scf_{field}_{species}"][mask], color=colors[field], lw=1.7, ls="--", label=f"SCF {field}")
            axes[1, column].plot(r[mask], rows[f"delta_{field}_{species}"][mask], color=colors[field], lw=1.8, label=f"Delta {field}")
        axes[0, column].set_title("Neutron potentials" if species == "n" else "Proton potentials")
        axes[0, column].set_ylabel("Potential (MeV)"); axes[0, column].grid(alpha=0.25); axes[0, column].legend(frameon=False, ncol=2)
        axes[1, column].set_xlabel("r (fm)"); axes[1, column].set_ylabel("DPL - SCF (MeV)"); axes[1, column].grid(alpha=0.25); axes[1, column].legend(frameon=False, ncol=2)
    fig.suptitle(title, fontsize=14)
    fig.savefig(out / "vs_dpl_scf.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "vs_dpl_scf.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), sharex=True, constrained_layout=True)
    for axis, species in zip(axes, ("n", "p")):
        for prefix, label, style in (("dpl", "DPL", "-"), ("scf", "Core SCF", "--")):
            scalar = rows[f"{prefix}_S_{species}"]
            vector = rows[f"{prefix}_V_{species}"]
            axis.plot(r[mask], (scalar + vector)[mask], ls=style, lw=2.0, label=f"{label} S+V")
            axis.plot(r[mask], (scalar - vector)[mask], ls=style, lw=1.7, label=f"{label} S-V")
        axis.set_title("Neutron" if species == "n" else "Proton")
        axis.set_xlabel("r (fm)")
        axis.set_ylabel("Potential combination (MeV)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(title + ": S+V and S-V", fontsize=13)
    fig.savefig(out / "s_plus_minus_v_dpl_scf.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "s_plus_minus_v_dpl_scf.pdf", bbox_inches="tight")
    plt.close(fig)

    combination_rows: dict[str, np.ndarray] = {"r": r}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), sharex=True, constrained_layout=True)
    for axis, species in zip(axes, ("n", "p")):
        for prefix, label, style in (("dpl", "DPL", "-"), ("scf", "Core SCF", "--")):
            scalar = rows[f"{prefix}_S_{species}"]
            vector = rows[f"{prefix}_V_{species}"]
            combination_rows[f"{prefix}_V_plus_S_{species}"] = vector + scalar
            combination_rows[f"{prefix}_V_minus_S_{species}"] = vector - scalar
            axis.plot(r[mask], (vector + scalar)[mask], ls=style, lw=2.0, label=f"{label} V+S")
            axis.plot(r[mask], (vector - scalar)[mask], ls=style, lw=1.7, label=f"{label} V-S")
        axis.set_title("Neutron" if species == "n" else "Proton")
        axis.set_xlabel("r (fm)")
        axis.set_ylabel("Potential combination (MeV)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(title + ": V+S and V-S", fontsize=13)
    fig.savefig(out / "v_plus_minus_s_dpl_scf.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "v_plus_minus_s_dpl_scf.pdf", bbox_inches="tight")
    plt.close(fig)
    np.savez(out / "v_plus_minus_s_dpl_scf_data.npz", **combination_rows)


def metrics(actual: np.ndarray, reference: np.ndarray, r: np.ndarray) -> dict[str, float]:
    # Core-1204 uses r[0]=0.001 fm as an origin extrapolation placeholder; its
    # equidistant physical integration mesh starts at r[1]=0.1 fm.
    delta = np.asarray(actual)[1:] - np.asarray(reference)[1:]
    physical_r = np.asarray(r)[1:]
    index = int(np.argmax(np.abs(delta)))
    return {
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(abs(delta[index])),
        "max_abs_r_fm": float(physical_r[index]),
        "signed_at_max": float(delta[index]),
    }


def occupied_levels(wavefunctions: dict[str, np.ndarray]) -> dict[tuple[str, str, int], dict]:
    result = {}
    for i, occupation in enumerate(wavefunctions["occupancy"]):
        if float(occupation) <= 1.0e-10:
            continue
        key = (
            str(wavefunctions["species"][i]),
            str(wavefunctions["name"][i]).strip(),
            int(wavefunctions["kappa"][i]),
        )
        result[key] = {
            "energy": float(wavefunctions["energy"][i]),
            "occupation": float(occupation),
        }
    return result


def run_core_scf(
    case: NucleusCase, tolerance: float, max_iterations: int, core_dir: str | None = None
) -> tuple[RHFCore, list[dict]]:
    core = RHFCore(core_dir)
    core.init(case.model, case.z, case.n, case.a)
    history = []
    for iteration in range(1, max_iterations + 1):
        residual = core.step()
        energy = core.energy()
        history.append({"iteration": iteration, "residual": residual, "energy_per_a_no_com": energy.e_per_A_no_com})
        if iteration >= 5 and abs(residual) <= tolerance:
            break
    return core, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Fresh Core-1204 SCF versus differentiable DPL comparison")
    parser.add_argument("--dpl-dir", required=True)
    parser.add_argument("--model", default="PKDD")
    parser.add_argument("--z", type=int, default=8)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--a", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument(
        "--core-dir",
        default=None,
        help="Core-1204 directory; defaults to DPL_RHF_CORE_DIR or the bundled sibling directory",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    case = NucleusCase(args.model, args.z, args.n, args.a or args.z + args.n)
    source = Path(args.dpl_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    archive = np.load(source / "dpl_hamiltonian.npz")
    dpl_stack = np.asarray(archive["stack"], dtype=np.float64)

    fixed = RHFCore(args.core_dir)
    fixed.init(case.model, case.z, case.n, case.a)
    fixed.set_local_stack(dpl_stack)
    fixed.solve_fixed_potential()
    fixed_energy = fixed.energy()
    fixed_waves = fixed.wavefunctions()

    scf, history = run_core_scf(case, args.tolerance, args.max_iterations, args.core_dir)
    scf_energy = scf.energy()
    scf_stack = scf.local_stack()
    scf_fields = scf.fields()
    scf_densities = scf.densities()
    scf_waves = scf.wavefunctions()

    backend = TorchRMFBackend(case, device=torch.device("cpu"), derivative_order=7)
    tensor = torch.as_tensor(dpl_stack, dtype=torch.float64).requires_grad_(True)
    dpl = backend.evaluate_tensor(tensor)
    direct = backend.functional.direct_energy_terms({**dpl.densities, **{k: dpl.fields[k] for k in ("sigma", "omega", "rho", "coul")}})
    kinetic = backend.functional.exact_kinetic_energy(dpl.orbitals)
    dpl_fields = {k: v.detach().cpu().numpy() for k, v in dpl.fields.items()}
    dpl_densities = {k: v.detach().cpu().numpy() for k, v in dpl.densities.items()}
    r = backend.r

    channel_names = ("vps_n", "vms_n", "vps_p", "vms_p")
    potential_metrics = {name: metrics(dpl_stack[i], scf_stack[i], r) for i, name in enumerate(channel_names)}
    field_metrics = {name: metrics(dpl_fields[name], scf_fields[name], r) for name in ("sigma", "omega", "rho", "coul")}
    density_profiles = {
        "rho_s": dpl_densities["rho_s_n"] + dpl_densities["rho_s_p"],
        "rho_b": dpl_densities["rho_v_n"] + dpl_densities["rho_v_p"],
        "rho_b3": dpl_densities["rho_v_n"] - dpl_densities["rho_v_p"],
    }
    density_metrics = {name: metrics(value, scf_densities[name], r) for name, value in density_profiles.items()}

    fixed_occupied = occupied_levels(fixed_waves)
    scf_occupied = occupied_levels(scf_waves)
    level_rows = []
    for key in sorted(set(fixed_occupied) & set(scf_occupied)):
        dpl_e = fixed_occupied[key]["energy"]
        scf_e = scf_occupied[key]["energy"]
        level_rows.append({
            "species": key[0], "name": key[1], "kappa": key[2],
            "occupation": fixed_occupied[key]["occupation"],
            "dpl_mev": dpl_e, "scf_mev": scf_e, "delta_mev": dpl_e - scf_e,
        })
    level_delta = np.asarray([row["delta_mev"] for row in level_rows])

    dpl_energy_terms = {
        "kinetic": float(kinetic.detach()),
        "sigma": float(direct["E_sigma"].detach()),
        "omega": float(direct["E_omega"].detach()),
        "rho": float(direct["E_rho"].detach()),
        "coulomb": float(direct["E_coul"].detach()),
        "direct_total": float(direct["E_direct"].detach()),
        "total_no_com": dpl.energy_total_no_com,
        "per_a_no_com": dpl.energy_per_a_no_com,
    }
    report = {
        "case": {"model": case.model, "z": case.z, "n": case.n, "a": case.a},
        "core1204_scf": {"iterations": len(history), "final_residual": history[-1]["residual"], "energy": scf_energy.__dict__},
        "dpl_differentiable_rmf": {"energy_terms_mev": dpl_energy_terms, "diagnostics": dpl.diagnostics},
        "dpl_core1204_fixed_potential": {
            "energy": fixed_energy.__dict__,
            "warning": "Core fixed-H energy contains fields retained from initialization; use it for eigenvalues only, not DPL total energy.",
        },
        "differences": {
            "energy_per_a_no_com_mev": dpl.energy_per_a_no_com - scf_energy.e_per_A_no_com,
            "matter_radius_no_com_fm": dpl.diagnostics["rms_matter_no_com"] - scf_energy.rms_matter_no_com,
            "charge_radius_no_com_fm": dpl.diagnostics["charge_radius_no_com"] - scf_energy.charge_radius_no_com,
            "occupied_level_mae_mev": float(np.mean(np.abs(level_delta))),
            "occupied_level_rmse_mev": float(np.sqrt(np.mean(level_delta * level_delta))),
            "potentials_fm_inverse": potential_metrics,
            "fields_fm_inverse": field_metrics,
            "densities_fm_inverse_cubed": density_metrics,
        },
    }
    (out / "detailed_comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    observable_rows = []
    for source_name, values in (
        ("dpl_energy", dpl_energy_terms),
        ("dpl_diagnostics", dpl.diagnostics),
        ("core_scf", scf_energy.__dict__),
        ("difference", report["differences"]),
    ):
        for name, value in values.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                observable_rows.append(
                    {"source": source_name, "observable": name, "value": float(value)}
                )
    with (out / "all_observables.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source", "observable", "value"))
        writer.writeheader()
        writer.writerows(observable_rows)
    with (out / "scf_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    with (out / "occupied_levels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=level_rows[0].keys()); writer.writeheader(); writer.writerows(level_rows)
    np.savez(out / "core1204_scf_profiles.npz", r=r, stack=scf_stack, **scf_fields, **scf_densities)
    write_vs_comparison(
        out, r, dpl_stack, scf_stack, backend.functional.params.mass_n,
        backend.functional.params.mass_p, f"{case.model} Z={case.z} N={case.n}: DPL versus SCF",
    )

    lines = [
        "# DPL 与 Core-1204 SCF 详细对比", "",
        f"参数集：{case.model}；核素：Z={case.z}, N={case.n}, A={case.a}。Core-1204 重新运行 {len(history)} 轮，最终势残差 {history[-1]['residual']:.3e}。", "",
        "## 总体物理量", "",
        "| 物理量 | DPL | Core-1204 SCF | DPL-SCF |", "|---|---:|---:|---:|",
        f"| E/A no CoM (MeV) | {dpl.energy_per_a_no_com:.8f} | {scf_energy.e_per_A_no_com:.8f} | {dpl.energy_per_a_no_com-scf_energy.e_per_A_no_com:+.8f} |",
        f"| 物质半径 no CoM (fm) | {dpl.diagnostics['rms_matter_no_com']:.8f} | {scf_energy.rms_matter_no_com:.8f} | {dpl.diagnostics['rms_matter_no_com']-scf_energy.rms_matter_no_com:+.8f} |",
        f"| 电荷半径 no CoM (fm) | {dpl.diagnostics['charge_radius_no_com']:.8f} | {scf_energy.charge_radius_no_com:.8f} | {dpl.diagnostics['charge_radius_no_com']-scf_energy.charge_radius_no_com:+.8f} |", "",
        "## 能量分解", "",
        "| 能量项 | DPL (MeV) | Core-1204 SCF (MeV) | DPL-SCF (MeV) |", "|---|---:|---:|---:|",
        f"| 动能 | {dpl_energy_terms['kinetic']:.8f} | {scf_energy.e_kin:.8f} | {dpl_energy_terms['kinetic']-scf_energy.e_kin:+.8f} |",
        f"| 直接相互作用能 | {dpl_energy_terms['direct_total']:.8f} | {scf_energy.e_dir:.8f} | {dpl_energy_terms['direct_total']-scf_energy.e_dir:+.8f} |",
        f"| 总能量 no CoM | {dpl_energy_terms['total_no_com']:.8f} | {scf_energy.e_total_no_com:.8f} | {dpl_energy_terms['total_no_com']-scf_energy.e_total_no_com:+.8f} |", "",
        "DPL 直接能细分："
        f"sigma={dpl_energy_terms['sigma']:.6f}, omega={dpl_energy_terms['omega']:.6f}, "
        f"rho={dpl_energy_terms['rho']:.6f}, Coulomb={dpl_energy_terms['coulomb']:.6f} MeV。", "",
        "> Core-1204 固定 DPL 势模式仅用于计算单粒子谱。该模式没有按 DPL 密度同步重建全部能量场，"
        "因此它打印的固定势总能量不是 DPL 变分总能量。", "",
        "## 占据单粒子能级", "", "| 核子 | 能级 | DPL (MeV) | SCF (MeV) | 差值 (MeV) |", "|---|---|---:|---:|---:|",
    ]
    lines.extend(f"| {x['species']} | {x['name']} | {x['dpl_mev']:.6f} | {x['scf_mev']:.6f} | {x['delta_mev']:+.6f} |" for x in level_rows)
    lines += ["", f"占据态 MAE = {np.mean(np.abs(level_delta)):.6f} MeV，RMSE = {np.sqrt(np.mean(level_delta*level_delta)):.6f} MeV。", ""]
    for title, values, unit in (("Dirac 势", potential_metrics, "fm^-1"), ("介子与库仑场", field_metrics, "fm^-1"), ("密度", density_metrics, "fm^-3")):
        lines += [f"## {title}", "", "误差统计使用 Core-1204 的物理网格 r>=0.1 fm，排除 r=0.001 fm 原点占位值。", "", f"| 通道 | MAE ({unit}) | RMSE ({unit}) | 最大差 ({unit}) | 位置 (fm) |", "|---|---:|---:|---:|---:|"]
        lines.extend(f"| {name} | {value['mae']:.8g} | {value['rmse']:.8g} | {value['max_abs']:.8g} | {value['max_abs_r_fm']:.3f} |" for name, value in values.items())
        lines.append("")
    (out / "DETAILED_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
