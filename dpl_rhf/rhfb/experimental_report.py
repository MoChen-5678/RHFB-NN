"""Readable data and scientific plots from an actual uncertified network run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .basis import AngularRadialBasis
from .currents import local_bilinears, invariant_baryon_density, _gamma_matrices
from .functional import _couplings
from .gaussian_pairing import GaussianPairingAction
from .operators import DensityFactors
from .spatial import GreenOperator
from dpl_rhf.rhf.parameters import pka1_parameters


def export_fields(out, summary):
    """Export implemented Hartree fields and active D1S gap, not full Fock h.

    Fock h is nonlocal and remains available as an operator through saved Q
    and rho.  These local arrays must not be labelled the complete normal h.
    """
    torch.set_num_threads(2)
    cfg = summary["config"]
    basis = AngularRadialBasis(radius_fm=cfg["radius_fm"], spacing_fm=cfg["spacing_fm"],
                              kappa_max=cfg["kappa_max"], n_theta=7, n_phi=14,
                              derivative_order=5)
    factors = np.load(out / "density_factors.npz")
    states = {s: DensityFactors(*[torch.as_tensor(factors[f"{s}_{name}"])
              for name in ("Q", "rho_active", "chi_active")]) for s in ("n", "p")}
    params = pka1_parameters()
    green = GreenOperator(basis, lmax=cfg["lmax"])
    fields = {}
    with torch.no_grad():
        samples = {s: basis._sample_range(v.q, 0, basis.point_count) for s, v in states.items()}
        current = {s: local_bilinears(samples[s], states[s].rho_active) for s in states}
        total_j = current["n"]["baryon_current"] + current["p"]["baryon_current"]
        rho_b = invariant_baryon_density(total_j)
        vertices = _couplings(rho_b, params)
        scalar = current["n"]["scalar"] + current["p"]["scalar"]
        iso_j = current["n"]["baryon_current"] - current["p"]["baryon_current"]
        fields["rho_invariant_fm3"] = rho_b.numpy()
        for key, value in vertices.items():
            fields[f"coupling_{key}"] = value.numpy()
        fields["sigma_hartree_scalar_potential_mev"] = (
            -params.hbar_c * vertices["gsig"] * green.apply(vertices["gsig"]*scalar,
                                                           params.m_sigma)).numpy()
        fields["omega_hartree_contravariant_potential_mev"] = (
            params.hbar_c * vertices["gome"][:, None] * green.apply(
                vertices["gome"][:, None]*total_j, params.m_omega)).numpy()
        fields["rho_vector_hartree_isovector_potential_mev"] = (
            params.hbar_c * vertices["grho"][:, None] * green.apply(
                vertices["grho"][:, None]*iso_j, params.m_rho)).numpy()
        e = (4*np.pi/params.alpha_inv)**0.5
        fields["coulomb_hartree_proton_potential_mev"] = (
            params.hbar_c * e**2 * green.apply(current["p"]["baryon_current"], 0.)).numpy()
        for s, state in states.items():
            for key in ("scalar", "baryon_current", "axial_current", "tensor"):
                if key in current[s]:
                    fields[f"{s}_{key}"] = current[s][key].numpy()
            pair = GaussianPairingAction(basis, state.q, state.chi_active,
                                        species=s, lmax=cfg["lmax"], checkpoint_blocks=False)
            ep, gap = pair.energy_and_active_gap()
            fields[f"{s}_delta_active_mev"] = gap.numpy()
            fields[f"{s}_pair_energy_mev"] = np.asarray(float(ep))
        gamma, gamma5 = _gamma_matrices(device="cpu", dtype=torch.complex128)
        # Tensor and pion derivative fields are meson amplitudes: their Dirac
        # Hamiltonian vertices include subsequent spatial differentiation.
        for mu in range(4):
            source_current = torch.zeros((basis.point_count, 3), dtype=torch.float64)
            for s, sign in (("n", 1), ("p", -1)):
                for axis in range(3):
                    tensor = .5j*(gamma[axis+1]@gamma[mu]-gamma[mu]@gamma[axis+1])
                    matrix = gamma[0]@tensor
                    value = torch.einsum("pia,ij,pjb,ba->p", samples[s].conj(), matrix,
                                         samples[s], states[s].rho_active).real
                    source_current[:, axis] += sign*value
            source = green.gradient_adjoint(vertices["grtn"][:, None]*source_current)
            fields[f"rho_tensor_meson_amplitude_mu{mu}_fm_inverse"] = green.apply(
                source, params.m_rho).numpy()
        pion_current = torch.zeros((basis.point_count, 3), dtype=torch.float64)
        for s, sign in (("n", 1), ("p", -1)):
            for axis in range(3):
                matrix = gamma[0]@gamma5@gamma[axis+1]
                pion_current[:, axis] += sign*torch.einsum(
                    "pia,ij,pjb,ba->p", samples[s].conj(), matrix,
                    samples[s], states[s].rho_active).real
        pion_source = green.gradient_adjoint(
            (vertices["fpio"]*params.hbar_c/params.m_pi)[:, None]*pion_current)
        fields["pion_pv_meson_amplitude_fm_inverse"] = green.apply(pion_source, params.m_pi).numpy()
    reference_pair_energy = summary["final"]["energy_terms_mev"]["d1s_pairing"]
    reconstructed_pair_energy = sum(float(fields[f"{s}_pair_energy_mev"]) for s in states)
    pair_error = abs(reference_pair_energy - reconstructed_pair_energy)
    if pair_error > 1e-10 * max(1., abs(reference_pair_energy)):
        raise RuntimeError("exported D1S gap disagrees with the saved training energy")
    fields["metadata_json"] = np.asarray(json.dumps({
        "status": "experimental_uncertified", "physical_result_accepted": False,
        "scope": "local Hartree fields, meson amplitudes, currents, couplings and active D1S gap",
        "nonlocal_fock_normal_field": "operator not exported as a local array",
        "rearrangement_field": "implicit energy autograd derivative, not exported separately",
        "reevaluated_from_saved_density_factors": True,
        "pair_energy_reconstruction_absolute_error_mev": pair_error,
        "export_source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in Path(__file__).parent.glob("*.py")},
    }))
    np.savez_compressed(out / "experimental_fields.npz", **fields)


def report(output):
    out = Path(output).resolve()
    summary = json.loads((out / "summary.json").read_text())
    rows = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    if not rows:
        raise ValueError("no actual evaluated network states")
    scalar_rows = []
    for row in rows:
        values = {"step": row["step"], "experimental_energy_mev": row["experimental_total_energy_mev"],
                  "parameter_gradient_l2": row["parameter_gradient_l2"],
                  "neural_layer_gradient_l2": row["neural_layer_gradient_l2"]}
        for s in ("n", "p"):
            values.update({f"number_{s}": row["particle_numbers"][s],
                           f"number_error_{s}": row["particle_number_errors"][s],
                           f"chi_norm_{s}": row["chi_frobenius"][s],
                           f"free_sea_fraction_{s}": row["sea_diagnostic"][s]["free_dirac_sea_particle_fraction"]})
        values.update(row["energy_terms_mev"])
        scalar_rows.append(values)
    with (out / "training_data.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)
    step = [r["step"] for r in rows]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].plot(step, [r["experimental_total_energy_mev"] for r in rows])
    axes[0, 0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0, 0].set_ylabel("Experimental EDF energy (MeV)")
    for s, name in (("n", "neutron"), ("p", "proton")):
        axes[0, 1].plot(step, [r["particle_number_errors"][s] for r in rows], label=name)
        axes[1, 0].plot(step, [r["chi_frobenius"][s] for r in rows], label=name)
        axes[1, 1].plot(step, [r["sea_diagnostic"][s]["free_dirac_sea_particle_fraction"]
                             for r in rows], label=name)
    axes[0, 1].set_ylabel("Particle-number error")
    axes[1, 0].set_ylabel("Anomalous-density Frobenius norm")
    axes[1, 1].set_ylabel("Free-Dirac sea fraction (diagnostic)")
    for axis in axes.ravel():
        axis.set_xlabel("Completed optimizer updates")
        axis.grid(alpha=.3)
    for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
        axis.legend()
    figure.suptitle("O-17 neural-network experiment — UNCERTIFIED")
    figure.savefig(out / "training_curves.png", dpi=170)
    plt.close(figure)
    arrays = np.load(out / "observables.npz")
    position, weight = arrays["positions_fm"], arrays["weights_fm3"]
    radius = np.linalg.norm(position, axis=1)
    shells = np.unique(np.round(radius, decimals=10))
    radial_rows = []
    for r in shells:
        mask = np.abs(radius-r) < 1e-9
        radial_rows.append([r] + [float(np.dot(weight[mask], arrays[key][mask]) / weight[mask].sum())
                                for key in ("rho_n_fm3", "rho_p_fm3", "rho_matter_fm3")])
    np.savetxt(out / "radial_density.csv", radial_rows, delimiter=",",
               header="r_fm,angular_average_rho_n_fm3,angular_average_rho_p_fm3,angular_average_rho_matter_fm3",
               comments="")
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    radial = np.asarray(radial_rows)
    for col, label in ((1, "neutron"), (2, "proton"), (3, "matter")):
        axis.plot(radial[:, 0], radial[:, col], marker="o", label=label)
    axis.set(xlabel="Radius (fm)", ylabel="Angular mean density (fm$^{-3}$)",
             title="O-17 neural-network density — UNCERTIFIED")
    axis.legend(); axis.grid(alpha=.3)
    figure.savefig(out / "radial_density.png", dpi=170)
    plt.close(figure)
    export_fields(out, summary)
    network_payload = torch.load(out / "experimental_network.pt", map_location="cpu",
                                 weights_only=False)
    network_evidence = {s: {
        "output_head_weight_norm": float(torch.linalg.vector_norm(
            weights["output_head.weight"])),
        "initial_output_head_weight_norm": 0.0,
        "evidence": "output head is initialized to zero; nonzero saved weights show neural optimizer updates",
    } for s, weights in network_payload["network_state"].items()}
    (out / "neural_parameter_evidence.json").write_text(json.dumps(network_evidence, indent=2)+"\n")
    initial, final = rows[0], rows[-1]
    moments = summary["observables"]
    lines = ["# O17 神经网络实验数据（未验收）", "",
             "这是 Python/PyTorch 神经网络的实际训练输出，未使用 Fortran。",
             "**尚未得到可信核解：完整物理闭合、相互作用场的无海约束及空间收敛未验收。**", "",
             f"优化更新：{summary['optimizer_updates_completed']}；停止原因：{summary['stop_reason']}。",
             f"空间：盒半径 {summary['config']['radius_fm']} fm，径向步长 {summary['config']['spacing_fm']} fm，|κ|≤3，每种粒子 14 个活动轨道。径向使用张颖论文的非对称有限差分。", "",
             "|量|初始|最后一次评估|", "|---|---:|---:|"]
    for label, a, b in [
        ("实验泛函能量 / MeV", initial["experimental_total_energy_mev"], final["experimental_total_energy_mev"]),
        ("中子数", initial["particle_numbers"]["n"], final["particle_numbers"]["n"]),
        ("质子数", initial["particle_numbers"]["p"], final["particle_numbers"]["p"]),
        ("中子反常密度范数", initial["chi_frobenius"]["n"], final["chi_frobenius"]["n"]),
        ("质子反常密度范数", initial["chi_frobenius"]["p"], final["chi_frobenius"]["p"]),
        ("网络层梯度范数", initial["neural_layer_gradient_l2"], final["neural_layer_gradient_l2"]),
    ]:
        lines.append(f"|{label}|{a:.12g}|{b:.12g}|")
    lines += ["", "|能量分项 / MeV|初始|最后一次评估|", "|---|---:|---:|"]
    for key, value in final["energy_terms_mev"].items():
        lines.append(f"|{key}|{initial['energy_terms_mev'][key]:.12g}|{value:.12g}|")
    lines += ["", "density_rearrangement=0 是能量记账占位：重排场来自密度依赖能量的导数，它不是独立的零物理场。", "",
              f"最后点中子、质子、物质 rms 半径：{moments['rms_radii_fm']} fm。",
              f"四极矩：{moments['quadrupole_moments_fm2']}。",
              f"训练对角化计数：{summary['solver_counts']}。", "",
              f"神经网络权重更新证据：{network_evidence}。", "",
              "自由 Dirac 海分量只用于诊断，不证明相互作用场的无海条件。参数梯度不等于全空间 RHFB 驻定残差；后者本轮未验收。", "",
              "数据文件：training_data.csv（逐步指标和全部能量分项）、observables.csv/.npz（三维密度及流）、radial_density.csv（径向平均）、density_factors.npz（Q、ρ、χ）、experimental_fields.npz（Hartree 场、介子幅、耦合及活动空间配对场）、experimental_network.pt（实验网络状态）、source_snapshot（本轮源代码）。",
              "非局域 Fock 场须使用算符表示，不能由局域 Hartree 场替代。实验检查点不兼容正式已验收续训。"]
    (out / "DATA_REPORT.md").write_text("\n".join(lines) + "\n")
    return out / "DATA_REPORT.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    args = parser.parse_args()
    print(report(args.output))
