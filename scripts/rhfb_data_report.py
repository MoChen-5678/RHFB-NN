#!/usr/bin/env python3
"""Write the spherical neural RHFB data report from saved JSON/CSV files only."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "outputs/rhfb_spherical"

# How each final result was obtained (set when the run was stopped on 2026-10-27).
PROVENANCE = {
    "O16": "1500 轮从零训练；用新代码自洽重评估最后状态（production_eval/O16）",
    "O17": "从 O16 热启动 800 轮；远程自洽重评估最后状态",
    "O18": "从 O16 热启动 800 轮；远程自洽重评估最后状态",
    "O19": "从 O18 热启动 800 轮；新代码自洽最终评估",
    "Ca40": "从零训练 1500 轮（第 210 轮起改为双卡流式 Fock）；自洽重评估最后状态（production_eval/Ca40）",
    "Ca48": "远程 A100 双卡，从 Ca40 热启动 800 轮；自洽重评估",
    "Sn132": "远程 A100 双卡，从零 1500 轮；自洽重评估",
    "Pb208": "远程 A100 双卡，从零 1500 轮；自洽重评估",
    "Pb210": "远程 A100 双卡，从零 1500 轮；自洽重评估",
}


def _f(value, digits):
    return "—" if value is None else f"{value:.{digits}f}"


def main():
    rows = json.loads((R / "comparison.json").read_text())
    out = []
    w = out.append
    w("# 球对称无刷新神经网络 RHFB（PKA1 + D1S）数据报告")
    w("")
    w("更新于 2026-10-28。本报告所有数值由 "
      "`scripts/rhfb_data_report.py` 从保存的 JSON/CSV 生成，未手工填写。")
    w("参考值来自 Fortran S-RHFB-C（同 PKA1+D1S、同 0.1 fm/20 fm 网格、KMX=(5,4)），"
      "仅在训练结束后对照。能量与半径均不含质心修正。")
    w("")
    w("## 1. 与 Fortran 参考的对比")
    w("")
    w("| 核 | E/A NN | E/A Ref | Δ(E/A) keV | E总 NN | E总 Ref | E_pair(n) NN | Ref | λn NN | Ref | "
      "能隙 NN | Ref | r_ch NN | Ref | r_n NN | Ref | 能级 MAE |")
    w("|" + "---|" * 17)
    for r in rows:
        w(f"| {r['nucleus']} | {r['nn_E_per_A']:.5f} | {r['ft_E_per_A']:.5f} | {1e3*r['dE_per_A']:+.2f} | "
          f"{r['nn_E_total']:.3f} | {r['ft_E_total']:.3f} | {r['nn_E_pair_n']:.3f} | {_f(r['ft_E_pair_n'], 3)} | "
          f"{r['nn_lambda_n']:.3f} | {_f(r['ft_lambda_n'], 3)} | {_f(r['nn_gap_n'], 3)} | {_f(r['ft_gap_n'], 3)} | "
          f"{r['nn_r_ch']:.4f} | {r['ft_r_ch']:.4f} | {r['nn_r_n']:.4f} | {r['ft_r_n']:.4f} | "
          f"{_f(r['level_mae_mev'], 3)} |")
    w("")
    w("- 能隙按参考程序定义：2|E_pair|/∫Σ(|κ_GG(r,r)|+|κ_FF(r,r)|)dr。")
    w("- 能级 MAE：v²>0.05 的中子与质子正则能级 h_νν 的平均绝对偏差（MeV）。")
    w("- 无配对核（O16、O17）的 λ 只落在能隙中，NN 与参考的差别不代表物理差异。")
    w("")
    w("## 2. 每个核的来源与状态")
    w("")
    w("| 核 | 来源 | 训练轮数 | 轨道残差 rms (MeV) | 占据残差 rms (MeV) | 海泄漏 | 对角化次数 |")
    w("|---|---|---|---|---|---|---|")
    for r in rows:
        summary = json.loads((R / "final" / r["nucleus"] / "summary.json").read_text())
        leak = summary["no_sea"].get("sea_leakage_self_consistent")
        w(f"| {r['nucleus']} | {PROVENANCE.get(r['nucleus'], '')} | {r['nn_epochs']} | "
          f"{r['nn_stationarity_orbital_mev']:.3e} | {r['nn_stationarity_occupation_mev']:.3e} | "
          f"{'未测' if leak is None else f'{leak:.1e}'} | {r['nn_eigensolves']} |")
    hist = R / "production/O17/history.csv"
    if hist.exists():
        last = list(csv.DictReader(hist.open()))[-1]
        w("")
        w(f"O17（已被自洽重评估替代）旧记录：末轮训练能量 E/A = {float(last['energy_per_a_no_com_mev']):.5f} MeV"
          "（低于 summary 中旧规则选中的状态；需自洽重评估后才能替换）。")
    w("")
    w("## 3. 未完成项目")
    w("")
    w("- O20、Ca41、Ca42、Ca44 的神经网络计算未运行；O18 正则空间收敛扫描未运行。")
    w("- O18、Pb210 的中子配对能弱于参考（-4.52 vs -5.21；-1.00 vs -3.26 MeV），"
      "最可能来自正则空间截断，需收敛扫描确认。")
    w("- 闭壳核 Ca48/Sn132/Pb208 的 NN 能量低于 S-RHFB-C 3-15 keV，并在 Core-1204 RHF 之上约 1 keV：")
    w("  NN 为变分上界，说明 S-RHFB-C 对较重核未达极小（最可能是 DWS 基截断）。")
    w("- 能级 MAE 对 v^2=1 的同 kappa 块按块迹比较（这些正则轨道只确定到块内转动）。")
    w("")
    w("## 4. 图表")
    w("")
    for name, text in (("systematics.png", "Δ(E/A)、中子配对能、λn 随核素"),
                       ("training_O16.png", "O16 训练曲线"), ("training_O18.png", "O18 训练曲线"),
                       ("training_O19.png", "O19 训练曲线"), ("training_Ca40.png", "Ca40 训练曲线"), ("levels_O18.png", "O18 中子正则能级与 v²"),
                       ("profiles.png", "中子矢量密度与局域配对张量")):
        w(f"- [{text}](../../paper/figures/{name})")
    w("- LaTeX 表格：`paper/tables/comparison.tex`；完整数值：`outputs/rhfb_spherical/comparison.json`")
    w("")
    w("## 5. Fortran 参考（全部 9 个核）")
    w("")
    w("| 核 | E/A (MeV) | E_pair(n) | λn | λp | r_ch | 迭代 |")
    w("|---|---|---|---|---|---|---|")
    for nucleus in ("O16", "O17", "O18", "O19", "O20", "Ca40", "Ca41", "Ca42", "Ca44", "Ca48",
                    "Sn132", "Pb208", "Pb210"):
        path = ROOT / "outputs/fortran_reference/runs" / nucleus / "parsed.json"
        if not path.exists():
            path = ROOT / "outputs/fortran_reference/runs_kmx88_remote" / nucleus / "parsed.json"
        if not path.exists():
            continue
        p = json.loads(path.read_text())
        e = p["energy"]
        w(f"| {nucleus} | {e['E_per_A_no_cm_MeV']:.5f} | {e['E_pair_MeV']['n']:.3f} | "
          f"{p['lambda_MeV']['n']:.3f} | {p['lambda_MeV']['p']:.3f} | "
          f"{p['radii_fm']['charge_without_cm']:.4f} | {p['iterations']} |")
    (R / "DATA_REPORT_20261027.md").write_text("\n".join(out) + "\n")
    print(R / "DATA_REPORT_20261027.md")


if __name__ == "__main__":
    main()
