# 严格 PRL-RMF 主路径替换说明

## 正式入口

新版已覆盖项目的正式 DPL 求解路径：

```bash
python -m dpl_rhf.cli.prl_train --model PKDD --z 8 --n 8 \
  --mode direct --backend torch-rmf --epochs 500 \
  --lambda-reconstruct 0.001 --derivative-order 7 \
  --activation silu --device cuda --out outputs/prl_pkdd_O16
```

`dpl_rhf/functionals/pkdd_action.py` 是唯一受支持的 PKDD-RMF 作用量实现。`legacy/` 中的旧
fieldspace、固定点和轨道参数化训练器不再用于论文结果，只保留兼容导入及训练后 Core 对比。

## 被替换的旧行为

1. 轨道 ADF 使用普通 Hermitian 均匀坐标度量；删除 Boole 权重与矩阵坐标的混用。
2. `S,V0,V3,VC` 直接使用物理单位；删除人为通道输出尺度。
3. massive 场只施加有限盒 Dirichlet 条件；删除同时固定边界导数的平方 envelope。
4. 正式网络只使用 SiLU；删除 SwiGLU 入口。
5. 删除 Thomas-Fermi 动能、经验能量窗口、经验半径、饱和密度和平滑罚函数。
6. 总能量固定为完整无质心 PKDD-RMF 作用量；SCF 只允许训练后盲测。
7. 径向 Dirac 块按 `B1=-D_backward+kappa/r`、`B2=D_forward+kappa/r` 显式构造。

## O16/PKDD 数值验收

2026-07-23 在 CPU 上运行 500 轮直接变分，输出目录为
`outputs/line_audit_uniform_metric_O16_cpu_500`（输出不提交 Git，可由上述命令复现）：

| 指标 | 新版 DPL | Core 无质心参考 | 差值 |
|---|---:|---:|---:|
| `E/A` (MeV) | -7.315948809 | -7.3158555 | -0.0000933 |
| 物质半径 (fm) | 2.5663302 | 2.5663369 | -0.0000067 |

内部公式验收：Hermiticity 误差为 0，作用量 on-shell 约化误差 `7.93e-12 MeV`，最大谱
残差 `1.45e-13`，局域势双线性误差 `5.55e-17`，autograd 有限差分相对误差 `5.53e-6`；
`formula_invariants.json` 全部通过。

500 轮的联合优化 monitor 为 `6.45e-3`，仍高于默认网络放行门 `1e-3`。这表示公式链已通过，
但直接变分优化尚未达到长网络训练的预设残差门；程序因此继续拒绝把该结果当作已收敛网络门文件。

## 验收文件

- `formula_invariants.json`：离散公式总门。
- `discrete_metric_check.json`：轨道矩阵元与密度积分恒等式。
- `gradient_check.json`：作用量 autograd 与中心有限差分。
- `channel_gradient_projection.csv`：`S,V0,V3,VC` 的能量/重构梯度关系。
- `physics_residuals.csv`：能量、重构、谱和 Euler-Lagrange 残差。
