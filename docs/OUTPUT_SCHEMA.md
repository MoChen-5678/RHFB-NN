# 输出文件定义

## 核心文件

| 文件 | 内容 | 是否必须 |
|---|---|---|
| `model.pt` | 网络 state dict、训练摘要、checkpoint 格式版本 | 是 |
| `summary.json` | 核素、结构、超参数、能量、gate 和诊断 | 是 |
| `history.csv` | 每 epoch 能量、驻定性、梯度、学习率和刷新标记 | 是 |
| `dpl_hamiltonian.npz` | `r`、四通道名和 `(4,Nr)` 局域势栈 | 是 |
| `density_profiles.npz` | 四个中子/质子标量与矢量密度 | 是 |
| `physical_components.npz` | S、V0、V3、VC 分量 | 是 |
| `density_matrix_factors.npz` | 各量子数分块的系数、因子和占据 | 是 |
| `final_wavefunctions.npz` | 最终谱验证态，不参与普通 epoch | 是 |

`dpl_hamiltonian.npz` 的通道顺序严格为 `vps_n, vms_n, vps_p, vms_p`，单位为 `fm^-1`。绘图时恢复 MeV 制的 S/V，不可直接把四通道当 MeV。

## 诊断文件

- `formula_invariants.json`：粒子数、正交、幂等和训练求解器计数。
- `training_solver_counts.json`：初始化、刷新、最终验证的对角化次数和耗时。
- `final_spectral_validation.json`：神经密度和最终谱密度的相对误差。
- `field_diagnostics.npz`：介子场、库仑场、重排势和物理势分量。
- `action_diagnostics.npz`：动能和各相互作用能分解。

## SCF 对比目录

- `detailed_comparison.json`：机器可读的完整差异。
- `DETAILED_COMPARISON.md`：组会可读表格。
- `core1204_scf_profiles.npz`：SCF 势、场和密度。
- `occupied_levels.csv`：占据单粒子能级逐态差异。
- `scf_history.csv`：SCF 收敛过程。
- `vs_dpl_scf.png/pdf/data.npz`：S/V 图和原始曲线。

## Checkpoint 兼容性

v1 格式名为 `dpl-rmf-density-matrix-v1`。评估器兼容当前旧式 density-matrix checkpoint，但缺失的修正尺度按 0.01 解释。模型类型、核素或张量形状不一致时严格拒绝加载。
