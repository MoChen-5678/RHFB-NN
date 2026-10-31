# DPL-RMF 使用说明

## 目标

DPL-RMF 的输入是参数集、质子数和中子数。新版主路线按 PRL/AI2DFT 的
`E[H_theta]` 模式构建：默认先直接变分独立物理势分量，验证物理路径后再启用神经网络。`torch-rmf`
后端在同一个 PyTorch 计算图中求解 fixed-H Dirac 方程、密度、RMF 场、无质心能量、
重构势和 `grad_E_H`，不使用外部 SCF 标签数据。

新版网络目标：

```text
parameterization(r, Z, N) -> {S,V0,V3,VC} -> H_theta
backend(H_theta) -> E[H], grad_E_H, H_tilde, densities, fields
grad_H = w_E * grad_E_H + lambda * (H_theta - H_tilde)
```

`torch-rmf` 后端使用 7 点非中心差分构造 Hermitian 径向 Dirac 矩阵，并用
`torch.autograd.grad(E/A, H_theta)` 得到真正可微 RMF 能量泛函梯度。旧 Fortran
fixed-H 后端只保留为诊断和最终 SCF 对比。

## PRL Hamiltonian 命令

```bash
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate torch_env
python -m dpl_rhf.cli.prl_train \
  --model PKDD --z 8 --n 8 \
  --mode direct --direct-order 32 \
  --backend torch-rmf \
  --epochs 500 \
  --lambda-reconstruct 0.001 \
  --energy-gradient-weight 1.0 \
  --derivative-order 7 \
  --device cuda --compare-scf \
  --out outputs/prl_pkdd_O16
```

正式入口只接受 `--activation silu`。激活函数对照属于已归档实验，不再进入严格求解器。
默认 `--checkpoint-policy physics` 恢复分开的内部物理残差最佳点；`--checkpoint-policy last`
只用于复现指定历史轮次，不能作为通过 SCF 后验挑选模型的默认方法。

直接变分通过后才运行长网络训练：

```bash
python -m dpl_rhf.cli.prl_train \
  --model PKDD --z 8 --n 8 \
  --backend torch-rmf --mode network \
  --direct-gate outputs/prl_pkdd_O16/direct_variation_summary.json \
  --epochs 1000 \
  --lambda-reconstruct 0.001 \
  --energy-gradient-weight 1.0 \
  --derivative-order 7 \
  --device cuda \
  --compare-scf \
  --out outputs/prl_pkdd_O16_compare
```

PRL 模式主要输出：

- `summary.json`：模式、核素、能量、半径和诊断。
- `history.csv`：`E/A`、`H-H_tilde` RMSE、后端梯度范数。
- `dpl_hamiltonian.npz`：网络输出的 `H_theta`。
- `reconstructed_hamiltonian.npz`：由密度重构的 `H_tilde`。
- `autograd_gradient.npz`：同一 PyTorch RMF 图中的 `d(E/A)/dH`。
- `gradient_check.json`：随机方向有限差分与 autograd 梯度误差。
- `formula_invariants.json`：公式不变量的统一通过/失败结论。
- `discrete_metric_check.json`：局域势矩阵元与密度积分的一致性。
- `channel_gradient_projection.csv`：`S,V0,V3,VC` 中能量与重构梯度的投影。
- `profile_compare.json`：开启 `--compare-scf` 后输出势场和密度的逐通道 MAE/RMSE/最大误差。
- `adf_shooting_validation.json`：ADF 与 Fortran shooting 在相同 SCF 势上的逐轨道盲测。
- `direct_variation_summary.json`：物理门结果；门失败时不得进入长网络训练。
- `density_diagnostics.npz`、`field_diagnostics.npz`：后端物理诊断。

O16/PKDD 已跑过的 2000 轮对照：

| 输出目录 | `--energy-gradient-weight` | DPL `E/A` no CoM | SCF `E/A` no CoM | 占据态 MAE |
|---|---:|---:|---:|---:|
| `outputs/prl_guarded_O16_2000_compare` | 1e-3 | -9.005372 | -7.315856 | 3.440116 |
| `outputs/prl_reconstruct_only_O16_2000_compare` | 0 | -7.882023 | -7.315856 | 3.166386 |

这些目录是旧 Fortran 梯度/重构模式的历史结果。当前默认 `torch-rmf` 已经接入真正可微 RMF
能量泛函梯度，推荐使用 `--energy-gradient-weight 1.0`，并用 `gradient_check.json` 验证梯度。

## Legacy 边界

`variational_rmf.py`、`fieldspace_rmf.py` 和旧 `dpl_rmf.py run` 已归档，不属于受支持的物理解算
入口。它们不应再用于生成论文结果。外部 Core 对比工具仍可通过严格训练器的 `--compare-scf`
调用，但 SCF 数据不会进入训练图或 checkpoint 选择。

## 参数组

PRL v1 当前只支持 PKDD/RMF。旧固定点程序支持多个 `IE=1` RMF 参数组：

| 名称 | 索引 |
|---|---:|
| DD-ME1 | 4 |
| DD-ME2 | 5 |
| PKDD | 6 |
| TW99 | 7 |
| DD-LZ1 | 8 |

RHF 参数组 `PKA1/PKO1/PKO2/PKO3` 会被拒绝。第二版再处理 Fock/非局域势。

## Core 外部诊断

检查 Core：

```bash
python dpl_rmf.py inspect-core
```

传统 SCF 基线：

```bash
python dpl_rmf.py scf-baseline --model DD-ME2 --z 8 --n 8 --max-iter 200 --out outputs/o16_scf
```

下面的字段只用于读取历史 Core 结果，不是严格 PRL 训练损失。

## Core 输出字段

`history.csv`：

- `residual`：`sqrt(mean(((P_theta - P_RMF) / scale)^2))`。
- `loss`：固定点残差加正则项。
- `e_total, e_per_A`：默认是不含质心修正的能量，用于和不做 CoM 修正的结果对比。
- `e_total_no_com, e_per_A_no_com`：不含 Fortran `E-cm` 的能量。
- `e_total_with_com, e_per_A_with_com`：包含 Fortran `E-cm` 的能量，对应 `Expect.f90` 打印的 `Energy per Particle`。
- `e_cm`：Fortran 计算的质心修正项。
- `rms_*_no_com, charge_radius_no_com`：未做质心修正的半径。
- `rms_*_with_com, charge_radius_with_com`：Fortran 质心修正后的半径。

`summary.json` 和 `observables.json`：

- `summary.json` 保存运行参数、收敛状态和最终观测量。
- `observables.json` 只保存最终观测量，便于直接对比 `no_com` 和 `with_com`。

`final_potentials.npz`：

- `vps_n, vms_n, vtt_n`
- `vps_p, vms_p, vtt_p`
- `sigma, omega, rho, coul`

`final_densities.npz`：

- `rho_s`
- `rho_b`
- `rho_b3`

`final_wavefunctions.npz`：

- `r`：径向网格。
- `G, F`：最终一次求解得到的所有中子/质子轨道径向波函数，形状为 `(n_orbits, n_grid)`。
- `species, it, index, name, kappa, l, n, energy, occupancy, degeneracy`：每个轨道的元数据。

`dpl_prediction.npz`（仅 `run` 模式）：

- `vps_n, vms_n, vps_p, vms_p`：最佳 loss 模型直接输出的神经网络势场。

## 注意

严格变分版没有实验能量目标、经验 `E/A=-8` 目标、Thomas-Fermi 动能、经验半径或
饱和密度损失。总能量默认不含质心修正。传统 SCF 只在训练结束后生成对照结果。
