# PRL/AI2DFT 核物理 Hamiltonian 模式

新版主路线采用 PRL `E[H_theta]` 思路：可训练参数只表示局域 Dirac Hamiltonian，训练器不把
SCF 数据作为标签。默认先用 Chebyshev 系数做无网络直接变分；验证通过后才把参数化替换为
SiLU 网络。`torch-rmf` 后端会在同一个 PyTorch 计算图里完成 fixed-H Dirac
广义 Hermitian 本征求解、密度、RMF 场重构、无质心能量和 `grad_E_H = d(E/A)/dH`。

完整 PKDD-RMF 作用量、有限盒子库仑边界项和约束变分定义见
[`COMPLETE_RMF_ACTION.md`](COMPLETE_RMF_ACTION.md)。默认能量梯度已改为该完整作用量的梯度。

## 程序边界

- 参数化层输出独立物理分量 `[S,V0,V3,VC]`，再严格组装四通道
  `H_theta = [vps_n, vms_n, vps_p, vms_p]`。
- `torch-rmf` 后端内部求解 Dirac 方程、构造密度、计算无质心能量、重构 `H_tilde`，并用
  autograd 给出完整 RMF 能量梯度。
- Fortran fixed-H backend 只作为可选诊断和训练后 SCF 对比，不再作为默认训练梯度来源。
- RHF/Fock 接口已预留，当前只实现 PKDD/RMF 局域 Hartree。
- Hamiltonian 网络所有隐藏激活统一使用 `SiLU`。输出经过硬物理表示层：中心偶函数、有限程
  强作用通道、库仑 `Z alpha/r` 尾、同位旋因子及中子/质子质量严格组装。
- 每个训练步只有一次可微 Dirac 谱分解和一次场方程线性求解，不进行 Python SCF 外迭代。
- Core-1204 只在训练后对照阶段运行，不参与初始化、损失或 checkpoint 选择。

## Dirac ADF 矩阵

径向 Dirac Hamiltonian 不通过弱形式或事后整体对称化构造。程序按论文在坐标点上直接建立：

```text
B1 = -D_backward + kappa/r
B2 =  D_forward  + kappa/r
H  = [[V+S, B1], [B2, V-S-2M]]
```

论文的明确规则是：径向大分量 `G(r)` 为奇函数时使用 forward ADF，为偶函数时使用 backward
ADF；小分量使用相反方向。注意 `G(r)~r^(l+1)`，这里不是轨道空间宇称。离散算符严格满足
`-D_backward=D_forward^T`，所以 `B1=B2^T`，Hermiticity 来自两个物理微分块本身。
普通 Hermitian ADF 的坐标度量为均匀 `h`；轨道归一、密度作用量和局域势双线性统一使用该
度量。Simpson/Boole 非均匀权重不进入坐标空间本征矩阵，也不与该离散作用量混用。

O16/PKDD 的同一 Core SCF 势验证中，六个占据态相对 Fortran shooting 的能量 MAE 为
`0.000243 MeV`，最大误差为 `0.000612 MeV`。

## PRL 广义梯度

训练严格在 MeV Hamiltonian 通道中使用：

```text
H_MeV = hbar_c * H_fm^-1
grad_H = w_E * d E_total_no_com/d H_MeV + lambda_MeV^-1 * (H_MeV - H_tilde_MeV)
L_surrogate = mean(H_MeV * stopgrad(grad_H))
```

`L_surrogate` 只用于把后端梯度传回网络参数，不是物理总能量。日志中的 `E/A` 来自后端固定势能量。

默认 `torch-rmf` 中 `grad_E_H` 由同一 PyTorch RMF 总能量泛函图自动微分得到，包含 fixed-H
Dirac 本征态对 Hamiltonian 的响应、密度、PKDD 密度依赖耦合、重排项、场重构和书中占据轨道动能。
因此默认 `w_E=1`。旧 Fortran 裸密度型梯度仍可通过 `--backend fortran-fixed` 做诊断，但不能视为完整
PRL 梯度。

## 运行

```bash
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate torch_env
python -m dpl_rhf.cli.prl_train \
  --model PKDD --z 8 --n 8 \
  --mode direct --direct-order 32 \
  --backend torch-rmf \
  --epochs 200 \
  --lambda-reconstruct 0.001 \
  --energy-gradient-weight 1.0 \
  --derivative-order 7 \
  --device cuda --compare-scf \
  --out outputs/prl_pkdd_O16
```

`--compare-scf` 在训练结束后运行外部盲测并决定物理门是否通过。SCF 不参与训练。长于 200 轮
的网络训练必须显式使用 `--mode network --direct-gate <direct_variation_summary.json>`。

## 输出

- `dpl_hamiltonian.npz`：网络输出的局域 Hamiltonian。
- `physical_components.npz`：独立物理分量 `S,V0,V3,VC`。
- `reconstructed_hamiltonian.npz`：由后端密度重构的 `H_tilde`。
- `density_diagnostics.npz`：后端密度。
- `field_diagnostics.npz`：后端场。
- `autograd_gradient.npz`：`d(E/A)/dH` 的可微 RMF 梯度。
- `gradient_check.json`：随机方向中心有限差分与 autograd 梯度对比。
- `physics_residuals.csv`：逐轮能量、重建、谱和场驻点残差。
- `spectral_diagnostics.json`：最终谱残差、度量归一和粒子数误差。
- `occupation_projector_history.csv`：全局无海占据、节点数和本征分支。
- `component_residuals.csv`：独立物理分量中的重构残差。
- `network_physical_constraints.json`：硬边界与通道恒等式诊断。
- `adf_shooting_validation.json`：同一 SCF 势下 ADF 与 Fortran shooting 的外部验证。
- `direct_variation_summary.json`：直接变分内部门和外部门；网络长训练的唯一门文件。
- `action_identity.json`：完整作用量、约化恒等式和场驻点残差。
- `profile_compare.json`：训练后才计算的 DPL/SCF 势、场、密度逐通道误差；不进入训练损失。
- `history.csv`：训练曲线。
- `summary.json`：训练配置和最终诊断。

## 过绑定/过拟合处理

旧 Fortran 裸梯度路径的 O16/PKDD 2000 轮对照结果：

| 模式 | `w_E` | DPL `E/A` no CoM | SCF `E/A` no CoM | 差值 | 占据态 MAE |
|---|---:|---:|---:|---:|---:|
| 不完整能量梯度保护版 | 1e-3 | -9.005372 | -7.315856 | -1.689516 | 3.440116 |
| 重构固定点默认版 | 0 | -7.882023 | -7.315856 | -0.566167 | 3.166386 |

严格裸梯度曾把 O16 推到约 `E/A=-31.996710`，这是非物理过绑定，不是普通机器学习训练集过拟合。
因此当前默认策略已改为：

- 不把 SCF 当训练标签，只在训练后对比。
- 用 `torch-rmf` 的 autograd `d(E/A)/dH` 作为 PRL 能量梯度。
- 用 `H_theta-H_tilde` 作为 Hamiltonian 重构残差。
- 用无质心修正能量和半径做诊断。
- 分开记录能量梯度和 Hamiltonian 重建残差，用二者归一值的最大值选择 checkpoint，避免
  广义梯度向量抵消造成伪收敛；不使用经验能量或半径窗口选模。
- 若联合残差超过历史最佳值两倍，训练恢复真实的步进前最佳状态、清空 Adam 动量并将学习率
  减半。该回退只依赖内部物理残差，用于防止优化器跨越占据谱分支。
- `lambda` 是 PRL 文中允许调节的 Hamiltonian 重构系数；当前 Hamiltonian 和梯度统一在 MeV
  坐标中，默认 `0.001 MeV^-1`，不是 SCF 拟合权重。
- 用 `gradient_check.json` 验证梯度不是 Fortran 裸密度近似。

## RHF 预留

`dpl_rhf/functionals/rhf_interfaces.py` 和 `dpl_rhf/backends/fortran_rhf_stub.py` 保留了未来 RHF 开发的
通道命名和后端边界。当前调用 RHF backend 会明确报错，避免把 RMF 结果误认为 RHF。
