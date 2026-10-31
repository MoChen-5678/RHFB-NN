# PRL/Core 主路径逐行物理与数学审计

## 审计边界

本表只覆盖 `dpl_rhf.cli.prl_train` 实际调用的严格路径。训练不运行或读取 SCF；Core-1204 只用于
核对 PKDD 参数、符号、单位和通道定义。PRL 决定 `E[H]` 与广义梯度，张颖论文决定 ADF，核物理书
决定连续 RMF 作用量。旧 PINN、TF 动能、经验半径/饱和密度损失和 Fortran SCF 混合不属于主路径。

## 行级映射

| Python 位置 | 论文/Core 依据 | 必须满足的恒等式 | 状态 |
|---|---|---|---|
| `functionals/pkdd_action.py:10-31` | `Define.f90:113-118` | PKDD 参数、`Mn != Mp`、`hbar*c` 单位一致 | 通过 |
| `pkdd_action.py:102-162` | 离散 RMF 作用量 | ADF、轨道双线性和场源使用同一均匀坐标度量 | 已修复 |
| `pkdd_action.py:179-220` | `Density.f90:118-139` | `g_i(rho_b)` 及 `dg_i/dzeta` 无 clamp | 通过 |
| `pkdd_action.py:233-270` | `Meanfield.f90:54-57` | sigma 源负号；omega/rho/Coulomb 源正号 | 通过 |
| `pkdd_action.py:299-322` | 核物理书 (3.6)-(3.7), `Density.f90` | `rho_s=G^2-F^2`, `rho_v=G^2+F^2` | 通过 |
| `pkdd_action.py:324-344` | 核物理书 (3.47), `Expect.f90:274-292` | 动能来自占据 ADF 旋量，不使用 TF | 已修复 |
| `pkdd_action.py:345-402` | 静态 RMF 拉格朗日量 | 标量为极小方向，时间分量矢量场为极大方向 | 通过 |
| `pkdd_action.py:431-507` | `PotelHF.f90:63-75`, `Meanfield.f90:228-230` | rearrangement 只进入共同矢量自能；`VPS/VMS` 符号正确 | 通过 |
| `models/hamiltonian_net.py:14-54` | Core 势场通道 | `[S,V0,V3,VC] -> [VPSn,VMSn,VPSp,VMSp]` 精确线性映射 | 通过 |
| `hamiltonian_net.py:57-133` | 球对称边界 | 中心偶函数；massive 只施加 Dirichlet；Coulomb Robin | 已修复 |
| `hamiltonian_net.py:134-194` | PRL Hamiltonian 网络 | SiLU-only；Woods-Saxon 仅初值；四通道可训练 | 已修复 |
| `backends/torch_rmf.py:134-160` | 张颖式 (11)-(14) | 7 点 forward ADF，多项式矩到 6 阶 | 通过 |
| `torch_rmf.py:163-233` | 张颖式 (15) | `B1=-Db+kappa/r`, `B2=Df+kappa/r`, `B1=B2^T` | 通过 |
| `torch_rmf.py:244-361` | no-sea 球对称 RMF | 正能支、节点、占据、归一化和粒子数严格 | 通过 |
| `torch_rmf.py:393-508` | PRL `E[H]` | 一次谱传播得到密度、场、总能量和 `dE/dH` | 通过 |
| `training/surrogate_gradient.py:6-31` | PRL 广义梯度 | `dE_total/dH_MeV + lambda(H-H_tilde)`，无通道度量 | 通过 |
| `training/prl_hamiltonian_trainer.py:178-410` | PRL 无监督优化 | SCF 不进入损失；两项梯度分别监控 | 通过 |

## 本轮发现并修复的问题

1. **离散度量混用**：普通 Hermitian ADF 使用均匀坐标内积，但作用量曾使用 Boole 权重。这样
   `<psi|V|psi>` 与 `int rho V` 不相等，破坏 `E[H]` 的离散变分链。现统一为均匀 `h` 度量，
   并输出 `discrete_metric_check.json`。
2. **massive 场盒边过约束**：平方 envelope 同时强制场值和导数为零，而场方程只要求有限盒
   Dirichlet 值。现改为线性零点，恢复边界斜率变分自由度。
3. **人为通道尺度**：`S/V0/V3/VC` 曾乘固定经验尺度。现从物理表示删除，PRL 梯度只在真实
   Hamiltonian 单位中传播。
4. **非 SiLU 正式入口**：删除 SwiGLU 网络路径，CLI 只接受 SiLU。
5. **近似损失混入严格类**：删除 TF 动能、经验能量窗口、半径、饱和密度和平滑罚函数接口。
6. **主路径依赖 legacy**：严格 PKDD 作用量迁移到 `functionals/pkdd_action.py`；旧路径仅兼容导入，
   legacy 训练器不属于受支持求解器。

## 自动验收

- `formula_invariants.json`：Hermiticity、谱残差、节点、归一化、作用量退化和 autograd。
- `discrete_metric_check.json`：逐轨道局域势矩阵元与密度积分恒等式。
- `channel_gradient_projection.csv`：四个物理通道在 H 空间及参数空间的两项 PRL 梯度。
- `gradient_check.json`：总无质心能量的随机方向中心有限差分。

这些内部恒等式是物理正确性的门；SCF 数据只能在训练完成后作为外部结果对比。
