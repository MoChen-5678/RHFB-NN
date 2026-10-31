# PRL / 核物理书 / Core-1204 物理约束逐项审计

## 1. 审计边界与来源优先级

本审计只使用以下本地资料，不运行 SCF，也不读取 SCF 输出：

1. `PhysRevLett.133.076401(1).pdf`：决定 `E[H]` 和广义 Hamiltonian 梯度。
2. `张颖-Wang_2025_Chinese_Phys._C_49_014106.pdf`：决定非中心差分 Dirac 微分块。
3. `核物理(1).pdf` 与 `../Core-1204/*.f90`：决定 PKDD-RMF 拉格朗日量、密度、场方程、重排项和势场通道。

来源冲突时，物理连续方程以核物理书为准；Core 用于核对本项目采用的符号、单位和 PKDD 参数；离散 Dirac 算符以张颖论文为准；训练梯度以 PRL 为准。

当前范围是球对称、静态、时间反演、无海、无配对、无质心修正的 PKDD Hartree RMF。Core 中的 Fock、rho-tensor、pion 和配对代码不属于当前 RMF 泛函。

## 2. 梯度停滞结论

旧 5000 轮结果中，Hamiltonian 空间的能量梯度与重构梯度余弦约为 `-0.011`，并没有发生显著反向抵消。广义场梯度 RMS 仍约 `10.33`，参数梯度却只有约 `2.5e-4`。原因是物理梯度被错误表示层投影掉：

- `V3` 被乘以全局 `(N-Z)/A`，令 `16O` 的该通道 Jacobian 严格为零。
- 固定 Woods-Saxon `nuclear_shape` 不是 Euler-Lagrange 约束，使 32 阶直接基底数值秩下降。
- PRL 梯度在物理分量上除以人为尺度平方，改变了 Hamiltonian 的度量和各通道有效 `lambda`。
- 使用 `E/A` 对 `fm^-1` Hamiltonian 求梯度，量纲不同于 PRL 的总能量泛函。
- 动能曾使用欧氏归一的原始 `eigh` 向量，而密度使用径向积分归一后的旋量，导致作用量内部归一不一致。

这些问题已从主训练路径移除。程序继续记录两项梯度在 Hamiltonian 空间和参数空间的夹角，防止以后把代理损失的数值过零误判为梯度抵消。

## 3. 逐项公式映射

| 物理项 | 核物理书 / Core 参考 | Python 实现 | 审计结论 |
|---|---|---|---|
| PKDD 参数和中质子质量 | `Define.f90:112-118` | `PKDDParameters` | 数值一致；严格区分 `Mn`、`Mp` |
| 标量/矢量密度 | `Density.f90:48-70` | `densities_from_orbitals` | `G^2-F^2`、`G^2+F^2`、`4 pi r^2` 和原点外推一致 |
| 密度依赖耦合 | `Density.f90:118-139` | `couplings` | 函数及对 `zeta` 导数一致；已删除非物理 `clamp` |
| sigma 场 | `Meanfield.f90:54` | `reconstruct_fields` | 源为 `-g_sigma rho_s`，符号一致 |
| omega 场 | `Meanfield.f90:55` | `reconstruct_fields` | 源为 `+g_omega rho_b`，符号一致 |
| rho 场 | `Meanfield.f90:56` | `reconstruct_fields` | 源为 `+g_rho(rho_n-rho_p)`；不允许用全局不对称度锁死 |
| Coulomb 场 | `Meanfield.f90:57` | `reconstruct_fields` | 仅质子源，包含有限盒外 `Z alpha/r` Robin 项 |
| 重排自能 | `PotelHF.f90:53-65` | `rearrangement_potential` | 三个密度依赖通道进入共同矢量自能，不作为独立能量重复相加 |
| 势场组装 | `PotelHF.f90:58-76` | `compose_hamiltonian`, `potentials` | `VPS/VMS`、同位旋反号、质子 Coulomb 和 `-2M_tau` 一致 |
| 动能 | `BASE.f90:514-535`; `Expect.f90:262-292` | `solve`, `exact_kinetic_energy` | 使用普通 Hermitian ADF 本征向量对自由 Dirac 块的 Rayleigh 商；物理 `G,F` 另作径向积分归一，不使用 TF 近似 |
| 总能量 | `Expect.f90:266-292` | `off_shell_rmf_action` | 训练使用无质心完整作用量；重排项由变分产生而不重复计能 |
| 轨道归一 | 核物理书变分约束；`DiracB.f90:106-110` | `solve`, `off_shell_rmf_action` | `int(G^2+F^2)dr=1`，拉格朗日项保留 |
| 无海约束 | 正负能支分离 | `_physical_candidates` | 仅用 `epsilon>-M`；已删除经验 `upper_norm>0.5` |
| 轨道量子数 | 球对称 Dirac 节点结构 | `solve` | 节点数现在是硬门，不再只记录 |
| ADF 微分块 | 张颖论文式 (11)-(15) | `_block_system` | 严格使用 `B1=-D_backward+kappa/r`、`B2=D_forward+kappa/r`；径向奇函数 `G` 用 forward，径向偶函数 `G` 用 backward |
| PRL 能量变量 | PRL `E[H]` | `evaluate_tensor` | 已改为总无质心能量，不再用 `E/A` 生成训练梯度 |
| PRL 广义梯度 | PRL `grad_H E + lambda(H-H_tilde)` | `generalized_prl_gradient` | 在 MeV Hamiltonian 通道逐元素实现，无通道尺度矩阵 |
| 无监督边界 | PRL 变分流程 | `TorchRMFBackend.__init__` | Python 直接构造 Core 定义的网格；训练不加载 Fortran 或 SCF |

## 4. 离散作用量与残差

四个场由离散作用量 Hessian 一次求解，因此真正的场约束是输出中的 `weak_el_sigma/omega/rho/coul_rms`。`collocation_el_*` 使用另一套强形式差分，只用于观察截断误差，不进入损失、checkpoint 或物理门。

场消元后必须同时满足：

```text
Gamma_RMF = T + 1/2 int [g_sigma sigma rho_s
                        +g_omega omega rho_b
                        +g_rho rho rho_3
                        +coul rho_p] d^3r
```

`action_reduction_error_mev` 用于验证这一恒等式。自动微分还必须通过总能量随机方向中心差分检查。

## 5. 传播公式复查（2026-07-22）

沿 `H -> eigensystem -> G/F -> density -> coupling/field -> action -> grad_H` 逐项复查。论文要求

```text
G odd  : D_G = D_forward,  D_F = D_backward
G even : D_G = D_backward, D_F = D_forward
```

这里的 odd/even 是径向延拓的奇偶性，不是轨道空间宇称。因为 `G(r)~r^(l+1)`，`l` 为偶数时
径向 `G` 是奇函数，必须使用 forward；`l` 为奇数时径向 `G` 是偶函数，必须使用 backward。
代码现已把变量和测试改成 `radial_g_is_odd`，避免再次混淆。

当前实现直接令 `D_backward=-D_forward^T`，不把 Boole/Simpson 非均匀权重乘进本征矩阵。
把积分权重放进伴随关系虽然也可制造 Hermitian 矩阵，但会改变论文式 (11)-(14) 的 ADF 系数，
因此已经删除。积分权重只用于物理波函数归一化、密度、场作用量和可观测量。

曾尝试把 Boole 权重放入 ADF 伴随关系，并把径向奇偶误读为轨道 `l` 的奇偶；两种改动均被
固定势盲测否决，没有保留在程序中。最终实现的六个 O16 占据态 ADF/shooting MAE 为
`0.0002426 MeV`，最大误差 `0.0006117 MeV`，最小波函数重叠 `0.9999999`。

26 项公式测试全部通过，包括 ADF 矩条件、明确的径向奇偶方向、Hermiticity、物理归一化、粒子数、
PKDD 耦合导数、重排势、场作用量变分、通道组装和总能量自动微分。传播链一致性通过并不等于
长程优化已经收敛；能量梯度和重构残差仍必须分别通过训练物理门。

## 6. 网络表示和训练流程

```text
(Z,N, PKDD) -> Python radial mesh
             -> SiLU network/direct variation
             -> exact [S,V0,V3,VC] channel assembly
             -> H_theta in fm^-1
             -> 7-point ADF Dirac eigensystem
             -> occupied G,F and densities
             -> differentiable RMF field solve
             -> H_tilde and total no-CoM action
             -> convert H to MeV
             -> grad_H E_total + lambda(H-H_tilde)
             -> surrogate backpropagation to network parameters
```

Woods-Saxon 只提供初始 Hamiltonian。网络修正仅硬编码原点规则性、强作用通道盒边消失、Coulomb Robin 边界和 RMF 通道恒等式，不限制势场的内部形状或正负幅度。

`gradient_alignment.csv` 分别记录两项梯度的 H 空间和参数空间夹角。checkpoint 需要能量梯度和重构残差分别下降，不能只检查相加后的梯度，也不能检查代理损失的正负。

## 7. 未纳入当前版本的物理

- RHF 非局域 `X/Y` 核、pion、rho-tensor 和 Fock 重排项。
- BCS/HFB 配对及奇核阻塞。
- 质心修正、形变、时间反演破缺和有限温度。

这些项必须通过预留 RHF/扩展接口加入，不能以 RMF 罚函数近似代替。
