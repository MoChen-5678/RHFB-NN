# 可微密度依赖 RMF 变分求解器：组会物理汇报稿

## 0. 一句话说明工作

本程序不是用 SCF 结果监督拟合势场，而是把球对称、密度依赖相对论平均场（DD-RMF）的
能量泛函、场方程和 Dirac 哈密顿量写入同一个 PyTorch 可微计算图，在满足核物理边界条件的
局域哈密顿量空间中直接寻找变分驻点。Core-1204 的 SCF 结果只在训练结束后用于外部检验。

当前正式计算是 **RMF Hartree**，不是 RHF；交换 Fock 项尚未进入作用量。Chebyshev 直接
变分只作为物理和梯度诊断基线；正式单核素 DPL 使用不含 Chebyshev 旁路的纯 SiLU 网络。

---

## 1. 从拉格朗日量到程序中的四个物理自由度

### 1.1 采用的静态 RMF 通道

球对称、偶偶核、时间反演不变时，程序保留：

- 标量等标量通道：`S = g_sigma sigma`；
- 时间分量等标量矢量通道：`V0 = g_omega omega + Sigma_R`；
- 时间分量等矢量矢量通道：`V3 = g_rho rho_3field`；
- 质子 Coulomb 通道：`VC = e A0`。

因此真正的变分坐标为

```text
X(r) = [S(r), V0(r), V3(r), VC(r)].
```

代码位置：`dpl_rhf/models/hamiltonian_net.py::PHYSICAL_COMPONENT_NAMES`。

### 1.2 四个 Dirac 对角通道

程序把四个独立物理分量严格映射为

```text
V+n = V0 + S + V3
V-n = V0 - S + V3 - 2 Mn
V+p = V0 + S - V3 + VC
V-p = V0 - S - V3 + VC - 2 Mp
```

内部势单位为 `fm^-1`，所以代码中的质量项实际为 `2 M/(hbar*c)`。中子约定
`tau_3=+1`，质子约定 `tau_3=-1`，Coulomb 只作用于质子。

代码位置：

- `hamiltonian_net.py::compose_hamiltonian`：物理分量到四通道；
- `hamiltonian_net.py::decompose_hamiltonian`：四通道到物理分量；
- `hamiltonian_net.py::physical_component_gradient`：梯度的精确链式法则。

这一步是硬映射，不是损失罚项。优化器不能产生违反上述同位旋和质量结构的四条独立势。

---

## 2. 参数集、网格和单位

### 2.1 参数集

`dpl_rhf/functionals/pkdd_action.py::RMF_PARAMETER_DATA` 保存五个 DD-RMF 参数集：

```text
DD-ME1, DD-ME2, PKDD, TW99, DD-LZ1
```

每个参数集包含 `Mn, Mp, m_sigma, m_omega, m_rho, rho_sat` 以及密度依赖耦合参数。数值来自
Core-1204 的参数定义。`rmf_parameters(model)` 是唯一参数入口；后端、初值分解、哈密顿量
组装和最终评估必须共享同一个参数对象。

### 2.2 径向网格

与 Core 保持一致：

```text
npt = 201
h   = 0.1 fm
r   = 0.001, 0.1, 0.2, ..., 20.0 fm
```

`r[0]=0.001 fm` 只是原点占位，物理积分从 `r=0.1 fm` 开始。轨道归一化、局域双线性型、
密度积分和作用量收缩统一使用同一 ADF 均匀度量：

```text
w[0] = 0, w[i>0] = h
d^3r = 4 pi r^2 w.
```

这不是高阶 Newton-Cotes 积分的精度选择，而是为了让离散矩阵期望值与密度形式严格代表
同一个离散泛函，避免破坏 `delta E/delta H`。

代码位置：`pkdd_action.py::PKDDRMFFunctional.__init__`。

---

## 3. 物理空间的硬边界条件

### 3.1 原点

网络和直接变分都使用 `r^2` 特征。因此球对称场在原点满足

```text
d phi/dr |_(r=0) = 0
```

这是参数化恒等式，不是近似边界罚函数。

### 3.2 有质量介子场的无穷远条件

源外 Klein-Gordon 方程给出

```text
phi(r>R) = C exp(-mu r)/r,   mu = m_phi/(hbar*c).
```

盒边的等价 Dirichlet-to-Neumann 条件为

```text
phi'(R) + (mu + 1/R) phi(R) = 0.
```

程序用三条质量相关尾基函数保留可变振幅，并让内部修正的值和一阶导数在 `R` 同时消失。
这保证 `S,V0,V3` 可解析延拓为 Yukawa 尾，而不是简单令盒边势为零。

更重要的是，泛函中还加入源外 `R` 到无穷远的解析作用量：

```text
Delta K_R = 4 pi R^2 (mu + 1/R).
```

只修改参数化而不修改作用量会造成边界条件与场方程不一致；当前两层已同时实现。

代码位置：

- `hamiltonian_net.py::PhysicalHamiltonianRepresentation`；
- `pkdd_action.py::PKDDRMFFunctional.__init__` 的 massive field kernels。

### 3.3 Coulomb 无穷远条件

Coulomb 是无质量通道，不能使用指数衰减：

```text
VC(r>R) = C/r,
VC'(R) + VC(R)/R = 0.
```

基础势使用有限电荷球并连接 `Z alpha/r`；可训练修正及其导数在盒边消失。Coulomb 泛函含
解析外部边界项 `R/alpha`。

### 3.4 轨道边界的当前处理

Dirac 束缚态在无穷远应按 `exp(-k r)` 衰减。当前 ADF 本征问题在 `R=20 fm` 使用零外部值，
即在足够大盒上截断已经很小的指数尾。它不是能量依赖的精确轨道 Robin 边界。组会中应把
“介子场的解析无穷远外延”和“轨道的有限盒截断”明确区分。

---

## 4. 七点非中心差分 Dirac 算符

### 4.1 为什么不用中心差分

中心差分离散一阶 Dirac 算符会引入费米子加倍/假态。程序按论文使用七点非中心差分 ADF，
并根据径向分量的奇偶性选择前向或后向算符。

代码位置：`dpl_rhf/backends/torch_rmf.py::noncentral_derivative_matrix`。

### 4.2 两个微分块必须独立构造

径向 Dirac 哈密顿量的两个非对角块为

```text
B1 = -D_backward + kappa/r
B2 =  D_forward  + kappa/r.
```

并强制

```text
D_backward = -D_forward^T,
B1 = B2^T.
```

因此离散哈密顿量严格厄米。不能先构造一个中心差分哈密顿量再做数值对称化，因为那不会恢复
上述物理微分链。

代码位置：`torch_rmf.py::DifferentiableDiracMatrix._block_system`。

### 4.3 奇偶性选择

若上分量轨道角动量为 `l`，则原点附近

```text
G(r) ~ r^(l+1).
```

偶 `l` 对应奇径向延拓，使用 forward ADF；奇 `l` 对应偶径向延拓，交换 forward/backward。
小分量使用相反的角动量结构。该选择直接写在矩阵构造中。

### 4.4 状态识别和填充

每个 `kappa` 通道对角化后，程序依次执行：

1. no-sea 选择：只保留相对负能海之上的粒子态；
2. 与解析谐振子参考支的重叠匹配；
3. 要求节点数等于目标主量子数；
4. 中子和质子分别按能量全局填充到给定 `N,Z`。

解析参考态只用于识别本征支，不提供势场、能量或 SCF 标签。

代码位置：

- `torch_rmf.py::independent_orbital_data`；
- `torch_rmf.py::DifferentiableDiracMatrix.solve`。

---

## 5. 从轨道得到密度

对每个占据态，采用球对称 RMF 密度：

```text
rho_s,q(r) = sum_a n_a (2j_a+1) [G_a^2-F_a^2] / (4 pi r^2)
rho_v,q(r) = sum_a n_a (2j_a+1) [G_a^2+F_a^2] / (4 pi r^2)
```

然后定义

```text
rho_s = rho_s,n + rho_s,p
rho_b = rho_v,n + rho_v,p
rho_3 = rho_v,n - rho_v,p.
```

轨道用同一个均匀 ADF 度量严格归一化。程序同时检查积分后的中子数、质子数和每条轨道
归一化残差。

代码位置：`pkdd_action.py::densities_from_orbitals`。

---

## 6. 密度依赖耦合和重排自能

令

```text
zeta = rho_b/rho_sat.
```

等标量耦合采用

```text
g_i(zeta) = g_i(rho_sat) a_i
            [1+b_i(zeta+d_i)^2]/[1+c_i(zeta+d_i)^2],
i = sigma, omega,
```

等矢量耦合采用

```text
g_rho(zeta) = g_rho(rho_sat) exp(-a_rho zeta).
```

代码保留解析 `dg/dzeta`，不在低密度尾部对 `zeta` 做 clamp，因为截断会修改泛函导数。

密度依赖产生重排项

```text
Sigma_R =
 [rho_s g_sigma' sigma
 +rho_b g_omega' omega
 +rho_3 g_rho' rho_field]/rho_sat.
```

`Sigma_R` 进入共同的等标量矢量自能 `V0`，不进入标量自能或同位旋分裂项。这一项对于
热力学一致性、单粒子能量和总能量变分一致性是必要的。

代码位置：

- `pkdd_action.py::couplings`；
- `pkdd_action.py::rearrangement_potential`；
- `pkdd_action.py::potentials`。

---

## 7. 每轮如何重建介子场

由当前轨道密度形成源：

```text
(-Delta + m_sigma^2) sigma = -g_sigma rho_s
(-Delta + m_omega^2) omega =  g_omega rho_b
(-Delta + m_rho^2)   rho   =  g_rho rho_3
-Delta A0                   = 4 pi alpha rho_p.
```

程序不是再启动一个 SCF 循环，而是在当前密度上对离散 Euler-Lagrange 线性系统求解一次，
得到 `sigma, omega, rho, coul`，再重建

```text
H_tilde[H].
```

场算符直接来自离散作用量的 Hessian，并包含解析无穷远边界项，所以“场重建方程”和“用于
自动微分的能量泛函”是同一个离散对象。

代码位置：

- `pkdd_action.py::reconstruct_fields`；
- `pkdd_action.py::field_stationarity_residuals`。

---

## 8. 完整静态作用量与能量

### 8.1 轨道动能

动能由占据 Dirac 轨道对自由径向 Dirac 算符的矩阵期望值计算，并减去静止质量；不使用
Thomas-Fermi 或局域动能密度近似。

代码位置：`pkdd_action.py::exact_kinetic_energy`。

### 8.2 离壳场作用量

静态作用量保留 RMF 鞍点结构：

```text
E_sigma = +(hbar*c)[ 1/2 sigma K_sigma sigma
                     + integral g_sigma rho_s sigma ]

E_omega = +(hbar*c)[-1/2 omega K_omega omega
                     + integral g_omega rho_b omega ]

E_rho   = +(hbar*c)[-1/2 rho K_rho rho
                     + integral g_rho rho_3 rho ]

E_coul  = +(hbar*c)[-1/2 A K_coul A
                     + integral rho_p A ].
```

标量方向是极小方向，时间样矢量方向是极大方向。轨道归一化通过单粒子能量作为拉格朗日乘子
加入。场方程满足后，完整作用量必须约化为

```text
E = E_kin + 1/2 integral(source * field)
```

的直接能表达式。程序把两种表达式的差作为严格公式不变量。

代码位置：

- `pkdd_action.py::off_shell_rmf_action`；
- `pkdd_action.py::direct_energy_terms`。

### 8.3 质心修正

当前训练目标和报告能量均不包含质心修正。所有与 Core 的能量比较使用 `no_com` 数据，避免
把仅在 SCF 最终波函数上计算的 Fortran 质心修正混入变分路径。

---

## 9. PRL 型哈密顿量变分

### 9.1 一次前向传播

`TorchRMFBackend.evaluate_tensor(H)` 严格按以下顺序执行：

```text
trial H
 -> ADF Dirac eigenstates
 -> occupied orbitals
 -> scalar/vector densities
 -> density-dependent couplings
 -> exact discrete meson-field solve
 -> reconstructed H_tilde
 -> complete RMF action E[H]
 -> PyTorch autograd delta E/delta H.
```

Fortran 不参与这条训练计算图。

代码位置：`torch_rmf.py::TorchRMFBackend.evaluate_tensor`。

### 9.2 广义梯度

实际送回变分参数的哈密顿量空间梯度为

```text
G_H = w_E delta E/delta H
      + lambda (H-H_tilde).
```

网络模式默认在第一轮按两项 Hamiltonian 梯度的 RMS 做一次尺度配平：

```text
lambda = RMS(w_E delta E/delta H) / RMS(H-H_tilde),
```

随后固定该 `lambda`，训练中不动态改变。这样实现补充材料所要求的两部分梯度量级可比。
直接变分基线仍可通过 `--lambda-mode fixed --lambda-reconstruct 0.001` 使用固定系数。
第一项寻找能量泛函驻点，第二项要求试探 Hamiltonian 与由自身密度重建的 Hamiltonian 自洽。
`lambda` 有 `MeV^-1` 单位，所有梯度在 MeV Hamiltonian 坐标中组合。

代码位置：`dpl_rhf/training/surrogate_gradient.py::generalized_prl_gradient`。

### 9.3 surrogate loss 的含义

PyTorch 优化器需要一个标量调用 `backward()`。程序构造

```text
L_sur = sum[H * stop_gradient(G_H)].
```

它的唯一作用是令

```text
d L_sur/d H = G_H.
```

因此 `L_sur` 的数值不是物理能量，也不能用于判断收敛。物理判断只能看能量梯度、自洽残差、
公式不变量和最终观测量。

### 9.4 梯度对抗诊断

程序分别计算能量梯度与重建梯度在以下空间中的夹角：

- 完整四通道哈密顿量空间；
- `S,V0,V3,VC` 四个物理通道；
- Chebyshev 或神经网络参数空间。

若夹角接近 `-1`，说明两个目标发生梯度对抗。程序不允许用二者相加后的偶然抵消作为最佳点。

### 9.5 最佳检查点

以第一轮为尺度分别归一化：

```text
r_E = RMS(delta E/delta H)/RMS_1(delta E/delta H)
r_H = RMS(H-H_tilde)/RMS_1(H-H_tilde).
```

监控量为

```text
monitor = max(r_E, r_H).
```

最佳点取 `monitor` 最小的轮次。网络模式使用论文补充材料规定的 Adam
(`beta=(0.9,0.999)`) 和 `ReduceLROnPlateau`，初始学习率为 `6e-5`，平台期衰减因子为
`0.5`。Chebyshev 直接基线保留原来的物理残差回退。SCF 数据不参与调度或最佳轮选择。

代码位置：`dpl_rhf/training/prl_hamiltonian_trainer.py::train_prl_hamiltonian`。

---

## 10. 当前“网络”和“直接变分”必须区分

### 10.1 直接变分模式

当前正式任务使用

```text
4 physical channels x 64 Chebyshev coefficients = 256 parameters.
```

它用于检验作用量、梯度和变分路径本身是否正确。没有隐藏层，因此不能称为神经网络结果。

代码位置：`hamiltonian_net.py::DirectHamiltonianParameterization`。

### 10.2 神经网络模式

网络采用 DeepH-E3 在球对称 `0e` 扇区的严格约化：

```text
input: even-parity localized radial graph
edge embedding: Gaussian expansion of radial separation
hidden: 64 x 0e
updates: 3 vertex blocks + 3 edge blocks
output: local Hamiltonian blocks S, V0, V3, VC
activation: SiLU
```

球对称局域 RMF 势没有 `l>0` 输出；这些自由度由 `kappa` 分块的 Dirac ADF 求解器承担，
不能把量子化学原子轨道的高阶 irreps 生搬到一维核势中。网络不包含 Chebyshev 可训练旁路，
所有 Hamiltonian 变化都由消息传递网络权重产生。图在 `[-R,R]` 上严格镜像，正负节点具有
相同偶宇称特征和镜像边，因此网络输出解析地属于偶宇称标量扇区，原点零导数不是软罚项。

前向物理链是核物理而非量子化学：

```text
(Z,N,r) -> H_theta -> ADF Dirac occupied states -> nuclear densities
        -> DD-RMF complete action E[H] and reconstructed H_tilde.
```

最终 `physical_components.npz` 保存由最终密度重建的 `S,V0,V3,VC` 物理场；
`network_variational_components.npz` 仅保存组装 `H_theta` 的内部变分坐标。后者存在
Hamiltonian 分解冗余，不能直接作为势场对比结果。

代码位置：`hamiltonian_net.py::LocalHamiltonianNet`。

---

## 11. 内部物理门

每个最终结果必须检查：

```text
Hermiticity error                  < 1e-12
action reduction error             < 1e-8 MeV
orbital normalization residual     < 1e-10
spectral residual                  < 1e-9
occupied node mismatch             = 0
uniform metric identity error      < 1e-14
local bilinear identity error      < 1e-12
autograd/finite-difference relative < 1e-5
autograd/finite-difference absolute < 1e-7.
```

此外记录四条介子场方程和两条 Dirac 方程的 Euler-Lagrange 残差、粒子数、边界条件和梯度
通道分解。测试入口 `python test_prl_mode.py` 当前包含 31 项严格测试。

---

## 12. SCF 在本项目中的唯一角色

最佳 DPL 点确定并保存以后，程序才调用 Core-1204：

1. 独立运行同参数集、同核素 SCF；
2. 比较无质心修正的 `E/A`；
3. 比较中子、质子、物质和电荷半径；
4. 比较完整径向 `S(r)`、`V(r)` 势；
5. 将 DPL 势固定送入 Fortran，比较占据单粒子能级；
6. 在固定 SCF 势上比较 ADF 本征值与 Fortran shooting 本征值。

SCF 不进入：

- 损失函数；
- 梯度；
- 学习率回退；
- 最佳检查点选择；
- 网络或 Chebyshev 参数初始化。

代码位置：

- `training/prl_hamiltonian_trainer.py` 的训练后验证段；
- `cli/compare_core1204.py`；
- `cli/plot_vs_comparison.py`。

---

## 13. 批量计算和输出

正式矩阵为

```text
5 parameter sets x 5 nuclei = 25 cases
nuclei: O16, Ca40, Ca48, Sn132, Pb208
epochs: 5000
GPUs: 2
workers per GPU: 3.
```

每个核输出：

- `summary.json`：参数集、核子质量、最佳轮、能量、半径和物理门；
- `dpl_hamiltonian.npz`：四个 Dirac 对角通道；
- `dpl_fields.npz`：`sigma, omega, rho, coul` 及物理自能；
- `dpl_densities.npz`：标量和矢量密度；
- `occupation_history.csv`：占据态、节点和单粒子能量；
- `gradient_alignment.csv`：梯度对抗诊断；
- `network_physical_constraints.json`：硬边界诊断；
- `scf_comparison/vs_dpl_scf.png`：DPL/Core 的 `V,S` 对比。

批处理最后按参数集生成五核链的能量、半径、单粒子能级和势误差汇总。

代码位置：`dpl_rhf/cli/batch_rmf.py`。

---

## 14. 组会上必须主动说明的限制

1. 当前是球对称偶偶核 DD-RMF Hartree，不含形变、配对和 Fock 非局域交换。
2. Chebyshev 直接变分输出只能作为诊断基线；只有 `variation_mode=network` 才能称为网络结果。
3. 每轮仍对角化 Python 中的 ADF Dirac 矩阵；Fortran SCF 不参与训练。
4. 介子场和 Coulomb 已实现解析无穷远外部作用量；轨道仍采用 `R=20 fm` 有限盒截断。
5. Woods-Saxon 和谐振子函数只用于初值及本征支识别，不是监督数据。
6. “与 SCF 接近”是外部数值验收，不是变分原理正确的证明；内部公式门必须先独立通过。
7. 当前结果不含质心修正，所有表格必须标记 `no_com`。

---

## 15. 建议的 15 分钟汇报顺序

1. **2 分钟：问题**  
   SCF 是固定点迭代；目标是把相同 RMF 泛函改写成可微哈密顿量变分，而非学习 SCF 标签。

2. **3 分钟：物理泛函**  
   介绍 `S,V0,V3,VC`、密度依赖耦合、重排项和静态鞍点作用量。

3. **3 分钟：离散 Dirac 问题**  
   重点展示七点 ADF、`B1/B2`、厄米性和节点约束，说明如何避免假态。

4. **3 分钟：PRL 型更新**  
   展示 `delta E/delta H + lambda(H-H_tilde)`，强调 surrogate loss 不是物理损失值。

5. **2 分钟：严格约束与验证**  
   展示原点、Yukawa、Coulomb、公式不变量和 SCF 后验隔离。

6. **2 分钟：结果和限制**  
   展示五核链表格及 V/S 图；明确当前是 RMF 直接变分基线，RHF 和网络结果尚属后续工作。

---

## 16. 最适合作为结论的一页

```text
输入：参数集 + (Z,N)

物理空间：
  四个 RMF 自能分量
  + 原点正则
  + Yukawa/Coulomb 无穷远条件
  + no-sea、粒子数和节点约束

可微传播：
  H -> Dirac 轨道 -> 密度 -> 介子场 -> H_tilde -> E[H]

变分方向：
  delta E/delta H + lambda(H-H_tilde)

内部证明：
  厄米性、谱残差、作用量约化、场方程、梯度有限差分

外部验收：
  Core-1204 SCF 的能量、半径、单粒子能级和 V/S 势

当前结论：
  已建立无 SCF 标签的可微 DD-RMF 变分基线；
  神经网络与 RHF 扩展必须在该基线通过物理门后进行。
```
