# PKDD-RMF 完整作用量与 PRL Hamiltonian 变分流程

本文档定义新版程序实际使用的物理泛函。核物理部分采用本目录《核物理(1)》第 1、3 章；
神经网络变分结构采用 `PhysRevLett.133.076401`。SCF 数据不进入训练损失。

## 1. 变分自由度

球对称、静态、时间反演、无海近似下，自由度为：

- 占据核子径向旋量 `psi_a=(G_a,F_a)^T`；
- 标量场 `sigma(r)`；
- 时间样矢量场 `omega(r), rho(r)`；
- 电磁势 `A(r)`，程序变量 `coul=e A`；
- 轨道正交归一约束的 Hermitian 拉格朗日乘子矩阵 `Lambda_ab`。

PKDD 是局域 Hartree RMF，因此当前没有 Fock 非局域核 `X_a,Y_a`。RHF 接口保留，但不能把
当前结果称为 RHF。

## 2. 完整静态作用量

程序扣除核子静止质量，使用：

```text
Gamma_RMF = T[G,F] + Gamma_sigma + Gamma_omega + Gamma_rho + Gamma_C
            - sum_ab Lambda_ab ( <psi_a|psi_b> - delta_ab )
```

其中：

```text
Gamma_sigma = hbar*c int d^3r [
    + 1/2 (grad sigma)^2 + 1/2 m_sigma^2 sigma^2
    + g_sigma(rho_b) rho_s sigma ]

Gamma_omega = hbar*c int d^3r [
    - 1/2 (grad omega)^2 - 1/2 m_omega^2 omega^2
    + g_omega(rho_b) rho_b omega ]

Gamma_rho = hbar*c int d^3r [
    - 1/2 (grad rho)^2 - 1/2 m_rho^2 rho^2
    + g_rho(rho_b) rho_b3 rho ]

Gamma_C = hbar*c int_0^R d^3r [
    - (grad coul)^2/(8*pi*alpha) + rho_p coul ]
    - hbar*c R coul(R)^2/(2*alpha)
```

最后一项是有限盒子外 `coul(r)=C/r` 的解析作用量，不能遗漏。若遗漏，O16 在 `R=20 fm`
时会产生约 2.3 MeV 的系统误差。

`T[G,F]` 是书中径向 Dirac 动能，不使用 Thomas-Fermi 近似。轨道满足：

```text
int dr (G_a G_b + F_a F_b) = delta_ab
```

程序在普通 Hermitian ADF 坐标表示中使用均匀权重 `h`。同一权重同时用于轨道归一、局域势
矩阵元和密度作用量积分，从而严格满足 `<psi|V|psi> = int rho V`。本征分解后不修改波函数，
也不执行 Lowdin 后处理。非均匀 Newton-Cotes 权重不能混入这一坐标本征度量。

PKDD 裸质量严格区分：

```text
M_n = 939.5731 MeV
M_p = 938.2796 MeV
```

Dirac 矩阵采用与 Core-1204 相同的静止质量移位表示，因此质量出现在
`VMS_n=...-2M_n`、`VMS_p=...-2M_p` 中，本征值对应 `epsilon_tau=E_tau-M_tau`。
动能、静止质量扣除、Woods-Saxon 小分量初始化和 `V/S` 反演也分别使用相应质量。
输出 `effective_mass_n_mev/effective_mass_p_mev = M_tau+g_sigma sigma hbar*c` 用于检查有效质量。

## 3. 密度依赖和重排项

```text
rho_s  = sum_a v_a^2 (2j_a+1)/(4*pi*r^2) (G_a^2-F_a^2)
rho_b  = sum_a v_a^2 (2j_a+1)/(4*pi*r^2) (G_a^2+F_a^2)
rho_b3 = rho_n-rho_p
```

`g_sigma(rho_b), g_omega(rho_b), g_rho(rho_b)` 使用 PKDD 参数化。作用量对核子旋量变分时，
耦合常数对 `rho_b` 的导数自动产生：

```text
Sigma_R = sigma rho_s dg_sigma/drho_b
        + omega rho_b dg_omega/drho_b
        + rho rho_b3 dg_rho/drho_b
```

重排项进入矢量自能和四个 Dirac 势，但不作为独立项重复加入总能量。

## 4. Euler-Lagrange 方程

对四个场变分得到书中场方程：

```text
(-Delta+m_sigma^2) sigma = -g_sigma rho_s
(-Delta+m_omega^2) omega = +g_omega rho_b
(-Delta+m_rho^2)   rho   = +g_rho rho_b3
-Delta coul               = 4*pi*alpha rho_p
```

对轨道变分得到：

```text
[-d/dr+kappa/r] F_a + VPS_tau G_a = epsilon_a G_a
[+d/dr+kappa/r] G_a + VMS_tau F_a = epsilon_a F_a
```

其中程序采用静止质量平移后的通道：

```text
VPS_tau = +g_sigma sigma + g_omega omega + tau g_rho rho
          + tau_c coul + Sigma_R
VMS_tau = -g_sigma sigma + g_omega omega + tau g_rho rho
          + tau_c coul + Sigma_R - 2 M_tau
```

原点幂律通过解析延拓基底硬编码为 `G~r^(l_u+1), F~r^(l_d+1)`，盒边束缚态为零。径向 Dirac
算符使用 7 点非中心差分，在径向积分度量下以转置配对构造广义 Hermitian Hamiltonian。

## 5. 消去场后的能量

当四个场方程成立时，完整作用量退化为：

```text
E_on-shell = T + 1/2 int d^3r [
    g_sigma sigma rho_s + g_omega omega rho_b
  + g_rho rho rho_b3 + coul rho_p ]
```

程序同时计算完整作用量和该半源场表达式，并输出
`action_reduction_error_mev`。这是一项离散一致性检查，不是训练标签。

## 6. 与 PRL 的结合

神经网络使用 SiLU 隐藏层，输出局域 `H_theta=(VPS_n,VMS_n,VPS_p,VMS_p)`。完整前向图为：

```text
H_theta -> occupied eigenspace -> G,F -> densities
        -> discrete-action field solve -> Gamma_RMF[H_theta]
        -> autograd d Gamma_RMF / d H_theta
```

实现中网络的四个独立输出不是四个任意 Dirac 对角元，而是物理分量
`(S,V0,V3,VC)`，随后严格按书中式 (3.48) 组装：

```text
VPS_n = V0 + S + V3
VMS_n = V0 - S + V3 - 2 M_n
VPS_p = V0 + S - V3 + VC
VMS_p = V0 - S - V3 + VC - 2 M_p
```

因此中子和质子共享同一个标量自能，`rho` 通道严格反号，库仑通道严格只作用于质子。
这些是网络表示层的硬约束，不依赖损失权重。

只最小化能量不能确定未占据空间。按照 PRL，引入由自身密度重构的 `H_tilde`：

```text
grad_H Gamma_generalized
  = grad_H Gamma_RMF + lambda (H_theta-H_tilde)
```

第一项寻找作用量驻点，第二项确定完整 Hamiltonian。`H_tilde` 来自同一 PKDD 作用量的
Euler-Lagrange 场，不来自 SCF 标签。

注意静态矢量场在未消元作用量中是极大方向。程序从完整离散作用量的 Hessian 一次求解四个
Euler-Lagrange 场，再对 `H_theta` 做 PRL 变分；场重建与作用量使用完全相同的积分、微分和
库仑外部边界项。Green 表达式只保留作解析对照，不进入训练图。

## 7. 输出与验收

- `action_diagnostics.npz`：完整作用量各项、库仑外部项和轨道约束项；
- `summary.json/diagnostics`：作用量退化误差及每条 Euler-Lagrange 方程 RMS；
- `autograd_gradient.npz`：完整作用量对四个势通道的梯度；
- `gradient_check.json`：有限差分与自动微分一致性；
- `profile_compare.json`：训练结束后与 Core-1204 SCF 的外部比较，不进入训练。
