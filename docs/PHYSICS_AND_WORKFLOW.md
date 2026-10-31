# 物理约束与完整工作流程

## 1. 研究对象

当前求解器处理球对称、时间反演不变的偶偶核 DD-RMF Hartree 问题。网络寻找的是受约束单体密度矩阵 `P` 的驻定路径，而不是拟合 SCF 输出，也不是逐个 epoch 求解 Dirac 方程。

![密度矩阵约束](images/density_matrix_constraints.png)

## 2. 网络变量与硬约束

每个中子/质子和 `kappa` 分块使用低秩因子：

```text
P_b = Q_b n_b Q_b^T,   Q_b = U_b C_b,   C_b^T C_b = I.
```

`U_b` 是固定或稀疏刷新的正能子空间，`n_b` 是由核子数和壳层占据确定的对角占据矩阵。网络从 `r/R`、`(r/R)^2` 和 sinc 径向特征出发，叠加核子种类、`kappa` 和主量子数嵌入，输出上下分量修正。所有非线性层均为 SiLU。QR 投影直接施加正交约束。这里的径向特征只负责表示核内形状，不能代替下面由 Dirac 渐近解导出的密度边界。

由此严格保证 Hermitian 性、占据数固定、粒子数固定、满占据分块幂等、正能空间中的 no-sea 约束和轨道正交。这些不是可被权重牺牲的 penalty，而是网络前向图的构造性质。

## 2.1 从 Dirac 波函数推导密度边界

图片给出的无源远区解为

```text
G_a(r) -> r k_a h_l^(1)(k_a r),
F_a(r) -> sgn(kappa_a) r k_a^2/(epsilon_a+2M_q)
          h_ltilde^(1)(k_a r),
k_a^2 = epsilon_a(epsilon_a+2M_q)/(hbar*c)^2.
```

这里的能量定义必须固定为

```text
epsilon_a = E_a(total) - M_q,
```

而不是包含静质量的核子总能量 `E_a(total)`。占据束缚态满足
`E_a(total)<M_q`，所以可写成 `epsilon_a=-B_a`，其中
`B_a=M_q-E_a(total)>0`。此时

```text
epsilon_a + 2M_q = E_a(total) + M_q > 0,
k_a^2 = -B_a(2M_q-B_a)/(hbar*c)^2 < 0.
```

定义

```text
lambda_a = sqrt(B_a(2M_q-B_a))/(hbar*c),
beta_a   = (hbar*c) lambda_a/(2M_q-B_a)
         = sqrt(B_a/(2M_q-B_a)).
```

使用球 Hankel 函数在纯虚参数处的渐近式，或直接使用远区径向
Dirac 方程，可得

```text
G_a(r) = A_a exp(-lambda_a r) [1 + O(1/r)],
F_a(r) = -beta_a G_a(r) [1 + O(1/r)].
```

因此固定 `(q,kappa)` 的非局域径向密度矩阵

```text
R_kappa(r,r') = (2j+1) sum_n v_n^2 u_n(r) u_n(r')^T
```

在 `r,r' -> infinity` 时由该固定 `(q,kappa)` 分块内 `lambda` 最小的
占据态 `a_F(kappa)` 主导：

```text
R_kappa(r,r') -> C exp[-lambda_F(r+r')]
  [[1, -beta_F], [-beta_F, beta_F^2]].
```

所以对任一径向矩阵分量，在一个坐标趋于无穷远时有

```text
(partial_r + lambda_F) R^(mu,nu)(r,r') -> 0.
```

对总的局域粒子种类密度，还要再对所有 `kappa` 分块求和，因此最终
由该粒子种类全部占据态中 `lambda` 最小的态主导。局域物理密度含
球坐标因子 `1/(4*pi*r^2)`。记
`D_X(r)=4*pi*r^2 rho_X(r)`，则矢量、标量和张量密度满足

```text
(partial_r + 2 lambda_F) D_X(r) -> 0,
(partial_r + 2 lambda_F + 2/r) rho_X(r) -> 0.
```

三个局域通道的渐近比例不是独立参数：

```text
rho_s/rho_v -> (1-beta_F^2)/(1+beta_F^2)
              = (M_q-B_F)/M_q,
rho_t/rho_v -> -2 beta_F/(1+beta_F^2)
              = -(hbar*c) lambda_F/M_q.
```

中子和质子必须分别使用各自质量及最小衰减常数。质子的 `1/r`
Coulomb 尾会在上述主导指数外增加 Coulomb 幂次修正；不能把中子
的自由 Hankel 尾原样复制给质子，但主导指数仍由分离能决定。

实现不再把存储端点解释为物理零边界。每个占据
`(q,kappa,n)` 列携带正能分支的 `epsilon=E_total-M`，普通训练轮不为此
求解 Dirac 方程。程序由上式直接延拓 `GG/GF/FG/FF` 密度双线性，设置
20 fm 端点，并解析计算从 20 fm 到无穷远的粒子数和二阶矩。该延拓在
PyTorch 图内完成，不是经验 `exp(-r)` 包络。初始化或显式稀疏 no-sea
刷新才更新 `epsilon`。

原点也不再对所有通道统一外推。对每个 `kappa` 分块先使用
`G~r^(l_G+1)`、`F~r^(l_F+1)`：只有对应 `l=0` 的局域密度可在原点取
有限值，其余贡献严格为零；tensor 密度在原点为零。

## 3. 密度与动能

球对称标量密度和矢量密度分别使用 `G^2-F^2` 与 `G^2+F^2`，乘占据数和简并度后除以 `4*pi*r^2`。原点按 Core 的正规外推。中子、质子分别累加为 `rho_s_n/p` 和 `rho_v_n/p`。

动能不是 Thomas-Fermi 或 ETF 近似，而是自由 Dirac 算符上的精确迹：

```text
E_kin = sum_b g_b Tr(n_b Q_b^T T0_b Q_b).
```

中子和质子分别使用参数集中的物理质量。Dirac 径向耦合严格分为 `B1=-D_backward+kappa/r` 和 `B2=D_forward+kappa/r`，而不是用一个中心差分块替代两者。

## 4. DD-RMF 场与势

局域密度依赖耦合 `g_sigma(rho_b)`、`g_omega(rho_b)`、`g_rho(rho_b)` 由所选参数集给定。每次前向传播从网络密度解离散场方程：

```text
(-Laplace + m_sigma^2) sigma = -g_sigma rho_s
(-Laplace + m_omega^2) omega =  g_omega rho_b
(-Laplace + m_rho^2)   rho   =  g_rho rho_3
-Laplace A0 = e^2 rho_p
```

有质量场在无穷远采用指数衰减条件；库仑场与外部 `C/r` 解匹配。场通过可微线性求解进入 PyTorch 计算图。

密度依赖耦合产生重排势：

```text
Sigma_R = [rho_s dg_sigma sigma + rho_b dg_omega omega
           + rho_3 dg_rho rho] / rho_sat.
```

标量自能 `S=g_sigma*sigma`，等标矢量势 `V0=g_omega*omega+Sigma_R`，同位旋矢量势 `V3=g_rho*rho`，质子还包含库仑势。中子/质子的 `V+S` 和 `V-S-2M` 通道与 Core-1204 符号一致。

## 5. 变分目标

消去满足场 Euler-Lagrange 方程的介子场后：

```text
E[P] = E_kin[P] + E_sigma + E_omega + E_rho + E_Coulomb.
```

相互作用项使用半源场形式。质心修正不进入训练目标和主对比量，避免把只在最终波函数上定义的修正混进密度矩阵变分路径。

仅降低能量可能沿不完整切空间漂移，因此损失同时包含投影驻定性：

```text
R_b = H_b Q_b n_b - Q_b sym(Q_b^T H_b Q_b n_b)
L = w_E E[P] + w_R RMS(R)^2.
```

`R_b=0` 是物理流形上的 Euler-Lagrange 驻定条件。它直接嵌入损失并由 PyTorch 自动微分传播。自动微分保证离散泛函与参数梯度一致，但不能单独防止 Dirac 海假态；假态由正能子空间约束和最终谱验证共同排除。

## 6. 通用球形 N/Z 与稀疏谱刷新

命令行接受任意非负整数 `N/Z`。既有 Core 壳层次序作为严格前缀；当
容量不足时按球形谐振子主壳自动增加新的 `(n,kappa)` 分块。中子和
质子独立填充，最后一个简并壳采用球形等填充。网络的 `kappa` 与主量子
数嵌入尺寸由实际分块动态生成，不含固定的 `kappa<=7` 或 `n<=7` 假设。
本阶段不包含配对、奇核阻塞和形变，因此“任意 N/Z”指球形等填充模型
空间，而不是对这些额外关联的近似实现。

初始化时建立正能空间。普通 epoch 只执行网络、QR、密度、场重建、能量与驻定残差，不调用本征求解器。默认每 100 轮刷新一次正能空间，500 轮训练实际刷新 4 次；这些块对角化与普通 epoch 分开计数。最终 Python 谱求解默认关闭；仅在显式给出 `--final-spectral-validation` 时运行。正式对照使用训练完成后的 Core。

![程序数据流](images/architecture_workflow.png)

## 7. 一次完整运行

1. 解析参数集和 `Z/N/A`，构造 201 点径向网格。
2. 构造七点 ADF Dirac 离散算符和场算符。
3. 初始化占据壳层与各 `kappa` 正能子空间。
4. 网络生成密度矩阵因子，QR 施加硬约束。
5. 计算密度、密度依赖耦合和四个局域场。
6. 构造 S/V 通道和完整局域哈密顿量。
7. 计算精确迹动能、直接相互作用能和驻定残差。
8. 反向传播更新网络；按设定间隔刷新正能空间。
9. 选择驻定监视量最佳 checkpoint。
10. 写出密度、场、势、能量、边界和求解器计数诊断。
11. 可选调用 Core-1204 SCF 进行外部对比；仅显式请求时导出 Python 谱波函数。

## 8. 能力边界

RMF 当前没有配对、形变和时间奇通道。独立 RHF 包实现 PKO2/PKA1 的
非局域 Fock、pion 赝矢量和 rho-tensor 通道；两条路径共享直接密度边界。
