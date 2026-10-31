# 无子空间刷新 DPL-RMF/RHF：物理原理与工作流程

## 1. 修改动机

旧版每隔固定轮数对当前 Hamiltonian 的各个 `(species, kappa)` 块执行完整
对角化，以新正能本征空间替换网络基底，并清零网络输出头。五个 PKA1 核的历史
曲线均显示：残差主要在刷新轮次突降，而刷新之间网络带来的下降很小。因此旧版的
主要收敛机制是谱空间重建，而不是神经网络连续变分。这种方法无法自然推广到形变
和三维大规模稀疏问题。

本版本将旧程序保留为只读基线。新版的训练路径禁止 `eig/eigh`、Dirac 求解、
正能基替换和输出层重置。

## 2. 变分对象

每个核子种类和球形 `kappa` 块由网络输出占据态因子

\[
Q_q^{\kappa}\in\mathbb R^{2N_r\times N_{\rm occ}},\qquad
P_q^{\kappa}=Q_q^{\kappa}f_q^{\kappa}Q_q^{\kappa T}.
\]

其中上下两半分别为 Dirac 大、小分量的内点网格值，`f` 是由核子数确定的占据
矩阵。网络输出不是 Hamiltonian 的本征矢修正系数，而是固定完整 Dirac 网格空间
中的修正。

初始 `Q` 来自解析谐振子/Woods-Saxon径向函数，不求解 Dirac 方程。神经修正
显式乘以边界包络：

\[
\delta G(r)\sim r^{l_G+1}(1-r/R),\qquad
\delta F(r)\sim r^{l_F+1}(1-r/R).
\]

盒外密度继续使用由
\(\epsilon=E_{\rm total}-M\) 导出的自由 Dirac 指数尾。所有隐藏非线性均为 SiLU。

前向末端实施薄 QR：

\[
Q^TQ=I.
\]

QR 只是同一个固定原始空间内的 Stiefel 回缩，不计算或替换 Hamiltonian 子空间。

## 3. 严格能量泛函

主损失直接使用现有可微核物理泛函：

\[
\mathcal L_E=E_{\rm RMF/RHF}[P]/A.
\]

RMF 保留密度依赖 `sigma/omega/rho/Coulomb`、重排项和精确 Dirac 动能。
PKA1 RHF 继续使用完整 Hartree、Fock、pion-PV、rho-tensor、
rho-vector-tensor、接触补偿及密度依赖重排项。Fock 非局域密度保持

\[
R_q^\kappa(r,r')=\sum_n f_{qn\kappa}
Q_{qn\kappa}(r)Q_{qn\kappa}^T(r'),
\]

先在同一 `kappa` 下对 `n` 求和，再对不同 `kappa` 求和；程序不把 Fock 项局域化。

对每个块，从能量泛函梯度得到自洽单粒子 Hamiltonian：

\[
H[P]=\frac{\delta E}{\delta P}.
\]

全空间 Euler-Lagrange 驻定残差为

\[
R_H=(I-QQ^T)H[P]Q.
\]

旧版只在当时的正能投影基内计算该量；新版在完整 Dirac 网格空间计算。

## 4. 逆 Hamiltonian 无海约束

直接最小化相对论能量会发生变分塌缩。对中子和质子分别取

\[
W_q=-M_q
\]

（Hamiltonian 已减去静质量），它位于物理正能支与 Dirac 海之间。每轮只做线性
求解：

\[
(H_q[P]-W_qI)X_q=Q_q.
\]

不求解 Dirac Hamiltonian 的本征值或本征矢。定义占据空间中的小矩阵

\[
K_q=Q_q^T(H_q-W_qI)^{-1}Q_q.
\]

约束包括：

\[
R_I=(I-QQ^T)X,
\qquad
s_i=\operatorname{eig}_i[(-W)K_q]\ge 1.
\]

`R_I` 保证占据投影同时是不变子空间。对已经减去静质量的 Hamiltonian，
`W=-M`，矩阵不等式 `(-W)K_q >= I` 同时排除 Dirac 海，并要求占据态处于
`epsilon=E_total-M <= 0` 的束缚支。必须检查 `K_q` 的全部本征值，不能只检查
对角元；后者随占据轨道的任意正交旋转而改变，可能掩盖某个落入 Dirac 海的
线性组合。这里处理的只是维数等于同一 `(q,kappa)` 占据态数的小矩阵，不是
Dirac Hamiltonian，也不产生或刷新计算子空间。损失为

\[
\mathcal L=\mathcal L_E+lambda_H\|R_H\|^2
+\lambda_I\left(\|R_I\|^2+\langle\max(0,1-s_i)^2\rangle
-10^{-2}\langle s_i\rangle\right).
\]

`lambda_H` 默认在第一轮按网络参数梯度范数自动平衡；`lambda_I` 默认 1。严格
按照逆 Hamiltonian 演化，当前密度产生的 `H[rho_k]` 在无海辅助更新内固定，梯度
只通过 `X=(H-WI)^(-1)Q` 回到占据因子 `Q`；总能量项仍保留完整的密度依赖和
`dH/drho` 自动微分。球形版本使用 float64 稠密求解，三维扩展时该接口
可替换为矩阵无关 MINRES，而不改变网络、密度或泛函接口。

## 5. 每轮数据流

1. 根据 `Z/N` 构造占据壳层、解析初态和固定七点 ADF Dirac 算符。
2. SiLU 网络生成每个 `(q,kappa,n)` 的全空间径向修正。
3. 边界包络和 QR 给出严格正交的占据因子 `Q`。
4. 由 `P=QfQ^T` 计算局域密度和 RHF 非局域密度 `R`。
5. 重建全部 Hartree/Fock 场并计算严格总能量。
6. 自动微分得到 `H[P]` 和全空间驻定残差。
7. 求解逆 Hamiltonian 线性方程，计算无海分支约束。
8. 总物理损失反传至网络，Adam 更新参数；不刷新基底或优化器。
9. 按驻定残差选择最佳 checkpoint。
10. 训练结束后才允许调用 Core-1204 做外部验证。

## 6. 输出与诊断

`history.csv` 至少包含能量、驻定残差、逆 Hamiltonian 残差、最小分支分数、
线性求解残差、参数梯度范数、学习率以及三类禁止操作计数。

必须始终满足：

```text
eigensolve_count = 0
refresh_count = 0
training_dirac_solves = 0
minimum_branch_score > 0
```

RMF 输出 `S`、`V`、`V-S`、`V+S` 所需场文件。RHF 额外输出每个
`(species,kappa)` 的总/Fock Hamiltonian、完整密度因子以及用于构造非局域
`R(r,r')` 的数据。Core 对比及绘图不参与损失。

## 7. 实验顺序与验收

首先运行 PKDD/O16 500 轮，确认无刷新曲线连续下降和禁止操作计数为零；之后运行
PKA1/O16，检查所有 PKA1 通道梯度有限且非零；最后扩展到 O16、Ca40、Ca48、
Sn132 和 Pb208。

目标为结合能和半径相对 Core 偏差不超过 1%，单粒子能量 MAE 不超过 1 MeV。
同时必须报告势场及 Fock 非局域密度误差，不能仅利用通道间能量抵消判断正确性。

## 8. 理论来源

- `PhysRevLett.133.076401(1).pdf`：以物理能量变分和自动微分训练网络。
- `核物理(1).pdf`：无海近似、Dirac径向变分方程和边界条件。
- `RHF-e.pdf`、`0204872龙文辉_5.0.pdf`：完整RHF/Fock泛函及变分导数。
- Tanimura, Hagino, Liang, PTEP 2015：逆Hamiltonian方法避免变分塌缩。
- D4FT：用低秩正交因子直接变分密度矩阵。

本版本结合这些思想，但不使用 SCF 数据监督网络，也不把周期性谱刷新伪装成训练。
