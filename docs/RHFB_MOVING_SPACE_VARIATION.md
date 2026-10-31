# 无海核子空间的允许变分与验收方程

2026-10-06。这里推导的是本项目的候选约束处方，不能称作论文已经证明的完整奇核 RHFB 等价算法。训练不使用 Dirac 或 BdG 对角化；独立小空间参考只用 Python。

## 文献支持与尚未支持的部分

[Geng–Long](https://arxiv.org/pdf/2202.00482) Eq.(10) 以正能 HF 态定义核子算符，Eq.(12) 在其粒子空间进行 Bogoliubov 变换。计算时又用含正、负能态的 DWS 基函数展开旋量，Eq.(23)。这支持区分物理核子空间与数值展开基函数；它没有直接给出本项目场依赖投影的变分导数或神经网络算法。

[Barbaroux–Esteban–Séré](https://arxiv.org/pdf/math-ph/0402058) 的原子 Dirac–Fock max–min 研究在特定闭壳极限有结论，但 Theorem 11 给出开壳反例。不能将那里的外层投影 max–min 假说当成奇核、配对或完整 PKA1 的通用证明。

因此，本节以下的驻定公式是从本项目明确的 EDF 与约束直接推导；其与核文献参考真空及常规 RHFB 求解的物理对应还需比较。

## 固定核子空间

令 B†B=I，rho=B rho_a B†，chi=B chi_a Bᵀ。活动空间 R_a 通过正则变换保证纯态与奇偶性，嵌入时补上非活动孔空间恒等块。

本项目约定 Delta=.5 Vbar chi，对 L=E−mu Tr(rho) 有

\[
\delta L=\operatorname{Tr}[(h-\mu)\delta\rho]
+\operatorname{Re}\operatorname{Tr}(\Delta^\dagger\delta\chi)
=\tfrac12\operatorname{Re}\operatorname{Tr}(H_{\rm BdG}\delta R).
\]

固定 B 的允许变分 delta R_a=[K,R_a] 导出

\[
[W^\dagger H_{\rm BdG}W,R_a]=0,\quad W=\operatorname{diag}(B,B^*).
\]

它仅是活动空间的驻定方程。若空间是受约束的，全原始空间对易子不必为零；若只是在有限活动空间内近似无约束问题，则还必须检查外部驻定残差。

## 场依赖核子空间

两端点轨道变分给出

\[
\delta L_B=2\operatorname{Re}\operatorname{Tr}(G_B^\dagger\delta B),\qquad
G_B=(h-\mu)B\rho_a+\Delta B^*\chi_a^\dagger.
\]

配对项含两个端点；忽略其中一个会使梯度差一倍。对完整正能空间 P=BB†，其外侧变化满足 delta B_external=(delta P)B。记 G_perp=(I−P)G_B，则

\[
\delta L_{\rm external}=\operatorname{Tr}(C\delta P),\quad
C=BG_{\perp}^\dagger+G_{\perp}B^\dagger.
\]

如果 P=1_(s,infinity)(h)，则 delta P=D P_h[delta h]。小空间参考的谱分量公式为

\[
(U^\dagger\delta P U)_{ij}
=\frac{p_i-p_j}{e_i-e_j}(U^\dagger\delta h U)_{ij}
\]

仅在两侧分支不同的指标间非零。训练用有理符号函数及线性求解保持其导数。delta h 又依赖 delta rho，而 rho 含 P，因此需要隐式场响应；有限次数场回代的自动微分不自动等于这个响应。

在自洽约束真正成立时 `[h,P]=0`，正常场的外侧贡献 `(I−P)(h−mu)B rho_a` 为零。因而

\[
G_{\perp}=(I-P)\Delta B^*\chi_a^\dagger.
\]

这是两种候选驻定处方可能产生差别的具体配对源。chi=0、投影固定，或完整配对场不耦合到排除空间时，响应项消失；这些退化情形的一致不能证明一般配对态等价。原型需报告该外侧源和正常场的正/负分支耦合，不能只报告 chi 非零。上述结论来自本节明示的变分公式，不是对核文献算法的额外归因。

## 可检验的耦合 KKT 系统

引入辅助正常场 H，构造 B(H)=QR[P(H) B_ref]，令 x 为活动空间正则坐标，

\[
F(H,x)=H-h[\rho(H,x)]=0,\qquad
f(H,x,\mu)=E[\rho(H,x),\chi(H,x)]-\mu N.
\]

在固定秩且 Dirac 间隙不闭合的局部图册中，完整约束驻定应求解

\[
f_x+F_x^T\lambda=0,\qquad
f_H+F_H^T\lambda=0,\qquad F=0,\qquad N=N_0.
\]

矩阵和复数变量需按独立实坐标展开。若 F_H 可逆，也可先解 F_Hᵀ lambda=−f_H，再形成隐式驻定梯度。规范冗余不应被误报为物理不收敛，但不可用参数梯度很小替代上述方程。

常规 frozen-P 内层 HFB 加 P(h) 外层更新满足另一组嵌套方程：活动对易子为零、F 为零。除非投影响应项消失或推导证明抵消，两组方程没有自动等价性。原型必须并列报告这两个参考的能量、密度、粒子数、配对、投影和驻定残差；不同则停在原型阶段说明原因。

## 当前直接核验

[no_sea_variation.py](../dpl_rhf/rhfb/no_sea_variation.py) 将 G_B 与 C 公式实现为独立代数助手。两个小空间测试核对了：实际未缩放 D1S、非零 chi、奇宇称纯态下的轨道一阶变分；Nambu 的 .5 Tr(H delta R) 约定；外侧 delta P 响应；固定空间正则对易子公式。2026-10-06 两项测试通过，允许误差 2e−11，记录为 `outputs/validation/no_sea_first_variation_20261006.xml`。

这证明上述变分公式和当前共轭约定一致，不证明 D1S 角动量核、完整 PKA1、移动空间物理等价或 O17 已验收。移动空间 KKT 与嵌套参考的数值比较正在独立原型中推进。
