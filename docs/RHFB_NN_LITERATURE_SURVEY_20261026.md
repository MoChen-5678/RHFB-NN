# RHFB 无刷新 NN 求解：文献调研（2026-10-26）

联网检索整理。"已开"＝实际打开过 abs/html；"检索"＝仅核对检索元数据；"未核实"须再查。
本文件只是文献依据，不代表任何数值验收。

## 1. 无海与变分塌缩（不对角化）

| 文献 | 要点 | 与本项目关系 |
|---|---|---|
| Hagino–Tanimura, PRC 82, 057301 (2010), [arXiv:1008.4995](https://arxiv.org/abs/1008.4995)（检索） | 逆 Hamiltonian 法 ψ=lim e^{τ/(h−W)}ψ0，W 位于费米海与狄拉克海之间 | 支持逆 H 无海项；仅单粒子固定势 |
| Tanimura–Hagino–Ring, PRC 88, 017301 (2013), [arXiv:1304.4697](https://arxiv.org/abs/1304.4697)（已开） | IHM 用于上下无界的 HFB 算符（非相对论球形） | 支持对 H_BdG 移位求逆；无 RHB/RHFB 先例 |
| Ren–Zhang–Meng, PRC 95, 024313 (2017), [arXiv:1612.09429](https://arxiv.org/abs/1612.09429)（已开） | 3D 格点 IHM＋谱方法，避免费米子倍增 | 网格/ADF 参考 |
| Li–Ren–Zhao, PRC 102, 044307 (2020), [arXiv:2007.09414](https://arxiv.org/abs/2007.09414)（已开） | Eq.(11) 多项式滤波 F(h)=(h−C)²/D²，C=−2m_N；Eq.(12) 试探态×F、残差×F^{N_F}(N_F=4) 后做 CG；RHB 推广列为未来工作 | 最直接的"不对角化无海"先例，比 sign 有理求积简单，但只是滤波，不是精确投影 |
| Talman, PRL 57, 1091 (1986)；Dolbeault–Esteban–Séré, JFA 174, 208 (2000)；Esteban–Lewin–Séré, [arXiv:0706.3309](https://arxiv.org/abs/0706.3309)（检索） | Dirac min–max 原理 | 理论背景 |
| Séré, CMP 404, 1275 (2023), [arXiv:2211.10196](https://arxiv.org/abs/2211.10196)（已开 abs；不动点方程全文未核实） | 先定义满足非线性不动点的可容许密度集，再在其上极小 Dirac–Fock 能量 | "移动投影上约束极小"的严格数学先例；电子、无配对、弱耦合 |
| Zhang–Liang–Meng, CPL 26, 092401 (2009), [arXiv:0910.4078](https://arxiv.org/abs/0910.4078)（检索） | 上分量类 Schrödinger 方程虚时步，含 Fock 非局域势 | 另一条无海路线 |

结论：文献支持逆 H／滤波作为无海手段。没有文献证明 sign(h−W) 投影与
Serra–Ring／Geng–Long 的 RHFB 无海处方等价。

## 2. 梯度法中的奇核阻塞

- Robledo–Bernard–Bertsch, PRC 86, 064313 (2012), [arXiv:1210.2768](https://arxiv.org/abs/1210.2768)（检索）：
  梯度法＋Gogny D1S＋时间奇场，直接极小奇 A 核 HFB，不用 EFA。与"显式奇参考＋正则梯度"同构。
- Robledo–Bertsch 1104.5453：奇 A 只需交换初始 U/V 的一列。
- Kasuya–Yoshida, PTEP 013D01 (2021), [arXiv:2005.03276](https://arxiv.org/abs/2005.03276)（已开）：
  时间反演对称且无零本征值时，最低 HFB 解属数目偶宇称；加时间奇约束可直接得到奇核解。可作替代方案或交叉验证。
- HFODD v2.38j [arXiv:0903.1020](https://arxiv.org/abs/0903.1020)、v3.06h [arXiv:2104.08255](https://arxiv.org/abs/2104.08255)（检索）：
  重叠只用于追踪阻塞态。Schunck 等 0910.2164：时间奇场带来约 100–200 keV 的 EFA 差别。
- DRHBc 奇核：Pan 等 [arXiv:2205.01329](https://arxiv.org/abs/2205.01329)、质量表 [arXiv:2402.02935](https://arxiv.org/abs/2402.02935)（检索）：
  EFA＋自动阻塞（逐候选取最低能），可作为 O17 比对基准。

结论：在固定数目宇称扇区内做梯度极小即可，不需要对角化选准粒子。
要指定 1d5/2，应借助守恒量扇区（如轴对称 Ω=5/2）或多初态取最低。
列重叠认证身份没有文献依据。

## 3. NN 求解 HFB／相对论 DFT

- Li 等, PRL 133, 076401 (2024)：能量变分 NN-DFT（本项目 RHF 版依据）。
- Yoshimura–Sekizawa 2026, [arXiv:2609.00836](https://arxiv.org/abs/2609.00836)（已开）：KS 版，后处理正交化，无配对。
  按检索摘要，HFB 列为后续工作，难点在准粒子正交（摘要页本身未提 HFB）。
  ETF 版 [arXiv:2604.25759](https://arxiv.org/abs/2604.25759)（检索）。
- Wang–Naito–Li–Liang, EPJA (2025), [arXiv:2412.03090](https://arxiv.org/abs/2412.03090)（已开）：DNN 解 Dirac 方程，用 IHM 避免塌缩，单粒子固定势。
- Dirac Woods–Saxon PINN, CTP 2025, [doi:10.1088/1572-9494/ae1a5c](https://doi.org/10.1088/1572-9494/ae1a5c)（检索）。
- Lou 等, PRX 14, 021030, [arXiv:2305.06989](https://arxiv.org/abs/2305.06989)（检索）：AGPs FermiNet（VMC），可作配对参数化参考。

未找到含配对、以能量变分训练的 NN 核 HFB／RHB／RHFB 先例，本项目属于新构造。

## 4. 固定基 HFB 的数目约束

- Egido–Lessing–Martin–Robledo, NPA 594, 70 (1995)（检索）：约束 HFB 共轭梯度。
- Staszczak–Stoitsov–Baran–Nazarewicz, EPJA 46, 85 (2010), [arXiv:1006.4137](https://arxiv.org/abs/1006.4137)（检索）：
  增广拉格朗日（ALM）可精确满足约束；二次罚项不精确。
- Ryssens–Bender–Heenen, EPJA 55, 93 (2019)（heavy-ball；arXiv 号未核实）。
- Sala–Colò 2026, [arXiv:2606.17652](https://arxiv.org/abs/2606.17652)（摘要无数目约束细节）。
- 未找到梯度／NN 框架下的 VAP 先例；Robledo–Bertsch 正则回缩仍是平均数目约束的主要依据。

## 推荐路线（按 Codex 现有思路）

1. 参数化：复 QR 的 Q＋活动空间 Thouless/指数参数；奇态交换列＋Pfaffian 检查。依据：Robledo–Bertsch；Robledo–Bernard–Bertsch。
2. 无海：先实现 F(h[ρ])^{N_F} 多项式滤波（Li–Ren–Zhao）或逆 H 损失项；sign 有理投影保留为对照（PEXSI 极点展开 [arXiv:0906.1319](https://arxiv.org/abs/0906.1319)）。
3. 投影自洽的数学表述：参照 Séré 2023；需独立证明在核场强下成立。
4. 数目：正则回缩写回网络＋ALM 乘子（训练器已有 `number_augmented_lagrangian` 选项，O17 驱动当前设为 False）。
5. 阻塞：按 Ω 扇区多初态取最低；与 DRHBc EFA 自动阻塞和 Kasuya–Yoshida 时间奇约束交叉验证。

## 无文献直接依据，需独立数值验证

- H_BdG 的双侧狄拉克海剔除（如 sign(H)·sign(H²−Ec²)）。
- 对 PKA1 Fock＋D1S 的 IHM／滤波与 Serra–Ring／Geng–Long 无海处方的等价性。
- 全空间 [H,R] 残差作损失的极小性与分支保持。
- NN 生成 Bogoliubov 态的整体方案。
- Séré 框架前提在核强标量/矢量场下是否成立。
