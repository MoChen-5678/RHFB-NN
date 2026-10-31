# 基于 PRL 广义变分梯度的神经网络 DD-RMF 求解器

## 组会汇报报告

### 一句话结论

当前程序不是用神经网络拟合 SCF 数据，也不是用 Chebyshev 系数冒充网络。它使用一个球对称
`0e` 标量消息传递网络直接参数化局域 Dirac Hamiltonian，通过完整可微 DD-RMF 核物理前向链
计算总能量和自洽重构 Hamiltonian，再用 PRL 的广义梯度

```text
G_H = dE/dH + lambda (H - H_tilde)
```

无监督优化网络参数。

五个 RMF 参数集、五个双幻核共 25 组的 2000 轮计算已经完成。与独立 Core-1204 SCF 比较，
能量和半径精度很好，单粒子谱与 S/V 势总体接近；但是当前严格内部驻定门仍未通过，因此现阶段
应表述为“得到高精度外部一致性结果的可微神经网络 RMF 原型”，不能表述为“已经数学完备地
证明所有变分残差收敛”。

---

## 1. 研究问题

传统球形 RMF 计算通过外部 SCF 固定点迭代：

```text
初始势 -> Dirac 方程 -> 波函数与密度 -> 场方程 -> 新势 -> 混合 -> 重复
```

本工作的目标不是重写或替代已经正确的 Core-1204 SCF，而是研究另一条求解路径：

1. 用神经网络表示局域 Hamiltonian；
2. 把核物理作用量和所有约束放进可微前向传播；
3. 直接对 Hamiltonian 的变分路径做优化；
4. 训练不读取 SCF 能量、势场或波函数标签；
5. 训练完成后才调用 Core-1204 做外部盲测。

这与普通 PINN 的区别是：网络不直接拟合某个波函数，也不把若干近似 PDE 残差任意加权成
经验损失。变分对象是局域 Hamiltonian，物理梯度来自完整能量泛函。

---

## 2. 方法边界

当前正式版本实现：

- 球对称；
- 静态、时间反演；
- no-sea；
- Hartree RMF；
- 密度依赖 DD-RMF 参数集；
- 单参数集、单核素独立训练。

支持的五个参数集：

```text
DD-ME1, DD-ME2, PKDD, TW99, DD-LZ1
```

测试核素：

```text
O16, Ca40, Ca48, Sn132, Pb208
```

当前版本不包含：

- RHF 的非局域 Fock 交换；
- 配对；
- 形变；
- 三维空间网络；
- 一个网络跨核素直接泛化。

RHF 接口保留，但不能把当前 RMF 结果称为 RHF 结果。

---

## 3. PRL 思路如何迁移到核物理

PRL 133, 076401 的核心不是某个量子化学基组，而是三步结构：

```text
结构信息 -> 神经网络 H_theta
H_theta -> density matrix / density -> E[H_theta], H_tilde
dE/dH + lambda(H-H_tilde) -> 网络参数梯度
```

本程序保留这三个逻辑层，但将前向物理替换为核物理：

```text
(参数集, Z, N, 径向网格)
 -> 球对称消息传递网络
 -> 局域 Dirac Hamiltonian H_theta
 -> 7点 ADF Dirac 谱
 -> no-sea 占据态
 -> 标量/矢量核密度
 -> 密度依赖耦合与重排项
 -> sigma/omega/rho/Coulomb 场
 -> 完整 DD-RMF 作用量 E[H_theta]
 -> 重构 Hamiltonian H_tilde
```

因此：

- 神经网络结构借鉴 PRL/DeepH-E3 的 Hamiltonian 消息传递思想；
- 前向链严格使用核物理 DD-RMF，而不是量子化学 DFT；
- Core-1204 不在训练图中；
- 每个 epoch 只有一次可微物理前向传播，不执行 Python SCF 外迭代。

---

## 4. 网络输入

这是单参数集、单核素模型，因此所有径向节点上恒定的输入都被删除：

```text
不输入 Z/A
不输入 N/A
不输入 A/208
不输入常数 A^(1/3)/R
```

`Z` 和 `N` 仍然严格进入：

- 中子、质子占据数；
- 密度源；
- Coulomb 电荷；
- 参数集质量和耦合；
- DD-RMF 作用量。

它们只是不作为对单核素网络无信息量的恒定节点特征。

每个正半轴径向节点的 18 维输入为：

```text
r/R
(r/R)^2
16个 l=0 球贝塞尔型径向嵌入 sinc(k r/R), k=1,...,16
```

固定径向嵌入仅用于表达坐标，不是可训练 Chebyshev 变分旁路。

---

## 5. 球对称等变消息传递网络

### 5.1 为什么只保留 0e

球对称 RMF 的局域 `S,V0,V3,VC` 都是旋转群下的偶宇称标量，即 `0e`。在这个子空间内，
E(3) 等变张量积严格退化为标量乘法和标量消息传递。

角动量信息并没有被删除，而是由核物理 Dirac 求解器中的 `kappa` 分块承担。把量子化学
原子轨道网络的 `1o,2e,...` 特征直接搬到一维球形局域势上，反而会引入当前物理问题中不存在
的自由度。

### 5.2 镜像径向图

Core 网格正半轴有 201 点。网络内部构造镜像图：

```text
[-R,...,-r_min, +r_min,...,+R]
```

共 402 个节点。正负节点使用完全相同的偶宇称特征，边集合严格镜像，所以网络输出满足：

```text
H(-r) = H(r)
dH/dr |_(r=0) = 0
```

原点规则性是表示层硬约束，不是损失函数中的软罚项。

### 5.3 图连接

每个节点连接自身和左右各三个邻居：

```text
j in [i-3, ..., i+3]
```

这与 7 点非中心 ADF 的局域作用范围一致。完整镜像图含约 2802 条有向边。

边长使用 16 个 Gaussian 基展开，再映射到隐藏边特征。

### 5.4 网络主体

```text
节点输入维数: 18
节点隐藏表示: 64 x 0e
边隐藏表示:   64 x 0e
消息块数量:   3
每个消息块:   1个边更新 + 1个节点更新
激活函数:     SiLU
归一化:       LayerNorm
输出头:       4个线性 Hamiltonian 物理通道头
参数量:       139652
```

每个消息块先更新边：

```text
e_ij' = SiLU[Norm(e_ij + F_e(e_ij,h_i,h_j)/sqrt(2))]
```

再聚合边消息更新节点：

```text
m_ij = F_m(h_i,h_j,e_ij')
m_i  = mean_j(m_ij)
h_i' = SiLU[Norm(h_i + F_v(h_i,m_i)/sqrt(2))]
```

所有 `F_e,F_m,F_v` 都由线性层和 SiLU 组成。

四个输出头零初始化，因此第一个 epoch 严格从解析 Woods-Saxon 初始 Hamiltonian 开始，不需要
减去第二次网络前向输出。

---

## 6. 网络输出与 RMF 通道

网络先给出四个物理坐标：

```text
S  : 等标量自能
V0 : 等标量时间样矢量自能，含重排
V3 : 同位旋第三分量时间样矢量自能
VC : 质子 Coulomb 势
```

它们在前向传播中严格组装为四个质量移位 Dirac 通道：

```text
VPS_n = V0 + S + V3
VMS_n = V0 - S + V3 - 2 M_n/(hbar*c)

VPS_p = V0 + S - V3 + VC
VMS_p = V0 - S - V3 + VC - 2 M_p/(hbar*c)
```

中子和质子使用各参数集定义的不同裸质量，不能用同一个近似质量替代。

Hamiltonian 对 `V0,V3,VC` 的分解存在内部冗余。因此：

- `network_variational_components.npz` 保存组装 `H_theta` 的内部网络坐标；
- `physical_components.npz` 保存由最终密度重建的物理 `S,V0,V3,VC`；
- 组会中的势场物理比较使用后者或直接使用中子/质子的总 `S,V`。

---

## 7. 势场的硬边界

### 7.1 原点

镜像图保证所有局域势为偶函数，解析满足中心零导数。

### 7.2 有质量场

`S,V0,V3` 的源外行为按 Yukawa 解处理：

```text
phi(r) = C exp(-mu r)/r
mu = m_phi/(hbar*c)
```

盒边满足 Robin 条件：

```text
dphi/dr + (mu + 1/R) phi = 0
```

作用量还包含从有限盒边到无穷远的解析外部项。

### 7.3 Coulomb 场

源外满足：

```text
VC(r) = C/r
dVC/dr + VC/R = 0
```

基础势使用有限电荷球并连接 `Z alpha/r`，可训练修正及其导数在盒边消失。

这些边界不通过经验罚系数训练，网络不能违反。

---

## 8. 7 点非中心 ADF Dirac 块

对每个核子种类和 `kappa`，显式构造：

```text
B1 = -D_backward + kappa/r
B2 =  D_forward  + kappa/r

H_kappa = [[V+S,       B1],
           [B2,   V-S-2M]]
```

离散算符满足：

```text
-D_backward = D_forward^T
B1 = B2^T
H_kappa = H_kappa^T
```

这样避免中心差分产生的费米子倍增/假态问题。程序同时约束：

- 大、小分量原点奇偶；
- no-sea 正能分支；
- 节点数；
- 中子数和质子数；
- 轨道归一；
- 谱残差；
- Hamiltonian Hermiticity。

PyTorch 自动微分只负责传播已选物理分支的梯度。自动微分本身不能消除假态，假态由 ADF 结构、
Hermitian 块和谱分支约束共同控制。

---

## 9. DD-RMF 密度、耦合与重排

占据轨道生成：

```text
rho_s  = sum_a v_a^2 (2j_a+1)/(4*pi*r^2) (G_a^2-F_a^2)
rho_b  = sum_a v_a^2 (2j_a+1)/(4*pi*r^2) (G_a^2+F_a^2)
rho_b3 = rho_n-rho_p
```

密度依赖耦合为：

```text
g_sigma(rho_b), g_omega(rho_b), g_rho(rho_b)
```

对密度依赖耦合的变分自动产生重排自能：

```text
Sigma_R =
    sigma rho_s  dg_sigma/drho_b
  + omega rho_b  dg_omega/drho_b
  + rho   rho_b3 dg_rho/drho_b
```

`Sigma_R` 只进入共同时间样矢量自能，不作为一个独立能量项重复加入。

场方程为：

```text
(-Delta+m_sigma^2) sigma = -g_sigma rho_s
(-Delta+m_omega^2) omega = +g_omega rho_b
(-Delta+m_rho^2)   rho   = +g_rho rho_b3
-Delta A0                 = 4*pi*alpha*rho_p
```

---

## 10. 完整作用量与质心修正

训练能量来自完整静态 DD-RMF 作用量，包含：

- 占据 Dirac 轨道的精确 ADF 动能；
- sigma、omega、rho、Coulomb 场能；
- 场与密度源耦合；
- 密度依赖耦合及重排；
- Yukawa/Coulomb 盒外解析作用量；
- 轨道归一拉格朗日乘子。

不使用：

- Thomas-Fermi 动能；
- 实验 `E/A` 标签；
- 经验 `-8 MeV` 目标；
- SCF 势场监督；
- 经验半径窗口；
- 人工平滑损失；
- 质心修正训练项。

训练和主对比统一使用：

```text
E_total_no_com
```

Core-1204 的质心修正只作为最终独立观测量，不能混入训练作用量。

---

## 11. PRL 广义梯度

每轮可微传播得到：

```text
E[H_theta]
H_tilde[rho(H_theta)]
dE/dH_theta
```

网络接收的 Hamiltonian 空间梯度为：

```text
G_H = w_E dE/dH + lambda(H_theta-H_tilde)
w_E = 1
```

第一项寻找能量泛函驻点。第二项补全仅靠占据态能量不能唯一确定的 Hamiltonian 信息，并要求
Hamiltonian 与其自身密度重构结果一致。

### 11.1 lambda

不同核素的 Hamiltonian 尺度不同。网络模式在第一轮计算：

```text
lambda =
RMS(w_E dE/dH) / RMS(H_theta-H_tilde)
```

随后固定该值，不在训练中动态追逐残差。

25 组计算中 `lambda` 范围约为：

```text
2.28e-4 到 2.71e-3 MeV^-1
```

### 11.2 PyTorch 反向传播

构造只用于 VJP 的标量：

```text
L_sur = sum[H_theta * stop_gradient(G_H)]
```

因此：

```text
dL_sur/dH_theta = G_H
```

不能再除以 Hamiltonian 元素数。此前的 `/Ngrid` 会把论文规定的梯度任意缩小。

`L_sur` 的数值不是物理能量，不能用它选择最佳轮。最佳轮监控：

```text
r_E = RMS(dE/dH) / RMS_1(dE/dH)
r_H = RMS(H-H_tilde) / RMS_1(H-H_tilde)
monitor = max(r_E,r_H)
```

这样避免两项梯度相消后产生“联合梯度很小”的假收敛。

---

## 12. 优化器和训练设置

当前 25 组统一设置：

```text
epochs               = 2000
optimizer            = Adam
beta                 = (0.9, 0.999)
initial learning rate= 2.5e-4
scheduler            = ReduceLROnPlateau
decay factor         = 0.5
patience             = 20
cooldown             = 10
activation           = SiLU
dtype                = float64
```

所有 25 组的最佳 monitor 都出现在第 2000 轮，最终学习率都降至约 `1.53e-8`。这说明：

1. 训练是稳定下降的；
2. 当前调度器已经把步长降得很小；
3. 2000 轮仍是边界最优点，后续应研究更合理的停止准则和二阶段优化，而不是简单无限增加轮数。

---

## 13. 25 组批量结果

### 13.1 总体误差

相对于独立 Core-1204 SCF，25 组统计如下：

| 指标 | 平均绝对误差 | 中位绝对误差 | 最大绝对误差 |
|---|---:|---:|---:|
| `E/A no CoM` | `4.51e-4 MeV` | `3.16e-4 MeV` | `1.35e-3 MeV` |
| 物质半径 | `5.88e-5 fm` | `5.05e-5 fm` | `2.13e-4 fm` |
| 电荷半径 | `8.50e-5 fm` | `4.20e-5 fm` | `4.06e-4 fm` |
| 占据单粒子能级 MAE | `2.08e-2 MeV` | `8.18e-3 MeV` | `7.57e-2 MeV` |
| 中子 `V` RMSE | `0.359 MeV` | `0.292 MeV` | `0.903 MeV` |
| 中子 `S` RMSE | `0.445 MeV` | `0.332 MeV` | `1.402 MeV` |
| 质子 `V` RMSE | `0.348 MeV` | `0.266 MeV` | `0.954 MeV` |
| 质子 `S` RMSE | `0.445 MeV` | `0.332 MeV` | `1.402 MeV` |

最大误差位置：

- `E/A`：DD-ME2/Pb208，约 `1.346 keV/A`；
- 物质半径：DD-LZ1/Ca48，约 `2.13e-4 fm`；
- 电荷半径：DD-LZ1/Sn132，约 `4.06e-4 fm`；
- 占据能级 MAE：DD-LZ1/O16，约 `75.7 keV`；
- S/V 势 RMSE：DD-LZ1/Ca40，`S` 最大约 `1.402 MeV`。

### 13.2 PKDD/O16 代表结果

以下数字来自独立单核素验证目录
`outputs/pkdd_o16_prl_e3_rmf_2000`，用于同时展示能量、半径、
ADF/shooting 和波函数检查。昨天完成的 25 组批量目录采用各自生成的
Core-1204 参考文件，因此 PKDD/O16 的最后几位略有差异；批量表中对应结果为
`DPL E/A=-7.31593332 MeV`、`SCF E/A=-7.31584846 MeV`、
`Delta E/A=-8.49e-5 MeV`。两组结论一致，但汇报时不应混用参考文件。

```text
DPL E/A no CoM = -7.315933323 MeV
SCF E/A no CoM = -7.315855519 MeV
差值             = -7.78e-5 MeV/A

DPL 物质半径     = 2.566356865 fm
SCF 物质半径     = 2.566336879 fm
差值             = 2.00e-5 fm

DPL 电荷半径     = 2.699894757 fm
SCF 电荷半径     = 2.699861173 fm
差值             = 3.36e-5 fm

占据能级 MAE     = 0.01424 MeV
```

PKDD/O16 的 S/V RMSE：

```text
V_n = 0.323 MeV
S_n = 0.315 MeV
V_p = 0.308 MeV
S_p = 0.315 MeV
```

固定 SCF 势上的 ADF/shooting 检查：

```text
6/6 占据态匹配
能级 MAE = 2.43e-4 MeV
最小波函数重叠 = 0.9999978
节点数错配 = 0
```

这说明网络误差和 ADF 离散误差可以分开：ADF 本身在同一势上的误差远小于最终 DPL-SCF 势差。

---

## 14. 如何正确解读结果

### 14.1 已经成立的结论

1. 纯消息传递神经网络参与了 Hamiltonian 优化；
2. 正式网络没有可训练 Chebyshev 旁路；
3. 训练没有读取 SCF 标签；
4. 五个参数集、五个核素都得到了接近 SCF 的能量、半径和势场；
5. Dirac 块 Hermitian，谱残差约 `1e-13`；
6. no-sea、粒子数、节点和归一约束正常；
7. PKDD/O16 外部 SCF 验收通过。

### 14.2 不能过度声明的结论

25 组的当前内部物理门：

```text
internal stationarity gate: 0/25
formula-invariant gate:    10/25
```

原因不是能量或谱已经明显错误，而是当前门限非常严格：

- 内部门要求归一化联合残差 `monitor <= 1e-3`；
- 实际 25 组最佳 monitor 范围约 `0.0142 到 0.0992`；
- 即相对第一轮下降约 90% 到 98.6%，但未达到设定门；
- 部分有限差分/autograd 梯度检查相对或绝对误差略高于硬阈值；
- 外部物理量接近 SCF 不能替代内部驻定性证明。

因此组会上应说：

> 当前网络已经找到与 SCF 物理观测高度一致的 Hamiltonian 区域，但严格变分驻点和梯度检查
> 仍需继续收紧。外部结果好不等于内部证明已经完成。

---

## 15. Chebyshev 基线在当前项目中的角色

Chebyshev `direct` 模式仍保留，但只用于：

- 验证离散作用量；
- 验证 Hamiltonian 梯度；
- 检查物理空间是否存在可达驻点；
- 为神经网络结果提供数值基线。

它不是当前 25 组结果的训练主体。当前 `variation_mode` 均为：

```text
network
```

神经网络参数量为 139652，而不是四通道 Chebyshev 系数。

---

## 16. 代码流程

### 16.1 主入口

```text
python -m dpl_rhf.cli.prl_train
```

### 16.2 关键文件

| 文件 | 作用 |
|---|---|
| `dpl_rhf/models/hamiltonian_net.py` | 镜像径向图、消息传递网络、硬边界和通道组装 |
| `dpl_rhf/backends/torch_rmf.py` | 7点 ADF Dirac 谱、占据、密度和可微 RMF 前向链 |
| `dpl_rhf/functionals/pkdd_action.py` | 五参数集 DD-RMF 完整作用量、耦合、重排和场方程 |
| `dpl_rhf/training/surrogate_gradient.py` | PRL 广义 Hamiltonian 梯度 |
| `dpl_rhf/training/prl_hamiltonian_trainer.py` | Adam 训练、梯度诊断、检查点和输出 |
| `dpl_rhf/cli/compare_core1204.py` | 训练后 Core-1204 SCF 与单粒子谱盲测 |
| `dpl_rhf/cli/plot_vs_comparison.py` | 中子/质子 S/V 对比图 |
| `dpl_rhf/cli/batch_rmf.py` | 5×5、多 GPU、每卡三进程批量调度 |

### 16.3 输出

每个核素目录包含：

```text
model.pt
dpl_hamiltonian.npz
reconstructed_hamiltonian.npz
network_variational_components.npz
physical_components.npz
dpl_orbitals.npz
density_diagnostics.npz
field_diagnostics.npz
history.csv
physics_residuals.csv
gradient_alignment.csv
summary.json
formula_invariants.json
scf_comparison/
  detailed_comparison.json
  occupied_levels.csv
  vs_dpl_scf.png
  vs_dpl_scf.pdf
```

---

## 17. 下一步研究计划

### 优先级 1：内部驻定性

- 重新定义对不同核素更可比的绝对/相对物理门；
- 检查最后学习率过小但 monitor 仍未过门的原因；
- 对能量梯度和重构梯度分别做通道预条件；
- 保持 PRL 梯度形式，不增加经验 SCF 拟合项；
- 增加可恢复 checkpoint，分析 500、1000、2000 轮路径。

### 优先级 2：梯度验证

- 改进有限差分步长扫描；
- 区分相对误差在近零梯度下的放大；
- 对每个 `S,V0,V3,VC` 通道做方向导数检查；
- 将门限建立在浮点误差和物理尺度上，而不是单一固定阈值。

### 优先级 3：模型泛化

- 当前一个模型只对应一个参数集和一个核素；
- 下一阶段才考虑加入真正有变化的结构条件；
- 使用多个核素共同无监督训练时，再引入 `Z,N` 的结构嵌入；
- 不能重新加入对单核素恒定且无意义的 `A/208`。

### 优先级 4：三维和 RHF

- 三维版本恢复完整 E(3) irreps 和角向消息传递；
- 保留当前球形 `0e` 网络作为严格基准；
- RHF 需要非局域 Fock Hamiltonian 块和相应可微能量泛函；
- 在 RMF 内部门通过前，不宣称 RHF 已实现。

---

## 18. 组会建议讲法

建议 20 分钟：

1. **2 分钟：问题和目标**  
   不是拟合 SCF，而是用网络优化 Hamiltonian 的变分路径。

2. **3 分钟：PRL 框架与核物理迁移**  
   强调只迁移优化框架，前向链是 DD-RMF。

3. **4 分钟：网络结构**  
   镜像径向图、64×0e、3 个边块和 3 个节点块、四通道头。

4. **4 分钟：严格物理约束**  
   通道公式、Yukawa/Coulomb 边界、7点 ADF、no-sea、完整作用量。

5. **3 分钟：PRL 广义梯度**  
   解释 `dE/dH + lambda(H-H_tilde)` 和 exact VJP。

6. **3 分钟：25 组结果**  
   先总体表，再展示 PKDD/O16 的 S/V 图和单粒子谱。

7. **1 分钟：局限和下一步**  
   外部误差好，但内部驻定门 0/25，必须继续解决。

---

## 19. 汇报中应避免的表述

不要说：

- “网络完全不求解 Dirac 问题。”  
  当前可微前向链每轮要做 fixed-H Dirac 谱求解。

- “已经完全替代 SCF。”  
  当前展示的是另一种变分求解路径，Core SCF 仍用于盲测。

- “25 组所有物理约束都通过。”  
  外部结果很好，但严格内部驻定门没有通过。

- “这是 RHF 网络。”  
  当前是 Hartree RMF。

- “这是一个跨核素通用模型。”  
  当前每个参数集、每个核素独立训练。

- “S/V 内部四场分解由能量唯一决定。”  
  Hamiltonian 分解有冗余，最终物理场取密度重构结果。

---

## 20. 结果与图表位置

批量总报告：

```text
outputs/rmf_five_by_five_prl_network_gpu_2000/BATCH_COMPARISON.md
outputs/rmf_five_by_five_prl_network_gpu_2000/batch_summary.csv
outputs/rmf_five_by_five_prl_network_gpu_2000/batch_summary.json
```

各参数集核素链图：

```text
outputs/rmf_five_by_five_prl_network_gpu_2000/<model>/nuclear_chain_observables.png
outputs/rmf_five_by_five_prl_network_gpu_2000/<model>/nuclear_chain_vs_errors.png
```

PKDD/O16 代表图：

```text
outputs/rmf_five_by_five_prl_network_gpu_2000/pkdd/O16/scf_comparison/vs_dpl_scf.png
```

---

## 参考

1. Y. Li et al., “Neural-network Density Functional Theory Based on Variational Energy
   Minimization,” *Phys. Rev. Lett.* **133**, 076401 (2024).
2. DeepH-E3: E(3)-equivariant neural-network representation of DFT Hamiltonian.
3. 当前目录核物理书目、ADF 文献和 Core-1204 RMF 实现。
