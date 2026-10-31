# 组会 PPT 生成提示词

下面内容可整体提交给支持长文本和本地图片引用的 PPT 生成模型。

---

## 总提示词

请生成一份 **16:9、18页、面向核理论与计算核物理研究者、汇报时长20分钟** 的中文学术 PPT。

题目：

```text
基于 PRL 广义变分梯度的神经网络 DD-RMF 求解器
——从局域 Dirac Hamiltonian 到五参数集五核素验证
```

### 听众

- 熟悉 RMF/RHF、Dirac 方程、密度泛函和 SCF；
- 不一定熟悉图神经网络和自动微分；
- 需要公式严谨、结果诚实、方法边界清晰；
- 不需要科普式人工智能介绍。

### 核心叙事

按以下逻辑展开：

```text
为什么不拟合 SCF数据
 -> PRL 为什么选择 H 作为变分对象
 -> 如何把量子化学 PRL 框架替换成核物理 DD-RMF 前向链
 -> 球对称 0e 消息传递网络如何构造
 -> 严格物理约束如何嵌入
 -> 广义梯度如何回传
 -> 25组结果好在哪里
 -> 内部物理门为什么仍未完全通过
 -> 下一步如何推进到三维和 RHF
```

### 视觉风格

- 白色背景，深灰正文；
- 主色使用深红 `#9E2A2B`，辅助色使用蓝色 `#2457A6` 和绿色 `#2A7F62`；
- 不使用大面积深蓝、紫色渐变、发光球或装饰性背景；
- 公式使用 LaTeX 风格排版；
- 图表直接使用项目已有 PNG，不重绘或美化数据；
- 结果页以大图为主，每页最多 3 条结论；
- 标题 30–34 pt，正文 18–22 pt，图注 14–16 pt；
- 页面底部统一放简短来源或本地文件名；
- 禁止使用营销式口号、卡通 AI 图标和人形机器人图片。

### 学术准确性要求

必须明确：

1. 当前是 Hartree RMF，不是 RHF；
2. 当前是球对称 1D 径向版本，不是完整 3D E(3) 网络；
3. 网络不读取 SCF 标签；
4. 每个 epoch 仍执行 fixed-H Dirac 谱求解；
5. Chebyshev 只作为独立基线，不参与当前网络输出；
6. 25 组外部 SCF 误差很好，但内部驻定门为 0/25、公式不变量门为 10/25；
7. 不得写“所有物理约束均通过”或“已经完全替代 SCF”。

---

## 逐页内容

### 第1页：封面

标题：

```text
基于 PRL 广义变分梯度的神经网络 DD-RMF 求解器
```

副标题：

```text
从局域 Dirac Hamiltonian 到五参数集五核素验证
```

页脚：

```text
汇报人：______   课题组：______   日期：______
```

视觉：白底，右下角用细线绘制简化流程
`H_theta -> rho -> E,H_tilde -> gradient`，不要使用装饰图片。

演讲提示：本工作不是训练网络复现 SCF 标签，而是用核物理作用量直接训练 Hamiltonian 网络。

---

### 第2页：我们要解决什么问题

标题：

```text
能否让网络直接优化 RMF 的变分路径？
```

左侧画传统流程：

```text
势 -> Dirac -> 密度 -> 新势 -> 混合 -> SCF迭代
```

右侧画本工作：

```text
网络 H_theta -> 可微核物理 -> E,H_tilde -> 反向传播
```

底部三条：

- 不使用 SCF 势场和能量标签；
- 不直接拟合波函数；
- Core-1204 只在训练后做盲测。

演讲提示：目标不是否定 Core，而是建立一条可扩展到高维 Hamiltonian 表示的新变分路径。

---

### 第3页：PRL 提供的是优化框架，不是核物理前向链

标题：

```text
保留 PRL 的 Hamiltonian 变分，替换其量子化学前向物理
```

中心公式：

```text
G_H = dE/dH + lambda(H-H_tilde)
```

上方写 PRL：

```text
结构 -> DeepH-E3 -> H -> density matrix -> DFT energy
```

下方写本工作：

```text
(参数集,Z,N,r) -> radial 0e GNN -> H
 -> ADF Dirac -> 核密度 -> DD-RMF作用量
```

右下角来源：

```text
Y. Li et al., PRL 133, 076401 (2024)
```

演讲提示：迁移的是网络表示 Hamiltonian 和广义梯度，不是量子化学基组与电子泛函。

---

### 第4页：完整程序流程

标题：

```text
一次 epoch 内的核物理可微前向链
```

用横向流程图：

```text
18维径向输入
 -> 0e消息传递网络
 -> {S,V0,V3,VC}
 -> 四通道Dirac H
 -> 7点ADF本征态
 -> no-sea占据
 -> rho_s,rho_b,rho_b3
 -> DD耦合与重排
 -> 场方程
 -> E[H],H_tilde
 -> PRL VJP
```

用红色框标网络，用蓝色框标核物理，用绿色框标梯度。

演讲提示：每轮没有 Python SCF 外循环，但仍有一次 fixed-H Dirac 谱求解。

---

### 第5页：网络输入为何只剩18维

标题：

```text
删除对单核素恒定且无信息量的输入
```

左侧“删除”：

```text
Z/A
N/A
A/208
A^(1/3)/R
```

右侧“保留”：

```text
r/R
(r/R)^2
16个 sinc(k r/R) 径向嵌入
```

底部说明：

```text
Z,N仍进入占据数、密度源、Coulomb和DD-RMF泛函，
只是不用作恒定节点特征。
```

演讲提示：A/208 对单核素训练不能提供任何节点间区分，只能重复偏置。

---

### 第6页：球对称 0e 消息传递网络

标题：

```text
DeepH-E3 思想在球对称标量扇区的约化
```

绘制镜像链：

```text
-R ... -r2 -r1 | +r1 +r2 ... +R
```

标注：

- 402 个镜像节点；
- 每点连接自身及左右各3点；
- 约2802条有向边；
- 16个 Gaussian 边长基；
- `H(-r)=H(r)`。

右侧网络参数：

```text
hidden = 64 x 0e
3 edge updates
3 node updates
4 output heads
SiLU
139652 parameters
```

演讲提示：球形局域势只有0e；角动量由后端的 kappa 分块承担。

---

### 第7页：消息块和输出初始化

标题：

```text
边更新和节点更新都必须真正参与 Hamiltonian 输出
```

展示公式：

```text
e_ij' = SiLU[Norm(e_ij + F_e(e_ij,h_i,h_j)/sqrt(2))]
m_i   = mean_j F_m(h_i,h_j,e_ij')
h_i'  = SiLU[Norm(h_i + F_v(h_i,m_i)/sqrt(2))]
```

下方：

```text
4个线性输出头零初始化
=> epoch 1 从解析 Woods-Saxon 初值开始
```

演讲提示：此前普通独立 MLP 收敛慢；共享消息传递让所有通道利用共同径向环境。

---

### 第8页：RMF 四通道是硬约束

标题：

```text
网络不能离开核物理 Hamiltonian 通道空间
```

展示四条公式：

```text
VPS_n = V0 + S + V3
VMS_n = V0 - S + V3 - 2M_n/(hbar c)
VPS_p = V0 + S - V3 + VC
VMS_p = V0 - S - V3 + VC - 2M_p/(hbar c)
```

旁注：

- 中子/质子质量严格区分；
- 最终物理 `S,V0,V3,VC` 取密度重构结果；
- 内部变分分解不直接作为物理输出。

---

### 第9页：无穷远和原点边界

标题：

```text
边界条件嵌入表示层，而不是加入经验罚函数
```

三栏：

1. 原点：
   ```text
   H(-r)=H(r), dH/dr|0=0
   ```
2. 有质量场：
   ```text
   phi~C exp(-mu r)/r
   phi'+(mu+1/R)phi=0
   ```
3. Coulomb：
   ```text
   VC~C/r
   VC'+VC/R=0
   ```

底部：

```text
有限盒外作用量解析补回到无穷远
```

---

### 第10页：7点 ADF 避免假态

标题：

```text
Hermitian 非中心差分块是 Dirac 求解的关键
```

主公式：

```text
B1 = -D_backward + kappa/r
B2 =  D_forward  + kappa/r
```

```text
H_kappa = [[V+S, B1],
           [B2, V-S-2M]]
```

列出：

- `B1=B2^T`；
- `H=H^T`；
- no-sea；
- 节点数；
- N/Z 守恒；
- 轨道归一；
- 谱残差约 `1e-13`。

演讲提示：自动微分不自动消除假态，假态由离散结构和分支约束控制。

---

### 第11页：完整 DD-RMF 作用量

标题：

```text
训练目标是完整无质心 DD-RMF 作用量
```

左侧列出密度：

```text
rho_s, rho_b, rho_b3
```

中间列出场：

```text
sigma, omega, rho, Coulomb
```

右侧突出重排：

```text
Sigma_R =
sigma rho_s dg_sigma/drho_b
+ omega rho_b dg_omega/drho_b
+ rho rho_b3 dg_rho/drho_b
```

底部红字：

```text
无 SCF 标签、无经验 E/A、无半径窗口、无质心修正训练项
```

---

### 第12页：PRL 梯度如何进入 PyTorch

标题：

```text
物理损失不是一个随意加权的标量损失
```

展示：

```text
G_H = dE/dH + lambda(H-H_tilde)
```

```text
lambda = RMS(dE/dH)/RMS(H-H_tilde)  # 第一轮确定后固定
```

```text
L_sur = sum[H stop_gradient(G_H)]
dL_sur/dH = G_H
```

右下角强调：

```text
不除以 Ngrid
L_sur 数值不用于判断收敛
```

演讲提示：最佳点同时监控能量梯度和 Hamiltonian 自洽残差，不能让两项互相抵消。

---

### 第13页：统一训练配置

标题：

```text
25组计算使用同一网络与优化配置
```

表格：

| 参数 | 数值 |
|---|---:|
| epochs | 2000 |
| optimizer | Adam |
| initial LR | 2.5e-4 |
| scheduler | ReduceLROnPlateau |
| factor | 0.5 |
| patience/cooldown | 20/10 |
| dtype | float64 |
| activation | SiLU |
| GPUs | 2×Tesla T4 |
| 并发 | 每卡3进程 |

旁注：

```text
所有25组 best_epoch=2000
最终 LR≈1.53e-8
```

---

### 第14页：25组总体结果

标题：

```text
能量、半径和占据谱与 Core-1204 SCF 高度一致
```

用大号数字：

```text
平均 |Delta(E/A)| = 4.51e-4 MeV
平均 |Delta R_m| = 5.88e-5 fm
平均 occupied-level MAE = 0.0208 MeV
```

下方表格：

```text
V_n RMSE = 0.359 MeV
S_n RMSE = 0.445 MeV
V_p RMSE = 0.348 MeV
S_p RMSE = 0.445 MeV
```

右下角：

```text
25/25 completed
```

---

### 第15页：参数集和核素链

标题：

```text
五个参数集上的核素链趋势
```

放置两张图，优先 PKDD：

```text
outputs/rmf_five_by_five_prl_network_gpu_2000/pkdd/nuclear_chain_observables.png
outputs/rmf_five_by_five_prl_network_gpu_2000/pkdd/nuclear_chain_vs_errors.png
```

图下注释：

- Ca40 在多个参数集上的 S/V 误差相对较大；
- 重核能量差约为 keV/A 量级；
- 半径误差保持在 `1e-4 fm` 左右或更低。

不要改变图中的数据或坐标。

---

### 第16页：PKDD/O16 S/V 与单粒子谱

标题：

```text
代表性案例：PKDD/O16
```

页脚注明：本页图和数值来自昨天完成的 25 组批量目录；独立单核素验证中的
ADF/shooting 与波函数检查使用另一份同设置参考文件，最后几位不可交叉混用。

左侧放图：

```text
outputs/rmf_five_by_five_prl_network_gpu_2000/pkdd/O16/scf_comparison/vs_dpl_scf.png
```

右侧数据：

```text
Delta(E/A) = -8.49e-5 MeV
Delta R_m  = +1.97e-5 fm
level MAE  = 0.014238 MeV
ADF/shooting MAE = 2.43e-4 MeV
min wavefunction overlap = 0.9999978
```

演讲提示：ADF 离散误差远小于最终网络与 SCF 的势差。

---

### 第17页：必须诚实报告的未完成问题

标题：

```text
外部结果很好，但严格内部驻定门尚未通过
```

用对比布局：

左侧绿色“已达到”：

- 25/25 有结果；
- 能量/半径/谱接近 SCF；
- Hermiticity 和谱残差正常；
- 网络无 SCF 标签。

右侧红色“未达到”：

```text
internal stationarity gate = 0/25
formula-invariant gate = 10/25
best monitor = 0.0142–0.0992
```

底部结论：

```text
外部物理量一致性不能替代内部变分驻点证明
```

演讲提示：这是下一阶段最重要的问题，不要把门限失败隐藏掉。

---

### 第18页：结论与下一步

标题：

```text
结论：PRL Hamiltonian 网络路径在球形 DD-RMF 上可行
```

三条结论：

1. 纯消息传递网络可以在无 SCF 标签下优化局域 RMF Hamiltonian；
2. 严格核物理通道、ADF、边界和作用量能够嵌入可微前向链；
3. 25组外部结果达到较高精度，但内部驻定性仍需继续解决。

下一步：

```text
梯度检查与停止准则
 -> 多核素联合训练
 -> 三维完整E(3)网络
 -> 可微RHF/Fock接口
```

页脚：

```text
Q&A
```

---

## 生成后的自检清单

生成 PPT 后逐项检查：

- 是否把当前方法误写成 RHF；
- 是否误写成不求解 Dirac 方程；
- 是否声称网络使用 SCF 标签；
- 是否把 Chebyshev 结果写成网络结果；
- 是否保留 `B1/B2` 的正确正负号；
- 是否保留中子/质子不同质量；
- 是否写清训练能量无质心修正；
- 是否展示 0/25 内部门和 10/25 公式门；
- 是否使用了正确的 25 组汇总数字；
- 是否直接引用本地 S/V 图而非生成虚构曲线。
