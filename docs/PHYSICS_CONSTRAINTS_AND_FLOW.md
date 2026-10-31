# DPL-RMF 物理约束与程序流程总结

本文档总结当前 `fieldspace_rmf.py` 的物理约束、变分原理实现、7 点非中心差分实现和完整程序流程。目标是方便逐项检查：哪些约束已经进程序，哪些只是诊断量，哪些地方可能导致当前模型跑不准。

## 1. 程序目标

当前程序不是用 SCF 数据监督拟合网络，而是做无监督变分优化：

1. 输入核子数 `Z,N` 和 PKDD 参数集。
2. 神经网络输出四个局域 Dirac 势通道：
   - 中子 `V+S`
   - 中子 `V-S-2M`
   - 质子 `V+S`
   - 质子 `V-S-2M`
3. 由这些势构造径向 Dirac Hamiltonian。
4. 解出占据态波函数。
5. 由波函数计算密度。
6. 由密度通过 RMF 场方程重构 `sigma, omega, rho, Coulomb` 场。
7. 由重构场再得到重构势。
8. 最小化能量与物理残差。

Fortran SCF 只作为训练后的外部比较，不参与训练损失。

## 2. 变分原理

图片中的基本形式是：

```text
E_B = <phi_0|H|phi_0> - A M = E - A M
delta [ E - sum_a E_a int f_a^dagger f_a dr ] = 0
```

这里 `E_a` 是单粒子能量，也是轨道归一化约束的拉格朗日乘子。对于球对称径向波函数，当前程序把这个约束落实为：

```text
int dr [G_a(r)^2 + F_a(r)^2] = 1
```

程序中 `G,F` 在每次构造密度前都会重新归一化，因此归一化残差理论上接近机器精度。

## 3. 径向 Dirac 方程约束

图片给出的局域 Hartree 情况可以写成：

```text
E_a G_a = - (d/dr - kappa_a/r) F_a
          + [M + Sigma_S(r) + Sigma_0(r)] G_a

E_a F_a = + (d/dr + kappa_a/r) G_a
          - [M + Sigma_S(r) - Sigma_0(r)] F_a
```

如果包含 RHF 非局域交换，右端还会有 `Y_a(r), X_a(r)`。当前 PKDD 程序是局域 RMF/Hartree 版本，因此：

```text
X_a(r) = 0
Y_a(r) = 0
```

程序中实际使用 shifted Dirac 势：

```text
VPS = V + S
VMS = V - S - 2M
```

对应矩阵形式为：

```text
H = [[ VPS,        D_upper ],
     [ D_lower,    VMS     ]]
```

其中 `D_lower = D+ + kappa/r`，`D_upper = D_lower^T`。这样保证离散 Hamiltonian 严格厄米。

## 4. 7 点非中心差分

张颖论文的核心结论是：中心差分缺少中点信息，会产生 Dirac 假态；使用 asymmetric difference formula 可以消除假态，同时需要交替使用前向/后向结构保证厄米性。

论文正文列出：

```text
3 点 ADF:  O(h^2)
5 点 ADF:  O(h^4)
```

当前程序新增了 7 点 ADF，形式为 7 点非对称模板，内部点系数满足：

```text
d f_i / dr =
[-49/20 f_i
 + 6 f_{i+1}
 - 15/2 f_{i+2}
 + 20/3 f_{i+3}
 - 15/4 f_{i+4}
 + 6/5 f_{i+5}
 - 1/6 f_{i+6}] / h
```

它对 0 到 6 次多项式满足一阶导数矩条件，因此是 7 点、六阶精度的非中心差分。程序参数：

```bash
--derivative-order 7
```

默认已改为 7。可选值：

```text
1, 2, 4, 5, 6, 7
```

其中 `4/5` 都走 5 点 ADF，`6/7` 都走 7 点 ADF。

## 5. 边界条件约束

图片中分部积分后的边界项要求束缚态满足：

```text
r^2 (partial E / partial G_a') delta G_a |_{0,infty} = 0
r^2 (partial E / partial F_a') delta F_a |_{0,infty} = 0
```

当前程序在有限盒子 `Rbox` 中落实为：

```text
G_a(0) = 0
F_a(0) = 0
G_a(Rbox) = 0
F_a(Rbox) = 0
```

注意：不能把完整 `V-S-2M` 势也强行约束为 0，因为它在无穷远含有 `-2M` 常数项，质子通道还包含库仑尾。这个错误已经修正，边界残差现在只约束波函数。

## 6. RMF 自洽场约束

由占据态构造密度：

```text
rho_s, rho_v, rho_3, rho_p
```

再由球对称 Green 函数求解介子场：

```text
sigma(r)
omega(r)
rho(r)
coul(r)
```

PKDD 是密度依赖耦合，程序包含：

```text
g_sigma(rho_b)
g_omega(rho_b)
g_rho(rho_b)
rearrangement term Sigma_R
```

重构场再生成重构势：

```text
H_tilde = H[rho(G,F)]
```

程序中的自洽约束为：

```text
mean( ((H_trial - H_tilde) / scale)^2 )
```

这是当前最关键的物理闭环。如果这个残差降不下去，说明网络给出的势和由自身波函数密度推出的 RMF 势不一致。

## 7. 当前损失函数

当前总损失为：

```text
L =
  E_no_com/A / 50
  + lambda_reconstruct * potential_reconstruction
  + lambda_dirac       * dirac_residual
  + lambda_norm        * normalization_residual
  + lambda_boundary    * boundary_residual
```

默认/常用参数：

```text
lambda_reconstruct = 命令行必填
lambda_dirac       = 1e-3
lambda_norm        = 1.0
lambda_boundary    = 1.0
```

各项含义：

```text
potential_reconstruction:
    网络试探势 H_trial 与密度重构势 H_tilde 的差。

dirac_residual:
    || H_tilde f_a - E_a f_a ||^2 / ||f_a||^2。
    这是图片中 Euler-Lagrange 方程对应的本征方程残差。

normalization_residual:
    mean_a (int (G_a^2 + F_a^2) dr - 1)^2。

boundary_residual:
    G,F 在 r=0 和 Rbox 的边界残差。
```

注意：当前 `dirac_residual` 是用重构势 `H_tilde` 检查网络解出的轨道是否仍满足 Dirac 方程。由于轨道本来是从 `H_trial` 精确解出来的，所以这个残差本质上也在推动 `H_trial = H_tilde`。

## 8. 能量定义

当前主能量不使用质心修正：

```text
E_no_com = E_kinetic + E_direct
E/A = E_no_com / A
```

这符合之前要求：DPL 主结果和 SCF 比较先看无质心修正版本。Fortran 中的质心修正只应作为单独模式，不应混进当前主损失。

## 9. 完整程序流程

一次训练迭代的流程：

```text
1. 网络输出 H_trial(r)
   channels = [vps_n, vms_n, vps_p, vms_p]

2. 构造 7 点 ADF Dirac 矩阵
   D_lower = D+ + kappa/r
   D_upper = D_lower^T
   H_trial is Hermitian

3. 对每个 occupied branch 求本征态
   H_trial f_a = E_a f_a

4. 选择 no-sea 占据支
   通过初始 Woods-Saxon 参考态 overlap 选分支

5. 归一化 G_a,F_a
   int (G_a^2 + F_a^2) dr = 1

6. 由 G,F 计算密度
   rho_s, rho_v_n, rho_v_p, rho_3, rho_p

7. 由密度计算 PKDD 密度依赖耦合
   g_sigma, g_omega, g_rho, rearrangement

8. 用 Green 函数重构场
   sigma, omega, rho, Coulomb

9. 由场生成 H_tilde
   vps_n, vms_n, vps_p, vms_p

10. 计算物理约束
    potential_reconstruction
    dirac_residual
    normalization_residual
    boundary_residual

11. 计算无质心能量 E_no_com/A

12. 反向传播优化网络参数
```

训练结束后才会：

```text
1. 保存 DPL 势场、密度、轨道。
2. 可选调用 Fortran 固定势求单粒子谱。
3. 可选运行完整 SCF 做外部对比。
```

## 10. 当前已发现的问题

### 10.1 7 点 ADF 后 LBFGS 不稳定

短测结果显示：

```text
7 点 ADF + Adam 阶段可以保持半径在物理区间。
加入 LBFGS 后容易跳到弥散态，质子半径变大。
```

因此当前建议：

```bash
--derivative-order 7
--lbfgs-steps 0
--lr 5e-4 或更小
--lambda-reconstruct 1.0 起步
```

LBFGS 暂时不要作为 7 点 ADF 的默认优化器，除非后续加入信赖域、步长限制或约束投影。

### 10.2 自洽约束仍是软约束

当前 `H_trial = H_tilde` 是罚函数，不是严格拉格朗日乘子或投影约束。若 `lambda_reconstruct` 太小，网络会优先降低能量，可能跑向非物理解。

下一步更严格的做法应该是：

```text
Augmented Lagrangian:
L_aug = E + lambda^T C + mu/2 ||C||^2
C = H_trial - H_tilde
```

而不是只调一个固定罚因子。

### 10.3 目前仍是局域 RMF，不含 RHF 交换核

图片中的一般形式有：

```text
X_a(r), Y_a(r)
```

当前 PKDD Hartree 版本设为 0。如果要完全对应 RHF 教材方程，需要加入非局域交换项和对应的非局域能量泛函。

### 10.4 7 点边界模板需要继续检查

当前 7 点 ADF 在靠近右边界时自动切换为后向 7 点模板，内部点使用前向 7 点模板。矩阵用 `D+` 和 `D+^T` 保证厄米。

需要进一步检查：

```text
1. 高阶边界模板是否引入边界局域态。
2. 不同 mesh-step 下单粒子能级是否稳定。
3. O16, Ca40, Pb208 的势场 RMSE 是否比一阶 ADF 改善。
```

## 11. 推荐排查顺序

建议先不要看总能量，而按下面顺序检查：

```text
1. normalization_residual 是否接近 0。
2. boundary_residual 是否接近 0。
3. dirac_residual 是否稳定小。
4. potential_reconstruction 是否下降。
5. rms_matter 是否保持物理区间。
6. 再看 E/A。
7. 最后才和 SCF 单粒子谱、势场 RMSE 对比。
```

如果 `E/A` 下降但 `potential_reconstruction` 或半径变坏，那不是物理收敛，而是软约束被能量项绕开了。

## 12. 当前建议命令

7 点 ADF 稳定性测试：

```bash
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate torch_env
python fieldspace_rmf.py \
  --z 8 --n 8 \
  --epochs 500 --lbfgs-steps 0 \
  --lambda-reconstruct 1.0 \
  --lambda-dirac 1e-3 \
  --lambda-norm 1.0 \
  --lambda-boundary 1.0 \
  --lr 5e-4 \
  --device cuda \
  --derivative-order 7 \
  --skip-compare \
  --out outputs/fieldspace_O16_order7_adam
```

如果半径和约束稳定，再做 SCF 外部对比：

```bash
python fieldspace_rmf.py \
  --z 8 --n 8 \
  --epochs 1000 --lbfgs-steps 0 \
  --lambda-reconstruct 1.0 \
  --lambda-dirac 1e-3 \
  --lambda-norm 1.0 \
  --lambda-boundary 1.0 \
  --lr 5e-4 \
  --device cuda \
  --derivative-order 7 \
  --out outputs/fieldspace_O16_order7_compare
```
