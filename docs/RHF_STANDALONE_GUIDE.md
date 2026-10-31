# DPL-RHF 独立发布包使用与工作流程

## 1. 发布包目标

该发布包用于在其他 Linux 服务器上独立训练和评估球对称、密度依赖的
PKO2/PKA1 神经变分 RHF 模型。训练代码不调用 Core-1204，不使用 Core
密度监督，也不在普通训练轮次求解 Dirac 方程。Core-1204 是可选的最终
验证程序，用于比较能量、半径、单粒子能级、局域势以及非局域密度矩阵
`R`。

包内包括：

- 完整 Python 源码及 PKA1/PKO2 参数和相互作用通道；
- 一键 Conda/Miniforge 环境配置脚本；
- CPU、CUDA、双精度二阶自动微分和磁盘缓存自检；
- 前台、`nohup` 后台、监控、停止和断点续训脚本；
- `memory`、`disk` 与 `recompute` 三种严格等价的自动微分图策略；
- 物理公式审计、程序工作流程、实验设计和输出定义；
- 可选 Core-1204 完整对比脚本，包含 `R` 的四分量图和误差数据。

参考论文、书籍、Core 二进制、训练缓存和历史大输出不包含在发布包中。
Core 动态库必须在目标平台单独编译。

## 2. 支持平台

- Linux x86-64 或 Linux aarch64；
- Python 3.10；
- CPU 模式或 NVIDIA CUDA 模式；
- CUDA 服务器推荐至少 16 GB 显存；PKA1 重核推荐 32 GB；
- 所有物理张量使用 `float64`，应优先选择双精度性能较好的 GPU；
- `disk` 模式推荐本地 NVMe，不推荐机械盘或共享网络盘。

程序使用 Linux `fcntl` 文件锁防止两个训练进程写入同一结果目录，因此
当前发布包不支持原生 Windows。Windows 用户应使用 WSL2/Linux。

## 3. 一键安装

解压后进入目录：

```bash
tar -xzf DPL-RHF-Standalone-v1.0.0.tar.gz
cd DPL-RHF-Standalone-v1.0.0
./setup_env.sh
```

脚本会：

1. 检测 NVIDIA 驱动并选择 CUDA 或 CPU；
2. 复用现有 Conda；若不存在则在当前目录安装 Miniforge；
3. 创建隔离环境 `.dpl-rhf-env`；
4. 安装固定的 PyTorch 2.5.1 和数值依赖；
5. 以 editable 模式安装 DPL-RHF；
6. 检查 float64 二阶梯度和 CUDA 磁盘卸载；
7. 检查 PKA1 完整通道参数是否可读取。

明确指定环境类型：

```bash
./setup_env.sh --cuda
./setup_env.sh --cpu
```

默认 CUDA wheel 为 CUDA 12.1。其他版本可指定官方 PyTorch wheel 索引：

```bash
./setup_env.sh --cuda \
  --torch-index-url https://download.pytorch.org/whl/cu121
```

激活环境：

```bash
source ./activate_env.sh
```

## 4. 三种自动微分图模式

### 4.1 纯内存模式

```bash
./run_rhf.sh PKA1 8 8 16 500 memory work/outputs/PKA1_O16
```

`memory` 使用 PyTorch `save_on_cpu` 保存重核自动微分中间张量。张量保留在
主机内存，速度通常高于磁盘模式，但 Sn132/Pb208 的完整 PKA1 二阶图可能
需要超过 100 GB 内存。旧参数名 `cpu` 仍可使用，但仅作为 `memory` 的兼容
别名。

### 4.2 磁盘模式

```bash
./run_rhf.sh PKA1 50 82 132 500 disk work/outputs/PKA1_Sn132
```

`disk` 将 autograd 保存的大张量写入 `work/graph-cache`，反向传播时恢复到
原始 GPU。每轮结束后缓存自动删除，强制中断留下的缓存会在下次启动时清理。

三种模式只改变中间张量的保存或精确重计算策略，不改变：

- 网络结构和 SiLU 激活函数；
- 正能空间投影和 QR 正交约束；
- 完整 PKA1/PKO2 Hartree、Fock 和 rearrangement 通道；
- `create_graph=True` 的二阶变分传播；
- 总能量与投影 Euler-Lagrange 驻点损失；
- 参数更新和 checkpoint 选择标准。

因此同一硬件和随机种子下，三种模式应只存在浮点舍入级差异。

### 4.3 精确重计算模式

```bash
./run_rhf.sh PKA1 50 82 132 500 recompute work/outputs/PKA1_Sn132
```

`recompute` 使用 PyTorch non-reentrant activation checkpoint。前向只保留
非局域密度块和密度依赖耦合，反向按完整 sigma、omega、rho-vector、
rho-tensor、rho vector-tensor、pion 和 Coulomb Fock 通道重新执行原公式。
它不压缩 float64，不截断 `R`，也不删除 rearrangement 或驻点二阶梯度。

V100 32 GB 实测中，Ca48 两次前向加一次二阶反向由磁盘模式的 294 秒
缩短到 39.5 秒；Sn132 一个优化步加一次最终评估为 81 秒，峰值显存
22.9 GiB。显存足够时，重核推荐使用 `recompute`。Pb208 在目标 GPU 上
完成峰值探针前应保留 `disk` 回退，不能按质量数线性假设其显存占用。

## 5. 命令格式

```text
./run_rhf.sh MODEL Z N A EPOCHS CACHE_MODE OUTPUT_DIR [额外参数]
```

示例：

```bash
# PKA1 O16，内存模式
./run_rhf.sh PKA1 8 8 16 500 memory work/outputs/PKA1_O16

# PKA1 Pb208，磁盘模式，每轮打印一次并每轮保存 checkpoint
./run_rhf.sh PKA1 82 126 208 500 disk work/outputs/PKA1_Pb208 \
  --print-every 1 --checkpoint-every 1

# PKA1 Sn132，精确重计算模式
./run_rhf.sh PKA1 50 82 132 500 recompute work/outputs/PKA1_Sn132

# PKO2 Ca40
./run_rhf.sh PKO2 20 20 40 500 memory work/outputs/PKO2_Ca40
```

脚本固定启用 `--resume`。结果目录存在 `training_checkpoint.pt` 时自动续训；
不存在时从头训练。输出目录中的 `.training.lock` 防止重复运行。

## 6. 后台运行

```bash
./run_background.sh PKA1 50 82 132 500 disk work/outputs/PKA1_Sn132
./monitor_rhf.sh
./stop_rhf.sh
```

后台日志保存在 `work/logs`。断开 SSH 不会结束 `nohup` 进程。停止脚本先发送
`SIGINT`，等待 30 秒后才发送 `SIGTERM`。checkpoint 默认每 10 轮保存一次。

## 7. 完整物理与训练流程

### 7.1 输入与壳层

输入为参数集、质子数 `Z`、中子数 `N` 和 `A=Z+N`。程序按球对称壳层生成
每个核子种类的 `(kappa,n)` 占据块，保持 `n` 和 `kappa` 标签。最后未满壳
采用 equal filling；当前模型不包含 pairing、blocking 和 deformation。

### 7.2 神经密度因子

SiLU 网络输出各占据块的径向密度因子修正。网络结果经过固定正能空间投影和
QR，直接满足粒子数、正交性、Hermiticity、占据谱和 no-sea 约束。普通 epoch
不求解 Dirac 方程。

### 7.3 局域和非局域密度

对每个 `(q,kappa)` 块先对主量子数 `n` 求和：

```text
R_kappa,q^(mu,mu')(r,r')
  = (2j_kappa+1) sum_n v_(n,kappa,q)^2
    u_(n,kappa,q)^mu(r) u_(n,kappa,q)^mu'(r').
```

保留 `GG/GF/FG/FF` 四个完整二维分量。Hartree 局域密度由同一个 `R` 的
对角收缩得到，Fock 项使用完整非局域 `R(r,r')`，两者不使用两套独立密度。

### 7.4 完整能量泛函

PKA1 首轮会检查 sigma、omega、rho-vector、rho-tensor、rho vector-tensor、
pion pseudovector、Coulomb 以及相应直接、交换、接触补偿和密度依赖
rearrangement 通道。缺失、断图或非有限通道会在第一个优化步骤前报错。

### 7.5 变分优化

能量路径为：

```text
network -> Q -> P -> local/nonlocal densities -> density-dependent vertices
        -> Hartree + Fock + rearrangement -> total RHF energy
        -> dE/dP -> projected Euler-Lagrange residual -> loss -> optimizer
```

块 Hamiltonian 由完整能量泛函梯度得到：

```text
h_a = [1/(2j_a+1)] dE_RHF/dP_a,
residual_a = (I - Q Q^T) h_a Q.
```

损失为每核子总能量与投影驻点残差平方的组合，驻点权重由首轮能量梯度和
残差梯度自动平衡。默认每 100 轮刷新一次正能空间；这是稀疏 no-sea 刷新，
不是每轮 Dirac 求解。输出必须保持 `training_dirac_solves=0`。

### 7.6 边界

原点使用每个 `kappa` 对应的上下分量角动量幂次。无穷远尾部使用由
`epsilon=E_total-M` 决定的指数衰减率，并对 `GG/GF/FG/FF` 的外部粒子数和
半径矩做解析延拓。

## 8. 训练输出

每个结果目录至少包含：

| 文件 | 内容 |
| --- | --- |
| `summary.json` | 能量、半径、完整通道、约束、求解器计数和配置 |
| `history.csv` | 每轮能量、驻点残差、学习率和耗时 |
| `training_checkpoint.pt` | 可续训状态，按 `checkpoint-every` 更新 |
| `best_checkpoint.pt` | 最佳投影驻点模型 |
| `local_observables.npz` | 局域密度、Hartree 场和局域 rearrangement |
| `density_matrix_factors.npz` | 每个 `(q,kappa,n)` 密度因子和占据 |
| `nonlocal_hamiltonians.npz` | 各块总 Hamiltonian 和 Fock 非局域算符 |

Fock 势不是唯一的局域 `S(r)` 或 `V(r)`，不能把 Fock 强行画成普通局域势。
它应通过块非局域 Hamiltonian 和构成交换项的 `R(r,r')` 进行验证。

## 9. Core-1204 最终验证与 R 对比

Core 不包含在发布包中。在目标服务器编译好 Core 并生成 `librhf.so` 后：

```bash
./compare_with_core.sh \
  work/outputs/PKA1_O16 8 8 16 /path/to/Core-1204 \
  work/outputs/PKA1_O16/core_comparison
```

比较程序运行独立 Core SCF，仅用于训练结束后的验证。必须输出：

- `detailed_comparison.json`：能量、半径、能级、局域势和 R 汇总误差；
- `single_particle_levels.csv`：按 `(q,kappa,n)` 匹配的单粒子能级；
- `local_s_v_comparison.npz`；
- `s_plus_minus_v_local_dpl_core.png`；
- `nonlocal_density_R_comparison.npz`：所有块的 DPL/Core/delta 四分量；
- `nonlocal_density_R_metrics.csv`：每个块和分量的 RMSE、MAE、最大误差、L2；
- `nonlocal_density_R_error_overview.png`；
- `nonlocal_density_R_blocks/R_*.png`：每个 `(q,kappa)` 的 4x3 图，逐行是
  `GG/GF/FG/FF`，逐列是 DPL、Core 和差值。

若 R 输出缺失，则 Fock 非局域输入尚未完成验证，不能将该比较视为完整 RHF
对比。

## 10. 推荐实验设计

1. 用 O16、5 轮、`memory` 模式完成安装冒烟测试。
2. 对 O16 分别运行 `memory`、`disk` 与 `recompute`，固定 seed，检查能量和梯度一致性。
3. 运行 O16/Ca40/Ca48/Sn132/Pb208，所有核使用相同网络和 500 轮配置。
4. 记录每核最佳能量、驻点 RMS、粒子数、半径和每通道能量。
5. 训练结束后才调用 Core；不得把 Core 数据加入损失或 checkpoint 选择。
6. 比较 `S+V`、`S-V`、单粒子能级和完整非局域 R 四分量。
7. 报告缓存模式、峰值内存、缓存峰值、总耗时和硬件型号。

## 11. 常见故障

- CUDA 自检失败：确认驱动可用，重新运行 `./setup_env.sh --cuda`。
- 主机内存不足：切换 `disk`，降低并行数，不要删除驻点损失。
- 磁盘模式很慢：检查 `iostat`；若磁盘 100% busy，增加 GPU 不会提速。
- 磁盘空间不足：停止训练后清理 `work/graph-cache/rhf-graph-*`；运行中禁止删除。
- 重复任务报锁错误：同一输出目录已有训练进程，使用 `./monitor_rhf.sh` 检查。
- Core 加载失败：在目标平台重新编译 `librhf.so` 并检查 GNU Fortran 运行库。
