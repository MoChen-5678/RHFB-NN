# RHFB-NN：无刷新神经网络相对论 Hartree-Fock-Bogoliubov（PKA1 + D1S）

用神经网络变分求解球形原子核的密度依赖相对论 Hartree-Fock-Bogoliubov
（DDRHFB）基态：粒子-空穴道为完整 PKA1，粒子-粒子道为固定强度 Gogny D1S
配对，奇核用显式阻塞。训练过程中**不对角化 Dirac 或 Bogoliubov
Hamiltonian、不刷新子空间**；Fortran S-RHFB-C 仅在训练结束后作参考对照。

无配对的 RHF/RMF 主程序是独立仓库
[RHF-NN](https://github.com/MoChen-5678/RHF-NN)，本仓库包含其完整副本并在其上扩展。

## 方法概要

- **正则基变分**：每个 `(q, kappa)` 块由网络给出正则轨道因子 `Q` 与占据
  logits，`v^2` 由 Fermi 位移保证粒子数；配对张量
  `kappa = Q (u v) Q^T`，阻塞态取 `(2j-1)` 等填充。
- **能量**：完整 PKA1 Hartree + Fock（含 rho-tensor、pion-PV、重排）与
  J=0 Gogny-D1S 有限程配对能（双高斯多极核，保留全部 Dirac 分量）。
- **无海近似**：网络因子在 QR 前乘以正能投影
  `P_+ = [1 + sign(h - W)]/2`（`W = -M`），sign 用 Newton-Schulz 迭代，
  只做矩阵乘法；投影子由当前场构造并固定，最终评估时迭代至自洽。
- **批量 Fock 与配对**：块对与 Dirac 分量的交换和配对收缩按
  (种类, kappa, 分量) 批量逐元素计算（`dpl_rhf/rhf/fock_batched.py`、
  `SphericalD1SPairing.energy`）；逐对循环保留为参考实现。

## 精度说明（与 Fortran S-RHFB-C 对比）

同 PKA1+D1S、同 0.1 fm / 20 fm 网格，不含质心修正（2026-10-28 数据报告）：

| 核 | E/A NN (MeV) | E/A Ref (MeV) | Δ(E/A) (keV) | r_ch NN / Ref (fm) | 能级 MAE (MeV) |
| --- | ---: | ---: | ---: | ---: | ---: |
| O16 | -7.34572 | -7.34892 | +3.20 | 2.8607 / 2.8588 | 0.017 |
| O17 | -7.23310 | -7.23287 | -0.23 | 2.8470 / 2.8458 | 0.024 |
| O18 | -7.29154 | -7.29958 | +8.04 | 2.8360 / 2.8374 | 0.061 |
| O19 | -7.14750 | -7.14642 | -1.08 | 2.8259 / 2.8273 | 0.078 |
| Ca40 | -8.33649 | -8.33772 | +1.23 | 3.5478 / 3.5479 | 0.007 |
| Ca48 | -8.49344 | -8.47872 | -14.71 | 3.5102 / 3.5087 | 0.086 |
| Sn132 | -8.30409 | -8.29735 | -6.74 | 4.7049 / 4.7063 | 0.042 |
| Pb208 | -7.83902 | -7.83561 | -3.41 | 5.5162 / 5.5154 | 0.027 |
| Pb210 | -7.79899 | -7.79777 | -1.22 | 5.5288 / 5.5304 | 0.036 |

全部结果的训练对角化次数为 0，海泄漏 ≤1e-10。已知差异：O18、Pb210 的
中子配对能弱于参考（疑为正则空间截断，待收敛扫描）；闭壳 Ca48/Sn132/Pb208
的 NN 能量比 S-RHFB-C 低 3–15 keV，而 NN 为变分上界，提示参考程序在重核
上受 DWS 基截断影响。

**数值实现一致性**

| 检查 | 结果 |
| --- | ---: |
| 批量 Fock 与流式/逐对 Fock：总能量差（Sn132、Pb248） | 4.5e-13 MeV |
| 批量 Fock 与流式 Fock：网络梯度相对差 | ~1e-15 |
| 批量 D1S 配对能与逐对循环：能量 / 梯度相对差 | < 1e-12 |

## 2026-10-31 性能更新

批量 Fock 取代逐 kappa 对流式计算成为默认路径，结果与流式一致（能量差
<1e-12 MeV，梯度相对差 ~1e-15）。A100 40GB 每轮（前向 + 反向）：

| 情形 | 流式（旧） | 批量（新） | 峰值显存（新） |
| --- | ---: | ---: | ---: |
| Sn132（capacity 184/58，kappa 8/7） | 16.2 s | 0.59 s | 7.3 GiB |
| Pb248（capacity 240/92，kappa 8/7） | 22.3 s | 0.77 s | 8.3 GiB |

drip-line 生产链实际训练约 0.7–0.8 s/轮（旧版 16–20 s/轮）。
`--fock-stream-devices` 仍可用；一旦设置，流式路径优先于批量路径。

## 安装

```bash
conda env create -f environment-rhf.yml
conda activate dpl-rhf
pip install -e .
```

## 训练与评估

`dpl-rhfb-sph-train` 启动时会检查资源限制，必须在下列任一启动器内运行：

- `scripts/run_rhfb_limited.py`：cgroup v2 内核内存限制（本地）；
- `scripts/run_rhfb_procguard.py`：进程级采样守护（无 cgroup 委派的主机）。
  需设置允许使用的 GPU，例如 `RHFB_AUTHORIZED_GPUS=0,1`。

```bash
export RHFB_AUTHORIZED_GPUS=0
python scripts/run_rhfb_procguard.py --memory-mib 8192 --gpu-indices 0 \
  --gpu-memory-mib 24576 --timeout-seconds 86400 \
  --report outputs/execution/train_Pb208.json -- \
  python -u -m dpl_rhf.cli.rhfb_spherical_train --z 82 --n 126 \
    --capacity-n 240 --capacity-p 92 --kappa-limits 8 7 \
    --epochs 1500 --lr 2e-3 --logit-lr 5e-2 \
    --device cuda --cuda-budget-mib 24576 \
    --out outputs/rhfb_spherical/Pb208

# 训练后自洽评估（迭代无海投影与配对符号至收敛）
python scripts/run_rhfb_procguard.py ... -- \
  python -u -m dpl_rhf.cli.rhfb_spherical_train --z 82 --n 126 \
    --capacity-n 240 --capacity-p 92 --kappa-limits 8 7 \
    --device cuda --cuda-budget-mib 24576 \
    --evaluate-only outputs/rhfb_spherical/Pb208/training_checkpoint.pt \
    --evaluate-state last --out outputs/rhfb_spherical/Pb208_eval
```

常用选项：`--block-n/--block-p`（阻塞轨道，奇核）、`--pairing {d1s,none}`、
`--initial-model`（从相邻核热启动）、`--resume`、`--graph-offload`。

**drip-line 链**：`scripts/remote_gpu04_dripline.sh` 在 8 张 GPU 上按同位素链
逐核训练并评估（O、Sn、Pb），以相邻核热启动；已训练或已评估的核在重启时
自动跳过，未完成的核从检查点续训。

**按 Σ± 收敛停止**（代替固定轮数）：

| 选项 | 含义 |
| --- | --- |
| `--converge-sigma-kev 10` | 中子、质子的 Σ₊ = V+S、Σ₋ = V−S 在全部内部网格点上的最大变化小于该值即视为收敛；`--epochs` 变为安全上限 |
| `--convergence-window 10` | 变化取当前与 10 个 epoch 前之差 |
| `--convergence-patience 20` | 上述条件须连续保持 20 个 epoch，避免振荡场偶然跌破阈值 |
| `--sigma-lr-patience 100` | Σ± 变化 100 个 epoch 无 10% 改善则学习率减半（代替能量平台调度） |
| `--min-lr 2e-5` | 学习率下限，防止学习率停滞造成“假收敛” |
| `--reset-lr` | 与 `--resume` 一起使用：从初始学习率重新开始调度，保留模型、优化器矩与历史 |

`history.csv` 逐轮记录 `sigma_step_change_kev`、`sigma_window_change_kev` 与连续满足轮数；
`summary.json` 的 `convergence` 字段记录是否收敛、总轮数、最终变化与学习率。
`scripts/remote_converge_sigma.py` 从各运行自身的检查点续训全部核直至收敛并自洽评估
（`--tag v2 --patience 20 --sigma-lr-patience 100`）；`scripts/build_rhfb_observables.py`
将评估结果与 Fortran 参考整理为观测量、单粒子能级、各耦合道能量三张 CSV。
2026-11-05 全部 70 个核在该判据下收敛（额外 29–1019 轮，停止时 Σ± 10 轮变化 ≤ 9.3 keV）。

## 测试

```bash
python -m pytest -q dpl_rhf/tests/test_spherical_d1s_pairing.py \
  dpl_rhf/tests/test_spherical_rhfb_backend.py dpl_rhf/tests/test_fock_streaming.py \
  dpl_rhf/tests/test_no_refresh_solver.py dpl_rhf/tests/test_density_matrix_rhf.py
```

## 文档

- [实施记录](docs/BLOCKED_RHFB_IMPLEMENTATION.md)、[物理决策](docs/PHYSICS_DECISIONS_20261015.md)
- [无海 sign 响应审查](docs/NO_SEA_SIGN_RESPONSE_REVIEW_20261015.md)
- [无刷新求解器原理](docs/NO_REFRESH_THEORY_AND_WORKFLOW.md)

## 未完成项目

- O20、Ca41、Ca42、Ca44 尚未计算；O18 正则空间收敛扫描未运行。
- 当前为球形实现；形变与时间奇场的一般奇核推广见实施记录。

训练输出、检查点、参考文献 PDF 与数据压缩包不纳入版本库（见 `.gitignore`）。
