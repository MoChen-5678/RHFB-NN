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

同 PKA1+D1S、同 0.1 fm / 20 fm 网格，不含质心修正。
以下使用 NN v2 生产组结果及 **2026-11-05 修正 PotelHF 后重新自洽求解的 S-RHFB-C 参考**。
正则轨道对角元 MAE 对 NN 占据数 v² > 0.05 的态统计，按保存的 `(species,kappa,principal)` 标签对应。

| 核 | E/A NN (MeV) | E/A Ref (MeV) | Δ(E/A) (keV) | r_ch NN / Ref (fm) | hνν 对照 MAE (MeV) |
| --- | ---: | ---: | ---: | ---: | ---: |
| O16 | -7.34602 | -7.34894 | +2.93 | 2.8593 / 2.8589 | 0.004 |
| O17 | -7.23381 | -7.23412 | +0.31 | 2.8465 / 2.8465 | 0.001 |
| O18 | -7.29192 | -7.30362 | +11.71 | 2.8360 / 2.8379 | 0.035 |
| O19 | -7.14753 | -7.15466 | +7.13 | 2.8259 / 2.8275 | 0.034 |
| Ca40 | -8.33694 | -8.33778 | +0.84 | 3.5481 / 3.5481 | 0.068 |
| Ca48 | -8.49387 | -8.49493 | +1.05 | 3.5099 / 3.5093 | 0.350 |
| Sn132 | -8.30433 | -8.30493 | +0.61 | 4.7049 / 4.7047 | 2.177 |
| Pb208 | -7.83938 | -7.83991 | +0.52 | 5.5148 / 5.5145 | 1.768 |
| Pb210 | -7.79931 | -7.80237 | +3.06 | 5.5289 / 5.5294 | 2.225 |

`hνν` 是保存轨道上的 Hamiltonian 对角元，不是 Hamiltonian 本征值；占据接近
0 或 1 时，轨道内部旋转可改变这些对角元。表中未作波函数重叠匹配，因此
这一列不能直接解释为本征能谱误差，最新 v2 数据也不支持沿用旧版的重核能级 MAE 宣称。

旧版 `PotelHF.f90` 中，omega 张量的 `gotn=0` 分支提前 `cycle`，跳过了
rho 张量有限程及矢量–张量直接项。按用户提供的修正版交换两段代码后，
全部 69 个不同核素重新收敛，更新了 72 条生产组/同位素链记录。
在同一 Fortran 密度上，NN 与修正参考的两个 rho 张量直接项最大误差为
4.96e-7 MeV；此前把主要偏差归因于 DWS 基截断的推断不再作为解释。

NN v2 的原始摘要保持不变；训练对角化/刷新次数为 0，无海投影均收敛，
最大海泄漏约 1.13e-10。场变化停止阈值不等于变分残差为零，开壳核的配对
与有限正则空间误差仍需独立收敛研究。此次实现检查为 38 项通过、1 项双 GPU
测试跳过，详情见 [修复与验证记录](docs/POTELHF_REFERENCE_FIX_20261105.md)。

### 2026-11-07 配对关联区修复（NN v3）

PotelHF 修正后，配对核的结合能误差明显大于闭壳核（v2：62 个配对核平均
NN − Fortran = +0.96 MeV，闭壳核 +0.08 MeV），且与配对能不足量相关（r = 0.76）。
把 Fortran 的正则态代入 NN 的 D1S 泛函可复现 Fortran 配对能（误差 1–4 keV），
因此泛函正确，原因有两处：

1. **正则空间截断**：容量按壳层顺序填满固定核子数，高 |κ| 块只有 1–2 个正则态。
   用 Fortran 态估计，每个 κ 块约需 6 个态配对能才收敛到 99%（2 个态时只有 65–75%）。
   新选项 `--min-states-per-kappa 6` 为每个 κ 块补足空正则态；Sn、Pb 的 κ 扩到 ±8。
2. **配对坍缩**：配对振幅符号初值全为 +1，而时间反演约定下最优相位为
   (−1)^{|κ|−1}；偶、奇 |κ| 块的配对相互抵消（只剩约 10%），优化随后把 uv 推到 0。
   现默认 `--pair-sign-init phase`。

v3 从各核 v2 最终模型映射热启动（`--initial-model`：匹配态的轨道逐位保留，
新增态近乎空占据，logit 按 t+μ 截断以便坍缩核重新长出配对），再按 Σ± 判据收敛。
72 条运行全部收敛（NN − 修正 Fortran，不含质心修正）：

| | v2 | v3 |
| --- | ---: | ---: |
| 配对核（62）平均 ΔE_total | +0.962 MeV | **+0.217 MeV** |
| 配对核平均配对能不足 | 4.67 MeV | **0.29 MeV** |
| 闭壳核（10）平均 ΔE_total | +0.082 MeV | +0.067 MeV |
| 全部 \|Δ(E/A)\| 平均 | — | 2.9 keV |

例如 Pb242：+1.795 → +0.214 MeV（配对能 −27.24 / −27.65 MeV）；Pb218 原配对坍缩：
+0.720 → +0.077 MeV（−9.08 / −9.79 MeV）。仍需注意：

- **O30** 在 v3 落入更差的局部极小（+3.43 MeV，v2 为 +0.67 MeV，配对能 −8.60 / −5.88 MeV），
  O28 亦由 +0.05 变为 +0.28 MeV；v3 空间包含 v2 态，属优化问题，待单独处理。
- **Sn176** 配对仍未恢复（−0.18 / −2.77 MeV），但其总能量差仅 +0.09 MeV。
- 各能量分项（动能、Hartree、Fock）在重核上仍有 0.3–2 MeV 的差异（总能量差约 0.1–0.2 MeV）。
  已核实把 Fortran 态代入 NN 泛函时各分项与 Fortran 吻合到几 keV，差异来自 NN 态本身。
  Pb208 上降低学习率（2e-4、5e-5）、L-BFGS 精修、增大径向基（32、48）都不再降低能量，
  轨道残差稳定在约 0.04 MeV：NN 态已是网络可表示函数集合内的驻点，剩余偏差属于
  参数化（各轨道修正共用 64 维编码器输出）的限制。对比 Fortran 分项时注意：其输出中
  5 行 "(Tensor ...)" 交换项是张量力部分的再分解（和为 `E-B: Tensor energy functional`），
  不能与各通道交换项重复相加。

另：Fermi 位移二分法改在主机端计算（与原实现逐位一致），扩大空间后每轮约 2.5–3 s；
`--radial-basis-size` 可调径向 sinc 特征数（测试表明 Pb208 上 32/48 并不优于 16）。

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
`--initial-model`（从相邻核或较小正则空间的结果热启动；结构不同时自动按
(species, κ, n) 映射）、`--resume`、`--graph-offload`、
`--min-states-per-kappa`（每个 κ 块的最少正则态数，配对核建议 6）、
`--pair-sign-init {phase,ones}`（配对振幅符号初值，默认 phase）、
`--radial-basis-size`（径向 sinc 特征数）。

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
2026-11-05 最新数据包的 72 条 NN 运行记录（69 个不同核素）在该判据下收敛（额外 29–1019 轮，停止时 Σ± 10 轮变化 ≤ 9.3 keV）。

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

- 同位素链现已包含 O20；Ca41、Ca42、Ca44 未纳入本次对照包，O18 正则空间收敛扫描未运行。
- 当前为球形实现；形变与时间奇场的一般奇核推广见实施记录。

训练输出、检查点、参考文献 PDF 与数据压缩包不纳入版本库（见 `.gitignore`）。
