# 完整使用说明

## 1. 单核训练

```bash
dpl-rmf-train --mode density-matrix --functional rmf-dd \
  --model PKDD --z 8 --n 8 --a 16 \
  --epochs 500 --device cuda --seed 20240623 \
  --hidden 64 --network-depth 3 --radial-basis-size 16 \
  --activation silu --lr 1e-4 --derivative-order 7 \
  --density-stationarity-weight 1000 --density-energy-weight 1 \
  --density-refresh-interval 100 \
  --density-stationarity-gate 1e-2 --density-spectral-gate 1e-2 \
  --checkpoint-policy physics --out outputs/PKDD_O16
```

`A` 默认取 `Z+N`，不是网络输入归一化量。`--checkpoint-policy physics` 保存驻定性监视量最小的 epoch。`--device auto` 自动选择 CUDA 或 CPU；为可复现实验建议明确指定。

## 2. 关键参数

| 参数 | 作用 | 推荐值 |
|---|---|---:|
| `--hidden` | 每层隐藏神经元数 | 64 |
| `--network-depth` | SiLU 隐藏层数 | 3 |
| `--radial-basis-size` | sinc 径向特征数 | 16 |
| `--density-correction-scale` | 相对初始物理投影子的网络修正尺度 | 0.01 |
| `--density-stationarity-weight` | 驻定性残差平方权重 | 1000 MeV^-1 |
| `--density-energy-weight` | 总能量项权重 | 1 |
| `--density-refresh-interval` | 正能空间稀疏刷新间隔 | 100 |
| `--derivative-order` | ADF 微分设置 | 7 |
| `--grad-clip` | 参数梯度裁剪范数 | 10 |

训练损失不包含 SCF 标签。不要用 SCF 能量差直接反向传播，否则实验不再是无监督物理变分。

## 3. 加载模型

```bash
dpl-rmf-evaluate examples/PKDD_O16/model.pt \
  --device cpu --out outputs/PKDD_O16_reproduced
```

评估器从 checkpoint 读取核素、参数集、网络宽度、深度、径向基数量、微分阶数和修正尺度。它重新生成密度、场、哈密顿量、最终谱验证波函数和 `evaluation_summary.json`。checkpoint 与当前模型形状不兼容时会立即失败，不进行部分加载。

## 4. SCF 外部对比

```bash
dpl-rmf-compare --dpl-dir outputs/PKDD_O16 \
  --model PKDD --z 8 --n 8 --a 16 \
  --tolerance 1e-10 --max-iterations 500 \
  --core-dir ../Core-1204 \
  --out outputs/PKDD_O16/scf_comparison
```

Core 固定 DPL 势得到的总能量不能作为 DPL 总能量；该调用只用于单粒子谱。DPL 总能量必须由同一密度和同一可微 RMF 泛函计算。

## 5. 批量运行

```bash
nohup dpl-rmf-batch --mode density-matrix --epochs 5000 \
  --models PKDD DD-ME1 DD-ME2 TW99 DD-LZ1 \
  --nuclei O16 Ca40 Ca48 Sn132 Pb208 \
  --gpus 0,1 --slots-per-gpu 3 \
  --out outputs/rmf_5x5_density_matrix_5000 \
  > outputs/rmf_5x5_density_matrix_5000.nohup.log 2>&1 &
```

每张卡三个进程只是调度上限，不保证显存一定足够。先用一个核素测量峰值显存，再确定并发数。`--aggregate-only` 不训练，只重新汇总已完成案例。

## 6. 重绘和判读

```bash
dpl-rmf-plot outputs/PKDD_O16
```

先检查 `formula_invariants.json`，再检查 `training_solver_counts.json` 中训练求解次数为零，然后检查内部 1% gate 和最终谱一致性，最后才读取 SCF 外部误差。
