# 2.0.0-DPL_RMF-fast-1D

本目录为 `variants/blocked-rhfb` 独立开发副本。奇核扩展目前提供组件和
无海算法验收原型，尚未通过完整 PKA1 奇核计算验收。
实现范围、备份恢复、试运行及未完成项目见
[奇核扩展实施记录](docs/BLOCKED_RHFB_IMPLEMENTATION.md)。

这是与原 `DFT-RHF` 完全隔离的正式RMF版本。它保留经过核对的 RMF/RHF
能量泛函，但删除了训练中的正能子空间对角化、子空间刷新和优化器重置。

网络直接在固定的完整 Dirac 径向网格空间生成占据密度因子
`P = Q f Q^T`。QR 严格实施正交约束；RMF/RHF 总能量、全空间
Euler-Lagrange 驻定残差和逆 Hamiltonian 无海约束共同构成无监督物理损失。
SCF/Core-1204 数据只允许在训练结束后验证。

## 快速开始

```bash
conda env create -f environment.yml
conda activate dpl-rmf
pip install -e .

# PKDD RMF
dpl-no-refresh train-rmf --model PKDD --z 8 --n 8 --epochs 500 \
  --device cuda --out outputs/no_refresh/PKDD/O16

# PKA1 RHF
dpl-no-refresh train-rhf --model PKA1 --z 8 --n 8 --epochs 500 \
  --device cuda --out outputs/no_refresh/PKA1/O16
```

`--density-refresh-interval` 和 `--refresh-interval` 必须为零，传入其他值会
直接报错。训练日志应始终满足：

```text
eigensolve_count = 0
refresh_count = 0
training_dirac_solves = 0
minimum_branch_score > 0
```

## Core 验证

RHF Hamiltonian 的严格定义、闭合数据和验收门控见
[`docs/RHF_HAMILTONIAN_DEFINITION_AND_AUDIT.md`](docs/RHF_HAMILTONIAN_DEFINITION_AND_AUDIT.md)。

```bash
dpl-no-refresh compare-core --kind rmf \
  --result outputs/no_refresh/PKDD/O16 --model PKDD \
  --z 8 --n 8 --core-dir ../Core-1204 \
  --out outputs/no_refresh/PKDD/O16/core

dpl-no-refresh compare-core --kind rhf \
  --result outputs/no_refresh/PKA1/O16 --model PKA1 \
  --z 8 --n 8 --core-dir ../Core-1204 \
  --out outputs/no_refresh/PKA1/O16/core
```

完整理论、数据流、损失、输出和实验验收标准见
[无刷新求解器原理与工作流程](docs/NO_REFRESH_THEORY_AND_WORKFLOW.md)。
原刷新版本位于独立的 `DFT-RHF` 目录及冻结归档中，本项目不覆盖它。
