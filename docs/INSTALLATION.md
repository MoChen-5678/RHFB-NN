# 安装与运行环境

## 1. 平台要求

发布包目标平台为 Linux x86-64。Python 求解器至少需要 Python 3.10、NumPy 1.24、PyTorch 2.0 和 Matplotlib 3.7。SCF 对比还需要 `libgfortran.so.5`、`libgomp.so.1`、`libquadmath.so.0`。CUDA 训练需要宿主机 NVIDIA 驱动；驱动不随发布包分发。

## 2. Conda 安装（推荐）

```bash
tar -xzf DPL-RMF-DensityMatrix-v1.0.0-linux-x86_64.tar.gz
cd DPL-RMF-DensityMatrix-v1.0.0/DFT-RHF
conda env create -f environment.yml
conda activate dpl-rmf
dpl-rmf-check
```

已有 `torch_env` 时可直接安装当前源码：

```bash
conda activate torch_env
python -m pip install . --no-deps --no-build-isolation
dpl-rmf-check
```

若需要固定 CUDA 版本，应先按照 PyTorch 官方方式安装匹配驱动的构建，再运行 `pip install -e . --no-deps`。

## 3. CPU 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install . --no-deps --no-build-isolation
dpl-rmf-check
```

CPU 可以完成训练和推理。对当前 201 点球对称网格，小型模型在 CPU 上可能比启动多个 GPU 进程更稳定。

## 4. Core-1204 定位

程序依次检查 `--core-dir`、`DPL_RHF_CORE_DIR`、DFT-RHF 相邻的 `Core-1204`、当前目录附近的 `Core-1204`。

```bash
export DPL_RHF_CORE_DIR=/absolute/path/to/Core-1204
dpl-rmf-check
```

发布包内默认布局无需设置环境变量。预编译库仅适用于 Linux x86-64。

## 5. 重新编译 Core

需要 `gfortran`、BLAS 和 LAPACK 开发库：

```bash
cd ../Core-1204
make -f Makefile.build clean
make -f Makefile.build -j
cd ../DFT-RHF
dpl-rmf-check --core-dir ../Core-1204
```

重新编译不会改变 DPL 损失；Core 仅用于最终固定势谱和外部 SCF 对比。

## 6. 环境检查解释

`dpl-rmf-check` 返回 JSON。`passed=true` 表示 Python 平台和 Core 动态库可用。`cuda_available=false` 不影响 CPU 模式。使用 `--require-cuda` 可把 CUDA 不可用视为失败。
