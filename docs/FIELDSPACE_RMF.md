# PKDD FieldSpace 变分程序

## 目标

程序以核子数 `Z,N` 和 PKDD 参数集为输入。网络只表示四个球对称局域 Dirac 势
`V+S` 与 `V-S-2M`（中子、质子各两项），不输出波函数，也不使用 SCF 数据监督训练。
Fortran 仅在训练结束后求解固定势的单粒子谱，并另外运行一次完整 SCF 作为外部比较。

## 物理闭环

每次目标函数计算依次执行：

1. SiLU 网络给出试探局域势 `H(theta)`；解析 Woods-Saxon 只作为初始参考。
   网络修正在原点允许为有限值，在 `Rmax` 精确归零，以保持核力场衰减和质子库仑尾。
2. 非中心前向差分与其转置组成严格厄米的径向 Dirac 矩阵，求出 no-sea 占据态。
3. 按教材密度公式构造标量、矢量和同位旋密度，并严格归一化到 `N,Z`。
4. 按教材球对称 Green 函数公式重构 `sigma, omega, rho, Coulomb` 场。
5. 加入 PKDD 密度依赖耦合及 rearrangement 项，得到重构势 `H_tilde`。
6. 最小化广义目标

   `L = (E_no_com/A)/50 + lambda * mean[((H-H_tilde)/scale)^2]`。

质心修正不参与训练和主对比。`lambda` 是显式命令行参数，因为本地文献正文未给出
补充材料中的数值；当前扫描结果使用 `0.1`，不能把它表述成论文指定常数。

## 使用

```bash
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate torch_env
python fieldspace_rmf.py \
  --z 8 --n 8 \
  --epochs 500 --lbfgs-steps 50 \
  --lambda-reconstruct 0.1 \
  --derivative-order 1 \
  --device cuda \
  --out outputs/fieldspace_O16
```

高精度网格复核：

```bash
python fieldspace_rmf.py \
  --z 8 --n 8 --mesh-step 0.05 \
  --epochs 500 --lbfgs-steps 50 \
  --lambda-reconstruct 0.1 --derivative-order 1 --device cuda \
  --out outputs/fieldspace_O16_h005
```

`--derivative-order 2` 使用二阶上风非中心模板，并仍以 `D+`/`D+^T` 组成厄米 Dirac
矩阵。这个选项已通过算子和自动微分测试，但短跑结果还没有优于一阶基线，因此现在用于
网格误差研究，不作为默认生产设置。

主要输出：`summary.json`、`dpl_local_stack.npz`、`reconstructed_local_stack.npz`、
`dpl_orbitals.npz`、`dpl_densities.npz`、`single_particle_compare.csv` 和
`compare_summary.json`。

## 验证

```bash
python test_fieldspace_rmf.py
python test_nuclear_matter_rmf.py
```

第一组检查厄米性、轨道归一化、粒子数和自动微分方向导数。第二组检查 PKDD 无限核物质
场方程以及包含 rearrangement 的热力学一致性。非中心一阶差分必须用至少两组 `mesh-step`
做网格外推；它消除了中心差分的成对假态，但误差只按一阶收敛。
