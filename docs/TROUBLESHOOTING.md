# 故障排查

## CUDA 不可用

运行 `dpl-rmf-check --require-cuda`。若 PyTorch 显示 CPU 构建，重新安装匹配驱动的 CUDA 构建。不要仅通过设置 `CUDA_VISIBLE_DEVICES` 假定 CUDA 已启用。

## Core 动态库找不到

使用 `--core-dir` 或设置 `DPL_RHF_CORE_DIR`。若库存在但加载失败，执行 `ldd Core-1204/librhf.so` 检查 `libgfortran.so.5`、`libgomp.so.1` 和 `libquadmath.so.0`。

## 能量下降但 gate 失败

这表示网络没有达到变分驻点。优先检查归一化驻定残差、学习率、驻定权重和刷新间隔，不要根据 SCF 能量接近手工选择 checkpoint。

## 出现假态或谱密度不一致

确认 `density-matrix` 模式、正能投影和最终谱验证已启用。减小刷新间隔并检查七点 ADF 两个微分块。自动微分只保证当前离散表达式的梯度正确，不会纠正错误 Hamiltonian 或缺失 no-sea 限制。

## NaN 或梯度爆炸

先降低学习率，再检查 `--density-correction-scale` 和梯度裁剪。保留 float64。若修改了密度公式，检查 `r=0` 外推和所有分母的正规化处理。

## 多进程 GPU 更慢

每卡三个进程是最大调度配置，不是性能保证。398x398 的少量谱刷新在 CPU LAPACK 上执行，而网络和可微场在 GPU 上执行。对 O16 小模型，CPU 可能更快。

## RHF 训练耗尽主机内存

重核的二阶驻点损失需要保存很大的自动微分图。使用 `--graph-offload disk` 和位于本地 NVMe 上的 `--graph-offload-dir`，不要关闭 `create_graph` 或删除驻点项来节省内存。`--graph-offload-min-mib 0.25` 会将大部分重张量写盘；提高阈值可以减少磁盘 I/O，但会增加内存占用。每轮完成后缓存自动删除，因此目录容量不随训练轮数线性增加。

日志中的 `graph_disk_cache` 行给出本轮写入量和读取量。若进程被强制终止，下一次磁盘卸载训练会清理遗留的 `rhf-graph-*` 目录。运行过程中不要手工删除当前缓存文件，否则反向传播无法恢复张量。

## 固定势 Core 总能量异常

固定 DPL 势调用中的 Core 场仍来自初始化状态，该总能量没有与 DPL 密度同步，只能读取单粒子能级。使用 DPL 可微泛函输出作为 DPL 总能量。
