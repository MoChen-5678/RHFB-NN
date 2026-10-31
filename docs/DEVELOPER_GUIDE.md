# 开发与 RHF 扩展指南

## 模块边界

- `models/density_matrix_net.py`：只负责可训练参数化和硬约束投影。
- `backends/density_matrix_rmf.py`：把密度矩阵映射为密度、能量、哈密顿量和驻定残差。
- `functionals/pkdd_action.py`：参数集、耦合、场方程、重排项、势、能量和观测量。
- `training/density_matrix_trainer.py`：优化器、刷新调度、gate、checkpoint 和输出。
- `cli/`：参数解析、评估、SCF 对比、批量调度和绘图，不复制物理公式。

## 增加参数集

在 `RMF_PARAMETER_DATA` 中增加完整核子质量、介子质量、饱和密度、耦合常数和密度依赖参数；增加 Core 参数编号映射；编写耦合值、导数、势通道和闭壳层 SCF 对比测试。不得通过复制其他参数集缺省值补字段。

## 数值不变量

任何修改必须保持四通道顺序、统一径向度量、七点 ADF 的前后向 Dirac 块、原点处理、无穷远边界、质心修正口径和 float64。改变积分或微分时必须同时验证能量自动微分与离散 Hamiltonian 一致。

## RHF 扩展接口

`functionals/rhf_interfaces.py` 预留 `sigma_fock`、`omega_fock`、`rho_tensor`、`pi_pv` 和轨道源 `X_a/Y_a`。真正 RHF 需要非局域密度矩阵核和轨道相关 Fock 泛函，不能把局域 RMF 势改名为 RHF。建议新增独立 `RHFBackend` 实现相同的 `evaluate(blocks)` 契约。

## 测试策略

公式测试覆盖参数、耦合导数、场方程、重排势和能量梯度；结构测试覆盖 SiLU、硬约束和无逐步 eigensolver；集成测试覆盖短训练、checkpoint 往返和 Core 动态库；验收测试固定 PKDD/O16 发布结果容差。
