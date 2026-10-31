# PKDD-RMF 势通道审计

依据《核物理(1)》式 (3.42)-(3.48) 和 Core-1204 `PotelHF.f90/Meanfield.f90`，新版通道定义如下。

| 分量 | 书中定义 | 程序定义 | 审计结果 |
|---|---|---|---|
| 标量 `S` | `g_sigma sigma`，`sigma<0` | `scalar_self_energy` | 符号正确 |
| 等标量矢量 `V0` | `g_omega omega + Sigma_R` | `vector_isoscalar` | 正确，含重排项 |
| 同位旋矢量 `V3` | `g_rho rho tau` | `vector_isovector` | 中子 `+`、质子 `-`，正确 |
| 库仑 `VC` | `e A tau_c` | `vector_coulomb` | 仅质子，正确 |
| 中子上通道 | `V0+S+V3` | `vps_n` | 正确 |
| 中子下通道 | `V0-S+V3-2M_n` | `vms_n` | 正确 |
| 质子上通道 | `V0+S-V3+VC` | `vps_p` | 正确 |
| 质子下通道 | `V0-S-V3+VC-2M_p` | `vms_p` | 正确 |

发现并修正的问题：旧网络把四个 Dirac 通道作为四个互相独立的函数输出，只有
`H-H_tilde` 软约束迫使其接近上述结构。这允许训练中间态出现 `S_n != S_p`，不属于严格 PKDD-RMF
变分空间。新版网络改为输出 `(S,V0,V3,VC)`，再按上表硬组装四通道。

其他严格条件：

- `M_n=939.5731 MeV`，`M_p=938.2796 MeV`，分别进入下通道、动能和静止质量扣除；
- `rho_b3=rho_n-rho_p`，与 `tau_n=+1,tau_p=-1` 一致；
- `Sigma_R=sigma rho_s dg_sigma/drho_b + omega rho_b dg_omega/drho_b + rho rho_b3 dg_rho/drho_b`；
- `VC` 使用 `alpha=1/137.03602`，场方程为 `-Delta VC=4 pi alpha rho_p`；
- 所有势通道单位为 `fm^-1`，乘 `hbar*c` 后为 MeV；
- `VMS` 含 `-2M/hbar/c`，网络本征值为扣除静止质量后的单粒子能量。
