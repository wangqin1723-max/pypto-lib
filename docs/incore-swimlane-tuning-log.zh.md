# 核内泳道（in-core op-simulator）调优记录

记录在 wq3（aarch64）上用 msprof op-simulator 做核内逐指令 profiling 的环境配方、
踩坑、以及由此驱动的 DeepSeek-V4 `hc_pre` 性能优化。

---

## 0. 环境配方（wq3，aarch64）

核内 profiling = 纯软件 camodel 仿真，**不需要 NPU 算力板卡**，但 `msprof` 启动时
会无条件初始化驱动平台。登录 shell 对 `/dev/davinci*`（属主 HwHiAiUser）无权限，
直接跑会报 `Init platform by driver faild!`，skill 会**误报成"缺 msopprof"**（其实
worker 在 `/usr/local/Ascend/cann-9.0.0/tools/msopprof/bin/`）。

**所有碰设备/驱动的命令必须包进 `task-submit`。** 不用换机器。

- **用 skill 工具，不要用官方 `tools/export_all_kernel_insight.py`**（见 §3）。
- 三步：
  1. build（不占设备）：`python <kernel>.py -p a2a3sim`
  2. profiling（task-submit）：payload 内 `conda activate wq3` + `source set_env.sh`
     + 3 个 ptoas export + `incore_profile.py --build-dir <dir> --func <k> --target a2a3`
  3. 清洗（不占设备）：`python -m pypto.tools.clean_sim_trace <OPPROF_dir> -o build_output/incore_<name>`
- 大 kernel 加大超时：`--msprof-timeout 1800 --step-timeout 1800 --build-timeout 1200`。

---

## 1. 泳道数据怎么读

`instr_metrics.json` 按子核拆：`core0.cubecore0`（AIC）/ `veccore0` / `veccore1`（两个
AIV 子块）。每条指令带 pipe / cycles / source / process_bytes / vector_utilization。

- **先看 stall 占比**：把 `BAR PIPE:*` / `WAIT_FLAG` / `SET_FLAG` 归为同步停顿，其余为真实工作。
  单核 `<<<1>>>` 隔离 trace 的 stall% 会被**严重高估**（无别的核/task 可重叠），但
  **结构性结论可靠**（哪条 pipe 主导、向量利用率、负载是否均衡）。
- `clean_sim_trace` 的 `trace.clean.json`：屏障重锚为流向箭头，per-lane busy% 给真实重叠视图。
- 退化空 trace 判据：CUBE=0 cycles + 只有 SCALAR/sync（数据依赖 kernel 被零输入坑成 0 迭代）。

---

## 2. skill 工具的 mix-kernel 签名修复

`gen_profiling_case.py` 给 cube+vector 混合 kernel 合成单核 dispatcher 时，**只解析
`_aic` 签名**，用同一份参数同时调 `_aic`/`_aiv`。当 AIV 端多出尾部标量（分块偏移）时
报 `no matching function for '<k>_aiv'`。

**修复**：单独解析 `_aiv` 签名，dispatcher 调 `_aiv` 时把多出的尾部标量补 `0`
（单核 = profile 第 0 分块，逐指令 cost 与分块无关 → 有代表性）。修后 linear、融合
hc_pre 等都能出泳道。

---

## 3. skill 工具 vs 官方 `tools/export_all_kernel_insight.py`

当前 ptoas 0.45 下，**官方工具两个 kernel 都失败**：
- 纯 cube kernel（matmul）：链接失败——官方 generator 没跟上 ptoas 的 `extern "C"`
  变更，前向声明缺 `extern "C"` → 符号 mangle 不匹配。
- mix kernel（linear）：`TimeoutExpired`——推断 buffer 偏小 + 非零分块偏移越界，camodel 卡死。

skill 自带的轻量 generator 后写、已处理 `extern "C"` + 按 .pto 静态 shape 取 buffer 上界
+ 单核最小用例（v6=0 减负），所以**现在用 skill 工具**。

**融合单 spmd 也能 op-simulate**（skill 单核 `<<<1>>>` ~40 秒出），**推翻旧结论"融合版
必 TimeoutExpired"**——那是官方工具全 grid + 真实标量的问题，不是融合本身的问题。

---

## 4. ✅ DeepSeek-V4 hc_pre：`LINEAR_K_TILE 128→256`（关键杠杆）

### 诊断
linear 是 hc_pre 关键路径主导（multiscope ~49–70%，唯一含 cube 核）。瓶颈**不是算力、
不是 HBM 带宽**（x 才 4MB ≈ 2.5us@1.6TB/s），而是 **K=16384 的串行 `matmul_acc` 链**：
`pl.pipeline(HC_DIM//LINEAR_K_TILE)` 共 128 步互相依赖，每步带固定 MTE2/MTE1/WAIT_FLAG
同步开销（in-core 已证同步占大头）。L2 还显示 linear 只用 8/24 cube 核。

### 改法
`LINEAR_K_TILE 128 → 256`：K-chain 步数 128→64，每步固定开销摊薄一半。

### 实测（同会话，decode + l2 swimlane，精度全 PASS）
| 文件 | 效果 |
|---|---|
| `hc_pre.py`（融合单 scope） | Total Test Time **−19.7%**（112.6→90.4us），aic Exec −33% |
| `hc_pre_multiscope.py` | linear scope **−29%**（64.7→46us），total −8% |

融合版收益更大：matmul 就是它唯一 spmd 的关键路径主体；multiscope 里 linear 提前结束后
被下游 scope 的串行/空闲 gap 稀释。

### 边界
- **K=512 编译失败**：FP32 `w_lin [32,512]=64KB` 撑爆 L0B 双缓冲 → **256 是甜点**。
- 推翻 memory 旧结论"hc_pre matmul-bound、改 scope/spmd 都没用"——**之前没人动 K 方向**。

---

## 5. ❌ sinkhorn 向量优化：被 32B tile 行对齐封死（负结果）

### 诊断（正确）
`comb_sinkhorn` 是 **sync/op-count bound**：单核 trace 92% barrier-stall、向量利用率 0.7%、
~20% 真实工作花在 pad 列 MOVEMASK。成本 = op 数量 × 每 op 固定同步开销（19 串行迭代 ×
~15 个 [16,8] 小 op × 4 行分开）。

### 尝试
4 个独立 `[16,8]` 行 tile 合成单个 `[16,16]` tile，行归一 `reshape→[64,4]` 单次 row_sum，
列归一 4-block 加 + `concat` 广播。目标每迭代 op ~15→~7 + 干掉 MOVEMASK。

### 为什么失败（硬约束）
`'pto.alloc_tile' expects row byte size to be 32-byte aligned, but got 16 bytes`：
**UB tile 每行必须 32B 对齐 → FP32 至少 8 列**。HC_MULT=4 的任何 4 宽子视图都不对齐
（reshape `[64,4]` 行 16B ✗；切 `[16,4]` 块 16B ✗）。原"4 个 `[16,8]` tile（HC_PAD=8=32B）
+ 列归一逐元素加"正是绕开这个约束的**已最优写法**。

### 结论
sinkhorn 在当前 tile 抽象下已到下限；再降只能靠编译器/ISA 支持非对齐窄 tile，或改算法
（减迭代数 = 数值问题）。**已干净 revert。**

---

## 6. 关键路径全景（multiscope, K=256）+ 为什么前端到顶了

用 L2 swimlane 的**命名 scope 表** + **per-task 时间线**把整张图盘清（decode, T=128,
Total ≈ 114–121us，同会话噪声）：

### 命名 scope Exec（来自 swimlane summary 表）
| scope | Exec(us) | Exec% | tail OH(us) |
|---|---|---|---|
| linear (aic+aiv) | ~39 | ~68% | **13.5** |
| comb_sinkhorn | 20.0 | 92% | 1.0 |
| mix_x | 20.6 | 80% | 3.0 |
| split_pre_post | 3.1 | | 1.5 |
| write_post | 3.7 | | 2.2 |

### per-task 时间线（关键路径）
| 阶段 | 窗口(us) | 时长 |
|---|---|---|
| linear | 0–44 | 44（K=256 后）|
| **空闲 gap** | **44–73** | **~29（≈25% 总时间）** |
| split_pre_post | 73–76 | 3.6 |
| write_post | 85–90 | 5.6 |
| sinkhorn ‖ mix_x（**并行重叠**）| 85–114 | ~24（mix_x 是尾巴）|

### 三个剩余大块，各有硬约束（前端动不了）
1. **29us gap ≠ 可调度空洞**：查 `aicpu_tasks` 发现 linear 的 AICPU 侧一直跑到 ~66us
   （aicore 部分 44us 就结束），即 linear 的 **tail OH 13.5us/task（远高于其它 scope 的
   1–3us）+ 8 个 task 错位完成 + ~6us redispatch**。是 spmd 任务拆解开销，属 AICPU/调度层。
2. **sinkhorn 20us** — 32B tile 对齐封死（§5）。
3. **mix_x 20us** — 与 sinkhorn **并行 co-limit**：单砍 mix_x，sinkhorn 立刻成新瓶颈。

### ❌ mix_x 并行化尝试（MIX_T_TILE 16→8, 8→16 tasks）：wash + 破坏精度
| 指标 | T_TILE=16 | MIX_T_TILE=8 |
|---|---|---|
| mix_x Exec | 20.6us | 14.2us ↓ |
| mix_x **tail OH** | 3.0us | **5.0us（翻倍）** |
| mix_x latency | 25.9us | 21.4us（几乎没动）|
| **Total** | 113.8us | **114.2us（wash）** |
| x_mixed 精度 | PASS | **FAIL** |

Exec 降的部分被 tail OH 翻倍吃掉（per-task body 太小 → OH 主导，见 memory
`feedback_pypto_fine_grained_parallel_overhead`）；且 `[8,8]` 方阵 transpose 路径取错
per-token 权重 → 精度挂。**已干净 revert。**

### 结论
**hc_pre 的干净前端杠杆只有 `LINEAR_K_TILE 128→256`**（§4，已验证落地）。剩下的 gap /
sinkhorn / mix_x 分别卡在 AICPU 调度开销、tile 对齐、per-task OH+co-limit——都需要
编译器/调度层（降 spmd tail OH、非对齐窄 tile）或算法层（减 sinkhorn 迭代）改动，
超出前端 kernel 调优范围。负结果已记此处，避免重复踩。
