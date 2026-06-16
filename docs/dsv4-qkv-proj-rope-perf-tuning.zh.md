# DeepSeek V4 `qkv_proj_rope` Decode 性能调优

[models/deepseek/v4/qkv_proj_rope.py](../models/deepseek/v4/qkv_proj_rope.py) 在 a2a3 平台上(带 `--enable-l2-swimlane`)的性能调优进展记录。

## 运行配置

- 模型配置:`FLASH` — `DECODE_BATCH=64, S=1, T=64, D=4096, H=64, HEAD_DIM=512, ROPE_DIM=64, NOPE=448, Q_LORA=1024`
- 命令:
  ```bash
  python models/deepseek/v4/qkv_proj_rope.py -p a2a3 -d 0 --enable-l2-swimlane
  ```

该 kernel 的 wall-clock 单次运行噪声约为 ±4%,即任何单次数据应视作 ±50us 误差。在声称有改进前需至少重跑 3 次。

## 调优进展

| Run | 日期 | Wall-clock | Tasks | Exec/Lat | Total Exec | 日志 |
|---|---|---:|---:|---:|---:|---|
| 基线 | 2026-05-18 | 1725 us | 1527 | 28.6% | 5696 us | `qkv_proj_rope_l2_swimlane.log` |
| **Opt A:从 3 个 vector scope 中移除 chunked_loop_optimizer** | 2026-05-18 | **1371 us**(单次) | **999**(−528) | **42.0%** | 5260 us | `qkv_proj_rope_optA.log` |
| Opt B 失败:`Q_PROJ_OUT_CHUNK=256`(N tile) | 2026-05-18 | — | — | — | — | `qkv_proj_rope_optB_outchunk256.log` — runtime 报错 507018 `ACL_ERROR_RT_AICORE_TIMEOUT` |
| Opt A + Opt B:`Q_PROJ_CHUNK=256`(K tile),3 次运行 | 2026-05-18 | **1371 / 1366 / 1313 us**(极差 58us,约 4.4%) | 999 | 40-43% | 4972-5190 us | `qkv_proj_rope_optB_kchunk256.log` + `qkv_proj_rope_l2_swimlane_3.log` |
| Opt C 回退:把 `attn_norm_rms` 并行拆为 16 partials | 2026-05-18 | 1466 us(+100us 回退) | 1015 | 41.9% | 5158 us | `qkv_proj_rope_optC_run1.log` — pypto 单任务启动开销吞掉了细粒度并行收益 |
| Opt D 回退:融合 `q_rope_reassemble` + `q_rope_write` | 2026-05-18 | 编译失败 | — | — | — | AIC matmul(NZ)无法与 AIV cast(ND)共享 scope |
| **Opt E:把 `token_x_cast_bf16` + `qr_cast_bf16` 融合进生产端 norm_apply scope**,3 次运行 | 2026-05-18 | **1184 / 1193 / 1220 us**(中位数 **1193**) | **951**(−48) | 41.7-43.0% | 5130-5262 us | `qkv_proj_rope_optE_run{1,2,3}.log` |
| Opt E 在不同 NPU 卡上重测(存在内存竞争) | 2026-05-18 | 1239 us | 951 | 45.4% | 5716 us | `qkv_proj_rope_l2_swimlane_4.log` — attn_norm_apply 17.5→39.5us(HBM 拥塞下的 memory-bound AIV op);仍在 Opt E 区间内 |
| Opt F-1 回退:matmul `out_dtype=BF16` + fused assemble | 2026-05-18 | 编译失败 | — | — | — | `pl.matmul_acc` 强制要求 FP32 累加器 |
| **Opt G:chunk size 调优**(`Q_LORA_TILE 32→128`、`KV_CHUNK 32→128`、新增 `KV_NOPE_CHUNK=64`),3 次运行 | 2026-05-18 | **1123 / 1174 / 1164 us**(中位数 **1164**) | **884**(−67) | 42-44% | 4342-4715 us | `qkv_proj_rope_l2_swimlane_5{,b,c}.log` |
| Opt H 回退:MPMD 重排(把 Stage 5/6 移到 attn_norm_apply 之后),3 次运行 | 2026-05-18 | 1166 / 1129 / 1196 us(中位数 **1166**) | 884 | 44-45% | 4652-4868 us | `qkv_proj_rope_l2_swimlane_6{,b,c}.log` — 验证了此前的时间线分析;还发现了 `attn_norm_apply` 上的 HBM 拥塞回退(+19us/task = +304us Σ Exec) |

**Opt B 的真实效果(噪声修正后):** `qproj_matmul` 单任务 Exec **8.55us → 7.95us(约 −7%)**,并非最初宣称的 −16%。−16% 是两次不相关的单次运行(一次较慢的 Opt A + 一次较快的 Opt B)之间的差。Opt B 对 wall-clock 的影响 *在噪声范围内* —— 可能略为正向但未确认。

## Opt A — 从 vector scope 中移除 `chunked_loop_optimizer`(接受,2026-05-18)

从 [models/deepseek/v4/qkv_proj_rope.py](../models/deepseek/v4/qkv_proj_rope.py) 的三个纯 vector `pl.at` 块中移除了 `optimization=pl.chunked_loop_optimizer`:

- `attn_norm_apply`(约第 87 行)
- `qproj_dequant`(约第 181 行)
- `q_head_rms_rope`(约第 190 行)

`qproj_matmul` 上保留 —— matmul scope 仍受益。

**根因确认:**`chunked_loop_optimizer` 哨兵会触发 `OutlineIncoreScopes`(编译 pass 11)把 `pl.cast` op 剥离成单独的 outlined 子函数:

- `attn_norm_apply` → cast(BF16→FP32) + vector mul 被拆成 2 个子任务
- `qproj_dequant` → cast(INT32→FP32) + row/col_expand_mul 被拆成 2 个
- `q_head_rms_rope` → RMS reduce + NOPE-loop + rope-prelude + cos cast + rope-math 被拆成 5 个;子 scope 之间的 `pl.slice(rope_cos/sin)` 还充当了额外的 hoist 边界

**经验法则:**只在热点 op 是 `pl.matmul` / `pl.matmul_acc` 的 scope 上使用 `chunked_loop_optimizer`。在纯 vector scope 上,它唯一可观测的效果就是 cast-split 开销。

## Opt B — Q_PROJ tile 调优(部分接受,2026-05-18)

**`Q_PROJ_OUT_CHUNK=256`(N tile)在 runtime 阶段失败**,报错 `ACL_ERROR_RT_AICORE_TIMEOUT`(507018)。编译通过,但 N=256 INT8 matmul 时 AICore 挂死。可能是 a2a3 上 CANN kernel 模板对 `INT8 [64, 128] × [128, 256] → INT32 [64, 256]` 的限制。**在该平台上不要再尝试 N=256,除非先确认 matmul 模板支持。**

**`Q_PROJ_CHUNK=256`(K tile,保留)有效:**把内层累加迭代次数减半(8 → 4)。`qproj_matmul` 单任务 Exec 改善约 7%(8.55 → 7.95 us,多次重跑后)。wall-clock 变化在噪声范围内 —— 微小但不确定为正向。之所以保留,是因为它对 Exec 是严格的改进且没有副作用。

**经验:**wall-clock 有约 4% 的单次运行噪声;声称微小改进需要至少 3 次重跑。另外:并行段中累计的 Exec 收益会被并行 slack 吸收,不一定能转化为 wall-clock 变化。要撬动 wall-clock,得针对串行单任务 scope(rms reduce、qr_quant),或针对 Exec 主导某个并行段的 scope。

## Opt E — 把 `cast_bf16` 融合进 `norm_apply`(接受,2026-05-18)

移除了两个只做 `FP32 → BF16` cast 的独立 `pl.at` scope 循环,将 cast 折叠进生产端 `attn_norm_apply` / `qr_norm_apply` scope:

- **`token_x_cast_bf16`(16 tasks)折叠进 `attn_norm_apply`**。同时彻底移除中间 GM buffer `token_x_fp32` —— `attn_norm_apply` 现在直接写 `token_x_bf16`。
- **`qr_cast_bf16`(32 tasks)折叠进 `qr_norm_apply`**。`qr_fp32` buffer 保留(仍被 `qr_rms` 读取),但 `qr_norm_apply` 不再重写它,而是直接写 `qr_bf16`。

**为什么这次成功了(而 Opt D 失败):**生产者和消费者都是 AIV vector op,不涉及 AIC matmul,所以没有 NZ/ND layout 边界。融合只是"扩展现有 AIV scope 让它再写一个 BF16 输出"。

**效果:**消除了 48 个 task(16 + 32),省掉 48 次启动开销(每次约 7us 尾部 OH)。同时省了 1 个中间 GM buffer(节省 DMA 带宽:每个 `token_x_fp32` chunk 写入 64KB,直接消除)。`attn_norm_apply` 单任务 Exec 由 20.6us → 17.6us(写 BF16 = DMA 减半 + 仅多一次 cast op)。`qr_norm_apply` 由 1.26us → 2.02us(cast 开销只占小部分)。

**净收益:**3 次运行 wall-clock 中位数 1366 → **1193 us(−173us,−12.7%)**。

## Opt G — chunk size 调优(有保留地接受,2026-05-18)

提高了 [models/deepseek/v4/qkv_proj_rope.py](../models/deepseek/v4/qkv_proj_rope.py) 中三个 N 轴 vector/matmul chunk:

- `Q_LORA_TILE = 32 → 128`(同时影响 `Q_LORA_CHUNK`,驱动 `Q_BLOCKS = 32 → 8`)
- `KV_CHUNK = 32 → 128`(驱动 `KV_BLOCKS = 16 → 4`)
- **新常量 `KV_NOPE_CHUNK = 64`** 仅用于 `kv_norm_nope` 循环 —— `NOPE_DIM = 448` 不能被 128 整除,所以 nope-norm vector chunk 必须取 448 的因数(64 = 448/7)。matmul / rms 路径仍用 `KV_CHUNK = 128`。

**为什么安全:**三个 chunk 都是纯 N 轴(输出列)tile size,不改变 dataflow 或累加器 dtype。

**效果 —— 任务数变化:**

| 函数 | 旧值(Opt E) | 新值(Opt G) |
|---|---:|---:|
| `qr_proj_matmul` | 32 tasks × 22.8us | **8 tasks × 31.3us** |
| `kv_proj_matmul` | 16 tasks × 20.4us | **4 tasks × 28.8us** |
| `kv_norm_nope` | 14 tasks × 9.4us | **7 tasks × 3.4us** |
| `qr_norm_apply` | 32 tasks × 1.3us | 8 tasks × 2.0us |
| **总任务数** | **951** | **884**(−67) |

并行 matmul scope 的单任务 Exec 略微上升(tile 更大 = 每任务工作更多,但 cube 利用率更好:`kv_proj_matmul` Exec/Latency 88.1% → 92.6%,`qr_proj_matmul` 58.9% → 85.4%)。Σ Exec 大幅下降,因为固定的单任务启动开销被更少更大的任务摊平。**不要尝试 `Q_PROJ_OUT_CHUNK=256`** —— 已知会触发 `ACL_ERROR_RT_AICORE_TIMEOUT`(见 Opt B)。

**效果 —— wall-clock(3 次运行):**

| Run | Wall-clock | Σ Exec | attn_norm_apply 单任务 Exec |
|---|---:|---:|---:|
| _5(运气好/无拥塞) | 1122.96 us | 4342 us | 18.18 us |
| _5b | 1173.96 us | 4715 us | 39.29 us |
| _5c | 1163.70 us | 4545 us | 37.21 us |
| **中位数** | **1164 us** | 4545 us | 37 us |

wall-clock 中位数 **1193 → 1164 us,−29 us(−2.4%)**——**在约 4% 的单次噪声带内**。Σ Exec 变化(−655us)和任务数变化(−67)都是明确的,但单看 wall-clock 收益在统计上不确定。

**HBM 拥塞观察:**`attn_norm_apply` 单任务 Exec 呈双峰分布 —— 无拥塞时约 18us,拥塞时约 37us —— Opt G 在 3 次中有 2 次落入拥塞模式(而 Opt E 基线稳定在约 17.6us)。假设:`qr_proj_matmul`(8 tasks × 1MB `wq_a` 读取)产生的 HBM 突发比旧的 32×256KB 任务更大,提高了与尾部 `attn_norm_apply` GM 写入发生竞争的概率。matmul 侧每任务的改进被 AIV 写侧每任务的退化部分抵消。

**为何在 wall-clock 模糊的情况下仍保留:**task 数和 Σ Exec 都是严格改进,无正确性或编译开销代价。未来的 scope 融合工作依赖合理的 chunk 几何(更大的 chunk 在后续融合时减少子任务碎片化)。作为新基线保留。

## Opt H — 显式 MPMD 重排(回退,2026-05-18)

把整个 Stage 5/6(KV 路径:`kv_proj_matmul` → `kv_rms` → `kv_norm_nope` → `kv_rope_*`)从函数末尾位置移到 Stage 0.2(`attn_norm_apply`)之后、Stage 1(`qr_proj_matmul`)之前。KV 路径只依赖 `token_x_bf16`,所以两种顺序下 dataflow 都正确;重排意图是给 runtime scheduler 最早的机会去和 Q 侧并行发射 KV 任务。

**结果(3 次运行):**wall-clock 1166 / 1129 / 1196 us,中位数 **1166 us** —— 与单独的 Opt G(中位数 1164 us)无法区分。

**意外子发现:**`attn_norm_apply` 单任务 Exec 在三次重排运行中一致跃升至约 40us(对比 Opt G 的 18-39us 双峰分布)。仅此一个 scope 让 Σ Exec 上升 +295us。机理:把 `kv_proj_matmul`(读取 4MB `wkv` 权重)与 `attn_norm_apply`(AIV 写 `token_x_bf16`)放成显式数据依赖相邻,scheduler 会并发发射两者,导致 HBM 过载并将 `attn_norm_apply` 钉在拥塞执行区。

**结论:**原始源码顺序下的 pypto dataflow scheduler **已经**会把 KV 任务排入 Q 侧空闲槽,但**避开了 `attn_norm_apply` 附近的 HBM 拥塞窗口**。显式源码层重排会破坏这种隐式的拥塞规避。**不要靠重排源码去"强制" MPMD 并行 —— pypto 已经做得更聪明了。**

## 当前状态(Opt A + B + E + G),3 次运行中位数

| 指标 | 值 |
|---|---|
| **Wall-clock** | **1164 us**(范围 1123-1174,±2%) |
| Total Exec | ~4342-4715 us |
| Total tasks | 884 |
| 平均每 task Exec | ~5.1 us |
| Exec / Latency | ~43% |

## 对比原始基线

**1725 us → 1164 us,−561 us,−32.5%**(显著;远超噪声)。2026-05-18 接受作为终点。

## 为什么优化停止于此

剩余约 100us 的 wall-clock 目标是 `q_rope_reassemble`(128 × 2.38us,AIC)+ `q_rope_write`(128 × 1.08us,AIV)。要融合就需要消除基于 matmul 的置换交错,改用纯 AIV 交错(3D reshape 技巧 / scatter_update)。这是算法级别的重写,不只是 scope 边界变化。预估 ROI 约为 −50 到 −100us wall-clock(4-8%);用户接受当前状态,不再追求。

根本性障碍在平台层面:`pypto` 不会在单个 `pl.at` scope 内插入隐式 NZ→ND layout 转换,所以任何 AIC matmul 后接 AIV cast/assemble 必须占用 2 个 runtime task。要干净地消除这个开销,要么:

- 后端特性:在 matmul→vector 边界增加 scope 内 NZ→ND 转换
- 源码重写:用纯 vector 交错替换置换 matmul

## Opt A+B+E+G 下的热点 scope(按累计 Exec 排序)

1. `qproj_matmul`:仍占主导
2. `qr_proj_matmul`:仍是沿 D_BLOCKS=16 的串行链(1 个 matmul + 15 个 matmul_acc)
3. `kv_proj_matmul`
4. `qproj_dequant`
5. `q_head_rms_rope`
6. `attn_norm_apply`:现约 17.6us(原 20.6us)
7. `attn_norm_rms`:单任务约 35us,确认就当前结构而言已接近最优

## 调查并排除的方案(2026-05-18,Opt E 之后)

以下方案看起来有希望,但经验证未通过:

1. **融合 `qproj_matmul` + `qproj_dequant`** —— 与 Opt D 同样的 NZ/ND scope 边界障碍。AIC INT8 matmul 输出在 L0C 中是 NZ;AIV dequant 在 GM 上需要 ND。无法共享一个 `pl.at`。
2. **Q/KV 路径并行(MPMD)** —— 对 Opt E 基线做时间线分析,`kv_proj_matmul` 已在约 1070us 完成,而 `q_rope_write`(实际 wall-clock 尾部)在约 1282us 完成。KV 不在关键路径上;提前它并不改变 wall-clock,反而会与 Q 侧争 dispatch loop(已有 50% 时间空转在 dispatch)。**Opt H(2026-05-18)经验证:**显式源码重排无 wall-clock 收益,且引入了 `attn_norm_apply` 的 HBM 拥塞(每任务 +19us,Σ Exec +304us)。原始源码顺序的 pypto dataflow scheduler 已经在重叠 KV 与 Q 的同时规避拥塞窗口 —— 显式重排破坏了这种隐式规避。
3. **`cube_nbuffer_mode` / `vec_nbuffer_mode` 编译选项** —— 这些参数名在 pypto、ptoas、simpler 或 pypto-lib 树中都不存在。无效。
4. **`Q_PROJ_OUT_CHUNK=512` + `enable_split_k=True`** —— N=256 已经超时;N=512 只会更糟。`split_k` 会在已经 dispatch-bound 的工作负载上增加 reduce 任务(swimlane report Part 2:scheduler 50% idle),净负效果。

## 剩余的优化方向(按优先级)

1. **`q_rope_reassemble` + `q_rope_write` 的算法级重写** —— 用纯 AIV 3D-reshape / scatter_update 替换置换 matmul 交错。目标是唯一剩余的 wall-clock 尾部(在 1239us 运行中 `q_rope_write` 跨度 610us)。剩余 ROI 最大。
2. **并行化关键路径上的单任务串行 reduce**:`attn_norm_rms`(35us)、`qr_rms`(17us)、`kv_rms`(9us)、`qr_quant`(19us)。把内部串行 reduce(16 / 32 / 16 blocks)转为 partial-parallel + final reduce。**注意:**Opt C 已经在 `attn_norm_rms` 拆 16 份时失败 —— pypto 单任务启动开销(约 5-10us)吃掉了细粒度 partial 的收益。用粗粒度 4 路分组,而非 N 路。
3. **`q_head_rms_rope` 每 head 内部的串行 8-chunk reduce**:在 head 内并行。
4. **后端特性请求**:AIC→AIV 边界的 scope 内 NZ→ND 转换可以解锁整个 codebase 的 Opt D 式融合。


## S=2(T=128)后续 — 2026-05-19

commit `05127d2` 把 `DECODE_SEQ` 从 1 切到 2(T=64 → T=128)后,在同一 kernel 上重新验证了上面所有 accepted/reverted 的 opt,参数保持不变(明确跳过 Opt B/G 的 chunk 调优,只评估结构性 opt)。

### 运行配置
- 同 FLASH 预设,但 `DECODE_BATCH=64, S=2, T=128`。其他全部相同(D=4096, H=64, HEAD_DIM=512, Q_LORA=1024)。
- 同命令:`python models/deepseek/v4/qkv_proj_rope.py -p a2a3 -d 0 --enable-l2-swimlane`

### 进展

| Step | Wall-clock | Tasks | Exec/Lat | 备注 |
|---|---:|---:|---:|---|
| S=2 基线(干净) | 1867.8 us | 1575 | 39.7% | 参考;T 翻倍工作量,比 S=1 基线慢约 8% |
| **Opt A**(去掉 3 个 `chunked_loop_optimizer`) | **1733 us** | 1095 | 53.0% | **−7.2% vs S=2 基线** — 保留 |
| Opt C(16-partial `attn_norm_rms`) | 中位数 1714 us(范围 1684-1751) | 1111 | 51.9% | 在 Opt A 噪声内;Σ Exec **上升 +600us** — 回退 |
| **Opt E**(把 `cast_bf16` 融合进 norm_apply) | **中位数 1564 us**(范围 1552-1607) | 1031 | 55.6% | **−9.7% vs Opt A** — 保留 |
| Opt H(MPMD 重排 Stage 5/6) | 中位数 1576 us(范围 1573-1589) | 1031 | 54.9% | 比 Opt E 中位数 +12us,在噪声内 — 回退 |
| **Final(A + E)** | **~1560 us** | 1031 | ~55% | **−307us, −16.4% vs S=2 基线** |

### Opt A 在 T=128 下需要一个结构性前置

`q_head_rms_rope` 在 T=128 下不能去掉 `chunked_loop_optimizer` 而仍保留单 `pl.at` scope —— 融合的 [RMS + NOPE + RoPE] 主体在 RoPE 块中持有约 7 个 FP32 `[T, ROPE_HALF|ROPE_DIM]` tensor(`q_rope_norm`、`q_even`、`q_odd`、`cos`、`sin`、`q_rot_even`、`q_rot_odd`),消耗约 **230 KB**,超过 192 KB Vec 预算。S=1(T=64)时同 scope 峰值约 115 KB,可以容纳;optimizer 的 cast-split 在那里是额外开销,这正是 Opt A 在 S=1 收益的来源。

**S=2 修复:**在每 head 的 `pl.parallel(0, H)` 循环内拆为两个 scope:

- `q_head_rms_nope` —— RMS sq-sum + inv_rms 计算 + NOPE 方向 normalize-and-cast
- `q_head_rope` —— RoPE 方向(norm → gather even/odd → cos/sin → rotate → cast)

`inv_rms` 通过一个 `[H, T]` FP32 暂存 tensor 跨 scope 边界传递。scope 拆分代价是一次 `[H, T]` GM 往返(约 32 KB)和 `H` 次额外 task 启动 —— 远小于去掉 optimizer 的收益。

### Opt C:与 S=1 同样结论(回退),机制不同

S=1 时串行 `attn_norm_rms` 是 35us —— 太短,16 路 partial 无法克服单任务启动开销(上面文档约 5-10us/launch)。S=2 时是 120us,理论上能摊平启动开销,16 个 partial 各运行约 45us 并行。但是:

- wall-clock 中位数实际*改善*约 20us(在约 4% 噪声内)。
- Σ Exec **上升约 +600 us**,因为 partial scope 的单任务工作(约 45us)包含原本作为单任务时没有的 head OH。

净:微小的 wall-clock 收益与噪声在统计上不可区分,而资源开销明确上升。同样回退。

### Opt H:同样结论(回退),原因不同

S=1 时,源码层重排导致 `attn_norm_apply` 因与 `kv_proj_matmul` 的 HBM 过载而每任务 +19us 退化。S=2 下,这个具体的退化*没有*重现 —— 单任务 `attn_norm_apply` 反而下降(93→82us)。但 wall-clock 中位数仍未改善(比 Opt E +12us)。所以来自 `feedback_pypto_dataflow_implicit_contention_avoidance.md` 的根本经验仍成立:原始源码顺序下的 pypto dataflow scheduler 已经在重叠 KV 与 Q,显式重排没有上行收益,这个例子里只是搬动了方差而没有移动中位数。

### 为什么这一轮跳过参数调优(Opt B / Opt G)

用户范围明确只做结构性变更。Opt G 的 chunk 大小(`Q_LORA_TILE = 128`、`KV_CHUNK = 128`、`KV_NOPE_CHUNK = 64`)很可能在 S=2 下也能干净地迁移,但本轮未重新验证。如继续推进,预计在 S=2 下还能再拿约 30-60us wall-clock(与 S=1 边缘的 −2.4% 类似的相对收益)。

### S=2 下的净结果

**1867.8 us → ~1560 us,−307 us,−16.4%**,只用了 `Opt A + Opt E + q_head_rms_rope scope 拆分`。q / kv / qr / qr_scale 在与 S=1 相同容差下验证 PASS。

### Opt Q — `kv_norm_nope` KV_NOPE_GROUP=2(回退,2026-05-20)

变更前:`kv_norm_nope` 是这个文件里唯一没做 GROUP-chunk 的 vec-norm scope:14 逻辑 iter × runtime fanout 2 = 28 tasks × 37us μ,span 60us,在 KV 分支上。

应用了标准 `pl.parallel(0, N, GROUP) + pl.range(GROUP)` 模式,设 `KV_NOPE_GROUP = 2`(NOPE_DIM/KV_CHUNK = 14 = 2×7,候选为 1/2/7/14)。单一跨 iter loop-carried tensor(`kv`)—— 与 Opt J/L/N 成功条件吻合。把 `(nbg + n_inner) * KV_CHUNK` 在所有地方内联,以避开 pypto AST loop-carry 陷阱(初始时报 SSAVerify "Variable 'n0' used outside its defining scope";内联后修复编译)。

**局部效果(scope 自身大幅改进):**

| 指标 | 基线 | Opt Q | Δ |
|---|---:|---:|---:|
| `kv_norm_nope` tasks | 28 | 14 | −14 |
| `kv_norm_nope` 单任务 μ | 37 us | **8.5 us** | (GROUP=2 时编译器生成了更紧凑的代码) |
| `kv_norm_nope` span | 60 us | **14 us** | **−46 us** |
| `kv_norm_nope` Σ Exec | 1041 us | **119 us** | **−922 us** |
| KV 分支结束时间 | ~476 us | **~406 us** | −70 us |

**但 wall 反而退化:**

| Run | 基线(重测) | Opt Q |
|---|---:|---:|
| run 1 | 624.3 us | 655.0 us |
| run 2 | 621.0 us | 660.1 us |
| run 3 | 641.8 us | 660.0 us |
| 中位数 | **624 us** | **660 us(+36 us, +5.8%)** |

**机理 —— 与 Opt H 相同。** KV 分支不在关键路径上(原本 KV 在约 476us 结束,而 Q 尾部在 670us → 约 194us slack)。KV 提前结束不改变 wall,但 scheduler 重排了发射顺序:`kv_proj_matmul` 现在更靠近 Q 侧关键路径入口,`qr_proj_matmul` 起始时刻向后漂移 +21us(195 → 195/228/216us),其 span 增长 +25us(75 → run 3 的 100us)。这个延迟向下游传播到 `qproj_matmul` 起始(基线 389us → run 1/2/3 中 389/405/416us),整条关键路径中位数右移约 36us。

**经验印证 `feedback_pypto_dataflow_implicit_contention_avoidance.md`:** KV 分支已经有约 200us slack。对非关键 scope 的局部改进会通过破坏隐式拥塞规避调度而退化 wall。scheduler 选择把 `kv_norm_nope` 保持在 GRP=1(28 个小 task)本身就在做隐式优化 —— 让这些 task 填进缝隙,而不是聚成大组去争 HBM。

**回退。** 状态:文件以 `kv_norm_nope` GRP=1 发布。

### Opt R — `q_rope_reassemble + q_rope_write` AIV 交错重写(编译失败,2026-05-20)

尝试了之前"剩余优化"小节标注的算法级重写 —— 唯一剩余约 100us wall-clock 路径。目标:把基于 cube 的 even/odd 置换 matmul(`q_rope_reassemble`)+ FP32→BF16 vector 写(`q_rope_write`)折叠进现有的 `q_head_rope` AIV scope,通过纯 AIV op 直接以交错 RoPE layout 写入 `q_flat`。

**尝试的方法:** 3D-assemble + reshape。交错 layout `[e0, o0, e1, o1, ..., e_{ROPE_HALF-1}, o_{ROPE_HALF-1}]` 等价于 `[T, ROPE_HALF, 2]` view 的 C-order 内存,其中 `[..., 0]` 放 even,`[..., 1]` 放 odd。计划:

```python
q_rope_buf = pl.create_tensor([T, ROPE_HALF, 2], dtype=pl.BF16)
q_rope_buf = pl.assemble(q_rope_buf, pl.reshape(q_rot_even_bf16, [T, ROPE_HALF, 1]), [0, 0, 0])
q_rope_buf = pl.assemble(q_rope_buf, pl.reshape(q_rot_odd_bf16,  [T, ROPE_HALF, 1]), [0, 0, 1])
q_flat[:, h*HEAD_DIM+NOPE_DIM : (h+1)*HEAD_DIM] = pl.reshape(q_rope_buf, [T, ROPE_DIM])
```

**编译失败:**
```
'pto.alloc_tile' op expects result row-major none_box tile row byte size
(cols * sizeof(dtype)) to be 32-byte aligned, but got 4 bytes
```

`[T, ROPE_HALF, 1]` BF16 源 tile 行字节数 = 1 × 2 B = 2 B(或者按 ROPE_HALF 当行为 4 B);无论哪种都远低于 **32 字节 AIV tile 对齐要求**。这是 MTE3 流水的硬件级约束 —— vector store 需要 ≥ 32 字节连续块。以 stride-2 元素粒度写 BF16(交错 layout 的固有要求)本质上违反此约束。

**调研过的替代方案:**

- **`pl.transpose` of `[T, 2, ROPE_HALF]` → `[T, ROPE_HALF, 2]`:** 转置后内层 tile 仍以 trailing axis 2 BF16 = 4 字节结尾。下游 reshape 到 `[T, ROPE_DIM]` 要么插入物理拷贝重新撞同一对齐墙,要么直接被拒。
- **`pl.tile.mscatter`:** 只支持 `FP16/FP32/INT16/INT32` —— 无 BF16 路径。且按元素 scatter 到 GM 在硬件上慢(约 1 elem/cycle),对比对齐 MTE3 的 32 elem/cycle 即使 cast 到 FP32 也比 matmul 方案更贵。
- **`pl.tensor.scatter_update`:** 只支持行 scatter(dim=-2)。无法 scatter 到跨步列位置。
- **将 pair 位打包为 INT32 + reinterpret_cast:** pypto 没有 bitcast API(`pl.cast` 只做数值转换)。

**结论 —— 是后端特性缺口,不是源码改写问题。** 印证了文档之前的判断(本节之前的 224-227 行):消除 `q_rope_reassemble + q_rope_write` 的唯一办法是 (a) pypto 暴露硬件原生的 vector 交错原语(tile 层封装的 `vinterleave` / `vmerge` 类指令),或 (b) 编译器在 AIC→AIV 边界插入隐式 NZ→ND 转换,以便现有 matmul 能融合进 `q_head_rope` 的 scope。3D-assemble 方法被 `pto.alloc_tile` 的对齐保护拒绝,且在源码级别没有可行的绕行方案。

**文件回到基线。** 无变更发布。

### Opt M 在 Opt P 之后重测,出现退化(2026-05-20)

接受 Opt P 后,尝试在 M-revert+N+O+P 状态上重新加回 Opt M(q_head_rms_nope GRP=8)。3 次测量:

| Run | wall |
|---|---:|
| run 1 | 671.9 us |
| run 2 | 661.6 us |
| run 3 | 652.8 us |
| 中位数 | **661.6 us** |

vs 单独 P 中位数 624 us → **叠加 M 上去 +38us 退化。**

**机理:** Opt P 缩小 qproj_dequant 的 span(100→35us),dequant 现约 465us 结束。下游入口提前后,`q_head_rms_nope` 的 span 成为新的关键路径主导:

- GRP=1: 64 tasks × 14us μ 分布到 48 cores → span 41us → 约 506us 结束
- GRP=8: 8 tasks × 97us μ 分布到 8 cores → span 102us → 约 580us 结束

GRP=1 显著更短的 span 在新瓶颈位置上获胜。GRP=8 的更大 μ 此前被下游 slack 隐藏,现在被暴露。

**M 永久回退。** 最终状态 `q_head_rms_nope = GRP=1`。

**通用经验:** 在隔离条件下边缘的 chunking 选择,会因为相邻优化加速完成而反转符号。每次相邻改进可能移动关键路径后,要重新验证此前接受的每个 CHUNK 选择。memory 条目 [`feedback_pypto_head_group_chunking_loop_carried.md`](../../.claude/projects/-data-<user>-newpto-pypto-lib/memory/feedback_pypto_head_group_chunking_loop_carried.md) 已加上此规则。


## Opt M 回退 + Opt P — qproj_dequant 解耦 chunking(接受,2026-05-20)

### Opt M 按用户要求回退

Opt M(`q_head_rms_nope` HEAD_GROUP=8)有边缘 wall 改进(中位数约 6us),但把单任务 μ 推到 97us —— 远超约 50us 的 dispatch 目标。此前保留 Opt M 的决策只基于 wall 中位数;用户要求回退,因为单任务 kernel 过大。恢复 `pl.parallel(0, H, 1)` 后,scope 回到 64 tasks × μ=14us、span=39us,跨 48 个 core —— 干净的 fan-out,更贴合设计目标。回退后单次 wall 从 681 → 694us(在噪声内;Opt M 的收益始终 ≤ 10us)。

### Opt P — qproj_dequant 与 qproj_matmul 解耦

之前在 Opt M/N/O 那一轮跳过了,因为它需要一个 **16 MB** 的全局 INT32 暂存 tensor(`col_acc_all` = `[Q_PROJ_HEAD_BLOCKS × T, Q_PROJ_OUT_CHUNK]` = `[256 × 128, 128]`)。再回看,这在 a3 上可以接受(HBM 充足,无 DMA 带宽热点),用户接受了这个代价。

实现:

```python
q_proj_fp32 = pl.create_tensor([T, H * HEAD_DIM], dtype=pl.FP32)
col_acc_all = pl.create_tensor([Q_PROJ_HEAD_BLOCKS * T, Q_PROJ_OUT_CHUNK], dtype=pl.INT32)

# Stage 3a — qproj_matmul 写入全局暂存
for hg in pl.parallel(0, Q_PROJ_HEAD_BLOCKS, Q_PROJ_GROUP):
    with pl.at(..., "qproj_matmul"):
        col_acc = pl.create_tensor([T, Q_PROJ_OUT_CHUNK], dtype=pl.INT32)  # pre-decl
        for h_inner in pl.range(Q_PROJ_GROUP):
            for qb in pl.pipeline(0, Q_PROJ_BLOCKS, stage=2):
                ... matmul/matmul_acc into col_acc ...
            col_acc_all = pl.assemble(col_acc_all, col_acc, [(hg + h_inner) * T, 0])

# Stage 3b — qproj_dequant 独立运行,使用自己更大的 GROUP
for hbg in pl.parallel(0, Q_PROJ_HEAD_BLOCKS, Q_PROJ_DEQUANT_GROUP):
    with pl.at(..., "qproj_dequant"):
        for h_inner in pl.range(Q_PROJ_DEQUANT_GROUP):
            col_acc_chunk = pl.slice(col_acc_all, ..., [(hbg + h_inner) * T, 0])
            ... cast + scale + assemble into q_proj_fp32 ...
```

为什么没掉进 Opt I "≥2 跨 iter loop-carried tensor" 陷阱:两个 `pl.parallel` 循环完全分离 —— `col_acc_all` 仅在 matmul 循环的 iter 内 loop-carried,`q_proj_fp32` 仅在 dequant 循环的 iter 内 loop-carried。每个循环只有 1 个跨 iter 输出,符合最新版的 [memory:feedback_pypto_head_group_chunking_loop_carried.md](../../.claude/projects/-data-<user>-newpto-pypto-lib/memory/feedback_pypto_head_group_chunking_loop_carried.md) 标准。

### DEQUANT_GROUP 扫参

| GROUP | wall(单次) | dequant tasks | dequant μ | dequant span |
|---:|---:|---:|---:|---:|
| 8(耦合,Opt J 状态) | 657 us | 32 | 13 us | 100 us |
| 8(解耦) | 657 us | 32 | 13 us | 33 us |
| **16(解耦)** | **628 us** | **16** | **23 us** | **35 us** |
| 32(解耦) | 635 us | 8 | 42 us | 45 us |

GRP=16 是甜点。GRP=32 单任务 μ=42us(最贴近 50us 目标)但失去并行度(只有 8 个外层 iter)。

### 结果(3 次运行中位数,对比 M-reverted+N+O 状态)

| Run | wall |
|---|---:|
| run 1 | 624.4 us |
| run 2 | 631.1 us |
| run 3 | 623.7 us |
| **中位数** | **624.4 us**(范围 7.4us,约 1.2%) |

| 指标 | M-reverted+N+O | Opt P(DEQUANT_GROUP=16) | Δ |
|---|---:|---:|---:|
| **wall 中位数** | 646 us | **624 us** | **−22 us(−3.4%)** |
| `qproj_dequant` tasks | 32 | 16 | −16 |
| `qproj_dequant` 单任务 μ | 13 us | 23 us | +77% |
| `qproj_dequant` span | 100 us | 35 us | −65 us |
| `qproj_dequant` 不同 core 数 | 22 | 16 | 干净 fan-out |

### 累计结果

| 状态 | wall(中位数) | 备注 |
|---|---:|---|
| 调优前 S=2 基线 | 1868 us | 参考 |
| Opt J 后 | 849 us | qproj_* chunked |
| Opt K 后 | 820 us | q_head_rope chunked |
| Opt L 后 | 687 us | qr_norm_apply chunked |
| Opt M 后(保留) | 681 us | q_head_rms_nope chunked GRP=8 |
| Opt N 后 | 614 us | attn_norm_apply chunked |
| Opt O 后 | 599 us | kv_proj_matmul chunked |
| **Opt M-revert + P 后** | **624 us** | rms_nope 回到 GRP=1;dequant 解耦 |

**1868 us → 624 us,−66%。** 验证 PASS。

## Opt M/N/O — 又三个 chunking 胜利(接受,2026-05-20)

Opt L 之后,用同样模式对三个 scope 做了 chunking:

| Opt | Scope | 新 GROUP | wall 前 | wall 后 | Δ |
|---|---|---:|---:|---:|---:|
| **M** | `q_head_rms_nope` | HEAD_GROUP=8 | 687 us | 681 us | −6 us |
| **N** | `attn_norm_apply` | ATTN_NORM_GROUP=8 | 681 us | 614 us | −67 us |
| **O** | `kv_proj_matmul` | KV_PROJ_GROUP=2 | 614 us | 599 us | −15 us |
| | (三者完成后 3 次中位数) | | | **646 us** | |

### Opt M — `q_head_rms_nope` HEAD_GROUP=8(意外的胜利)

预期会退化:变更前该 scope 已经处于"task 数 ≈ core 数"的甜点(64 tasks 分布到 48 个 AIV core,span 38us)。8x chunking 预期会过头。

实际:wall −6us。scope 自身按设计变慢(span 38 → 104us,8 tasks × 97us μ 跨 8 个 core),但结束时间几乎不变(约 577us),而下游 `q_head_rope` 现在可以提前约 25us 启动,因为从 64 个细粒度 task → 8 个大 task 的 scope 末尾屏障砍掉了一大段 dispatch 尾。

**重要经验** —— `q_head_rms_nope` 有 **2** 个跨 iter loop-carried tensor(`q_flat`、`q_head_inv_rms_all`),依然能跨 8 个 core 干净 chunk。这修正了 [memory:feedback_pypto_head_group_chunking_loop_carried.md](../../.claude/projects/-data-<user>-newpto-pypto-lib/memory/feedback_pypto_head_group_chunking_loop_carried.md) 的判断:"≥2 loop-carried tensor 必败"是错的。Opt I 的 3-tensor 失败可能是 `q_rope_pair_stage` 的 `[H*T, ROPE_DIM]` stride 模式特有问题,或者与 orch-tensor optimizer 的交互。**先试,再测,再决定** —— 根据"loop-carried tensor 数"预测不可靠。

GRP=16 退回到 772us(只有 4 个并行 iter,单任务 span 202us)。保留 GRP=8。

### Opt N — `attn_norm_apply` ATTN_NORM_GROUP=8(三者中最大胜利)

Stage 0.2 的融合 norm-and-cast scope 原本是 32 tasks × 46us。每个 task 已经在 50us 目标之上,所以不是显然的 dispatch-bound 候选。但该 scope 正好坐在**串行 `attn_norm_rms` 之后的关键路径**上,卡住所有 Q 侧下游工作(`qr_proj_matmul`、`qproj_matmul`)。

扫参:

| ATTN_NORM_GROUP | wall | 备注 |
|---:|---:|---|
| 1(基线) | 681 us | 32 tasks × 46us μ,span 75us |
| 2 | 646 us | −35us |
| 4 | 625 us | −56us |
| **8** | **614 us** | **−67us,甜点** |
| 16 | 626 us | 退化(只有 2 个外层 task) |

收益主要来自 dispatcher unblock:`qr_proj_matmul` 起始从 184us(Opt L)移到 152us(Opt N GRP=4),在 GRP=8 时还更早。task 数从 32 减半到 16(GRP=2)节省了超过自身的时间;再减半省下更多下游发射 slack。

### Opt O — `kv_proj_matmul` KV_PROJ_GROUP=2(现在 cube-bound)

16 tasks × 22us μ —— 在 50us 目标以下但已在噪声层以上。

| KV_PROJ_GROUP | wall | 备注 |
|---:|---:|---|
| 1(基线) | 614 us | 16 tasks × 22us,span 78us |
| **2** | **599 us** | **−15us,8 tasks × 47us** |
| 4 | 621 us | 退化(4 个外层 × 78us,并行度更低) |

GRP=2 达到每 task 47us —— 正中 50us 目标。

### 跳过的 opt — `qproj_dequant` 解耦 chunking

考虑过:把 `qproj_dequant` 从 `qproj_matmul` 的外层 `pl.parallel` 解耦,用自己的(更大)GROUP chunk。这需要一个全局 `[Q_PROJ_HEAD_BLOCKS × T, Q_PROJ_OUT_CHUNK]` INT32 暂存 buffer = 每次调用 **16MB**,且这成为一个跨 matmul 外层循环的 iter loop-carried tensor —— 重新引入 Opt I 风格的"≥2 跨 iter 输出"风险。未推进 —— `qproj_dequant` 的 13us μ 落在 `qproj_matmul` 的 span 之内(有显著重叠),对 wall 贡献已经很小。

### 累计结果

| 状态 | wall(中位数) | 备注 |
|---|---:|---|
| 调优前 S=2 基线 | 1868 us | 参考 |
| Opt A+E + scope-split 后 | 1560 us | 结构性 |
| Opt B-2 | 1224 us | K-tile 256 |
| Opt J | 849 us | qproj_* chunked |
| Opt K | 820 us | q_head_rope chunked |
| Opt L | 687 us | qr_norm_apply chunked |
| Opt M | 681 us | q_head_rms_nope chunked |
| Opt N | 614 us | attn_norm_apply chunked |
| Opt O | 599 us | kv_proj_matmul chunked |
| **最终 3 次中位数** | **646 us** | (范围 634–647,约 2% 噪声) |

**1868 us → 646 us,−65%。** q / kv / qr / qr_scale 在记录的容差下验证 PASS。


## Opt L — QR_NORM_GROUP-chunk `qr_norm_apply`(接受,2026-05-20)

Opt K 之后,`qr_norm_apply` 是下一个明显 dispatch-bound 的 scope:32 tasks × 2.7us μ,span 24.7us —— 几乎纯 AICPU dispatch 开销。虽然它自身对 wall 贡献微小,但 scope 坐在 `qr_rms` 与 `qr_quant_amax` 之间的关键路径上,其 span 推迟了下游 `kv_proj_matmul` / `qproj_matmul` 的启动。

与 Opt J/K 同样的 chunking 模式。单一跨 iter loop-carried tensor(`qr_bf16`);符合成功条件。

```python
for qbg in pl.parallel(0, Q_BLOCKS, QR_NORM_GROUP):
    with pl.at(..., "qr_norm_apply"):
        for q_inner in pl.range(QR_NORM_GROUP):
            qr_norm_col0 = (qbg + q_inner) * Q_LORA_CHUNK
            ... assemble into qr_bf16 ...
```

### `QR_NORM_GROUP` 扫参

| GRP | wall | qr_norm tasks | qr_norm μ | qr_norm span | 备注 |
|---:|---:|---:|---:|---:|---|
| 1(Opt K) | 820 us | 32 | 2.73 us | 24.7 us | 基线 |
| 4 | 775 us | 8 | 7.3 us | 13.8 us | −45 us |
| **8** | **687 us** | **4** | **7.8 us** | **9.2 us** | **−133 us vs Opt K** |
| 16 | 698 us | 2 | 14.4 us | 14.8 us | +11 us vs GRP=8(噪声内;并行度更低) |

接受 `QR_NORM_GROUP = 8`。GRP=16 把并行度降到 2 task 而无进一步收益。除了 `qr_norm_apply` 自身节省约 16us 之外,主要收益来自解锁下游链:

- `kv_proj_matmul` 起始 437 → 400us(GRP=4)→ GRP=8 还更早
- `qproj_matmul` span 105 → 121us(GRP=8)—— wallclock 略长但起始更早
- 整条 Q 侧关键路径压缩约 130us

### 结果(单次运行,vs Opt K)

| 指标 | Opt K | Opt L(GRP=8) | Δ |
|---|---:|---:|---:|
| **wall** | 820 us | **687 us** | **−133 us(−16.2%)** |
| `qr_norm_apply` tasks | 32 | 4 | −28 |
| `qr_norm_apply` span | 24.7 us | 9.2 us | −15.5 us |

验证 PASS。收益(−133us)远大于 `qr_norm_apply` 自身 Σ Exec 节省(约 30us),证实该 scope 是阻塞下游发射的 dispatch 瓶颈点。

## Opt K — HEAD_GROUP-chunk `q_head_rope`(与 rms_nope 解耦)(接受,2026-05-20)

Opt J 把关键路径从 Stage 3 移走后,新的尾部是 `q_head_rms_nope + q_head_rope`(合计约 180us span)。Opt I 之前曾尝试在两个 scope 都在同一外层 pl.parallel 内时对它们做 chunk,失败 —— 3 个跨 iter loop-carried tensor 把 scope 钉在 4 个 core。新做法:**解耦** 两个 scope:

- `q_head_rms_nope` 保持 `pl.parallel(0, H, 1)`(64 个细粒度 task 分布到 48 个 AIV core,span 36us —— 已经接近最优)
- `q_head_rope` 移到**自己的** `pl.parallel(0, H, HEAD_GROUP)` + `pl.range(HEAD_GROUP)`,只有**一个**跨 iter loop-carried tensor(`q_rope_pair_stage`)

这正好符合 Opt J 的成功条件:一个有单一 loop-carried 输出的 chunked scope 能干净地跨 HEAD_GROUP 个 core 调度。

```python
# rms_nope 不变(在 H 上细粒度)
for h in pl.parallel(0, H, 1):
    with pl.at(..., "q_head_rms_nope"):
        ... assemble into q_flat, q_head_inv_rms_all ...

# rope 单独 chunk —— 单一跨 iter 输出
for hg in pl.parallel(0, H, HEAD_GROUP):
    with pl.at(..., "q_head_rope"):
        q_head_inv_rms_t = pl.create_tensor([T, 1], dtype=pl.FP32)  # pre-decl
        for h_inner in pl.range(HEAD_GROUP):
            ... assemble into q_rope_pair_stage ...
```

### 结果(单次运行,vs Opt J)

| 指标 | Opt J | Opt K | Δ |
|---|---:|---:|---:|
| **wall** | 849 us | **820 us** | **−29 us(−3.4%)** |
| `q_head_rope` tasks | 64 | **8** | −56 |
| `q_head_rope` 单任务 μ | 5.8 us | **32.7 us** | +5.7×(高于 50us 目标带) |
| `q_head_rope` span | 103 us | **37 us** | **−66 us** |
| `q_head_rope` 不同 core 数 | 42 | **8** | 干净 fan-out |
| `q_head_rope` Σ Exec | 370 us | 262 us | −108 us |

`q_head_rms_nope` 也间接受益:span 123 → 36us,不同 core 数 41 → 48 —— 不再与 chunked rope scope 共享并行度预算。

### 为什么 Opt K 成功而 Opt I 失败

同一 `pl.parallel + pl.range(CHUNK)` 模式,结果相反:

| | Opt I(rms_nope + rope 合并) | Opt K(只 rope) |
|---|---:|---:|
| 跨 iter loop-carried tensor | **3**(q_flat、q_head_inv_rms_all、q_rope_pair_stage) | **1**(q_rope_pair_stage) |
| 使用 core / task | 4 / 8(被钉住,退化) | 8 / 8(干净) |

解耦的关键在于:Opt I 中的 3 个 tensor 在不同 iter 间有不同的写模式;尽管 pypto 看到它们 assemble 到不相交的 slice,runtime 仍保守地通过小 core 池序列化执行。

`q_head_rms_nope` **故意没** chunk:它有 2 个 loop-carried tensor(`q_flat` 用于 NOPE 写、`q_head_inv_rms_all`),且 64 task 细粒度版本已在 48 core 上以 36us span 饱和,无改进空间。

## Opt J — Q_PROJ_GROUP-chunk `qproj_matmul` + `qproj_dequant`(接受,2026-05-20)

**净结果:wall-clock 1224us → 约 850us,−31%(本 kernel 单次最大胜利)。**

Opt B-2 减少了单任务 cube 工作后,swimlane 仍显示 `qproj_matmul` 与 `qproj_dequant` 是主导 span 对(合计约 316us)。256 个外层 pl.parallel task,均值 9.4us/task,cube 利用率约 31%(24 个 cube core 中只有 8.5 个有效)—— 远低于"AICPU dispatch 是瓶颈"阈值(perf-doc 规则 2)。应用了 `q_rope_reassemble` / `q_rope_write` 已使用的 HEAD_GROUP 模式:

```python
for hg in pl.parallel(0, Q_PROJ_HEAD_BLOCKS, Q_PROJ_GROUP):    # 256 → 32 outer
    col_acc_grp = pl.create_tensor([Q_PROJ_GROUP * T, Q_PROJ_OUT_CHUNK], dtype=pl.INT32)
    with pl.at(..., "qproj_matmul"):
        col_acc = pl.create_tensor([T, Q_PROJ_OUT_CHUNK], dtype=pl.INT32)  # pre-decl
        for h_inner in pl.range(Q_PROJ_GROUP):
            for qb in pl.pipeline(0, Q_PROJ_BLOCKS, stage=2):
                ... matmul/matmul_acc into col_acc ...
            col_acc_grp = pl.assemble(col_acc_grp, col_acc, [h_inner * T, 0])
    with pl.at(..., "qproj_dequant"):
        col_acc_chunk = pl.create_tensor([T, Q_PROJ_OUT_CHUNK], dtype=pl.INT32)  # pre-decl
        for h_inner in pl.range(Q_PROJ_GROUP):
            col_acc_chunk = pl.slice(col_acc_grp, ...)
            ... cast + scale + assemble into q_proj_fp32 ...
```

避开了 [memory:feedback_pypto_head_group_chunking_loop_carried.md](../../.claude/projects/-data-<user>-newpto-pypto-lib/memory/feedback_pypto_head_group_chunking_loop_carried.md) 中的两个 AST 陷阱:在所有地方内联 `(hg + h_inner) * Q_PROJ_OUT_CHUNK`(在 `pl.range` 内不绑定 Python 局部),且 `col_acc` / `col_acc_chunk` 在 `pl.range` body 外预先 `pl.create_tensor`,以给 pypto 的 loop-carried init_values 线索一个有效的外部源。

### 结果(Opt B-2 基线 vs Opt A,标注处为 3 次中位数)

| 指标 | Opt B-2(3 次中位数) | Opt A(2 次) | Δ |
|---|---:|---:|---:|
| **wall** | **1224 us** | **848–850 us** | **−375 us(−31%)** |
| `qproj_matmul` tasks | 256 | **32** | −224 |
| `qproj_matmul` 单任务 μ | 9.4 us | **41–46 us** | +4×(高于 50us 目标) |
| `qproj_matmul` span | 318 us | **100–105 us** | **−215 us** |
| `qproj_matmul` 不同 core 数 | 24 | 16 | (核数变少但基本全并行) |
| `qproj_matmul` Σ Exec | 2426 us | 1317–1466 us | **−960 to −1109 us(−40 至 −46%)** |
| `qproj_dequant` tasks | 256 | 32 | −224 |
| `qproj_dequant` 单任务 μ | 2.6 us | 12.3 us | +4.7× |
| `qproj_dequant` span | 307 us | 98–107 us | **−200 us** |
| `qproj_dequant` Σ Exec | 727 us | 395 us | **−332 us(−46%)** |

两个二阶改进,都是 Stage 3 提前完成的连锁效应:

- **`q_head_rms_nope` span 344us → 123us(−221us)** —— Stage 4 的 AIV scope 链更早启动,发现 AIV core 拥塞更小,并行更干净(run 1 跨 41 个 core)。
- **`q_head_rope` span 354us → 103us(−251us)** —— 同一机制。

### 为什么 Opt I 失败而这里成功

同一 `pl.parallel + pl.range(CHUNK)` 形状,结果相反。差别正是 [memory:feedback_pypto_head_group_chunking_loop_carried.md](../../.claude/projects/-data-<user>-newpto-pypto-lib/memory/feedback_pypto_head_group_chunking_loop_carried.md) 警告过的 —— 跨 pl.parallel iter loop-carried tensor 的数量:

| Scope 对 | 跨 iter loop-carried tensor | 使用 core / task |
|---|---:|---|
| Opt I `q_head_rms_nope` + `q_head_rope` | **3**(q_flat、q_head_inv_rms_all、q_rope_pair_stage) | 4 / 8(被钉,退化) |
| Opt J `qproj_matmul` + `qproj_dequant` | **1**(q_proj_fp32;col_acc_grp 是单 outer-iter 局部) | 16 / 32(干净 fan-out) |

matmul scope 自身甚至零跨 iter 输出 —— col_acc_grp 在每个 pl.parallel iter 中都新 `pl.create_tensor` 出来,镜像了能正常工作的 `q_rope_grp_fp32` 模式。

### wall-clock 都去哪儿了

Opt J 把关键路径搬走了:Stage 3(`qproj_matmul + qproj_dequant`)不再是主导。新时间线(来自 run 1,wall=850us):

| 区间 | Scope | 耗时 |
|---|---|---:|
| 0–84 | `attn_norm_rms`(串行,阻塞) | 84 us |
| 99–179 | `attn_norm_apply`(并行 × 32 core) | 80 us |
| 202–384 | Q 侧 LoRA + quant 链 | 182 us |
| 393–541 | **`qproj_matmul + qproj_dequant`** | **148 us** |
| 571–750 | `q_head_rms_nope + q_head_rope` | 179 us |
| 721–850 | KV 路径 + RoPE reassemble/write | 129 us |

剩余关键路径成分:`attn_norm_rms`(串行 84us 前置)、`q_head_rms_nope/rope`(合计约 180us span)。两者都比新的 Stage 3 小,所以下一步 wall-clock 优化应针对它们。

### Q_PROJ_GROUP 扫参 —— 8 是甜点

试过 `Q_PROJ_GROUP = 16` 想把 `qproj_dequant` 单任务 μ 推近 50us dispatch 目标(实际 12us → 23us)。结果:wall **退化 +83us**(849 → 932us)。

| 指标 | GRP=8 | GRP=16 |
|---|---:|---:|
| wall | 849 us | **932 us**(+83) |
| `qproj_matmul` 单任务 μ | 41 us | 79 us |
| `qproj_matmul` span | 100 us | 106 us(平) |
| `qproj_dequant` 单任务 μ | 12 us | **23 us** |
| `qproj_dequant` span | 98 us | **56 us**(−42) |
| `q_head_rms_nope` span | 123 us | 39 us |
| `q_head_rope` span | 103 us | 50 us |

两个目标(dequant 和下游 Stage 4)确实改进,但 `qr_norm_apply` 与 `qr_quant_amax` 之间出现了一个新的 78us AICPU dispatch 间隙(GRP=8 时约 14us)—— 这两个 scope 与 qproj 没有数据依赖,所以这是 scheduler 级别的副作用:16 个大 qproj task 排队后,AICPU dispatcher 卡住了 Q 侧链。即使局部胜利,净 wall 更差。

保留 `Q_PROJ_GROUP = 8` 作为甜点。

### 评估并接受的风险

- **`col_acc_grp` GM 占用**: [Q_PROJ_GROUP × T, Q_PROJ_OUT_CHUNK] INT32 = 8 × 128 × 128 × 4 B = 每个 outer iter **512 KB**,作为 scope-local 在 pl.parallel body 内分配。镜像可工作的 `q_rope_grp_fp32` 模式(256 KB FP32)。验证 PASS,无 OOM。
- **NZ→ND fixpipe 数不变**:重构前每次调用做 256 × 1 = 256 次隐式 L0C→GM 转换;Opt J 在 matmul scope 内通过 `pl.assemble(col_acc_grp, col_acc, [h_inner * T, 0])` 做 32 × 8 = 256 次。同样的 per-element 转换工作量,但批为 32 个 task 而非 256。
- **验证 PASS** 在 q / kv / qr / qr_scale 上以记录的容差。

## Opt B-2 — 在 S=2 重新应用 `Q_PROJ_CHUNK = 128 → 256`(接受,2026-05-20)

Opt B 的 K-tile 变更最初在 S=1 验证过,后来显然回退了(当前 `models/deepseek/v4/qkv_proj_rope.py` 以 `Q_PROJ_CHUNK = 128` 发布)。在 S=2(T=128)下两侧各 3 次中位数重测:

| 指标 | 基线(K=128) | Opt B-2(K=256) | Δ |
|---|---:|---:|---:|
| **wall 中位数** | 1232 us | 1224 us | **−8 us(−0.6%,在噪声内)** |
| `qproj_matmul` Σ Exec | 2774 us | 2426 us | **−347 us(−12.5%)** |
| `qproj_matmul` 单任务 μ | 10.83 us | 9.40 us | −13.3% |
| `qproj_matmul` span | 321 us | 318 us | 平 |
| `qproj_dequant` Σ Exec | 729 us | 727 us | 平 |

把 K 流水阶段从 8 → 4 减半,使每任务 cube 工作下降约 13%,但 task 数不变(仍是 256,因为外层 pl.parallel 是 `Q_PROJ_HEAD_BLOCKS` 而非 `Q_PROJ_BLOCKS`)。span 不变是因为 `qproj_matmul` 关键路径不是 cube-bound —— cube 利用率约 31%(Σ Exec / (span × 24 cube core)),所以更小的 Σ Exec 不会转换为 span 缩短。

按原 Opt B 同样逻辑**保留**:严格 Σ Exec 改进,无编译/runtime 问题,验证 PASS。wall 收益单看在统计上不确定,但为未来可能把瓶颈推到 cube 上的优化提供了空间。

**注意事项不变:**`Q_PROJ_OUT_CHUNK = 256` 仍会触发 `ACL_ERROR_RT_AICORE_TIMEOUT`(见原 Opt B)—— **不要**尝试 N-tile 增大。只有 K-tile(`Q_PROJ_CHUNK`)是安全的。

## Opt I — HEAD_GROUP-chunk `q_head_rms_nope` + `q_head_rope`(回退,2026-05-20)

变更前在当前接受状态(Opt A+B+E+G + S=2 结构性拆分)下测得 swimlane wall=1291us,per-head Stage 4 链可见地受限:

| Scope | tasks | Σ Exec | span |
|---|---:|---:|---:|
| `q_head_rms_nope` | 64 | 798 us | 387 us |
| `q_head_rope` | 64 | 318 us | 381 us |

每个 per-head task 平均约 12–17us —— 远低于 perf-doc 约 50us/kernel 目标 —— 暗示 AICPU dispatch-bound(参 [docs/performance-tuning.md](performance-tuning.md) 规则 2)。兄弟 scope `q_rope_reassemble` / `q_rope_write` 已经成功使用 HEAD_GROUP-chunking(8 tasks × 8 head 每个),所以应用了同一模式:

```python
for hg in pl.parallel(0, H, HEAD_GROUP):
    with pl.at(..., "q_head_rms_nope"):
        for h_inner in pl.range(HEAD_GROUP):
            # 一个 head 的 RMS sq-sum + NOPE 方向 normalize-and-cast
            ...
    with pl.at(..., "q_head_rope"):
        for h_inner in pl.range(HEAD_GROUP):
            # 一个 head 的 RoPE body
            ...
```

实现期间出现两个 pypto AST 怪异点:

1. **不要在 `pl.range(HEAD_GROUP)` body 内把 `h = hg + h_inner` 绑定到 Python 局部。** AST 分析器把循环 body 中设置的任何 Python 局部通过 `init_values` 串起来,但 `h` 没有有效的外部初始值 → SSA 验证失败,报 `Variable 'h_inline20' used outside its defining scope`。**修复:** 在所有地方内联 `(hg + h_inner)` 和 `(hg + h_inner) * HEAD_DIM` —— 与可用的 `q_rope_reassemble` 完全一致。
2. **在 `pl.range(HEAD_GROUP)` body 之前预先声明 `q_head_inv_rms_t = pl.create_tensor([T, 1], dtype=pl.FP32)`。** 当一个 Python 局部 tensor 在外层循环 body 中定义,且被嵌套 `pl.range`(此处 `pl.range(NOPE_DIM // HEAD_CHUNK)`)使用时,pypto 会通过外层循环的 `init_values` 串它。占位 `create_tensor` 给该串联一个有效的预循环初始值。

**验证 PASS** 在 q/kv/qr/qr_scale 上以同样容差。单次运行 wall=**1388us** —— 明显退化(比变更前 1291us 测量 +97us,远超约 4% 噪声带)。

**为什么退化(根因):**

| Scope | tasks | Σ Exec | span | 不同 core 数 |
|---|---:|---:|---:|---:|
| `q_head_rms_nope`(变更前,64 tasks) | 64 | 798 us | 387 us | **27** |
| `q_head_rms_nope`(Opt I,8 tasks) | 8 | 763 us | **525 us** | **4** |
| `q_rope_reassemble`(已 chunk,8 tasks) | 8 | 82 us | 16 us | **8** |

chunking 在 Σ Exec 上达到了预期效果(每 scope −35us、−98us —— 启动开销摊平),但 8 个大并行 iter 只落到 **3-4 个不同 AIV core**,而非 8 个。每任务约 95us,4 个 core 服务 8 个 iter,理想 span 应是 `ceil(8/4) × 95 = 190us`;实测 525us 表明 scheduler 在每个 core 上每两个 task 之间另加了约 80us 空闲间隙。

与可工作的 `q_rope_reassemble`(8 tasks → 8 不同 core)对比是诊断关键。两个 scope 都有同样外层 `pl.parallel(0, H, HEAD_GROUP)` 形状,但区别在**有多少 tensor 在并行迭代间 loop-carried**:

- `q_rope_reassemble`:只写 scope-local `q_rope_grp_fp32`(在每个并行 iter body 中通过 `pl.create_tensor` 新建)。零跨 iter tensor 串联。在 8 个 core 上干净调度。
- `q_head_rms_nope` / `q_head_rope`:写**三个**跨 iter loop-carried tensor —— `q_flat`(NOPE 写)、`q_head_inv_rms_all`(每 head inv_rms)、`q_rope_pair_stage`(BF16 rope 输出)。即使每个 iter assemble 到每个 tensor 不相交的 slice,IR 还是把单一 SSA 值通过 pl.parallel 的 `init_values` 串起来,pypto runtime 保守地把迭代序列化到 3-4 个 core 池中。

64-iter 基线虽然实际并行也只有 2 个有效 core,但许多小 task 让 dispatcher 把它们轮转到 **27** 个不同 core 上,而不是钉住小池。净:同样的串联深度下,多个细粒度 chunk 比少数大 chunk 分布到更多 core 上。

**经验** —— `pl.parallel + pl.range(HEAD_GROUP)` chunking 只在被 chunk 的 scope 输出是 scope-local 时(例如通过并行 body 内的 `pl.create_tensor` 中介)才有收益。assemble 到多个跨 iter loop-carried tensor 的 scope 应保持 `pl.parallel(0, N, 1)`,即使每任务工作低于 50us dispatch 目标。

**状态 —— 文件已接近最优。** Opt A+B+E+G 的接受状态(S=1)加上 `q_head_rms_rope` scope 拆分(S=2 必需)是终点;`q_rope_reassemble` / `q_rope_write` 剩余约 100us 尾部需要算法级重写(用纯 AIV 交错替换基于 matmul 的 even/odd 置换),已经因为 ROI 与代价不匹配而排除。
