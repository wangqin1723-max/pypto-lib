# DeepSeek V4 Decode：PyPTO 方案 3 与 AscendC 性能对比

> 状态：已补充 EP8/TP4 PyPTO 正式 100 轮性能数据；任务在测量完成后的
> teardown 阶段超时，因此对比结论为预发布结果。
> 更新时间：2026-07-31
> 报告位置：本地 `docs/` 目录；当前文件未被 Git 跟踪，未纳入任何分支提交。

## 1. 结论摘要

本报告比较以下两个实现的 DeepSeek V4 decode-fwd 性能：

- **PyPTO 方案 3**：`hc_head + RMSNorm + LM head`，输出 FP32 logits，不包含采样。
- **AscendC 基线**：从第一个 `HcPre` 开始，到设备侧 ArgMax 前最后一个
  `aclnnInplaceCopy_CastAiCore_Cast` 结束。

当前已确认：

- PyPTO 方案 3 分支直接基于方案 2：
  `dsv4-ascendc-decode-compare2@a79c141`。
- 当前被测提交为
  `67235ef8f096ae36ca005c1bacd661580ab2f53a`。
- EP8/TP4 静态契约通过：全局 128 个专家、每卡 16 个专家、B=4、
  T=8、start_pos=8192。
- AscendC 单次 trace 的对齐区间为 **46.2275 ms**。
- PyPTO EP8/TP4 已完成 5 轮预热和 100 轮测量。按每轮 8 个 rank 中
  最慢 rank 的 `effective_us` 统计，min/median/mean/max/stdev 分别为
  **40.9805/41.2341/41.7217/59.7017/2.5835 ms**。
- 按 mean 比较，PyPTO 比 AscendC 低 **4.5058 ms**，延迟比为
  **0.9025**，即低 **9.7469%**；按 median 比较则低 **10.8017%**。
- 100 轮共得到 800 个完整 invocation，parser 未进入 flattened fallback。
  但任务在测量完成后的子进程关闭阶段 `exit=1`，且 `golden_fn=None`，
  因此该结果可用于性能分析，不代表端到端数值正确性或完整流程 PASS。

## 2. 对比范围

### 2.1 PyPTO 方案 3 边界

方案 3 的语义终点为：

```text
43-layer decode
  -> moe_signal_clear
  -> hc_head
  -> final RMSNorm
  -> distributed LM head
  -> FP32 logits
```

不包含：

- greedy sampling
- ArgMax
- sampled_ids 输出
- device 侧 embedding/输入打包
- device 侧 decode metadata 构造

### 2.2 AscendC 边界

AscendC trace 使用以下规则截取：

1. 选择 trace 中最早的 `HcPre`。
2. 限定在相同 device PID。
3. 找到其后第一个设备 ArgMax，作为排他截止点。
4. 在截止点之前，选择结束时间最晚的
   `aclnnInplaceCopy_CastAiCore_Cast` 作为区间终点。

该区间包含 86 个 `HcPre`，对应 43 层中每层两个 HC-pre 边界。

## 3. Workload 对齐

| 项目 | PyPTO 方案 3 | AscendC / 对齐要求 |
|---|---:|---:|
| EP | 8 | 8 |
| LM-head TP | 4 | 4 |
| 全局专家数 | 128 | 128 |
| 每卡专家数 | 16 | 16 |
| Batch | 4 | 4 |
| Active tokens | 8 | 8 |
| Decode start position | 8192 | 8192 |
| Top-k | 6 | 6 |
| 每层总路由数 | 384 | 384 |
| 每专家每层路由数 | 3 | 3 |
| 输出 | FP32 logits | ArgMax 前 logits 边界 |
| Sampling | 不包含 | 截止于设备 ArgMax 前 |

PyPTO 使用 host-prepared `x_hc`、slot mappings 和 sparse-attention
metadata。`embed_weight`、`block_counts` 和 `sampled_ids` 不属于方案 3
forward 边界。

## 4. 代码与版本

### 4.1 PyPTO

- 工作区：本地独立 worktree `pypto-lib_dsv4_ascendc_decode_compare3`
- 分支：
  `perf/dsv4-ascendc-decode-compare3`
- 被测提交：
  `67235ef8f096ae36ca005c1bacd661580ab2f53a`
- 方案 2 基线：
  `a79c141c30851e4adaaa5925f99db597fbb6082e`
- PTOAS：0.54
- CANN：9.0.0
- 目标平台：A2/A3
- 目标设备：
  `0,2,4,6,8,10,12,14`

### 4.2 AscendC trace

- 文件：仓库根目录 `trace_view_a3_decode.json`
- 文件大小：55,983,871 bytes
- SHA-256：
  `eefb2733f73215452c511177b613d51c16b575a23b2052f1a51424f44a0b244d`
- JSON records：212,324
- 完整 `ph=X` records：205,751

## 5. 正确性与结构证据

### 5.1 当前 EP8/TP4 提交的静态验证

在 `67235ef` 上已完成：

- Python compileall：PASS
- compare3 contract tests：**4 passed**
- Ruff：PASS
- header check：PASS
- English-only check：PASS
- EP8/TP4 import/signature contract：PASS

契约确认：

- `N_RANKS=8`
- `LM_HEAD_TP_SIZE=4`
- `N_EXPERTS_GLOBAL=128`
- `N_LOCAL=16`
- `B=4`
- `T=8`
- `DECODE_START_POS=8192`
- `decode_fwd` 和 `l3_decode_fwd` 均以 FP32 `x_hc` 开始
- 输出包含 FP32 `logits`
- 不包含 `embed_weight`、`block_counts`、`sampled_ids`

### 5.2 历史两卡正确性门禁

重放到方案 2 基线之前，曾在旧提交 `d70cf0e` 上完成 EP2/TP2/T8
两卡门禁：

- SWA layer 0 golden：PASS
- CSA layer 2 golden：PASS
- HCA layer 3 golden：PASS
- HC head golden：PASS
- decode RMSNorm golden：PASS
- TP2/DP2 LM head golden：PASS
- 完整 43 层 decode runtime：exit 0

该完整 decode 使用 `golden_fn=None`，因此只证明编译、运行和分布式通信
完成，不构成端到端数值 golden。由于该门禁早于当前 rebase 后提交，
它仅作为历史组件正确性证据，不能替代当前 `67235ef` 的 EP8/TP4
device 验证。

### 5.3 当前 EP8/TP4 device 结构验证

状态：**部分完成；独立 L2 图结构验证仍 Pending**

当前提交的源代码、静态契约、正式编译和 8-rank dispatch 已确认方案 3
包含 HC head、final RMSNorm 和 distributed LM head。独立 L2 trace 尚未
成功生成，因此以下逐 rank 编译图检查仍作为发布前门禁：

- 第一个任务为 `hc_pre_rms`
- `hc_pre_rms` 数量为 86
- `route_hash` 数量为 43
- `route_sort` 数量为 0
- 不包含 device metadata 构造或 `pack_x_hc`
- 不包含 greedy/sample/ArgMax
- 末尾连续任务为：

```text
moe_signal_clear
hc_head_rms
hc_head_linear
hc_head_reduce
rms_norm
lm_head_dispatch_push
lm_head_dispatch_wait
lm_head_dispatch_gather
lm_head_matmul
lm_head_combine_push
lm_head_combine_wait
lm_head_combine_gather
lm_head_signal_clear
```

## 6. 测量方法

### 6.1 PyPTO 正式测量

正式计时配置：

```text
PYPTO_BENCH=1
PYPTO_BENCH_WARMUP=5
PYPTO_BENCH_ROUNDS=100
PYPTO_BENCH_RAW=1
enable_l2_swimlane=0
EP=8
TP=4
T=8
start_pos=8192
```

统计规则：

- 每个 rank 每轮必须恰好有一次 dispatch。
- 8 个 rank × 100 轮，共 800 次 invocation。
- 每轮有效延迟取 8 个 rank 延迟的最大值。
- 正式结果报告 min、median、mean、max、stdev。
- L2 trace 单独运行，只用于图结构验证，其时间不得用于正式性能。

本次正式进程在离开 `prepare()` context 时发生 teardown 异常，导致 runner
未执行正常的统计打印代码。异常处理保留并回显了完整 STRACE marker；本文
使用 runner 相同的 `pypto.runtime.bench._parse_stats_from_strace` 重新解析：

- 预热 invocation：5 × 8 ranks
- 测量 invocation：100 × 8 ranks = 800
- round grids：100
- 每个 round 的 rank 数：8
- `fallback_flattened=False`
- headline metric：每轮 8 个 rank 中最大的 `effective_us`

因此性能样本本身完整；teardown 异常不计入 `effective_us`，但任务整体状态
仍为失败，不能写成完整流程 PASS。

### 6.2 AscendC 测量

AscendC 当前只有一个所选 trace 区间：

| 字段 | 值 |
|---|---:|
| 起点 | `HcPre` |
| 起点 timestamp | 1776181321247673.911 µs |
| 终点任务 | `aclnnInplaceCopy_CastAiCore_Cast` |
| 终点 timestamp（含 duration） | 1776181321293901.3554375 µs |
| HcPre 数量 | 86 |
| 精确十进制区间 | 46.2274444375 ms |
| 与现有 float 提取方式兼容的值 | **46.2275 ms** |
| 样本数 | 1 |

最终对比使用 46.2275 ms，以保持与现有提取脚本一致。精确十进制值与
该值相差 0.0555625 µs，来源是 epoch 级 timestamp 转 binary64 时的舍入。

## 7. 性能结果

### 7.1 原始结果

| 实现 | 样本 | Min (ms) | Median (ms) | Mean (ms) | Max (ms) | Stdev (ms) |
|---|---:|---:|---:|---:|---:|---:|
| AscendC | 1 | 46.2275 | 46.2275 | 46.2275 | 46.2275 | N/A |
| PyPTO 方案 3 EP8/TP4 | 100 | **40.9805** | **41.2341** | **41.7217** | **59.7017** | **2.5835** |

PyPTO 原始微秒统计为：

```text
effective_us min=40980.540 median=41234.140 mean=41721.735
             max=59701.680 stdev=2583.546
```

### 7.2 对比结论

| 指标 | 结果 |
|---|---:|
| PyPTO mean - AscendC | **-4.505765 ms** |
| PyPTO mean / AscendC | **0.902530647** |
| PyPTO mean 相对 AscendC | **-9.7469%** |
| PyPTO median - AscendC | **-4.993360 ms** |
| PyPTO median / AscendC | **0.891982911** |
| PyPTO median 相对 AscendC | **-10.8017%** |
| 当前更快实现 | **PyPTO 方案 3（预发布结论）** |

计算公式：

```text
absolute_delta_ms = pypto_mean_ms - 46.2275
ratio = pypto_mean_ms / 46.2275
percent_delta = (ratio - 1) * 100%
```

本次 `percent_delta=-9.7469%`，表示按 mean 计算，PyPTO 方案 3 的延迟
比 AscendC 单次 trace 区间低 9.7469%。

## 8. 统计解释限制

- AscendC 是一个 trace 中选取的单次区间，样本数为 1。
- PyPTO 是同一次正式任务中的 100 轮分布，并非 100 次独立进程测量。
- 因此两者可以比较中心延迟，但不能把 AscendC 单次值解释为稳定的
  mean、median 或方差。
- PyPTO 的 min/median/max/stdev 不应与 AscendC 的单点值进行
  对称统计推断。
- PyPTO 第 65、79 轮分别出现 58.6815、59.7017 ms 高延迟样本，使 mean
  高于 median；中心趋势优先同时报告 median 和 mean。
- PyPTO 的 800 个 marker 完整，但 runner 在测量后的 teardown 阶段
  `exit=1`；关闭异常不在计时窗口内，仍需单独修复或放宽关闭门禁。
- 当前 EP8/TP4 使用 `golden_fn=None`，没有端到端数值 golden。
- 独立 L2 图结构验证仍待完成。因此本文给出预发布倍率，不把它标记为
  最终发布结论。

## 9. 验证与发布门禁

- [x] EP8/TP4 compile-only 通过
- [ ] EP8/TP4 uninstrumented smoke 完整退出（device dispatch 已完成，teardown `exit=1`）
- [ ] 8-rank L2 图结构验证通过
- [x] 100 轮 formal benchmark 的 device dispatch 完成
- [x] 8 个 rank 每个 rank 恰好 100 个样本
- [x] 总测量 invocation 数为 800
- [x] 无 flattened parser fallback
- [x] 写入原始 `effective_us`
- [x] 写入 min/median/mean/max/stdev
- [x] 计算绝对差、倍率和百分比
- [x] 填写预发布对比结论
- [ ] teardown 完整退出并取得任务 `exit=0`
- [ ] 当前提交 EP8/TP4 端到端数值验证

## 10. 当前执行状态

当前相关任务均已结束，没有任务仍在运行：

- 正式性能任务：
  完成 5 轮预热和 100 轮测量，8 个 rank 均到达 invocation 105；随后
  8 个 chip child 未在 Simpler 的 10 秒 close budget 内退出，任务
  `completed (exit=1)`。
- 第一次 uninstrumented smoke：
  device dispatch 返回，1 个 chip child 关闭超时，`completed (exit=1)`。
- 第二次 uninstrumented smoke：
  device dispatch 返回，5 个 chip child 关闭超时，`completed (exit=1)`。

三次失败均发生在 `rt(...)` 返回后的 worker teardown，而不是编译或 device
dispatch 阶段。现有证据显示 HCCL communicator destroy 已完成；更深层的
device/context/stream cleanup 未在固定 10 秒预算内结束。该问题影响任务最终
状态，但不改变已完成的 100 轮 `effective_us` 计时样本。

## 附录 A：PyPTO 100 轮原始 effective_us

单位：µs。每行依次列出 10 轮；已排除 5 轮预热。

```text
001-010: 41010.740, 41635.520, 41264.359, 41494.680, 41203.619, 41296.360, 41199.419, 41138.120, 41270.019, 41160.580
011-020: 41062.000, 41024.480, 46253.140, 41066.480, 41321.960, 41284.079, 41087.680, 41207.019, 41186.520, 41171.540
021-030: 41226.320, 41085.740, 41151.019, 41257.840, 41198.320, 41142.320, 41198.659, 41093.760, 41109.760, 41103.920
031-040: 41145.540, 41286.260, 41038.460, 41931.180, 41699.261, 41783.760, 41691.920, 41667.420, 41420.880, 41436.560
041-050: 41177.280, 41284.120, 41599.740, 41277.000, 41309.360, 41462.900, 41430.000, 41364.280, 41336.160, 41528.480
051-060: 41292.660, 41439.759, 41471.020, 41315.000, 41519.700, 42099.740, 41637.499, 41354.619, 41443.920, 41314.040
061-070: 41791.760, 41422.980, 41222.260, 41338.960, 58681.480, 41091.200, 41065.280, 41040.120, 41139.960, 41143.000
071-080: 41103.040, 41130.200, 44242.040, 40980.540, 41120.260, 41040.540, 41074.080, 41144.400, 59701.680, 41023.499
081-090: 41056.760, 41073.200, 41372.580, 41302.820, 41222.001, 41118.020, 41206.100, 41293.980, 41148.160, 41128.740
091-100: 41558.840, 41252.500, 41210.560, 41174.720, 41120.760, 41192.680, 41241.960, 41304.960, 41195.580, 41836.859
```
