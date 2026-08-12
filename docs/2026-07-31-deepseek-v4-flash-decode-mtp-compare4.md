# DeepSeek V4 Flash 方案 4 Decode→MTP 与 MTP-core 性能对比报告

## 状态

本报告同时保留两个互不混用的口径：历史完整方案 4 Decode→MTP 链路，以及
本次新增的严格 MTP-core 独立区间。严格 runner、正确性验证、八卡 100 轮
回归、生成程序边界审计和八卡 system trace 均已完成。本次不做核内性能
分析。

## 本次严格 MTP-core 对比

### 计时边界

严格区间只包含：

```text
MTP projection -> SWA -> MoE -> HC head -> RMSNorm -> LM head logits
```

| 框架 | 起点 | 终点 |
|---|---|---|
| AscendC | Model47 Task6 首个 projection `RmsNorm` 开始 | Model47 Task66 `aclnnInplaceCopy_CastAiCore_Cast` 结束 |
| PyPTO 3.0 | 顶层 compute child `mtp_decode_core_logits` 开始，即首个 `mtp_hidden_norm_quant` | 同一 child 的 `lm_head_combine_gather` 完成 |

以下工作在两个框架的严格比较区间之外：

- MTP embedding；
- accepted hidden packing；
- SWA slot mapping、indices 和 lens 构造；
- 采样、ArgMax 和接受逻辑；
- MoE 与 LM-head signal cleanup。

PyPTO 把前三项的产物作为 host-prepared `TensorSpec` 传入，不提交采样，
并把 cleanup 拆成第二个顶层 child。因此 system trace 中第一个顶层
`MIX_AIC` 就是严格计算区间，第二个顶层 `MIX_AIC` 仅负责 cleanup，直接
比较时排除。

### 工作负载

| 配置项 | 值 |
|---|---:|
| EP | 8 |
| LM-head TP | 4 |
| 全局路由专家数 | 128 |
| 每个 rank 的本地路由专家数 | 16 |
| 每个 rank 的活跃 token 数 | 8 |
| 静态形状 | B=4, S=2, T=8 |
| Decode 起始位置 | 8192 |
| MoE top-k | 6 |
| 路由模式 | `trace-hash` |
| L2 swimlane | 关闭 |
| 设备 | 0,2,4,6,8,10,12,14 |

在 EP8、T8、top-k=6 下，每层共有 384 条活跃路由；固定的
`trace-hash` 输入让 128 个全局专家各被命中 3 次。专家拓扑沿用 compare2
已经确认的 128/16 配置，不复用旧的 256/32 拓扑。

### AscendC 严格参考数据

数据源为本地输入 `trace_view_a3_decode.json`，区间为 Model47
Task6–Task66。

| 稳态轮次 | projection 到最终 Cast |
|---|---:|
| 1 | 2.287126 ms |
| 2 | 2.260706 ms |
| 3 | 2.160163 ms |
| 中位数 | 2.260706 ms |
| 平均值 | 2.235998 ms |

Model47 embedding 到最终 Cast 的平均值 2.271312 ms 不是本次所选边界。
完整 43 层 decode、handoff 和 MTP 的 52.731107 ms 更不是严格 MTP-core
的比较分母。

### PyPTO 实现与程序边界

测试分支为 `perf/dsv4-ascendc-decode-mtp-compare4`，验证提交为
`2ccdfa0228c96cb199401c7d76f39ea7aeb0beed`，PTOAS 为 0.54。主要提交：

- `46f6cdd`：拆分 LM-head compute 与 signal cleanup，同时保留原兼容 wrapper；
- `ea61159`：新增严格 runner `decode_mtp_core.py`；
- `2ccdfa0`：补强 prepared metadata、EP8/TP4 golden group 和 buffer ABI 合约。

生成程序的 compute child 共 74 个 function ID：

| 阶段 | PyPTO function ID | 边界任务 |
|---|---:|---|
| MTP projection | 0–3 | `mtp_hidden_norm_quant` 开始 |
| SWA | 4–31 | attention 主体 |
| MoE | 32–62 | dispatch、expert、combine |
| HC head | 63–65 | HC head 主体 |
| 最终 RMSNorm | 66 | final norm |
| LM head | 67–73 | `lm_head_combine_gather` 结束 |

compute child 内没有 embedding、外部 hidden packing、SWA metadata builder、
sampling、`moe_signal_clear` 或 `lm_head_signal_clear`。cleanup child 只有
`moe_signal_clear` 和 `lm_head_signal_clear` 两个任务。host orchestration 先
提交 rank 0–7 的全部 compute child，再提交 rank 0–7 的全部 cleanup child。
function ID 只用于证明边界与顺序，AIC/AIV 可以重叠，不能把 74 个 ID 当成
串行耗时相加。

### 正确性验证

| 验证 | 运行 | 拓扑 | 结果 |
|---|---|---|---|
| 精确 golden | 独立验证运行 | EP2/TP2，两卡 0,2；每 rank 16 个本地专家 | `kv_cache`、`hidden_out`、`next_pre_hc_hidden`、`logits` 全部 PASS，exit=0 |
| 目标拓扑冒烟 | 独立验证运行 | EP8/TP4，八张偶数卡，128/16 专家 | 四个保留输出全部 finite PASS，exit=0 |

两卡任务用于低成本精确数值验证，虽然仍保持每 rank 16 个本地专家，但它
不是 EP8/TP4 性能对比数据。正式性能数据只来自八卡目标拓扑。

### PyPTO 5+100 回归数据

正式回归运行使用 5 轮预热和 100 轮正式测量，
共得到 800 个 compute 样本和 800 个 cleanup 样本，四个输出 finite PASS，
任务 exit=0。每轮先在八个 rank 中取最大值；compute 和 cleanup 分开统计。

| 指标 | 每轮最大 compute child | 每轮最大 cleanup child（排除项） |
|---|---:|---:|
| 最小值 | 3.872300 ms | 0.013300 ms |
| 中位数 | 4.004450 ms | 0.014200 ms |
| 平均值 | 4.009504 ms | 0.014307 ms |
| P90 | 4.091850 ms | 0.015110 ms |
| P95 | 4.112255 ms | 0.015400 ms |
| P99 | 4.131401 ms | 0.015621 ms |
| 最大值 | 4.161200 ms | 0.017700 ms |

NPU0 在 100/100 轮中都是 compute critical rank，因此它的 compute 分布与
上表第一列相同。该程序级计时用于回归和稳定性诊断；跨框架正式差值使用
下面的 system trace，而不是这组 `PYPTO_BENCH` 数据。

### PyPTO 严格 System Trace

严格 trace 运行使用 1 轮预热和 3 轮稳态执行，
finite 验证通过并以 exit=0 完成。每个 rank 恰好记录 8 个顶层 `MIX_AIC`：
一个预热 compute/cleanup 对，以及三个正式 compute/cleanup 对。正式数据只
选每个 pair 的第一个事件，并通过稳定提交顺序映射到
`mtp_decode_core_logits`；第二个事件对应 `mtp_decode_core_cleanup`，确认后
排除。该 CANN 版本把硬件事件命名为 `aicore_kernel_0_mix_aic`，配套 AICPU
事件仍是通用的 `simpler_aicpu_exec_<hash>`，因此映射依据是生成的 host
orchestration 顺序和 Task ID，而不是名称字符串或耗时大小。

事件筛选条件为 `.ph == "X"` 且 `args["Task Type"] == "MIX_AIC"`。每个
rank 的事件按时间排序后，Task ID 均完整对应 `0..7`：预热 compute/cleanup
为 `0/1`，三个正式轮次的 compute/cleanup 分别为 `2/3`、`4/5`、`6/7`。
因此正式 compute 只取 Task ID `2`、`4`、`6`，cleanup Task ID `3`、`5`、
`7` 全部排除。

| 稳态轮次 | NPU0 compute | 八个 rank 的最大 compute | 最大 rank | 最大 cleanup（排除） |
|---|---:|---:|---:|---:|
| 1 | 4.990800 ms | 4.990800 ms | NPU0 | 0.779860 ms |
| 2 | 4.993460 ms | 4.993460 ms | NPU0 | 0.771729 ms |
| 3 | 4.711020 ms | 4.711020 ms | NPU0 | 0.765829 ms |
| 中位数 | 4.990800 ms | 4.990800 ms | — | 0.771729 ms |
| 平均值 | 4.898427 ms | 4.898427 ms | — | 0.772473 ms |

NPU0 在三个正式 trace 轮次中均为 compute critical rank。cleanup 表中的值是
每轮八个 rank 的最大 cleanup 顶层包络；它只用于验证第二个 child 的独立性，
不进入直接性能比较。

### 严格跨框架结果

正式比较采用 AscendC NPU0 的 Task6–Task66 区间和 PyPTO 八个 rank 中的
最大 compute child。定义：

```text
delta_ms = PyPTO_max_rank_ms - AscendC_NPU0_ms
ratio = PyPTO_max_rank_ms / AscendC_NPU0_ms
```

ratio 大于 1 表示 PyPTO 更慢。

| 统计项 | AscendC NPU0 | PyPTO 最大 rank | 差值 | PyPTO/AscendC |
|---|---:|---:|---:|---:|
| 中位数 | 2.260706 ms | 4.990800 ms | +2.730094 ms | 2.207629× |
| 平均值 | 2.235998 ms | 4.898427 ms | +2.662429 ms | 2.190712× |

按三轮 system trace，PyPTO 严格 MTP-core 的最大 rank 中位数比 AscendC
高 120.763%，平均值高 119.071%。这是顶层 compute child 的 system-trace
包络结论；100 轮 `PYPTO_BENCH` 结果用于回归诊断，不替代该正式比较。

## 历史完整方案 4：对比目标

在相同的 EP8/TP4 工作负载下，对比 AscendC 与 PyPTO 3.0 的完整
DeepSeek V4 Flash 方案 4 Decode→MTP 链路：

```text
主模型 decode
  -> decode 到 MTP 的衔接阶段
  -> MTP embedding
  -> MTP projection
  -> MTP SWA
  -> MTP MoE
  -> MTP HC head
  -> MTP RMSNorm
  -> MTP LM head
  -> 最终 FP32 logits/cast
```

这不是方案 3 的 `decode_fwd.py` 工作负载。方案 3 在主模型的 HC head、
RMSNorm 和 LM head 尾部结束；方案 4 还包含设备侧 Decode→MTP 衔接以及
完整的一层 MTP decode。

## 历史完整方案 4：计时边界

对比区间包含起止算子本身，以及二者之间的全部工作。

| 框架 | 起点 | 终点 |
|---|---|---|
| AscendC | Model48 首个 `aclnnEmbedding_GatherV2AiCore_GatherV2` 开始 | Model47 Task66 `aclnnInplaceCopy_CastAiCore_Cast` 结束，包含该 Cast 自身耗时 |
| PyPTO 3.0 | 设备侧 `pack_x_hc` embedding 任务开始 | `mtp_decode_layer_logits_lm_head_combine_gather` 完成 |

不包含 PyPTO 在 MTP logits gather 之后的 `lm_head_signal_clear`。也不包含
AscendC 最终 Task66 Cast 之后的 ArgMax、采样或其他处理。

## 工作负载对齐

| 配置项 | AscendC | PyPTO 3.0 |
|---|---:|---:|
| 方案 | 4 | 4 |
| EP 并行度 | 8 | 8 |
| LM-head TP 并行度 | 4 | 4 |
| 全局路由专家数 | 128 | 128 |
| 每个 rank 的本地路由专家数 | 16 | 16 |
| 每个 rank 的活跃 token 数 | 8 | 8 |
| 静态 decode 容量 | B=4, S=2, T=8 | B=4, S=2, T=8 |
| Decode 起始位置 | 8192 | 8192 |
| MoE top-k | 6 | 6 |
| 路由负载 | 哈希路由 | `trace-hash` |
| 测量时的 L2 swimlane | trace 未提供 | 关闭 |
| 设备编号 | 参考 trace 记录 NPU0 | 0,2,4,6,8,10,12,14 |

PyPTO 使用 `perf/dsv4-ascendc-decode-mtp-compare4` 分支的提交
`5713db766ce3f279f3a72769fadd0b0a60ab3143`，PTOAS 版本为 0.54。

在 EP8、T8、top-k=6 的配置下，每层共有 384 条活跃路由。PyPTO 的
`trace-hash` 固定测试数据每层恰好覆盖 128 个全局专家各 3 次。

## 历史完整方案 4：AscendC 参考数据

数据源为本地对比工作区中的 `trace_view_a3_decode.json`。该 trace 是输入
数据，不提交到仓库。

| 轮次 | 主模型 decode | 衔接间隔 | MTP | 完整区间 |
|---|---:|---:|---:|---:|
| 1 | 46.325086 ms | 3.994020 ms | 2.322726 ms | 52.641832 ms |
| 2 | 46.401346 ms | 4.139383 ms | 2.295967 ms | 52.836696 ms |
| 3 | 46.317065 ms | 4.202484 ms | 2.195244 ms | 52.714793 ms |
| 平均值 | 46.347832 ms | 4.111962 ms | 2.271312 ms | 52.731107 ms |

完整区间统计如下：

| 指标 | 耗时 |
|---|---:|
| 最小值 | 52.641832 ms |
| 中位数 | 52.714793 ms |
| 平均值 | 52.731107 ms |
| 最大值 | 52.836696 ms |

AscendC trace 提供了 NPU0 的 3 个样本。该 trace 本身不能独立证明
EP8/TP4 拓扑，拓扑信息来自本次对比约定。

## 历史完整方案 4：PyPTO System Trace 结果

跨框架的直接对比数据必须来自 CANN system trace，不做核内性能分析，也不
启用 L2 swimlane 重复执行。

现有 trace 来自一次独立运行，包含 1 轮预热和
3 轮稳态数据。该 trace 在每个 rank 上只记录三个顶层 `MIX_AIC`：主模型
decode、MTP 输入衔接和 MTP decode。它没有记录顶层任务内部的子算子名或
时间戳，因此无法精确裁剪到本报告定义的起止点：

- 主模型 decode 顶层任务在 `pack_x_hc` 之前还包含 3 个 metadata 任务；
- MTP decode 顶层任务在目标 `lm_head_combine_gather` 之后还包含
  `lm_head_signal_clear`。

以下数据是从第一个顶层 `MIX_AIC` 开始到第三个顶层 `MIX_AIC` 结束的
程序级包络，包含起点前 metadata、终点后 signal clear 以及三个子程序之间
的间隔。它只能作为代理上界，不能冒充严格同边界结果。

| 稳态轮次 | NPU0 顶层程序包络 | 八个 rank 中的最大包络 |
|---|---:|---:|
| 1 | 47.951640 ms | 47.951640 ms |
| 2 | 47.739440 ms | 47.739440 ms |
| 3 | 68.664700 ms | 68.664700 ms |

| 指标 | NPU0 顶层程序包络 | 最大 rank 顶层程序包络 |
|---|---:|---:|
| 最小值 | 47.739440 ms | 47.739440 ms |
| 中位数 | 47.951640 ms | 47.951640 ms |
| 平均值 | 54.785260 ms | 54.785260 ms |
| 最大值 | 68.664700 ms | 68.664700 ms |

第 3 轮的最终 MTP 顶层任务为 23.864940 ms，前两轮分别为
2.618860 ms 和 2.618220 ms；八个 rank 均出现同方向的尾部膨胀。因此除了
边界不严格外，3 轮代理值的平均值还明显受到该异常长尾影响。

## 历史完整方案 4：PyPTO 100 轮回归性能测试

`PYPTO_BENCH effective_us` 是回归性能指标，不是跨框架对比的主要数据。
在组合式 L3 程序中，它表示每轮八个 rank 中最慢 rank 的有效设备时间窗，
可能不包含子任务调度之间的空闲间隔。

数据来自一次正式回归运行。每个 rank 每轮固定执行
3 个 dispatch；先对同一 rank 的 3 个 dispatch 求和，再取八个 rank 的最大
值，得到每轮的 `effective_us`。

| 指标 | 最慢 rank 的有效耗时 |
|---|---:|
| 最小值 | 41.6412 ms |
| 中位数 | 42.3684 ms |
| 平均值 | 44.1480 ms |
| P90 | 50.6394 ms |
| P95 | 52.8060 ms |
| P99 | 63.5038 ms |
| 最大值 | 80.9808 ms |

| 诊断项 | 值 |
|---|---:|
| 预热轮数 | 5 |
| 正式测量轮数 | 100 |
| 完整正式样本数 | 800/800 个 rank-round；2400/2400 个 dispatch |
| `fallback_flattened` | 否 |
| `host_union_mean_us` | 49,867 us（49.867 ms） |
| `host_mean_us` | 49,413 us（49.413 ms） |
| 输出验证 | `mtp_logits` PASS，shape=(8, 8, 129280)，FP32 |
| 任务退出状态 | `exit=0`，整体 `PASS` |

P90、P95 和 P99 由日志中保留到 0.1 us 的原始 dispatch 数据重建，采用
`h=(n-1)q` 的线性插值法。由显示精度引入的误差处于亚微秒量级。正式日志为
`decode_fwd_mtp_ep8_tp4_8k_perf_100r.log`。

## 历史完整方案 4：跨框架对比限制

以下限制只针对旧的完整 Decode→MTP runner 和旧 system trace。旧 runner
把边界外 metadata 与 signal clear 包在顶层程序里，无法从该 trace 恢复完整
链路的严格同边界数据，因此本报告不对历史完整链路计算正式差值或加速比。
这不影响前文已经由独立 MTP-core runner 得出的严格区间结论。

| 统计项 | AscendC NPU0 | PyPTO 严格同边界 | 差值 | 比值 |
|---|---:|---:|---:|---:|
| 中位数 | 52.714793 ms | 现有 trace 无法提取 | 不计算 | 不计算 |
| 平均值 | 52.731107 ms | 现有 trace 无法提取 | 不计算 | 不计算 |

## 验证证据

完整方案 4 和严格 MTP-core 实现已经完成以下验证：

- EP8/TP4、全局 128 专家、每个 rank 16 个本地专家的拓扑约束检查；
- 主模型 decode、设备侧衔接到 MTP logits 的完整计算图检查；
- 严格 runner 两卡精确 golden，四个保留输出全部 PASS；
- 严格 runner 八卡目标拓扑 finite 冒烟，四个保留输出全部 PASS；
- 严格 runner 八卡 5+100 回归，800/800 compute 与 800/800 cleanup 样本
  完整，`exit=0`；
- 严格 runner 八卡 1+3 system trace，每 rank 8/8 顶层事件完整，
  `exit=0`；
- 8 个严格 runner contract 与 25 个完整方案 4 legacy contract，共 33/33
  通过；
- header lint、English-only lint、Ruff 和全部 pre-commit hook 通过；
- compare4 分支工作区干净，生成的 build、日志和 trace 均未进入版本控制。

独立 runner 通过物理拆分顶层 child 解决了旧 trace 粒度不足的问题，不依赖
核内时间戳或 op-simulator 数据。

## 复现命令

以下命令中的 `<pypto-root>`、`<pto-isa-root>` 和 `<compare4-worktree>` 需要
替换为本机对应目录。

两卡精确 golden 命令：

```bash
task-submit --ptoas 0.54 \
  --device 0,2 \
  --timeout 0 --max-time 1800 --run '
export PYPTO_ROOT="<pypto-root>"
export SIMPLER_ROOT="$PYPTO_ROOT/runtime"
export SIMPLER_BINDINGS="$SIMPLER_ROOT/build/cp310-cp310-linux_aarch64/python/bindings"
export PTO_ISA_ROOT="<pto-isa-root>"
export PYTHONPATH="$SIMPLER_ROOT:$SIMPLER_ROOT/python:$SIMPLER_BINDINGS:$PYPTO_ROOT/python:$PYPTO_ROOT/build/python/bindings${PYTHONPATH:+:$PYTHONPATH}"
cd "<compare4-worktree>"
python models/deepseek/v4-flash/decode_mtp_core.py \
  -p a2a3 --ep 2 --tp 2 -d "$TASK_DEVICE" \
  --start-pos 8192 --num-tokens 8 \
  --routing-mode trace-hash --enable-l2-swimlane 0'
```

八卡 EP8/TP4 finite 冒烟命令：

```bash
task-submit --ptoas 0.54 \
  --device 0,2,4,6,8,10,12,14 \
  --timeout 0 --max-time 1800 --run '
export PYPTO_ROOT="<pypto-root>"
export SIMPLER_ROOT="$PYPTO_ROOT/runtime"
export SIMPLER_BINDINGS="$SIMPLER_ROOT/build/cp310-cp310-linux_aarch64/python/bindings"
export PTO_ISA_ROOT="<pto-isa-root>"
export PYTHONPATH="$SIMPLER_ROOT:$SIMPLER_ROOT/python:$SIMPLER_BINDINGS:$PYPTO_ROOT/python:$PYPTO_ROOT/build/python/bindings${PYTHONPATH:+:$PYTHONPATH}"
cd "<compare4-worktree>"
python models/deepseek/v4-flash/decode_mtp_core.py \
  -p a2a3 --ep 8 --tp 4 -d "$TASK_DEVICE" \
  --start-pos 8192 --num-tokens 8 \
  --routing-mode trace-hash --enable-l2-swimlane 0 --finite-only'
```

严格 runner 的 100 轮命令：

```bash
task-submit --ptoas 0.54 \
  --device 0,2,4,6,8,10,12,14 \
  --timeout 0 --max-time 1800 --run '
export PYPTO_ROOT="<pypto-root>"
export SIMPLER_ROOT="$PYPTO_ROOT/runtime"
export SIMPLER_BINDINGS="$SIMPLER_ROOT/build/cp310-cp310-linux_aarch64/python/bindings"
export PTO_ISA_ROOT="<pto-isa-root>"
export PYTHONPATH="$SIMPLER_ROOT:$SIMPLER_ROOT/python:$SIMPLER_BINDINGS:$PYPTO_ROOT/python:$PYPTO_ROOT/build/python/bindings${PYTHONPATH:+:$PYTHONPATH}"
cd "<compare4-worktree>"
PYPTO_BENCH=1 PYPTO_BENCH_WARMUP=5 PYPTO_BENCH_ROUNDS=100 \
PYPTO_BENCH_RAW=1 \
python models/deepseek/v4-flash/decode_mtp_core.py \
  -p a2a3 --ep 8 --tp 4 -d "$TASK_DEVICE" \
  --start-pos 8192 --num-tokens 8 \
  --routing-mode trace-hash --enable-l2-swimlane 0 --finite-only \
  > decode_mtp_core_ep8_tp4_8k_perf_100r.log 2>&1'
```

严格 runner 的 system trace 命令：

```bash
task-submit --ptoas 0.54 \
  --device 0,2,4,6,8,10,12,14 \
  --timeout 0 --max-time 1800 --run '
export PYPTO_ROOT="<pypto-root>"
export SIMPLER_ROOT="$PYPTO_ROOT/runtime"
export SIMPLER_BINDINGS="$SIMPLER_ROOT/build/cp310-cp310-linux_aarch64/python/bindings"
export PTO_ISA_ROOT="<pto-isa-root>"
export PYTHONPATH="$SIMPLER_ROOT:$SIMPLER_ROOT/python:$SIMPLER_BINDINGS:$PYPTO_ROOT/python:$PYPTO_ROOT/build/python/bindings${PYTHONPATH:+:$PYTHONPATH}"
export PYPTO_BENCH=1 PYPTO_BENCH_WARMUP=1 PYPTO_BENCH_ROUNDS=3 PYPTO_BENCH_RAW=1
cd "<compare4-worktree>"
msprof \
  --application="python models/deepseek/v4-flash/decode_mtp_core.py -p a2a3 --ep 8 --tp 4 -d $TASK_DEVICE --start-pos 8192 --num-tokens 8 --routing-mode trace-hash --enable-l2-swimlane 0 --finite-only --runtime-dir build_output/_jit_l3_mtp_decode_core_20260731_225822" \
  --output=build_output/system_trace_decode_mtp_core_20260731_2320 \
  --ascendcl=on --runtime-api=on --task-time=on --aicpu=on --ai-core=on'
```

历史完整方案 4 的 runtime 目录由提交 `5713db7` 生成；严格 MTP-core 的
runtime 目录由提交 `2ccdfa0` 生成。二者都采用 128/16 专家拓扑，旧版
256/32 runtime 不适用于本次数据。

## 结果解释限制

- AscendC 输入 trace 只记录一个 NPU rank；PyPTO 同时报告 NPU0 以及八个
  rank 中的最大区间。
- AscendC trace 本身不能独立证明 EP8/TP4；拓扑来自本次对比约定。
- 严格 runner 消费准备好的 embedding 输出、accepted hidden 和 SWA
  metadata，因此结论只适用于 MTP-core，不代表完整 Decode→MTP 延迟。
- 正式 system trace 只有 3 个稳态样本；100 轮回归用于补充稳定性分布，但
  两种计时机制不能混合作为同一统计总体。
- 本次不分析算子核内耗时。
- 历史完整 runner 的 `effective_us` 与旧 system trace 仍不能和 AscendC
  52.731107 ms 直接计算严格加速比。
- 52.731107 ms 是完整 43 层 decode、handoff 与 MTP 的历史均值，绝不能
  作为严格 MTP-core runner 的分母。

## 结论

本次已经把 PyPTO 方案 4 中的严格 MTP-core 物理隔离为单独 compute child：
其计算内容仅为 projection、SWA、MoE、HC head、RMSNorm 和 LM-head logits；
embedding、packing、metadata、sampling 和 cleanup 均不在计时 child 内。

三轮 system trace 中，PyPTO 最大 rank 的严格 compute 中位数为
4.990800 ms、平均值为 4.898427 ms；AscendC Task6–Task66 分别为
2.260706 ms 和 2.235998 ms。按相同的严格 MTP-core 口径，PyPTO 中位数
多 2.730094 ms、为 AscendC 的 2.207629×；平均值多 2.662429 ms、为
AscendC 的 2.190712×。PyPTO 的 100 轮程序级 compute 中位数为
4.004450 ms、平均值为 4.009504 ms，用于回归稳定性参考。

历史完整方案 4 的 AscendC 均值 52.731107 ms 继续保留在报告中，但不参与
上述严格区间的差值或倍率计算。
