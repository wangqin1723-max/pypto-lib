# DeepSeek V4 LM-Head TP4 性能对比报告

## 摘要

本报告对比以下两组数据：

- `trace_view_a3_decode.json` 中 Model ID 47 的 RMSNorm 结束到最终 Cast 结束区间；
- PyPTO LM-Head TP4 projection-only 正式测试结果。

| 对比项 | Trace 基线 | PyPTO |
|---|---:|---:|
| 中位耗时 | 779.316 us | 574.850 us |
| P90 耗时 | 未统计（仅 3 个样本） | 693.500 us |
| 中位耗时差值 | - | -204.466 us |
| 相对耗时 | 1.000 | 0.738 |
| Trace/PyPTO 中位数比值 | 1.000x | 1.356x |

在本次观测口径下，PyPTO 中位耗时比 Trace 基线低 **26.24%**。

## 对比口径

### Trace 基线

基线数据来自 `trace_view_a3_decode.json` 中 NPU Stream 240 上的 Model ID 47：

- 起点：Task 59 `RmsNorm` 的结束时刻；
- 终点：Task 66 `aclnnInplaceCopy_CastAiCore_Cast` 的结束时刻；
- 包含：同步等待、LM-Head MatMul、Transpose 和最终 Cast；
- 不包含：Task 59 RMSNorm 本身、Model ID 48，以及后续 ArgMax Cast。

该区间是 Stream 上的墙钟时间，包含 Task 60/61 和 Task 63/64 的
Capture Record/Wait，并非只对计算 Kernel 耗时求和。

Trace 中共有 3 次对应区间：

| 样本 | 耗时（us） |
|---|---:|
| 1 | 779.316125 |
| 2 | 752.535125 |
| 3 | 780.2360625 |
| 中位数 | **779.316125** |

### PyPTO 测试

PyPTO 测试从已经完成 RMSNorm 的 hidden states 开始，到组装后的 FP32 logits
可用为止。

| 配置项 | 配置值 |
|---|---|
| 设备 | 0、2、4、6 |
| TP / DP | 4 / 4 |
| Token 数 | 8 |
| 测试模式 | Projection only |
| 预热 / 正式轮数 | 5 / 100 |
| L2 Swimlane | 关闭 |
| 正确性 | PASS，FP32 logits `(4, 8, 129280)` |

PyPTO 测量范围包含以下 Kernel 区域：

1. `lm_head_dispatch_push`、`lm_head_dispatch_wait`、
   `lm_head_dispatch_gather`
2. `lm_head_matmul`
3. `lm_head_combine_push`、`lm_head_combine_wait`、
   `lm_head_combine_gather`
4. `lm_head_signal_clear`

测试不包含 Greedy Sampling 和 ArgMax。

## 测试结果

正式测试按每一轮 4 个 Rank 中的最慢 Rank 统计：

| 统计项 | 耗时（us） |
|---|---:|
| 最小值 | 399.400 |
| 中位数 | **574.850** |
| 平均值 | 596.634 |
| P90 | 693.500 |
| 最大值 | 964.800 |

与 Trace 中位数 779.316125 us 对比：

```text
差值 = 574.850 - 779.316125 = -204.466125 us
相对耗时 = 574.850 / 779.316125 = 0.737634
Trace/PyPTO 中位数比值 = 779.316125 / 574.850 = 1.355686x
```

## 结论

本次测试中，PyPTO TP4 projection-only 路径的中位耗时为 **574.850 us**，
比选定 Trace 区间的中位耗时 **779.316 us** 低 **204.466 us（26.24%）**，
观察到的 Trace/PyPTO 中位数比值为 **1.356x**。

该结果不能直接视为严格同环境下的端到端加速比：PyPTO 数据来自 100 轮
Standalone 测试，并按每轮最慢 Rank 统计；Trace 基线只有 3 个图内单 Stream
样本，两者的运行环境、任务结构和计时方式不同。二者对齐的是语义边界：
均排除 RMSNorm 和采样，并在 FP32 logits 可用时结束。

正式测试日志位于：
`build_output/dsv4_lmhead_tp4_plan_a/lmhead_tp4_projection_benchmark/formal_tp4_dp4_8token_even_0_2_4_6.log`。
