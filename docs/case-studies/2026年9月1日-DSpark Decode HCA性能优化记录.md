# DSpark Decode HCA 性能优化记录

> 归档日期：2026-09-15。来源：`pypto-lib/local_archive/2026年9月1日-DSpark Decode HCA性能优化记录.md`。
> 原文的实现、测量与计划保留其历史语境；阅读说明见[学习资料索引](../README.md)。

日期：2026 年 9 月 1 日

本文简单记录 `deepseek_v4_flash_dspark` 的 Decode HCA 性能优化过程、PR #1102 的 TP4 泳道图结论，以及后续优化计划。本文仅保存在本地，不提交到远程仓库。

## 1. 已完成的三步优化

### PR #1035：消除 raw KV 的 GM 转置

链接：[Perf: eliminate HCA raw KV GM transpose](https://github.com/hw-native-sys/pypto-lib/pull/1035)

原来的 raw attention 会先把 KV 写入一个转置后的 GM 临时张量，再分别用于 QK 和 PV。这个过程增加了一个独立转置任务和一次 GM 中间读写。

修改后：

- raw KV 直接加载到 Mat memory；
- QK 使用同一块 tile 的 `transpose_view`；
- PV 继续使用原始 tile；
- 删除 `raw_kv_t` GM 临时张量和独立转置任务。

性能记录：TP4、A2/A3、设备 3/5/7/9、global batch 64、sequence 8、local T=128、冻结 golden，5 轮预热后测试 100 轮；最快 rank 的 median 从 **3906.2 µs 降到 3453.9 µs，提升 11.6%**。

### PR #1098：多个推测 token 复用 raw KV

链接：[Perf: reuse HCA raw KV across speculative tokens](https://github.com/hw-native-sys/pypto-lib/pull/1098)

原来每个推测 token 都分别收集自己的 raw KV window，相邻 token 的窗口高度重叠，因此同一批 KV 被重复搬运。

修改后：

- 每个 request 只收集一次 `WIN + S - 1` 行的 raw KV；
- 四个推测 token 使用这块 buffer 的不同偏移视图；
- 根据每个 token 的 window length 保留短序列和未来 token 的 mask；
- 用一个向量化任务生成 raw-window validity mask。

性能记录：TP4 `decode_hca`、a2a3、8K context，10 轮预热后测试 100 轮；最快 rank 的 median 从 **899.1 µs 降到 842.7 µs，提升 6.3%**。

### PR #1102：融合 merge、逆 RoPE 和 pack

链接：[Perf: fuse HCA merge, inverse RoPE, and packing](https://github.com/hw-native-sys/pypto-lib/pull/1102)

原来的 HCA 后处理分成多步：合并 raw/compressed attention 的 softmax 状态、执行逆 RoPE、按 O projection group 重新排布，然后再发布到其他 TP rank。中间结果需要经过额外任务和 GM 读写。

修改后：

- 用 48 个 cyclic workers 完成 raw/compressed softmax merge；
- 在同一个 kernel 中完成归一化、逆 RoPE 和 grouped packing；
- distributed publisher 直接发送已经排布好的 group；
- TP1 路径复用相同的 packed 输出。

性能记录：isolated TP4-shaped HCA sparse attention、a2a3 device 3、B=16、S=8、64 个 compressed rows、fresh golden，10 轮预热后测试 100 轮；mean 从 **1179.3 µs 降到 829.0 µs，约提升 29.7%**。

PR #1102 当前 a2a3 CI 的失败不是代码错误：`decode_hca.py` 的四项精度检查和 `[RUN] PASS` 均已通过，随后 taskqueue daemon 停止或重启，导致 `task-submit` 以 exit 137 退出。a5sim 另有 buffer 限制和模拟器编译问题，不属于本次关注的 a2a3 真机结果。

> 三个 PR 的测试范围、代码基线和统计口径并不完全相同。#1035 和 #1098 是完整 TP4 `decode_hca` 的最快-rank median，#1102 是 isolated sparse-attention 的单卡 mean，因此不能直接串联绝对耗时或把三个百分比相乘。

## 2. PR #1102 的 rank2 泳道图分析

测试条件：wq3 环境、分支 `perf/fuse-hca-merge-rope-pack`、提交 `b1e4e4f`、TP=4、设备 3/5/7/9、8K start position、chip swimlane perf level 4。四项正确性检查全部通过。

按照“选择 dispatch 到 finish 总时间最短的 rank”这一规则，本次选择 **rank2**：

- dispatch → finish：**2283.000 µs**；
- AICore makespan：**2278.080 µs**；
- Observed critical-path compute：**2109 µs，占 92.6%**；
- Observed runtime-scheduling stall：**169 µs，占 7.4%**；
- Static CPM：**2028.480 µs**。

泳道图：rank2 merged swimlane（本地证据：`build_output/fuse-hca-merge-rope-pack-20260901_200645/rank2/merged_swimlane_20260901_200741.json`；未随文归档）

关键路径上的主要阶段：

| 阶段 | Observed contribution | 占 AICore makespan | 结论 |
| --- | ---: | ---: | --- |
| `hca_raw_attn` | 496.8 µs | 21.8% | 最大的本地计算热点 |
| `o_group_a2a_wait` | 333.3 µs | 14.6% | 等待其他 rank，不是 wait kernel 自身计算慢 |
| CP 输入 AllGather push | 216.1 µs | 9.5% | 后续独立的通信优化点 |
| `kv_proj_matmul` | 170.9 µs | 7.5% | 次级计算热点 |
| `scatter_softmax_pool` | 163.6 µs | 7.2% | 次级计算热点 |
| `hca_cmp_qk_pv` | 142.6 µs | 6.3% | 与 raw attention 部分重叠 |
| `hca_stream_merge_pack` | 92.8 µs | 4.1% | #1102 新的融合 kernel |
| `hca_stream_publish` | 62.6 µs | 2.7% | 可继续研究与 merge 的流水化 |

四个 rank 在 HCA 尾部的表现如下：

| rank | `hca_raw_attn` contribution | `hca_cmp_qk_pv` contribution | `o_group_a2a_wait` |
| --- | ---: | ---: | ---: |
| rank0 | 461 µs | 187 µs | 322 µs |
| rank1 | 555 µs | 135 µs | 287 µs |
| rank2 | 497 µs | 143 µs | 333 µs |
| rank3 | 712 µs | 212 µs | 1.3 µs |

rank3 的 raw/compressed attention 最慢，基本最后到达 AllToAll 同步点，所以它自身几乎不用等待；rank2 本地算得更快，却在 `o_group_a2a_wait` 中等待约 333 µs。由此判断，`o_group_a2a_wait` 是慢 rank 的表现，不是下一步应该直接修改的 kernel。

`hca_raw_attn` 和 `hca_cmp_qk_pv` 是并行分支，之后才进入 `hca_stream_merge_pack → hca_stream_publish → o_group_a2a_wait`。当前 raw 分支更长，因此单独加速 compressed 分支很可能仍被 raw 分支遮住，不能稳定降低 wall time。

## 3. 下一步计划

### 第一优先级：分析并优化 `hca_raw_attn`

先对 `hca_raw_attn` 做 in-core profiling，对照 rank2 对应设备和本次较慢的 rank3 对应设备，确认瓶颈属于：

- AIC：QK/PV MatMul 和 cube tile；
- AIV：softmax、mask 和 online-softmax merge；
- MTE：raw KV/Q 的搬运；
- 或 block/core 之间的负载不均衡。

有 PMU 和 in-core 泳道证据后，再决定是否调整 `H_TILE`、`RAW_K_TILE`、`ATTN_D_TILE` 或内部流水。当前不先猜 tile 参数。

这一项优先级最高，因为它既是 rank2 最大的本地计算热点，也很可能是其他 rank 到达 AllToAll 较晚的主要原因。优化它可能同时减少 Attention 计算和后面的跨 rank 等待。

### 第二优先级：让 merge 和 publish 更早重叠

当前执行顺序是：

```text
hca_stream_merge_pack 完全结束
    → hca_stream_publish 启动
    → o_group_a2a_wait
```

rank2 上 merge、间隙和 publish 串行约为 `92.8 + 5.5 + 62.6 = 160.9 µs`。可以单独研究一个小 PR：按 group/token block 产生 packed 数据后立即 publish，或者让 merge 直接写入 distributed window，以减少中间 GM 写回/读回并提前通知其他 rank。

这项改动涉及分布式发布协议，应该与 `hca_raw_attn` 的 kernel 调优分开 review。

### 第三优先级：单独优化 CP 输入 AllGather

rank2 的 AllGather push 为 216.1 µs，readback 约 55.7 µs，也是明显热点。但它属于 CP 通信基础设施，应独立于 HCA kernel 优化推进。

### 暂不优先

- 不先修改 `o_group_a2a_wait`：它主要在等较慢的 peer；
- 不先只优化 `hca_cmp_qk_pv`：当前大部分被 raw 分支覆盖；
- 不先做零散 early-dispatch/dummy：rank2 的总调度 stall 只有约 169 µs，收益上限低于 raw attention；
- 不把上述三类改动放进同一个 PR。

## 4. 后续性能验证口径

泳道图用于定位，最终性能结论使用关闭 DFX 的真实 benchmark：

- wq3、a2a3、固定同一组四张卡、TP=4；
- 固定 shape、8K start position 和 frozen golden；
- 同一进程中完成 10 轮预热和 100 轮计时；
- 打印 raw samples，同时记录最快 rank 的 median 和 mean；
- baseline 与优化版本使用完全相同的 PyPTO、simpler、PTOAS 和 PTO ISA pin；
- 正确性检查必须全部通过。

level-4 泳道图存在 observer cost，本文的 2283 µs 只用于拆解关键路径，不作为最终对外性能数字。
