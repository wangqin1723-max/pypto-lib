# DeepSeek V4 CSA TP4 精度问题排查记录

> 归档日期：2026-09-15。来源：`pypto-lib/local_archive/20260820-csa-precision-investigation.md`。
> 原文的实现、测量与计划保留其历史语境；阅读说明见[学习资料索引](../README.md)。

## 记录目的

记录近期 DeepSeek V4 CSA TP4 测试中出现的精度校验失败，区分已经确认的性能结果与尚未定位的环境/运行问题，供后续复现和定位使用。

## 当前代码与 PR 状态

- 仓库：`pypto-lib`
- 当前分支：`perf/fuse-deepseek-v4-csa-o-projection-rs`
- 当前 PR head：`229fc95`（两个 TP 优化提交）
- 本次没有修改 TP 优化代码，也没有提交新的代码变更。

## 已确认的有效结果

此前在同一组 TP4 workload 下，基线和融合路径都完成了数值校验并产生了有效 benchmark：

| 路径 | median | mean | max | 校验 |
| --- | ---: | ---: | ---: | --- |
| `upstream/main` 基线 | 4986.8 us | 5469.4 us | 10885.7 us | PASS |
| 当前两提交融合路径 | 4117.5 us | 4346.4 us | 8896.8 us | PASS |

测试条件为 TP4、`a2a3`、`case=max`，固定 4 张卡，benchmark 100 轮。

对应日志：

- 基线：tp4_two_commit_baseline_empty.log（本地证据：`$WORKSPACE/pypto-lib_tp4_full_baseline/tp4_two_commit_baseline_empty.log`；未随文归档）
- 融合后：tp4_two_commit_after_empty.log（本地证据：`20260820-csa-precision-logs/tp4_two_commit_after_empty.log`；未随文归档）

## 最近一次失败运行

使用 TP4 D-Spark CSA 入口重新运行时，先误加了该入口不支持的参数：

```text
--enable-l2-swimlane 0
```

该次只是参数解析失败。去掉参数后，TP4 入口按默认配置运行（默认关闭 swimlane），编译和 runtime 均完成，但最终校验失败。

失败日志：

- tp4_swimlane0_fullbench.log（本地证据：`20260820-csa-precision-logs/tp4_swimlane0_fullbench.log`；未随文归档）

校验结果：

```text
compress_state       PASS
inner_compress_state PASS
kv_cache             PASS
cmp_kv               PASS
idx_kv_cache         PASS
idx_kv_scale         PASS
x_out                FAIL
```

失败详情：

```text
ratio_reldiff fail: worst rdiff=1.985
```

日志中的示例绝对误差大约为 `0.004` 到 `0.007`，但相对误差检查在接近零的元素上被放大，因此触发了 `x_out` 的校验阈值。

本次没有产生可用的 `effective_us` benchmark 结论，不能拿这次运行与之前的 4 千多 us 数据比较。

## 已排除或暂时不作为结论的因素

### 不是中间缓存整体损坏

压缩状态、KV cache、索引 cache 等输出均 PASS，失败集中在最终 `x_out`，说明当前证据不能指向整个 CSA 图或通信结果全面错误。

### 不是单卡 MTP 结果与 TP4 不一致

单卡 `deepseek_v4_flash_mtp/decode_csa.py` 曾得到：

```text
effective_us = 1008.5 us
```

这是单卡 MTP workload，只用于验证 PTOAS/runtime 环境，不能与 TP4 D-Spark 的 4～5 ms 结果直接比较。

### `507018` 与本次 TP4 精度失败不是同一个现象

单卡 MTP 使用 `--enable-l2-swimlane 4` 时，benchmark 阶段出现过 `507018`，但数值校验仍 PASS；`swimlane=0` 单轮 benchmark 可以通过。该现象属于 benchmark/runtime 状态问题，不能直接解释 TP4 的 `x_out` 精度失败。

## 当前最可能的排查方向

目前还不能确认是代码回归。优先级如下：

1. **工具链或 runtime 不一致**：核对成功日志与失败日志使用的 PyPTO commit、runtime 构建目录、PTOAS 0.57、CPython 3.11 runtime 和 pto-isa commit。
2. **运行输入或物理卡差异**：确认 `case=max`、TP=4、权重/输入生成方式和卡组是否一致。
3. **数值路径边界差异**：若环境完全一致仍失败，再检查 O projection/ReduceScatter 融合后的 FP32 累加、量化/反量化和跨卡归并顺序。
4. **相对误差阈值放大**：记录绝对误差分布和失败元素位置，判断是系统性偏差还是接近零值的相对误差放大。

## 建议的下一步

暂不修改优化代码。先在与成功日志完全一致的环境下复现一次：

1. 记录并固定 PyPTO、runtime、pto-isa、PTOAS 版本和运行时路径。
2. 使用相同 TP4 `decode_csa.py -p a2a3 --tp 4 --case max` 命令和相同卡组。
3. 关闭 benchmark，仅验证 `x_out`，排除重复 dispatch 对结果的影响。
4. 若仍失败，再对比失败元素位置、绝对误差分布以及 O projection/ReduceScatter 中间张量。
5. 只有在精度稳定 PASS 后，才重新进行 TP4 性能 benchmark。

## 结论

此前的 TP4 融合性能结果仍然有效；最近一次 `x_out` 失败尚不足以证明 TP 优化代码回归。当前首要任务是复现并固定成功/失败运行之间的环境、输入和工具链差异。

## TP4 偶现复现结果（2026-08-20）

在关闭 benchmark、固定 `TP=4`、`case=max` 的条件下，对当前融合代码各重复运行 3 次：

| 卡组 | 第 1 次 | 第 2 次 | 第 3 次 |
| --- | --- | --- | --- |
| `1,3,5,9`（此前失败卡组） | PASS | `x_out` FAIL | PASS |
| `5,7,9,11`（自动分配空闲卡组） | PASS | `x_out` FAIL | PASS |

两组卡的共同特征：`compress_state`、`inner_compress_state`、KV cache 和索引 cache 均 PASS，只有最终 `x_out` 偶发失败，失败率均为 1/3。说明问题不是某一组物理卡独有，当前证据更支持 TP4 融合路径或 runtime 调度/同步的偶发问题。

对应日志：

- 原失败卡组：`tp4_fused_precision_repeat_{1,2,3}.log`
- 空闲卡组：`tp4_fused_idle_precision_repeat_{1,2,3}.log`

这组结果确认后，下一步应使用同一工具链和空闲卡组做 TP2 对照；TP2 结果可帮助判断问题是否由 TP4 特有的跨卡归并路径触发。

## TP2 ring 参数对照结果（2026-08-20）

使用此前成功运行的 ring 参数重新测试 TP2，卡组为 `5,7`，`case=max`，关闭 benchmark，连续运行 3 次：

```text
PTO2_RING_DEP_POOL=262144
PTO2_RING_TASK_WINDOW=262144
PTO2_RING_HEAP=2147483648
```

三次均正常完成，`compress_state`、`inner_compress_state`、KV cache、索引 cache 以及最终 `x_out` 全部 PASS。对应任务：

```text
task_20260820_075845_112846121152
```

对应日志：

- `tp2_fused_ring_precision_repeat_{1,2,3}.log`

此前未设置 ring 参数的 TP2 运行并未进入精度校验，而是在运行时触发 `507018 / HEAP_RING_DEADLOCK`；因此不能将其当作精度失败。

当时的 3 次结果全部 PASS，但后续追加重复已经证明 TP2 也会偶发 `x_out` 失败；因此那 3 次不能作为 TP2 稳定通过的结论。

## TP2 扩展重复结果（2026-08-20）

在相同卡组 `5,7` 和相同 ring 参数下追加 5 次 TP2 `case=max` 精度运行：

| 重复次数 | 结果 |
| --- | --- |
| 1 | PASS |
| 2 | `x_out` FAIL |
| 3 | PASS |
| 4 | PASS |
| 5 | `x_out` FAIL |

失败仍只发生在最终 `x_out`，压缩状态、KV cache 和索引 cache 均 PASS。失败样例的绝对误差约为 `0.0042`～`0.0066`，触发 `ratio_reldiff` 阈值；不是 ring deadlock 或 507018。追加 5 次的任务号为：

```text
task_20260820_082012_7372113485
```

日志：`tp2_fused_ring_extra_repeat_{1,2,3,4,5}.log`。合并此前 3 次结果后，TP2 共 8 次中 6 次 PASS、2 次 `x_out` FAIL（失败率 25%）。因此当前问题并非 TP4 独有，而是 TP2/TP4 都可能出现的最终输出偶发数值偏差；下一步应优先对比同一 TP2/TP4 输入、运行时调度和融合前后输出。

## 后续原因定位计划

1. 固定同一物理卡、ring 参数、工具链和输入生成方式，分别重复运行融合路径与未融合基线，确认失败是否只发生在融合实现。
2. 对每次失败记录 mismatch 数量、最大绝对误差、最大相对误差和元素位置，判断误差是否集中在固定 token、rank 或输出分片。
3. 对比 PASS/FAIL 两次运行的调度和通信日志，重点检查 ReduceScatter、owner window 写入、同步依赖以及 FP32 累加顺序。
4. 若基线也偶发失败，优先排查 runtime、PTOAS、pto-isa 和输入生成的非确定性；若只有融合路径失败，再针对 O projection 与 ReduceScatter 融合逐段回退定位。
5. 原因确认前不修改阈值，不把 `507018` 与数值失败混为一类，也不重新进行性能结论测试。

## 新 runtime 对照（2026-08-20）

PyPTO/runtime 已重新编译并使用匹配的 `pto-isa=f51c92f6`、PTOAS 0.57。TP2 融合路径重新运行后仍出现 `x_out` FAIL；压缩状态、KV cache、索引 cache 均 PASS，失败绝对误差仍约为 `0.004`～`0.005`，触发相对误差阈值。任务日志：tp2_fused_newruntime_probe.log（本地证据：`20260820-csa-precision-logs/tp2_fused_newruntime_probe.log`；未随文归档），任务号 `task_20260820_185028_377633622935`。

当前每次普通运行都会重新生成随机输入，PASS/FAIL 之间并非同一输入。因此现有“偶发”比例不能直接证明 runtime 非确定性；下一步必须固定输入（固定 torch seed 或 golden replay）后重复运行，再区分输入敏感的误差阈值问题与同一输入下的执行非确定性。

## 新 runtime 未融合对照（2026-08-20）

在同一套 `PTOAS=0.57`、`pto-isa=f51c92f6` 和 ring 参数下，临时切回未融合 `decode_o_proj + o_proj_reduce_scatter`，TP2 `case=max` 仍出现 `x_out` FAIL。所有中间缓存检查 PASS，失败绝对误差约 `0.0041`～`0.0062`。任务号：`task_20260820_185555_76384511538`，日志：tp2_unfused_newruntime_probe.log（本地证据：`20260820-csa-precision-logs/tp2_unfused_newruntime_probe.log`；未随文归档）。测试后已恢复融合代码，工作区无源码 diff。

这进一步说明新旧 runtime 以及融合/未融合两条 TP2 路径都能触发同类最终输出误差；当前应优先固定输入做 replay，而不是继续把问题归因于单个 O projection 实现。

## TP2 未融合对照（2026-08-20）

为做基线对照，临时将 `decode_csa.py` 恢复为 `decode_o_proj + o_proj_reduce_scatter`，测试结束后已恢复融合代码，工作区无源码改动。使用 PTOAS 0.57、相同 ring 参数、TP2 `case=max`，自动分配空闲卡：

- tp2_unfused_auto_repeat_1.log（本地证据：`20260820-csa-precision-logs/tp2_unfused_auto_repeat_1.log`；未随文归档）：PASS
- tp2_unfused_auto_repeat_2.log（本地证据：`20260820-csa-precision-logs/tp2_unfused_auto_repeat_2.log`；未随文归档）：`x_out` FAIL
- tp2_unfused_single_extra.log（本地证据：`20260820-csa-precision-logs/tp2_unfused_single_extra.log`；未随文归档）：PASS

其中第 2 次的压缩状态、KV cache、索引 cache 均 PASS，仅最终 `x_out` 失败；第 3～5 次因同一 task 内重复启动时出现 `pld.system.defer_wait` 注册表缺失，未进入精度校验，不能计入精度统计。独立 task 的补测重新 PASS。

因此未融合路径也能复现 `x_out` 偶发失败（当前有效样本 2 次 PASS、1 次 FAIL），说明问题不能简单归因于新增的 O projection/ReduceScatter 融合代码；需要继续排查 runtime 调度、输入/执行非确定性或更早的公共 CSA 数值路径。
