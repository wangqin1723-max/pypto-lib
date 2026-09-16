# HCA TP1 CMP 任务切分实验（2026-09-15）

> 归档日期：2026-09-15。来源：`pypto-lib/local_archive/dspark-hca-tp1-cmp-split-results-20260915.md`。
> 原文的实现、测量与计划保留其历史语境；阅读说明见[学习资料索引](../README.md)。

## 结论

每 block 处理 2 个 query 可将 128K CMP 的 AIC block 耗时从 140.98–197.02 µs 缩短到 59.54–75.66 µs，达到了 50–100 µs 的任务粒度目标。但关闭 profiling 的 HCA 中位数只下降 4.25 µs（0.57%）；3 query 候选只下降 2.15 µs（0.29%）。本轮不将其作为已经证明稳定收益的优化提交 PR，两份补丁和全部样本保留。

## 配置与验证

- 基线：合入 #1228 的 main 提交 `1a9e487ce22a12c10e651b6b0f25fdf838068791`，并非历史 P4 工作区。
- 平台 a2a3，同一分配的偶数卡 12；TP1，batch=16，每请求本轮 8 个 token。
- 上下文分别为 16 个 8192、16 个 131072；每次共 128 个 query。
- 每配置一个 benchmark 进程，5 warmup / 100 samples，保留所有样本。PYPTO_BENCH=1、PYPTO_BENCH_RAW=1，实际 chip/dep-gen/PMU/dump-args/scope-stats 全部关闭。
- 沿用已通过的 8K/128K 输入和 golden；六个 benchmark 均有输入/输出缓存命中、x_out PASS 和 RUN PASS，未放宽精度阈值。
- 两候选的额外 attention 边界精度均通过：连续/非连续页、负页号、越界页号、零历史、部分页、混合长度。batch=2 的 16 query 还覆盖 3 query 分组的末尾不足一组。
- 六组 TP1 编译通过；额外 TP4 编译通过。未测 TP4 性能，也未验证所有 batch 的性能。
- 工具链：PyPTO f1bb086、simpler 15f5cbd、PTOAS 0.57、PTO ISA 96ba706、CANN 9.0.0，与历史环境相同。
- 任务 `task_20260915_203239_34667123538` 完成，exit=0，设备已释放。

## 改动

只修改 `decode_sparse_attn_hca.py`。TP1 且压缩 KV 超过一个 K tile 时，把原来 24 个 block 按步长 24 分配 query 的方式，改为每 block 处理 2 或 3 个连续 query，逻辑 block 数分别为 64、43。每个 block 拥有独立的三槽传递缓冲区。K=128、每个 query 的 QK/softmax/PV 计算及输出归属不变。

8K 的单 KV tile 快速路径保留 24 个 block；TP4 保留原分配方式。生成 C++ 去除注释及自动 inline 后缀后，仅 CMP AIC/AIV 两个计算 kernel 有变化，编排代码相应改变 block 数及缓冲区分配。

## 关闭 profiling 的 benchmark 中位数

| 方案 | 8K | 相对基线 | 128K | 相对基线 |
|---|---:|---:|---:|---:|
| 基线 | 545.10 µs | — | 747.64 µs | — |
| 每 block 2 query | 543.77 µs | -1.33 µs / 0.24% | 743.39 µs | -4.25 µs / 0.57% |
| 每 block 3 query | 544.94 µs | -0.16 µs / 0.03% | 745.49 µs | -2.15 µs / 0.29% |

## 128K 单次 level-4 泳道图

| 方案 | CMP 逻辑 block 数 | 单 AIC block min / median / max | CMP 整体窗口 | HCA AICore 端到端 |
|---|---:|---:|---:|---:|
| 基线 | 24 | 140.98 / 162.09 / 197.02 µs | 348.14 µs | 882.58 µs |
| 2 query | 64 | 59.54 / 68.14 / 75.66 µs | 212.82 µs | 772.98 µs |
| 3 query | 43 | 58.66 / 95.16 / 104.76 µs | 251.34 µs | 796.48 µs |

单次泳道图中的基线 CMP AIC 启动跨度为 196.06 µs，存在明显分批启动。两候选仍需在同样的 24 个 AIC 上分批完成全部 query；缩短单 block 不等于把整个 CMP 阶段降到 50–100 µs。基线的本次调度形态也不代表其 100 次无 profiling 样本的中位数。不能把泳道图 882.58→772.98 µs 写成 benchmark 收益。

传递缓冲区总量由 12.41 MiB 增加到 33.09 MiB（2 query）或 22.23 MiB（3 query）。目前没有减少 attention 总计算量，也没有测出相称的中位数收益。

## 产物

实验根目录：`build_output/hca_tp1_cmp_split_20260915_203011/`。

- `results.json`：完整汇总、100 个原始样本、逐 kernel 与逐引擎统计。
- `{baseline,q2,q3}/{8192,131072}_bench/`：实际配置和 benchmark 结果。
- `q2/candidate.patch`、`q3/candidate.patch`：独立实验补丁。
- `chip_swimlanes/{baseline,q2,q3}/131072/`：128K 前后泳道图。
- `chip_swimlanes/{baseline,q2}/8192/`：8K 前后泳道图。

五份捕获的原始 AICore/AICPU/合并记录数和依赖声明的 block 数全部一致，各目录均包含 records、deps、name map、CPM_static.json、CPM_observed.json 和完整关键路径报告。
