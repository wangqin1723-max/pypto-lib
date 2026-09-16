# PyPTO Worker 与 SPMD 工作分配说明

> 归档日期：2026-09-15。来源：`pypto-lib/local_archive/2026年9月3日-PyPTO-Worker与SPMD工作分配说明.md`。
> 原文的实现、测量与计划保留其历史语境；阅读说明见[学习资料索引](../README.md)。

日期：2026 年 9 月 3 日

本文用于解释 PyPTO kernel 中常说的 worker、`pl.spmd`、逻辑 block、物理计算核和 grid-stride 循环之间的关系，并结合 DSpark Decode HCA 中的实际代码说明“worker 摊薄”是怎样实现的。

## 1. 最重要的概念

下面几个名词容易混在一起：

| 名词 | 含义 |
| --- | --- |
| task | AICPU/runtime 调度的一次 kernel launch，是任务图中的一个节点 |
| SPMD block | 同一个 kernel 程序的一份逻辑实例，拥有独立的 block index |
| worker | 源码中对一个长期处理多份工作的 SPMD block 的称呼，不是新的硬件类型 |
| physical core/cluster | 真正执行 block 的 AIC、AIV 或 MIX 计算资源 |
| work item/tile | 算法上需要完成的一小份工作，例如一个 token、一组 head 或一个 KV tile |

因此，worker 本质上还是一个 SPMD block。只有当一个 block 在 kernel 内循环处理多份 work item 时，我们通常才强调它是一个“长期运行的 worker”。

## 2. `pl.spmd` 做了什么

```python
with pl.spmd(WORKERS, name_hint="demo"):
    worker = pl.tile.get_block_idx()
    ...
```

这段代码表示：

1. runtime 只调度一个名为 `demo` 的 task；
2. 该 task 展开为 `WORKERS` 个逻辑 block；
3. 每个 block 执行相同的程序；
4. `get_block_idx()` 为每个 block 返回不同的编号，范围是 `0` 到 `WORKERS - 1`。

例如 `WORKERS = 4` 时，同一个 kernel 程序会有四份逻辑实例：

```text
block/worker 0
block/worker 1
block/worker 2
block/worker 3
```

这些是逻辑 block，不保证在同一时刻启动，也不永久绑定到某个物理核。

## 3. 一份工作对应一个 block

最直接的写法是：

```python
with pl.spmd(TILE_COUNT, name_hint="one_block_one_tile"):
    tile = pl.tile.get_block_idx()
    process(tile)
```

这里启动 `TILE_COUNT` 个 block，每个 block 只处理一个 tile。

优点是工作天然并行；缺点是 block 数量很多时，每个 block 都要承担一遍 kernel 初始化、参数准备和资源装载。大量短 block 还可能表现为多波滚动派发，泳道图上会比较碎。

仓库里的 RMSNorm 接近这种形式：

```python
with pl.spmd(t_dim // T_TILE, name_hint="rms_norm"):
    worker = pl.tile.get_block_idx()
    row = worker * T_TILE
    ...
```

代码位置：`models/deepseek_v4_flash_mtp/rmsnorm.py:42`。

## 4. 固定 worker 数量，循环处理大量工作

worker 摊薄常用 grid-stride 循环实现：

```python
WORKERS = 4

with pl.spmd(WORKERS, name_hint="persistent_demo"):
    worker = pl.tile.get_block_idx()
    for task in pl.range(worker, task_count, WORKERS):
        process(task)
```

如果 `task_count = 11`，工作分配如下：

```text
worker 0 -> task 0, 4, 8
worker 1 -> task 1, 5, 9
worker 2 -> task 2, 6, 10
worker 3 -> task 3, 7
```

循环起点是 worker 编号，步长是 worker 总数。因此：

```text
worker i 处理 i, i + WORKERS, i + 2 * WORKERS, ...
```

每个 work item 只会被一个 worker 处理，同时所有合法 work item 都会被覆盖。

## 5. “摊薄”具体摊薄了什么

假设原来有 256 个 tile，并为每个 tile 启动一个 block：

```python
with pl.spmd(256):
    tile = pl.tile.get_block_idx()
    prepare()
    process(tile)
```

那么 `prepare()` 会执行 256 次。

改成 20 个长期 worker 后：

```python
with pl.spmd(20):
    worker = pl.tile.get_block_idx()
    prepare()
    for tile in pl.range(worker, 256, 20):
        process(tile)
```

此时 `prepare()` 只执行 20 次，每个 worker 平均处理 12～13 个 tile。能够被摊薄的开销通常包括：

- block/kernel 的固定初始化；
- 不随 tile 改变的索引计算；
- 可以跨循环复用的 KV、权重或 scratch；
- 重复创建的常量 tile；
- 逻辑 block 过多带来的启动与滚动派发开销。

真正的核心计算量通常没有减少：256 个 tile 仍然需要全部计算。性能收益来自固定成本减少和数据复用增加。

## 6. worker 与物理核不是一回事

`pl.spmd(20)` 表示 20 个逻辑 block，并不直接表示“占用前 20 个 AIC 核”。runtime 会根据 kernel 的资源类型，将 block 派发到可用的 AIC、AIV 或 MIX 计算资源。

对于同时包含 cube 和 vector 工作的 MIX kernel，一个可运行资源通常是一组配套的计算簇，而不是单独的一颗 AIC 或 AIV。逻辑 worker 只有在被 runtime 派发后才会占用相应资源。

因此需要区分：

```text
源码声明的 worker 数量
        ↓
一个 SPMD task 中的逻辑 block 数量
        ↓ runtime 按资源可用情况派发
设备上的物理 core/cluster
```

## 7. 为什么泳道图上不一定整齐同时开始

`pl.spmd(N)` 是一个 task，但它包含的 block 是逐个派发到可用资源的。默认 `sync_start=False`，runtime 不需要等全部 block 都具备资源后再统一启动。

所以泳道图可能表现为：

```text
core 0:   [block 0----------------]
core 1:      [block 1----------------]
core 2:         [block 2----------------]
...
```

这种现象叫滚动派发。出现错落并不自动说明性能有问题，可能只是：

- 各计算簇释放时间不同；
- runtime 逐个填充空闲 descriptor/resource slot；
- 前一个 kernel 在不同核上的结束时间不同；
- 不同 worker 实际处理的 work item 数量不同。

`sync_start=True` 会要求 block 同步启动，但它可能为了等待所有资源而推迟整个 task，不能仅为了让泳道图整齐就开启。性能判断仍应以 wall time 为准。

## 8. DSpark HCA raw attention 的 worker

当前 raw attention 使用 20 个 worker：

```python
RAW_WORKERS = 20

with pl.spmd(RAW_WORKERS, name_hint="hca_raw_attn"):
    worker = pl.tile.get_block_idx()
    for token in pl.range(worker, t_dim, RAW_WORKERS):
        ...
```

代码位置：`models/deepseek_v4_flash_dspark/decode_sparse_attn_hca.py:283`。

这里的算法工作单位是 token。每个 worker 循环处理：

```text
worker, worker + 20, worker + 40, ...
```

每处理一个 token 时，它会：

1. 确定该 token 对应的 request 和 raw-window 起点；
2. 将该 token 使用的 raw KV window 加载到 Mat memory；
3. 循环处理这个 token 的多个 head tile；
4. 完成 QK、softmax 和两段 PV；
5. 写出 softmax 的 `m/l/o` 状态。

相比一个 token/head tile 对应一个短 block，这种结构让同一个 worker 在更长时间内连续工作，并且 raw KV 可以被该 token 的多个 head tile 复用。

## 9. 仓库中的另一个实际例子

CP token AllGather 也使用 grid-stride worker：

```python
with pl.spmd(PUSH_WORKERS, name_hint="cp_token_allgather_push"):
    worker = pl.tile.get_block_idx()
    for band_row in pl.range(
        worker * COMM_ROW_TILE,
        full_local,
        PUSH_WORKERS * COMM_ROW_TILE,
    ):
        pld.tensor.put(...)
```

代码位置：`models/deepseek_v4_flash_dspark/decode_cp_token_allgather.py:94`。

这里每个 worker 循环发布多段 row band。它和 HCA 的计算内容不同，但工作分配机制相同。

## 10. worker 数量怎么选

worker 数量不是越多越好，也不是必须等于物理核数。

worker 太多：

- 固定初始化重复次数增加；
- KV/scratch 复用下降；
- 短 block 和滚动派发更明显。

worker 太少：

- 不能充分占满物理计算资源；
- 每个 worker 的串行循环过长；
- work item 数量不能整除时，尾部负载可能不均衡。

一般要在下面两者之间平衡：

```text
足够的并行度  <->  足够长的单 worker 生命周期和复用能力
```

选择方法是实际 sweep，例如比较 16、20、24、32 个 worker，并同时观察：

- 端到端最快-rank wall time；
- 目标 kernel 的 task duration；
- 核利用率和尾部；
- 每个 worker 分到的 work item 数量；
- Mat、Vec、Acc 等片上内存是否满足约束。

## 11. 阅读路径

建议依次阅读：

1. `docs/models/qwen3_14b/paged_attention_pypto.md:21`：最清楚的 24-worker + grid-stride 示意；
2. `docs/pypto-coding/pypto-coding-style.md:620`：`pl.spmd` 的官方语法；
3. `docs/debug-and-tune/dependency-and-scheduling.md:35`：一个 SPMD launch 为什么仍然只是一个 task；
4. `models/deepseek_v4_flash_dspark/decode_cp_token_allgather.py:94`：较短的实际 worker 代码；
5. `models/deepseek_v4_flash_dspark/decode_sparse_attn_hca.py:283`：HCA raw attention 的 persistent worker。

`examples/advanced/topk.py` 的注释虽然提到了 `pl.spmd block`，但当前实现实际使用 `pl.parallel + pl.at`，不适合作为 worker 模式的入门例子。

## 12. `hca_cmp_qk_pv` 的八 token 复用例子

`hca_cmp_qk_pv` 原来的逻辑 block 是一个 `token + compressed-KV work`：

```python
cmp_qk_block_count = t_dim * cmp_work_count

with pl.spmd(cmp_qk_block_count):
    token = block_idx // cmp_work_count
    work = block_idx % cmp_work_count
    kv = load_compressed_kv(work)
    compute_qk_softmax_pv(token, kv)
```

统一 8K、16 请求和每请求 8 个验证 token 时，`t_dim = 128`。当 `cmp_work_count = 1` 时，该结构会启动 128 个 MIX block。同一请求的 8 个 token 使用相同的 compressed-KV work，但是每个 block 都会重新装载一次 `[128, 512]` BF16 KV tile。

优化后将一个请求的 8 个 token 组成一个 token tile：

```python
CMP_T_TILE = 8
cmp_qk_block_count = (t_dim // CMP_T_TILE) * cmp_work_count

with pl.spmd(cmp_qk_block_count):
    token_start = token_block * CMP_T_TILE
    kv = load_compressed_kv(work)
    for token_offset in pl.range(CMP_T_TILE):
        compute_qk_softmax_pv(token_start + token_offset, kv)
```

上述统一 8K 条件下，逻辑 block 数从 128 个降到 16 个。每个 block 只将 `[128, 512]` compressed-KV tile 装载到 Mat memory 一次，然后在 kernel 内循环处理同一请求的 8 个 token。

这个优化同时减少了：

- compressed-KV 的 GM 到 Mat 装载次数，理论上从每请求 8 次降到 1 次；
- block 固定初始化和索引准备次数；
- runtime 为 MIX block 滚动派发的数量；
- 生成 orchestration 中与 block 数成正比的 GM pipe scratch，在该 case 中从约 16 MiB 降到 2 MiB。

QK、softmax 和 PV 的数学计算量没有减少；收益来自跨 token 的 KV/scratch 复用和 block 固定成本摊薄。这是一种“按可复用数据分组”的 worker 设计，和 raw attention 的 grid-stride worker 思想相同，但工作分配形式不同。

在 wq3、A2/A3、TP=4、16 请求统一 8K、固定设备 5/7/9/11、5 轮 warmup + 100 轮测量下，最快 rank 结果为：

| 版本 | median | mean |
| --- | ---: | ---: |
| 优化前 | 1336.5 µs | 1337.7 µs |
| `CMP_T_TILE = 8` | 1216.2 µs | 1233.7 µs |
| 提升 | 120.3 µs，9.001% | 104.0 µs，7.775% |

两个版本均复用同一份 frozen golden，`compress_state`、`kv_cache`、`cmp_kv` 和 `x_out` 全部 PASS。原始日志位于：

- `build_output/hca_cmp_qk_pv_tuning/baseline_100r.log`
- `build_output/hca_cmp_qk_pv_tuning/cmp_t8_100r.log`

生成代码的直接证据是，优化后 `hca_cmp_qk_pv.pto` 只在 token 循环前出现一次 compressed-KV `pto.tload`，紧接着是上界为 8 的 `scf.for`。
