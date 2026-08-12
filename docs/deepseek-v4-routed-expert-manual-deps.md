# DeepSeek V4 Routed Expert 手动依赖记录

本文记录 `models/deepseek/v4/expert_routed.py` 这次手动建依赖的过程。
核心目标是：打破 routed expert 内部由于动态 `recv_y` 写入导致的保守串行，同时保证后续 combine 仍然能正确等待每个非空 tile 的输出。

## 背景问题

`expert_routed` 的输出是 `recv_y`，shape 逻辑上是：

```text
[N_LOCAL_EXPERTS, RECV_MAX, D]
```

每个 local expert 写自己的 row slab，理论上不同 expert、不同 tile 之间互不冲突。但代码里 `n_rows` 来自运行时的 `recv_expert_count`，所以实际循环是动态的：

```python
for local_i in pl.parallel(N_LOCAL_EXPERTS):
    n_rows = pl.read(recv_expert_count, [local_i, 0])
    n_tiles = (n_rows + RECV_TILE - 1) // RECV_TILE
    for t in pl.parallel(n_tiles):
        ...
        recv_y_flat[...] = ...
```

自动依赖分析面对这种动态写入时，不能总是证明不同 tile 写的是完全不相交的区域。为了保证正确性，runtime 可能会保守地把一些本来可以并行的任务串起来。这个问题在 7 层 `decode_fwd.py` 泳道图里会放大，表现为 routed expert 相关任务出现明显串行尾巴，AICore 端到端时间变长。

## 代码模式 1：原始自动依赖模式

原始写法主要依赖自动依赖分析：

```python
for local_i in pl.parallel(N_LOCAL_EXPERTS):
    n_rows = pl.read(recv_expert_count, [local_i, 0])
    n_tiles = (n_rows + RECV_TILE - 1) // RECV_TILE

    for t in pl.parallel(n_tiles):
        h_tile_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32)
        gate_i32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT32)
        up_i32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT32)

        for nb_idx in pl.spmd(..., name_hint="exp_gate_mm"):
            ...

        for nb_idx in pl.spmd(..., name_hint="exp_up_mm"):
            ...

        for nb_idx in pl.spmd(..., name_hint="exp_gate_up_act"):
            ...

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_h_q"):
            ...

        for db_idx in pl.spmd(..., name_hint="exp_w2_mm"):
            ...

        for db_idx in pl.spmd(..., name_hint="exp_w2_act"):
            recv_y_flat[...] = ...
```

这个模式的优点是简单，所有依赖由编译/runtime 自动处理。缺点是：动态 `recv_y` 写入会让依赖分析偏保守，导致一些本该并行的 expert tile 被串行化。

standalone `expert_routed.py` 旧版的一个基准结果是：

```text
old auto-dependency version: 515.96 us, 350 AICore tasks
```

这个 standalone 结果本身不差，但 full graph 里会出现更明显的串行风险。

## 代码模式 2：错误的 full-launch 手动依赖模式

中间尝试过一种更激进的手动依赖方式：把 routed expert 全部放进 `manual_scope()`，并用固定上限循环发任务。

伪代码类似：

```python
with pl.manual_scope():
    for local_i in pl.parallel(N_LOCAL_EXPERTS):
        for t in pl.parallel(TILES_PER_EXPERT):
            ...
```

这个模式确实能绕开自动依赖导致的保守串行，但问题是它没有保留动态 `n_tiles`。即使某个 expert 没有 token，或者只有很少 tile，也会按固定上限发任务。

泳道图上表现为 AICore task 数明显增加：

```text
fixed full expert/tile manual version: about 661-671 us, 449 AICore tasks
```

这说明这个方案虽然解决了串行，但引入了空 expert 发任务的问题，因此不能作为最终方案。

## 代码模式 3：最终动态手动依赖模式

最终方案保留动态 `n_tiles`，只让非空 tile 发任务，同时把核心 compute chain 放入 `manual_scope()` 并显式串依赖。

核心结构是：

```python
TILES_PER_EXPERT = RECV_MAX // RECV_TILE

recv_y_flat = pl.reshape(recv_y, [N_LOCAL_EXPERTS * RECV_MAX, D])
w2_act_tids = pl.array.create(N_LOCAL_EXPERTS * TILES_PER_EXPERT, pl.TASK_ID)

with pl.manual_scope():
    for local_i in pl.parallel(N_LOCAL_EXPERTS):
        n_rows = pl.read(recv_expert_count, [local_i, 0])
        n_tiles = (n_rows + RECV_TILE - 1) // RECV_TILE

        for t in pl.parallel(n_tiles):
            tile_idx = local_i * TILES_PER_EXPERT + t
            ...
```

这里的关键点是：`n_tiles` 仍然来自运行时 `n_rows`，所以空 expert 不会发 routed compute task。

在 `manual_scope()` 里面，临时 GM tensor 需要声明 `manual_dep=True`：

```python
h_tile_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32, manual_dep=True)
gate_i32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT32, manual_dep=True)
up_i32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT32, manual_dep=True)
```

然后每个阶段显式接上依赖。

### gate/up 并行

`gate` 和 `up` 互相独立，可以并行发：

```python
with pl.spmd(..., name_hint="exp_gate_mm") as gate_tid:
    ...

with pl.spmd(..., name_hint="exp_up_mm") as up_tid:
    ...
```

### act 等待 gate/up

激活阶段依赖 `gate` 和 `up`：

```python
with pl.spmd(
    ...,
    name_hint="exp_gate_up_act",
    deps=[gate_tid, up_tid],
) as act_tid:
    ...
```

### h_q 等待 act

量化阶段依赖 act：

```python
h_tile_i8 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT8, manual_dep=True)

with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_h_q", deps=[act_tid]) as hq_tid:
    ...
```

### w2 等待 h_q

第二个 matmul 依赖量化结果：

```python
y_i32 = pl.create_tensor([RECV_TILE, D], dtype=pl.INT32, manual_dep=True)

with pl.spmd(..., name_hint="exp_w2_mm", deps=[hq_tid]) as w2_tid:
    ...
```

### w2_act 等待 w2，并写 recv_y

最终反量化、乘 routing weight、写 `recv_y`：

```python
with pl.spmd(..., name_hint="exp_w2_act", deps=[w2_tid]) as w2_act_tid:
    ...
    recv_y_flat[flat_t0 : flat_t0 + RECV_TILE, d0 : d0 + D_OUT_TILE_ACT] = ...

w2_act_tids[tile_idx] = w2_act_tid
```

到这里，manual scope 内部的计算依赖链是：

```text
exp_gate_mm ┐
            ├─> exp_gate_up_act -> exp_h_q -> exp_w2_mm -> exp_w2_act
exp_up_mm   ┘
```

不同 expert/tile 之间没有被自动依赖分析强行串起来，可以更自然地重叠执行。

## 代码模式 4：marker task 回到自动依赖模式

`manual_scope()` 解决了串行问题，但也带来一个新问题：manual scope 内部写 `recv_y` 后，外部自动依赖分析不一定能看到这些写入已经完成。

而 full graph 里后面还有 combine，需要正确等待 `recv_y`。所以在 `manual_scope()` 外面，再用一个很小的自动依赖 task 把每个非空 tile 的 `recv_y` 重新注册给自动依赖系统。

代码是：

```python
for local_i in pl.parallel(N_LOCAL_EXPERTS):
    n_rows = pl.read(recv_expert_count, [local_i, 0])
    n_tiles = (n_rows + RECV_TILE - 1) // RECV_TILE
    flat_base = local_i * RECV_MAX

    for t in pl.parallel(n_tiles):
        tile_idx = local_i * TILES_PER_EXPERT + t
        flat_t0 = flat_base + t * RECV_TILE

        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="exp_routed_tile_done",
            deps=[w2_act_tids[tile_idx]],
        ):
            recv_y_flat[flat_t0 : flat_t0 + RECV_TILE, 0:16] = recv_y_flat[
                flat_t0 : flat_t0 + RECV_TILE, 0:16
            ]
```

这个 `recv_y -> recv_y` 自赋值不是为了计算，而是为了建立依赖桥：

```text
manual compute chain -> exp_routed_tile_done -> downstream combine
```

这里故意只写 `0:16` 这一小段，减少 marker task 的实际开销。它只需要让自动依赖系统知道：这个 tile 的 `recv_y` 在 `w2_act` 后才 ready。

## 泳道图截图路径

下面这些都是 repo-relative 路径，可以在仓库根目录下直接找到。截图时优先打开
`merged_swimlane_*.json`，可以拖到 Perfetto UI 查看。

### standalone `expert_routed.py` 对比

旧版自动依赖模式，AICore 端到端 `515.96 us`，`350` 个 AICore task：

```text
build_output/_jit_expert_routed_test_20260702_150052/dfx_outputs/merged_swimlane_20260702_150114.json
build_output/_jit_expert_routed_test_20260702_150052/dfx_outputs/l2_swimlane_records.json
```

错误的 fixed full-launch 手动依赖模式，AICore 端到端约 `661.52 us`，`449`
个 AICore task，用来截图说明“空 expert 也发任务”的问题：

```text
build_output/_jit_expert_routed_test_20260702_152705/dfx_outputs/merged_swimlane_20260702_152724.json
build_output/_jit_expert_routed_test_20260702_152705/dfx_outputs/l2_swimlane_records.json
```

最终动态手动依赖模式，和旧版同样是 25 个非空 tile，AICore 端到端
`521.30 us`，`375` 个 AICore task：

```text
build_output/_jit_expert_routed_test_20260702_153950/dfx_outputs/merged_swimlane_20260702_154009.json
build_output/_jit_expert_routed_test_20260702_153950/dfx_outputs/l2_swimlane_records.json
```

这一组截图重点看：

```text
old:      没有 exp_routed_tile_done，task 数少一些
fixed:    空 expert/tile 也发任务，task 数明显变多
current:  只对非空 tile 多出 exp_routed_tile_done，用小 marker 换下游依赖可见性
```

### 7 层 `decode_fwd.py` rank1 对比

旧的正常版本之一，rank1 AICore 端到端 `11057.26 us`：

```text
build_output/_jit_l3_decode_fwd_20260702_105132/dfx_outputs/rank1/d0/merged_swimlane_20260702_105246.json
build_output/_jit_l3_decode_fwd_20260702_105132/dfx_outputs/rank1/d0/l2_swimlane_records.json
```

旧的 manual-dep 版本之一，rank1 AICore 端到端 `10779.18 us`：

```text
build_output/_jit_l3_decode_fwd_20260702_110829/dfx_outputs/rank1/d0/merged_swimlane_20260702_110949.json
build_output/_jit_l3_decode_fwd_20260702_110829/dfx_outputs/rank1/d0/l2_swimlane_records.json
```

之前最好的一次正常结果，rank1 AICore 端到端 `10194.30 us`：

```text
build_output/_jit_l3_decode_fwd_20260702_112926/dfx_outputs/rank1/d0/merged_swimlane_20260702_113058.json
build_output/_jit_l3_decode_fwd_20260702_112926/dfx_outputs/rank1/d0/l2_swimlane_records.json
```

串行异常样例，rank1 AICore 端到端 `38670.98 us`：

```text
build_output/_jit_l3_decode_fwd_20260702_144702/dfx_outputs/rank1/d0/merged_swimlane_20260702_144827.json
build_output/_jit_l3_decode_fwd_20260702_144702/dfx_outputs/rank1/d0/l2_swimlane_records.json
```

另一个串行异常样例，rank1 AICore 端到端 `46190.78 us`：

```text
build_output/_jit_l3_decode_fwd_20260702_145458/dfx_outputs/rank1/d0/merged_swimlane_20260702_145622.json
build_output/_jit_l3_decode_fwd_20260702_145458/dfx_outputs/rank1/d0/l2_swimlane_records.json
```

当前最终版本，rank1 AICore 端到端 `10240.10 us`：

```text
build_output/_jit_l3_decode_fwd_20260702_155153/dfx_outputs/rank1/d0/merged_swimlane_20260702_155318.json
build_output/_jit_l3_decode_fwd_20260702_155153/dfx_outputs/rank1/d0/l2_swimlane_records.json
```

这一组截图重点看：

```text
normal old:       10.8-11.4 ms 档位
best old:         10.19 ms，和当前基本持平
serialized old:   38-46 ms，routed expert 串行异常明显
current:          10.24 ms，规避大串行异常，同时保持正常档位性能
```

## 泳道图里的体现

### standalone `expert_routed.py`

公平对比选择同样 25 个非空 tile 的结果：

```text
old auto-dependency version:        515.96 us, 350 AICore tasks
dynamic manual-dependency version:  521.30 us, 375 AICore tasks
```

当前版本多了 25 个 AICore tasks，正好对应 25 个非空 tile 的 marker：

```text
exp_routed_tile_done(r3t0)
exp_routed_tile_done(r3t1)
...
exp_routed_tile_done(r3t24)
```

泳道图里这些 marker 在尾部形成一串很短的小块，例如：

```text
exp_routed_tile_done(r3t0)  ts=448.62  dur=1.12  fanin=[r2t6]        fanout=[r3t1]
exp_routed_tile_done(r3t1)  ts=454.36  dur=0.98  fanin=[r2t13,r3t0]  fanout=[r3t2]
...
exp_routed_tile_done(r3t24) ts=521.78  dur=0.60  fanin=[r2t174,r3t23]
```

所以 standalone 变慢一点的原因很明确：不是 matmul 或 activation 本身变慢，而是多了这些 `exp_routed_tile_done` marker task。单个 marker 大约 `0.6-1.1 us`，总 dur 约 `18.22 us`，但它们和前面计算有重叠，所以最终 AICore 端到端只多了约 `5.34 us`。

### 7 层 `decode_fwd.py`

7 层 full graph 才是这次修改更关心的场景，因为它包含后续 combine，也最容易暴露 routed expert 串行问题。

rank1 AICore 端到端时间对比：

```text
prior normal runs:      11057.26 us, 10779.18 us, 11422.14 us
best prior normal run:  10194.30 us
current run:            10240.10 us
serialized outliers:    38670.98 us, 46190.78 us
```

结论是：当前版本比多个正常旧结果快约 `5%-10%`，和最好一次正常结果基本持平；更重要的是，它规避了 `38 ms / 46 ms` 这种明显串行异常。

## 最终权衡

这次修改的本质是用一小段可控开销换 full graph 里的依赖正确性和并行度：

```text
收益：
- 保留动态 n_tiles，空 expert 不发 routed compute task
- routed expert 内部显式依赖，减少自动依赖导致的保守串行
- downstream combine 通过 marker task 仍然能正确等待 recv_y

代价：
- 每个非空 tile 多一个 exp_routed_tile_done marker task
- standalone expert_routed.py 微基准会看到约 5 us 级别的小幅退化
```

因此这个方案适合当前目标：优化 7 层 decode full graph 的 routed expert 串行问题，而不是单纯追求 standalone `expert_routed.py` 的最小 task 数。

## 验证命令

本次使用过的主要验证命令：

```bash
python -m py_compile models/deepseek/v4/expert_routed.py
ruff check --config ruff.toml models/deepseek/v4/expert_routed.py
python tests/lint/check_headers.py
python tests/lint/check_english_only.py
python models/deepseek/v4/expert_routed.py -p a2a3 -d 0 --enable-l2-swimlane
python models/deepseek/v4/decode_fwd.py -p a2a3 --ep 2 -d 0,1 --enable-l2-swimlane
```

standalone `expert_routed.py` 的 `recv_y` 校验通过；7 层 `decode_fwd.py` 成功生成 rank1 泳道图并用于 AICore 端到端时间对比。
