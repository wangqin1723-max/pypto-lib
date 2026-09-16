# HCA 生成 Kernel 与任务结构分析

> 归档日期：2026-09-15。来源：`pypto-lib/local_archive/2026年9月2日-HCA生成Kernel与任务结构分析.md`。
> 原文的实现、测量与计划保留其历史语境；阅读说明见[学习资料索引](../README.md)。

日期：2026 年 9 月 2 日

本文记录如何从一次完整的 `decode_hca_test` build output 中理解 `hca_raw_attn` 的任务数量、SPMD block 数、AIC/AIV 物理执行数量、单 block 内部计算以及泳道耗时。本文仅保存在 `local_archive/`，不作为公共文档提交。

关联记录：[DSpark Decode HCA 性能优化记录](../case-studies/2026%E5%B9%B49%E6%9C%881%E6%97%A5-DSpark%20Decode%20HCA%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96%E8%AE%B0%E5%BD%95.md)

## 1. 分析对象

本文分析下面这次运行：

```text
build_output/_jit_l3_decode_hca_20260902_013402
```

运行条件：

- DSpark Decode HCA；
- A2/A3，TP=4；
- 16 个请求；
- 16 个请求统一从 8192 开始；
- chip swimlane perf level 4；
- 优化版本提交 `94c5fa0`；
- 按最快 rank 口径选择 rank2。

rank2 的主要分析文件：

```text
build_output/_jit_l3_decode_hca_20260902_013402/
├── next_levels/decode_hca_test/
│   ├── kernel_config.py
│   ├── kernels/aic/hca_raw_attn_aic.cpp
│   ├── kernels/aiv/hca_raw_attn_aiv.cpp
│   └── ptoas/
│       ├── hca_raw_attn.cpp
│       └── hca_raw_attn.pto
└── dfx_outputs/rank2/d0/
    ├── deps.json
    ├── name_map.json
    ├── chip_swimlane_records.json
    └── merged_swimlane_20260902_013413.json
```

## 2. 先区分四个概念

生成文件很容易让人把 task、block 和 kernel execution 混为一谈。对本次 `hca_raw_attn`，正确关系是：

```text
1 个运行时逻辑 task
    └── 256 个 SPMD logical block
            ├── 每个 block 执行 1 个 AIC kernel
            └── 每个 block 执行 2 个 AIV kernel
```

因此：

```text
hca_raw_attn
= 1 个逻辑 task
= 256 个 logical block
= 256 条 AIC 物理执行记录
 + 512 条 AIV 物理执行记录
= 768 条泳道物理记录
```

几个概念的证据来源不同：

| 问题 | 应查看的文件 |
| --- | --- |
| 有多少逻辑 task、每个 task 有多少 block | `deps.json` |
| kernel ID 对应什么函数、运行在 AIC 还是 AIV | `kernel_config.py`、`name_map.json` |
| 一个 block 内部执行哪些 tile、MatMul、Vector 和搬运指令 | `kernels/aic/*.cpp`、`kernels/aiv/*.cpp` |
| 编译器 IR、tile shape 和源代码位置 | `ptoas/*.pto` |
| block 最终在哪个物理核上、何时开始和结束 | `chip_swimlane_records.json` |
| 任务之间的依赖和可视化调度 | `deps.json`、`merged_swimlane*.json` |

## 3. `deps.json` 给出的任务结构

rank2 的 `deps.json` 中，`hca_raw_attn` 对应：

```json
{
  "task_id": "12884901905",
  "early_dispatch": false,
  "kernel_ids": [30, 31, 31],
  "block_num": 256
}
```

其中：

- `task_id=12884901905`：这是一个运行时任务节点；
- `block_num=256`：这个任务包含 256 个逻辑 SPMD block；
- `kernel_ids=[30,31,31]`：每个 block 有一个 AIC subslot 和两个 AIV subslot；
- `early_dispatch=false`：这个 task 本身没有作为 early-resolve producer 被标记。

`kernel_config.py` 和 `name_map.json` 给出：

```text
func_id 30 = hca_raw_attn_aic
func_id 31 = hca_raw_attn_aiv
```

整个 `decode_hca_test` 有 73 个逻辑 task。HCA 本地计算及其尾部链路如下：

| 逻辑 task | block 数 | 资源形状 | 预期物理记录数 | rank2 实际记录数 |
| --- | ---: | --- | ---: | ---: |
| `hca_cache_writeback` | 64 | AIV | 64 | 64 |
| `hca_gather_kv` | 16 | AIV | 16 | 16 |
| `hca_raw_valid` | 16 | AIV | 16 | 16 |
| `rope_cs` | 2 | AIV | 2 | 2 |
| `hca_raw_attn` | 256 | MIX | 768 | 768 |
| `hca_cmp_work_gather` | 16 | AIV | 16 | 16 |
| `hca_cmp_qk_pv` | 128 | MIX | 384 | 384 |
| `hca_stream_merge_pack` | 48 | AIV | 48 | 48 |
| `hca_stream_publish` | 48 | AIV | 48 | 48 |
| `o_group_a2a_wait` | 1 | AIV | 1 | 1 |
| `o_group_a2a_gather` | 48 | AIV | 48 | 48 |
| `o_group_a2a_complete` | 1 | AIV | 1 | 1 |

MIX task 的物理记录数计算方法是：

```text
physical rows = block_num × active kernel slots
```

例如：

```text
hca_raw_attn   = 256 × 3 = 768
hca_cmp_qk_pv  = 128 × 3 = 384
```

## 4. 为什么生成 C++ 里不直接写着 256

`hca_raw_attn_aic.cpp` 描述的是“一个 AIC block 收到某个 `block_idx` 后做什么”，不是整个 SPMD fan-out 的调度程序。

文件底部的 `kernel_entry` 从运行时 dispatch payload 中读取：

```cpp
int32_t __pypto_spmd_block_idx = get_block_idx(args);
int32_t __pypto_spmd_block_num = get_block_num(args);
```

然后将它们传给生成函数：

```cpp
hca_raw_attn_aic(..., __pypto_spmd_block_idx, __pypto_spmd_block_num);
```

在函数签名中，它们被匿名化为：

```text
v12 = block_idx
v13 = block_num
```

因此，同一份 AIC 二进制会带着不同的 `block_idx` 执行 256 次。`block_num=256` 是运行时任务属性，应以 `deps.json` 为准。

## 5. 从生成 C++ 还原 block 到数据的映射

生成 AIC 代码中的关键常量可以还原为：

```text
T              = 128 tokens
H              = 64 heads
RAW_H_TILE     = 32 heads
H / RAW_H_TILE = 2 head tiles per token
S              = 8 speculative tokens per request
WIN            = 128 rows
HEAD_DIM       = 512
REQUEST_KV_ROWS= 135 = WIN + S - 1
request_count  = 16
```

因此：

```text
raw_block_count = T × (H / RAW_H_TILE)
                = 128 × 2
                = 256
```

生成代码中的 block 映射等价于：

```python
stream_t = block_idx // 2
stream_h_tile = block_idx % 2
stream_h0 = stream_h_tile * 32
stream_state_row = stream_t * 64 + stream_h0
```

举例：

| block_idx | token | head 范围 | state 起始行 |
| ---: | ---: | --- | ---: |
| 0 | 0 | 0～31 | 0 |
| 1 | 0 | 32～63 | 32 |
| 2 | 1 | 0～31 | 64 |
| 3 | 1 | 32～63 | 96 |
| 254 | 127 | 0～31 | 8128 |
| 255 | 127 | 32～63 | 8160 |

request 和 raw KV 起始地址等价于：

```python
stream_request = stream_t // 8
stream_token = stream_t % 8
stream_first_t = stream_request * 8
stream_first_len = window_swa_lens[stream_first_t]
stream_drop = max(stream_first_len + stream_token - 128, 0)
stream_raw_base = stream_request * 135 + stream_drop
```

这也解释了 `raw_kv` 的形状：

```text
[request_count × REQUEST_KV_ROWS, HEAD_DIM]
= [16 × 135, 512]
= [2160, 512]
```

## 6. 一个 `hca_raw_attn` block 内部做什么

一个 logical block 处理一个 token 的 32 个 head：

```text
Q       [32, 512] BF16
raw KV  [128, 512] BF16
score   [32, 128] FP32
output  [32, 512] FP32
m/l     [32, 1] FP32
```

总体数据流：

```text
AIC：加载 Q 和 raw KV，计算 QK
  ↓ TPipe
AIV：scale、valid mask、row_max、exp、row_sum、BF16 cast
  ↓ TPipe
AIC：计算 PV-left 和 PV-right
  ↓
写出 m[32,1]、l[32,1]、o[32,512]
```

### 6.1 AIC 侧

源代码层面的 QK 是：

```text
[32, 512] × [512, 128] → [32, 128]
```

PTOAS 将 K=512 拆成 4 个 128，所以生成代码为：

```text
1 × TMATMUL
3 × TMATMUL_ACC
```

两个 PV 分别为：

```text
PV-left  [32,128] × [128,256] → [32,256]
PV-right [32,128] × [128,256] → [32,256]
```

每个 PV 的 K=128 被拆成两个 64，因此各自产生：

```text
1 × TMATMUL
1 × TMATMUL_ACC
```

一个 AIC block 合计：

```text
3 × TMATMUL
5 × TMATMUL_ACC
= 8 条 Cube MatMul 指令
```

### 6.2 AIV 侧

每个 MIX block 有两个 AIV subblock。AIV 的 `kernel_entry` 额外读取：

```cpp
int32_t __pypto_spmd_subblock_idx = get_sub_block_id(args);
```

生成函数通过 `subblock_idx` 区分两个 AIV lane。主要 Vector 工作包括：

- 加载 `[1,128]` validity mask；
- 扩展为 `[32,128]`；
- 对 QK score 执行 scale 和 bias；
- `TROWMAX`；
- `TEXP`；
- validity mask；
- `TROWSUM`；
- cast 为 BF16；
- 经 TPipe 送回 AIC，供两段 PV 使用；
- 写出 softmax 的 `m` 和 `l` 状态。

### 6.3 为什么 AIC 文件里还能看到 AIV 函数

`kernels/aic/hca_raw_attn_aic.cpp` 中同时存在 AIC 和 AIV 的静态生成函数。这是 MIX PTO 源经过代码生成后的组织形式，不表示 AIC 会执行 Vector 分支：

- AIC 二进制启用 `__DAV_CUBE__`，执行 Cube body；
- AIV 二进制启用 `__DAV_VEC__`，执行 Vector body；
- `kernel_entry` 最终调用对应的 AIC 或 AIV 函数。

因此不能简单按整个 `.cpp` 文件中的字符串出现次数统计实际指令，必须看编译宏和最终调用入口。

## 7. 输入输出元数据

`deps.json` 中 `hca_raw_attn` 的主要参数为：

| 参数 | 方向 | dtype | shape | 含义 |
| ---: | --- | --- | --- | --- |
| 0 | INPUT | INT32 | `[128]` | 每个 token 的 raw-window length |
| 1 | INPUT | BF16 | `[8192,512]` | 展平的 Q，`128 × 64` 行 |
| 2 | INPUT | BF16 | `[2160,512]` | 16 个 request 复用的 raw KV |
| 3 | INPUT | FP32 | `[128,128]` | raw validity mask |
| 4 | OUTPUT_EXISTING | FP32 | `[8192,1]` | softmax max state |
| 5 | OUTPUT_EXISTING | FP32 | `[8192,1]` | softmax sum state |
| 6 | OUTPUT_EXISTING | FP32 | `[8192,512]` | raw attention output |
| 7 | OUTPUT_EXISTING | FP32 | `[4194304]` | MIX TPipe 的 GM workspace |

这些文件可以看到 dtype、shape、stride、offset 和访问公式，但看不到本次真实 tensor 的数值内容。本次复制的 build output 没有附带 frozen input/golden 数据目录。如果需要查看具体输入值，需要保留 `--save-data` 产物或增加有边界的 tensor dump，不能从生成 C++ 反推出实际数值。

## 8. rank2 实际泳道结果

rank2 的 level-4 chip swimlane 中，task `12884901905` 的记录完整匹配 `deps.json`：

| lane | 记录数 | 使用物理核数 | 单条最短 | 单条 median | 单条 mean | 单条最长 | lane wall span |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AIC | 256 | 24 | 5.52 µs | 7.06 µs | 7.30 µs | 10.72 µs | 434.32 µs |
| AIV | 512 | 48 | 4.46 µs | 6.12 µs | 6.36 µs | 9.56 µs | 433.38 µs |

这里要区分：

- 单条 `duration` 是某个 block 在某条 AIC/AIV lane 上真正执行的时间；
- `434.32 µs` 是 256 个 AIC block 经多批滚动派发后的整体跨度；
- 不能把 256 个 AIC duration 简单相加当作 wall time；
- 也不能把 core busy time 的下降直接称为端到端性能提升。

## 9. 推荐的阅读顺序

排查一个生成 kernel 时，推荐按下面顺序阅读：

1. 原始 PyPTO：先理解算法、SPMD block 公式和 tile 含义；
2. `deps.json`：确认 task ID、block 数、资源形状和参数 shape；
3. `kernel_config.py`：把 kernel ID 映射到 AIC/AIV 函数；
4. `ptoas/*.pto`：查看编译器 IR、tile shape 和源代码位置；
5. `kernels/aic|aiv/*.cpp`：查看最终 PTO 指令、同步和 TPipe；
6. chip swimlane：确认这些 block 实际如何派发和运行；
7. in-core simulator：只有需要分析单 block 的 CUBE/VECTOR/MTE pipe 时再采集。

生成 C++ 最适合回答：

```text
一个 block 内部到底做了什么？
```

`deps.json` 最适合回答：

```text
一共有多少 task、每个 task 有多少 block？
```

chip swimlane 最适合回答：

```text
这些 block 实际在什么时间、什么物理核上运行？
```

## 10. 常用查看命令

```bash
HCA_BUILD=build_output/_jit_l3_decode_hca_20260902_013402
HCA_RANK_DIR="$HCA_BUILD/dfx_outputs/rank2/d0"
HCA_KERNEL_DIR="$HCA_BUILD/next_levels/decode_hca_test"
```

查看 `hca_raw_attn` 的任务属性：

```bash
jq '.tasks[] | select(.kernel_ids == [30,31,31]) |
    {task_id, early_dispatch, kernel_ids, block_num, args}' \
  "$HCA_RANK_DIR/deps.json"
```

查看 kernel ID 映射：

```bash
jq '.callable_id_to_name | {
      "30": .["30"],
      "31": .["31"]
    }' \
  "$HCA_RANK_DIR/name_map.json"
```

查看 AIC block identity 和入口：

```bash
rg -n 'get_block_idx|get_block_num|kernel_entry|hca_raw_attn_aic' \
  "$HCA_KERNEL_DIR/kernels/aic/hca_raw_attn_aic.cpp"
```

查看 AIC 的 Cube 指令：

```bash
rg -n 'TMATMUL|TMATMUL_ACC|TLOAD|TSTORE|TPUSH|TPOP' \
  "$HCA_KERNEL_DIR/kernels/aic/hca_raw_attn_aic.cpp"
```

查看 AIV 的 subblock 和 Vector 指令：

```bash
rg -n 'get_sub_block_id|subblock|TROWMAX|TROWSUM|TEXP|TPUSH|TPOP' \
  "$HCA_KERNEL_DIR/kernels/aiv/hca_raw_attn_aiv.cpp"
```

查看带原始变量名和源码位置的 PTO IR：

```bash
rg -n 'stream_idx|stream_state_row|stream_raw|matmul|row_max|row_sum|exp' \
  "$HCA_KERNEL_DIR/ptoas/hca_raw_attn.pto"
```

## 11. 明天对照 24-worker 版本时看什么

旧版本是：

```text
block_num = 256
physical rows = 256 × 3 = 768
```

如果 24-worker 改造按预期生效，新 build output 应看到：

```text
block_num = 24
physical rows = 24 × 3 = 72
```

同时检查生成 C++ 或 PTO IR：

- `block_idx` 的含义应从“token/head tile ID”变成“worker ID”；
- 每个 worker 内部应出现遍历多个 raw tile 的循环或其编译展开形式；
- 256 个 tile 应分配为 16 个 worker 各处理 11 个、8 个 worker 各处理 10 个；
- AIC/AIV 单条 duration 会明显变长，这是任务变粗后的正常现象；
- 关键指标是整体 wall span 是否下降，而不是单条 kernel 是否变长；
- `sync_start=False` 时，24 个 worker 仍可能由三个 scheduler thread 分批启动；
- 需要确认 compressed 分支的并行重叠没有因为 raw worker 长时间占据 MIX cluster 而恶化。

性能判断仍以统一 8K、相同 frozen golden、关闭 DFX、TP=4、相同轮数的最快-rank wall time 为准。生成 C++ 和泳道图用于解释结果，不能替代最终 benchmark。
