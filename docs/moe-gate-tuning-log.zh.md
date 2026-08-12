# MoE Gate 核内性能调优记录

日期：2026-07-02

本文记录 DeepSeek V4 MoE EP1 路径中，围绕 `models/deepseek/v4/gate.py`
做过的核内性能分析、尝试、上板数据和最终结论。

## 目标

希望通过优化 `gate.py` 内部 kernel 性能，最终带来完整 MoE 的 AICore
端到端时间下降。

主要验证命令：

```bash
task-submit --device auto --max-time 1800 --run \
  "python models/deepseek/v4/moe.py -p a2a3 --ep 1 --enable-l2-swimlane > moe_l2_swimlane.log 2>&1" \
  2>&1 | tail -3
```

MoE 性能口径：

- 使用 `--enable-l2-swimlane` 生成的 `merged_swimlane*.json`
- 从第一个实际 AICore kernel event 开始，到最后一个实际 AICore kernel event 结束
- 不包含 `setup` event

## 前置背景

### exp_gate_mm / exp_up_mm

`exp_gate_mm` / `exp_up_mm` 侧看到的主要问题仍然是
`MOV_OUT_TO_L1_MULTI_ND2NZ` 阻塞。尝试过在 Python kernel 层表达 DN layout：

```python
create_l1([K_TILE, MM_INTER_TILE], transpose=True)
gather_row(..., transpose=True)
matmul(..., b_trans=False)
```

编译能过，也能看到 `layout<dn>` 和 `partition_tensor_view<512x1xi8>`，但
`x_next` mismatch 约 71%。判断是 Python 层 view/layout 表达没有真正让后端
dense load 走到更快的 DN2ZN/ZN-friendly lowering，运行时仍无法消除关键
ND2NZ 成本。

结论：这条线需要 PyPTO backend/codegen 支持 dense DN2ZN/ZN-friendly load，
不是只改 Python kernel 就能稳定解决。

### sh_gate_up

`sh_gate_up` 的核内性能也主要受 ND2NZ 搬运影响，而不是 MMAD 本身。相关 fused
尝试上板后没有形成可保留的正确性和性能收益，因此未保留改动。

## gate.py 热点

独立看 `gate.py`，热点不是 ND2NZ，而是：

- `x_norm_quant`
- `gate_aiv` / `gate_aic`
- `ffn_norm`

baseline PMU 里，hash/sort 两条路径的主要 cycles 如下：

```text
hash baseline:
x_norm_quant       84817
gate_aiv           74229
gate_aic           70826
ffn_norm           44694
route_hash         12724
gate_inactive_zero  1276
SUM_MAX           288566

sort baseline:
x_norm_quant       82436
gate_aiv           71395
gate_aic           67970
ffn_norm           44846
route_sort         16271
gate_inactive_zero  1013
SUM_MAX           283931
```

## 尝试过的 gate tile 参数

原始参数：

```python
D_TILE = 128
QUANT_TILE = 32
```

尝试过：

```python
QUANT_TILE = 128
D_TILE = 256
D_TILE = 512
D_TILE = 1024
D_TILE = 2048
```

`D_TILE=2048` 编译验证失败：

```text
Function 'ffn_norm': Vec buffer usage (200736 bytes) exceeds platform limit (188416 bytes)
```

独立 gate PMU 上，`D_TILE=1024, QUANT_TILE=128` 的局部收益最好：

```text
hash:
baseline q32/d128     288566
q128/d1024            199558   (-30.8%)

sort:
baseline q32/d128     283931
q128/d1024            202145   (-28.8%)
```

主要收益：

```text
x_norm_quant:
hash 84817 -> 30712
sort 82436 -> 30739

ffn_norm:
hash 44694 -> 18934
sort 44846 -> 19332
```

这说明 `gate.py` 局部核内优化是成立的。

## 放进 MoE 后的结果

### 单次结果

曾有一次 MoE swimlane 单测看到：

```text
优化版 AICore kernel-only: 832.66 us
改前 baseline:            874.94 us
```

单次看起来收益约：

```text
42.28 us, 约 4.8%
```

### 多次随机输入

随后按相同命令多跑，结果如下：

```text
优化版:
1) 861.32 us
2) 872.42 us
3) 830.00 us
mean = 854.58 us

改前 baseline:
1) 837.18 us
2) 855.54 us
3) 853.58 us
mean = 848.77 us
```

随机输入下，MoE 端到端没有稳定收益，均值反而慢约：

```text
5.81 us, 约 0.7%
```

### same-input A/B

固定同一份 `--golden-data` 做 A/B 时，能看到 gate 优化确实会把后续链路提前：

```text
baseline same-input: 924.74 us
gate opt same-input: 879.38 us
收益: 45.36 us
```

same-input 下，关键窗口也能看到 `ffn_norm/x_norm_quant` 明显提前：

```text
baseline:
ffn_norm      17.66 us
x_norm_quant  44.20 us

gate opt:
ffn_norm       8.04 us
x_norm_quant  15.52 us
```

但随机输入 repeat 时，expert 路由分布、expert kernel 数量和后续 combine 路径波动
会淹没这部分收益。

## 最终结论

`gate.py` 局部核内性能可以明显变好，但当前这版 tile 改动没有稳定转化为完整
MoE 的 AICore 端到端收益。

因此没有保留改动，`gate.py` 已恢复为：

```python
D_TILE = 128
QUANT_TILE = 32
```

最终判断：

- `D_TILE=1024, QUANT_TILE=128` 可作为局部 gate PMU 优化候选
- 但不能作为 MoE 端到端优化合入
- 后续所有 gate 优化都应先固定输入做 A/B，再用随机输入 repeat 复核

## 后续建议

如果继续通过 `gate.py` 推动 MoE 端到端收益，建议采用以下准入方式：

1. 固定 `--golden-data`，每个方案跑 3 到 5 次，先确认 AICore kernel-only 端到端稳定下降。
2. 再跑随机输入 repeat，确认收益没有被路由分布波动吞掉。
3. 不只看 `ffn_norm/x_norm_quant` 局部时间，还要看关键链：

```text
ffn_norm -> x_norm_quant -> gate_aic/aiv -> route_hash -> dispatch -> expert -> combine -> hc_post
```

更值得继续看的方向：

- `QUANT_TILE=128` 配更保守的 `D_TILE=512`，看是否比 `1024` 更稳
- 优化 `gate_aic/aiv` 和 `route_hash/dispatch` 的衔接，而不是只压 norm/quant
- 对 MoE 使用固定输入和固定路由分布的 benchmark，避免随机 expert count 干扰判断
