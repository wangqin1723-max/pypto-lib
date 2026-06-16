# DSv4 CSA decode UB 占用与性能空间分析

> 分析对象：`models/deepseek/v4/decode_attention_csa.py`
> 数据来源：`build_output/_jit_attention_csa_test_20260611_190043/`
> - 静态 UB/Cube 分配：`report/memory_after_AllocateMemoryAddr.txt`
> - 实测时间：`dfx_outputs/merged_swimlane_20260611_190110.json`（wall ≈ 2200µs）

## 核心结论（先看这个）

1. **UB 占用率本身不能判断瓶颈**，必须和 swimlane 实测时间对照。只看那份
   `memory_after_AllocateMemoryAddr.txt` 会得出错误结论。
2. **UB 占用率不是当前的性能瓶颈。** 占用最高的几个 scope（84%）全部在前段、
   已并行展开、不在关键路径——再 coarsen 既被 buffer 卡住、也不动 wall-clock。
   报告里的 84% 含义是“这块别再碰了”，不是“没空间提升”。
3. **还有提升空间，但杠杆在 `decode_sparse_attn.py`，不在 orchestrator 本身。**
   关键路径上的 `gather_kv` / `qk_pv` 任务过细、利用率极低，且 UB 有大量余量，
   做大 tile / 粗化分组在 buffer 上完全可行。
4. **`decode_attention_csa.py` 自身的 scope**（csa_rope_step / csa_cmp_rope /
   csa_sparse_idx_tile / csa_cache_writeback）都是 µs 级，调它榨不出东西。

## 一、整体时间结构（swimlane 实测，wall ≈ 2200µs）

| 阶段 | 时间窗 | 占 wall |
|------|--------|---------|
| 前段（hc_pre→attn_norm→qkv_proj_rope→compressor∥indexer） | 0 – ~800µs | ~36% |
| **sparse_attn 段**（gather_kv→qk_pv→merge_norm/rope→o_proj(proj_a/b)→hc_post） | ~800 – 2200µs | **~64%** |
| orchestrator 自有 scope（csa_rope_step / csa_cmp_rope / csa_sparse_idx_tile / csa_cache_writeback） | 散落 | 各 <5µs busy，可忽略 |

关键点：orchestrator 本身几乎不耗时，wall 全在被它组合的子核里，主体是 sparse_attn。

## 二、UB 占用 vs 关键路径——两者错位

高 UB 占用的 scope **全部在前段、已并行、不在关键路径**（buffer-capped，别再碰）：

| scope | UB | 在关键路径? |
|------|----|-----------|
| q_head_rms_nope | 84.1% | 否（前段） |
| qr_proj_aiv | 83.9% | 否 |
| mix_x | 75.5% | 否 |
| qr_hadamard_quant_aiv / score_aiv | 62% | 否 |
| merge_norm | 58.7% | 部分（sparse_attn 内，但非主瓶颈） |

真正在**关键路径（sparse_attn 段）的 scope，UB 全有余量**：

| scope | UB | busy / window / task 数 | 性质 |
|------|----|------------------------|------|
| gather_kv | **1.0%** | 32µs / ~300µs 窗 / 256 task | MTE2 gather，**dispatch/latency-bound** |
| qk_pv_aiv | **27.4%** | 49µs / ~400µs 窗 / 512 task | flash attn，利用率极低 |
| qk_pv_aic | Right **100%**(cube) | 24.9µs / 256 task | cube 侧 KV tile 满 |
| proj_a_aiv / proj_b_aiv | ~44% | o_proj 尾链 ~400µs | 有余量 |
| rope / hc_post | 35% / 2.6% | — | 有余量 |

`gather_kv` / `qk_pv` 的 busy 只占窗口约 11%：256~512 个 ~0.1µs 的小任务被依赖串起来，
正是 memory 里记的“per-task launch overhead 吃掉细粒度并行”（每任务 ~5–10µs OH）的形态。

## 三、Cube 侧补充

Cube（Mat/Left/Right/Acc）满载的 scope 也都在前段，非瓶颈：
- `kv_score_proj` / `kv_score_proj_0`：Left 100% / Right 100%
- `qproj_matmul`：Left/Right/Acc 全 100%
- 关键路径上 `qk_pv_aic` 的 Right 100% 是 KV tile，属于 sparse_attn，可随分块策略一起调。

## 四、下一步建议

- **不要**再去动前段高 UB scope（84%/75%）：被 buffer 卡死且不在关键路径。
- **要**去 `decode_sparse_attn.py`：
  - `gather_kv`：减少 gather 任务数（fewer-bigger DMA），UB 占用仅 1%，空间充裕。
  - `qk_pv`：加粗 KV 分块 / 调 flash 分组，aiv 侧 UB 仅 27%。
- 与历史一致：CSA 已 2898→2556（rope writeback + kv_rope spmd）；
  q_head_rope 的 matmul-reinterleave 路线因 pypto#1648 被 park。

## 复现方法

```python
# 从 merged swimlane 重建 per-scope 时间窗（去掉 (r1tNN) 后缀按名聚合）
import json, collections, re
d = json.load(open('dfx_outputs/merged_swimlane_<ts>.json'))
win = collections.defaultdict(lambda: [1e18, -1e18, 0.0, 0])
for e in d['traceEvents']:
    if e.get('ph') == 'X' and 'dur' in e:
        nm = re.sub(r'\(r\d+t\d+\)', '', e['name'])
        ts, dur = e['ts'], e['dur']      # ts/dur 单位为 ms
        w = win[nm]
        w[0] = min(w[0], ts); w[1] = max(w[1], ts + dur)
        w[2] += dur; w[3] += 1
# w = [min_start, max_end, busy_sum, ntask]；window = max_end - min_start
```
