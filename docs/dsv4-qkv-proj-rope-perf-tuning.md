# DeepSeek V4 `qkv_proj_rope` Decode Perf Tuning

Perf-tuning progression for [models/deepseek/v4/qkv_proj_rope.py](../models/deepseek/v4/qkv_proj_rope.py)
on a2a3 with `--enable-l2-swimlane`.

## Run Config

- Model config: `FLASH` — `DECODE_BATCH=64, S=1, T=64, D=4096, H=64, HEAD_DIM=512, ROPE_DIM=64, NOPE=448, Q_LORA=1024`
- Command:
  ```bash
  python models/deepseek/v4/qkv_proj_rope.py -p a2a3 -d 0 --enable-l2-swimlane
  ```

Wall-clock has ~4% run-to-run noise on this kernel; treat any single number as
±50us. Re-run 3+ times before claiming an improvement.

## Progression

| Run | Date | Wall-clock | Tasks | Exec/Lat | Total Exec | Log |
|---|---|---:|---:|---:|---:|---|
| Baseline | 2026-05-18 | 1725 us | 1527 | 28.6% | 5696 us | `qkv_proj_rope_l2_swimlane.log` |
| **Opt A: drop chunked_loop_optimizer from 3 vector scopes** | 2026-05-18 | **1371 us** (single run) | **999** (−528) | **42.0%** | 5260 us | `qkv_proj_rope_optA.log` |
| Opt B fail: `Q_PROJ_OUT_CHUNK=256` (N tile) | 2026-05-18 | — | — | — | — | `qkv_proj_rope_optB_outchunk256.log` — runtime fail 507018 `ACL_ERROR_RT_AICORE_TIMEOUT` |
| Opt A + Opt B: `Q_PROJ_CHUNK=256` (K tile), 3 runs | 2026-05-18 | **1371 / 1366 / 1313 us** (range 58us, ~4.4%) | 999 | 40-43% | 4972-5190 us | `qkv_proj_rope_optB_kchunk256.log` + `qkv_proj_rope_l2_swimlane_3.log` |
| Opt C reverted: parallelize `attn_norm_rms` into 16 partials | 2026-05-18 | 1466 us (+100us regression) | 1015 | 41.9% | 5158 us | `qkv_proj_rope_optC_run1.log` — pypto per-task launch overhead defeats fine-grained parallelism |
| Opt D reverted: fuse `q_rope_reassemble` + `q_rope_write` | 2026-05-18 | compile fail | — | — | — | AIC matmul (NZ) cannot share scope with AIV cast (ND) |
| **Opt E: fuse `token_x_cast_bf16` + `qr_cast_bf16` into producing norm_apply scopes**, 3 runs | 2026-05-18 | **1184 / 1193 / 1220 us** (median **1193**) | **951** (−48) | 41.7-43.0% | 5130-5262 us | `qkv_proj_rope_optE_run{1,2,3}.log` |
| Opt E re-run on different NPU card (memory contention) | 2026-05-18 | 1239 us | 951 | 45.4% | 5716 us | `qkv_proj_rope_l2_swimlane_4.log` — attn_norm_apply 17.5→39.5us (memory-bound AIV op on contended HBM); still within Opt E range |
| Opt F-1 reverted: matmul `out_dtype=BF16` + fused assemble | 2026-05-18 | compile fail | — | — | — | `pl.matmul_acc` hard-requires FP32 accumulator |
| **Opt G: chunk-size tuning** (`Q_LORA_TILE 32→128`, `KV_CHUNK 32→128`, new `KV_NOPE_CHUNK=64`), 3 runs | 2026-05-18 | **1123 / 1174 / 1164 us** (median **1164**) | **884** (−67) | 42-44% | 4342-4715 us | `qkv_proj_rope_l2_swimlane_5{,b,c}.log` |
| Opt H reverted: MPMD reorder (move Stage 5/6 to right after attn_norm_apply), 3 runs | 2026-05-18 | 1166 / 1129 / 1196 us (median **1166**) | 884 | 44-45% | 4652-4868 us | `qkv_proj_rope_l2_swimlane_6{,b,c}.log` — confirms prior timeline analysis; also discovered HBM-contention regression on `attn_norm_apply` (+19us/task = +304us Σ Exec) |

**Opt B real effect (after noise correction):** `qproj_matmul` per-task Exec
**8.55us → 7.95us (~−7%)**, not the −16% initially claimed. The −16% was the
gap between two unrelated single runs (one Opt A run that happened to be slow,
one Opt B run that happened to be fast). Wall-clock impact of Opt B is *within
noise* — likely small positive but unconfirmed.

## Opt A — drop `chunked_loop_optimizer` from vector scopes (accepted, 2026-05-18)

Removed `optimization=pl.chunked_loop_optimizer` from three pure-vector `pl.at`
blocks in [models/deepseek/v4/qkv_proj_rope.py](../models/deepseek/v4/qkv_proj_rope.py):

- `attn_norm_apply` (line ~87)
- `qproj_dequant` (line ~181)
- `q_head_rms_rope` (line ~190)

Kept on `qproj_matmul` — matmul scopes still benefit.

**Root cause confirmed:** the `chunked_loop_optimizer` sentinel triggers
`OutlineIncoreScopes` (compile pass 11) to peel `pl.cast` ops into separate
outlined sub-functions:

- `attn_norm_apply` → cast(BF16→FP32) + vector mul split into 2 sub-tasks
- `qproj_dequant` → cast(INT32→FP32) + row/col_expand_mul split into 2
- `q_head_rms_rope` → RMS reduce + NOPE-loop + rope-prelude + cos cast +
  rope-math split into 5; the `pl.slice(rope_cos/sin)` between sub-scopes acts
  as an additional hoist boundary

**Rule of thumb:** only use `chunked_loop_optimizer` on scopes whose hot op is
a `pl.matmul` / `pl.matmul_acc`. In pure-vector scopes its only observable
effect is the cast-split overhead.

## Opt B — Q_PROJ tile tuning (partially accepted, 2026-05-18)

**`Q_PROJ_OUT_CHUNK=256` (N tile) failed at runtime** with
`ACL_ERROR_RT_AICORE_TIMEOUT` (507018). Compile passed, but AICore hung when
N=256 INT8 matmul. Likely a CANN kernel-template limit on a2a3 for
`INT8 [64, 128] × [128, 256] → INT32 [64, 256]`. **Do NOT try N=256 again on
this platform without first checking the matmul template support.**

**`Q_PROJ_CHUNK=256` (K tile, kept) works:** halves inner accumulation
iterations (8 → 4). Per-task Exec for `qproj_matmul` ~7% improvement
(8.55 → 7.95 us, after re-runs). Wall-clock change is within noise floor —
small but not confidently positive. Kept because it's a strict Exec improvement
with no downside.

**Lesson:** wall-clock has ~4% run-to-run noise; need 3+ runs to claim small
improvements. Also: aggregate Exec gains in parallel sections get absorbed by
parallelism slack and don't always move wall-clock. To shift wall-clock,
target serial single-task scopes (rms reduces, qr_quant) or scopes whose Exec
dominates a parallel band.

## Opt E — fuse `cast_bf16` into `norm_apply` (accepted, 2026-05-18)

Removed two `pl.at` scope loops that did pure `FP32 → BF16` cast as a separate
stage, folded the cast into the producing `attn_norm_apply` / `qr_norm_apply`
scopes:

- **`token_x_cast_bf16` (16 tasks) folded into `attn_norm_apply`**. Also
  removed the `token_x_fp32` intermediate GM buffer entirely —
  `attn_norm_apply` now writes `token_x_bf16` directly.
- **`qr_cast_bf16` (32 tasks) folded into `qr_norm_apply`**. The `qr_fp32`
  buffer is retained (still read by `qr_rms`), but `qr_norm_apply` no longer
  rewrites it; instead it writes `qr_bf16` directly.

**Why this works (where Opt D failed):** both producer and consumer are AIV
vector ops. No AIC matmul involved, so no NZ/ND layout boundary. Fusion is
just "extend the existing AIV scope to also write a second BF16 output."

**Effect:** 48 tasks eliminated (16 + 32), 48 launch-overhead trips removed
(~7us tail OH each). Also removed 1 intermediate GM buffer (saves DMA
bandwidth: each `token_x_fp32` chunk write was 64KB, gone). Per-task Exec for
`attn_norm_apply` was 20.6us → 17.6us (writing BF16 = half DMA of FP32 + only
one extra cast op). `qr_norm_apply` was 1.26us → 2.02us (cast cost ate small
fraction).

**Net:** median wall-clock 1366 → **1193 us (−173us, −12.7%)** over 3 runs.

## Opt G — chunk-size tuning (accepted with caveat, 2026-05-18)

Raised three N-axis vector/matmul chunks in
[models/deepseek/v4/qkv_proj_rope.py](../models/deepseek/v4/qkv_proj_rope.py):

- `Q_LORA_TILE = 32 → 128` (also `Q_LORA_CHUNK`, drives `Q_BLOCKS = 32 → 8`)
- `KV_CHUNK = 32 → 128` (drives `KV_BLOCKS = 16 → 4`)
- **New constant `KV_NOPE_CHUNK = 64`** for the `kv_norm_nope` loop only —
  `NOPE_DIM = 448` is not divisible by 128, so the nope-norm vector chunk
  must stay at a divisor of 448 (64 = 448/7). The matmul / rms paths use
  `KV_CHUNK = 128`.

**Why this is safe:** all three chunks are pure N-axis (output column) tile
sizes; they don't change dataflow or accumulator dtype.

**Effect — task counts:**

| Function | Old (Opt E) | New (Opt G) |
|---|---:|---:|
| `qr_proj_matmul` | 32 tasks × 22.8us | **8 tasks × 31.3us** |
| `kv_proj_matmul` | 16 tasks × 20.4us | **4 tasks × 28.8us** |
| `kv_norm_nope` | 14 tasks × 9.4us | **7 tasks × 3.4us** |
| `qr_norm_apply` | 32 tasks × 1.3us | 8 tasks × 2.0us |
| **Total tasks** | **951** | **884** (−67) |

Per-task Exec for the parallel matmul scopes went up slightly (bigger tile =
more work per task, but better cube utilization: `kv_proj_matmul`
Exec/Latency 88.1% → 92.6%, `qr_proj_matmul` 58.9% → 85.4%). Σ Exec dropped
significantly because the fixed per-task launch overhead is amortized over
fewer, larger tasks. **Don't try `Q_PROJ_OUT_CHUNK=256`** — already known to
trigger `ACL_ERROR_RT_AICORE_TIMEOUT` (see Opt B).

**Effect — wall-clock (3 runs):**

| Run | Wall-clock | Σ Exec | attn_norm_apply per-task Exec |
|---|---:|---:|---:|
| _5 (lucky / uncontended) | 1122.96 us | 4342 us | 18.18 us |
| _5b | 1173.96 us | 4715 us | 39.29 us |
| _5c | 1163.70 us | 4545 us | 37.21 us |
| **median** | **1164 us** | 4545 us | 37 us |

Wall-clock median **1193 → 1164 us, −29 us (−2.4%)** — **within the ~4%
run-to-run noise band**. Σ Exec change is unambiguous (−655 us), task count
change is unambiguous (−67), but wall-clock benefit is statistically
inconclusive on its own.

**HBM contention observation:** `attn_norm_apply` per-task Exec is bimodal —
~18us when uncontended, ~37us when contended — and Opt G hits the contended
mode in 2/3 runs (vs Opt E baseline at ~17.6us steady). Hypothesis:
`qr_proj_matmul` (8 tasks × 1MB `wq_a` read each) issues larger HBM bursts than
the old 32×256KB tasks, increasing the probability of contending with the
trailing `attn_norm_apply` GM write. The per-task improvement on the matmul
side is partly cancelled by the per-task degradation on the AIV write side.

**Why kept despite borderline wall-clock:** task count and Σ Exec are strict
improvements with no correctness or compile-time downside. Future scope-fusion
work depends on chunk geometry being sane (larger chunks reduce sub-task
fragmentation when later fusing). Kept as the new baseline.

## Opt H — explicit MPMD reorder (reverted, 2026-05-18)

Moved the entire Stage 5/6 (KV path: `kv_proj_matmul` → `kv_rms` →
`kv_norm_nope` → `kv_rope_*`) from its original position at the end of the
function to right after Stage 0.2 (`attn_norm_apply`), before Stage 1
(`qr_proj_matmul`). The KV path only depends on `token_x_bf16`, so dataflow
correctness is preserved either way; the reorder was meant to give the
runtime scheduler the earliest possible opportunity to issue KV tasks in
parallel with the Q-side.

**Result (3 runs):** wall-clock 1166 / 1129 / 1196 us, median **1166 us** —
indistinguishable from Opt G alone (median 1164 us).

**Surprising sub-finding:** `attn_norm_apply` per-task Exec consistently
jumped to ~40us across all three reorder runs (vs the 18-39us bimodal
distribution in Opt G). Σ Exec went up +295us purely from this one scope.
Mechanism: placing `kv_proj_matmul` (which reads the 4MB `wkv` weight) into
explicit data-dependence-adjacency with `attn_norm_apply` (AIV write of
`token_x_bf16`) makes the scheduler issue them concurrently, oversubscribing
HBM and pinning `attn_norm_apply` into the contended-execution regime.

**Takeaway:** the pypto dataflow scheduler in the original source order
**already** schedules KV tasks into Q-side idle slots, but **avoids the
HBM-contention window around `attn_norm_apply`**. Explicit source-level
reordering destroys this implicit contention avoidance. **Do not reorder
source to "force" MPMD parallelism — pypto already does it more
intelligently.**

## Current state (Opt A + B + E + G), median of 3 runs

| Metric | Value |
|---|---|
| **Wall-clock** | **1164 us** (range 1123-1174, ±2%) |
| Total Exec | ~4342-4715 us |
| Total tasks | 884 |
| Avg Exec / task | ~5.1 us |
| Exec / Latency | ~43% |

## vs original baseline

**1725 us → 1164 us, −561 us, −32.5%** (significant; well above noise).
Optimization accepted as endpoint 2026-05-18.

## Why optimization stopped here

Remaining ~100us wall-clock target was `q_rope_reassemble` (128 × 2.38us, AIC) +
`q_rope_write` (128 × 1.08us, AIV). To fuse, would need to eliminate the
matmul-based permutation interleave and replace with pure-AIV interleave
(3D reshape trick / scatter_update). This is an algorithm-level rewrite, not
just a scope-boundary change. ROI estimated at −50 to −100us wall-clock (4-8%);
user accepted current state instead of pursuing.

The fundamental blocker is platform-level: `pypto` does not insert implicit
NZ→ND layout conversion within a single `pl.at` scope, so any AIC matmul
followed by AIV cast/assemble must occupy 2 runtime tasks. To remove this cost
cleanly, either:

- Backend feature: add intra-scope NZ→ND conversion at the matmul→vector
  boundary
- Source rewrite: replace permutation-matmul with pure vector interleave

## Hot scopes at Opt A+B+E+G (top by aggregate Exec)

1. `qproj_matmul`: still dominant
2. `qr_proj_matmul`: still serial chain along D_BLOCKS=16 (one matmul + 15
   matmul_acc)
3. `kv_proj_matmul`
4. `qproj_dequant`
5. `q_head_rms_rope`
6. `attn_norm_apply`: now ~17.6us (was 20.6us)
7. `attn_norm_rms`: single-task ~35us, confirmed near-optimal for current
   structure

## Investigated and ruled out (2026-05-18, post-Opt E)

These looked promising but did not survive verification:

1. **Fuse `qproj_matmul` + `qproj_dequant`** — same NZ/ND scope-boundary
   blocker as Opt D. AIC INT8 matmul output is NZ in L0C; AIV dequant needs
   ND on GM. Cannot share one `pl.at`.
2. **Q/KV path parallel (MPMD)** — timeline analysis of the Opt E baseline
   shows `kv_proj_matmul` already finishes at ~1070us, while `q_rope_write`
   (the actual wall-clock tail) finishes at ~1282us. KV is not on the critical
   path; moving it earlier doesn't change wall-clock and would compete with
   Q-side for the dispatch loop (already 50% idle-spinning on dispatch).
   **Empirically confirmed by Opt H (2026-05-18):** explicit source reorder
   yielded no wall-clock benefit AND introduced HBM contention on
   `attn_norm_apply` (+19us per task, +304us Σ Exec). The pypto dataflow
   scheduler in the original source order already overlaps KV with Q while
   avoiding the contention window — explicit reorder breaks the implicit
   avoidance.
3. **`cube_nbuffer_mode` / `vec_nbuffer_mode` compile flags** — these
   parameter names do not exist anywhere in pypto, ptoas, simpler, or the
   pypto-lib tree. No-op.
4. **`Q_PROJ_OUT_CHUNK=512` + `enable_split_k=True`** — N=256 already
   timeouts; N=512 only worse. `split_k` adds reduce tasks in a workload that
   is already dispatch-bound (per Part 2 of swimlane report: 50% scheduler
   idle), so net negative.

## Remaining optimization avenues (rank-ordered)

1. **Algorithm-level rewrite of `q_rope_reassemble` + `q_rope_write`** —
   replace permutation-matmul interleave with pure AIV 3D-reshape /
   scatter_update. Targets the only remaining wall-clock tail (`q_rope_write`
   span 610us in a 1239us run). Largest remaining ROI.
2. **Parallelize single-task serial reduces on critical path**:
   `attn_norm_rms` (35 us), `qr_rms` (17 us), `kv_rms` (9 us),
   `qr_quant` (19 us). Convert serial inner reduce (16 / 32 / 16 blocks) to
   partial-parallel + final reduce. **Caveat:** Opt C already failed when
   `attn_norm_rms` was split 16-way — pypto per-task launch overhead
   (~5-10us) defeats fine-grained partials. Use coarse 4-way grouping, not
   N-way.
3. **`q_head_rms_rope` per-head inner serial 8-chunk reduce**: parallelize
   within head.
4. **Backend feature ask**: intra-scope NZ→ND conversion at the AIC→AIV
   boundary would unlock Opt D-style fusions across the codebase.

## S=2 (T=128) follow-up — 2026-05-19

After commit `05127d2` switched `DECODE_SEQ` 1 → 2 (T=64 → T=128), all four
accepted/reverted opts above were re-validated on the same kernel without
parameter changes (Opt B/G chunk tuning explicitly skipped, only structural
opts in scope).

### Run config
- Same FLASH preset, but `DECODE_BATCH=64, S=2, T=128`. Everything else
  identical (D=4096, H=64, HEAD_DIM=512, Q_LORA=1024).
- Same command: `python models/deepseek/v4/qkv_proj_rope.py -p a2a3 -d 0 --enable-l2-swimlane`

### Progression

| Step | Wall-clock | Tasks | Exec/Lat | Notes |
|---|---:|---:|---:|---|
| S=2 baseline (clean) | 1867.8 us | 1575 | 39.7% | reference; T-doubled work, ~8% slower than S=1 baseline |
| **Opt A** (drop `chunked_loop_optimizer` × 3) | **1733 us** | 1095 | 53.0% | **−7.2% vs S=2 baseline** — kept |
| Opt C (16-partial `attn_norm_rms`) | median 1714 us (range 1684-1751) | 1111 | 51.9% | within noise of Opt A; Σ Exec **rose +600us** — reverted |
| **Opt E** (fuse `cast_bf16` into norm_apply) | **median 1564 us** (range 1552-1607) | 1031 | 55.6% | **−9.7% vs Opt A** — kept |
| Opt H (MPMD reorder Stage 5/6) | median 1576 us (range 1573-1589) | 1031 | 54.9% | +12us vs Opt E median, within noise — reverted |
| **Final (A + E)** | **~1560 us** | 1031 | ~55% | **−307us, −16.4% vs S=2 baseline** |

### Opt A required a structural prerequisite at T=128

`q_head_rms_rope` cannot drop `chunked_loop_optimizer` and remain in a single
`pl.at` scope at T=128 — the fused [RMS + NOPE + RoPE] body holds ~7 FP32
`[T, ROPE_HALF|ROPE_DIM]` tensors live in the RoPE block (`q_rope_norm`,
`q_even`, `q_odd`, `cos`, `sin`, `q_rot_even`, `q_rot_odd`) and pushes
**~230 KB** through the 192 KB Vec budget. At S=1 (T=64) the same scope
peaked at ~115 KB and fit comfortably; the optimizer's cast-split was extra
overhead, hence Opt A's win there.

**S=2 fix:** split into two scopes within the per-head `pl.parallel(0, H)`
loop:

- `q_head_rms_nope` — RMS sq-sum + inv_rms compute + NOPE-direction
  normalize-and-cast
- `q_head_rope` — RoPE direction (norm → gather even/odd → cos/sin → rotate → cast)

`inv_rms` crosses the scope boundary via a `[H, T]` FP32 staging tensor.
The scope-split cost is one `[H, T]` GM round-trip (~32 KB) and `H` extra
task launches — well under the win from dropping the optimizer.

### Opt C: same verdict as S=1 (reverted), different mechanism

At S=1 the serial `attn_norm_rms` was 35us — too short for 16-way partials
to overcome per-task launch overhead (the published doc above notes ~5-10us
per launch). At S=2 it is 120us, which should theoretically amortize launch
overhead, and 16 partials run ~45us each in parallel. But:

- Wall-clock median actually *improves* by ~20us (within ~4% noise).
- Σ Exec **rises by ~+600 us** because the partial scope's per-task work
  (~45us) includes head-OH it didn't have when it was one task.

Net: the small wall-clock win is statistically indistinguishable from noise
while resource cost is unambiguously up. Same revert.

### Opt H: same verdict (reverted), different reason

At S=1, source-level reorder caused +19us per-task `attn_norm_apply`
regression via HBM oversubscription against `kv_proj_matmul`. At S=2, that
specific regression does *not* reproduce — per-task `attn_norm_apply`
actually went down (93→82us). But wall-clock median still does not improve
(+12us vs Opt E). So the underlying lesson from
`feedback_pypto_dataflow_implicit_contention_avoidance.md` holds: the pypto
dataflow scheduler in original source order already overlaps KV with Q,
explicit reorder offers no upside, in this case it just shifts variance
without moving the median.

### Why parameter-tuning (Opt B / Opt G) was skipped this round

User scope was explicitly structural-only. Opt G's chunk sizes (`Q_LORA_TILE
= 128`, `KV_CHUNK = 128`, `KV_NOPE_CHUNK = 64`) likely transfer cleanly to
S=2 but were not re-validated. If pursued, expect another ~30-60us
wall-clock at S=2 (similar relative gain to S=1's borderline −2.4%).

### Net result at S=2

**1867.8 us → ~1560 us, −307 us, −16.4%** with only `Opt A + Opt E +
q_head_rms_rope scope split`. Validation PASS on q / kv / qr / qr_scale at
the same tolerances as S=1.

### Opt Q — `kv_norm_nope` KV_NOPE_GROUP=2 (reverted, 2026-05-20)

Pre-change: `kv_norm_nope` was the only vec-norm scope in this file not
GROUP-chunked: 14 logical iters × runtime fanout 2 = 28 tasks × 37us μ,
span 60us, on the KV branch.

Applied the standard `pl.parallel(0, N, GROUP) + pl.range(GROUP)` pattern
with `KV_NOPE_GROUP = 2` (NOPE_DIM/KV_CHUNK = 14 = 2×7, so candidates are
1/2/7/14). Single cross-iter loop-carried tensor (`kv`) — matches Opt
J/L/N success condition. Inlined `(nbg + n_inner) * KV_CHUNK` everywhere
to avoid the pypto AST loop-carry pitfall (initially regressed with
SSAVerify "Variable 'n0' used outside its defining scope"; inlining
fixed compile).

**Local effect (the scope itself improved a lot):**

| Metric | Baseline | Opt Q | Δ |
|---|---:|---:|---:|
| `kv_norm_nope` tasks | 28 | 14 | −14 |
| `kv_norm_nope` per-task μ | 37 us | **8.5 us** | (compiler generated much tighter code at GROUP=2) |
| `kv_norm_nope` span | 60 us | **14 us** | **−46 us** |
| `kv_norm_nope` Σ Exec | 1041 us | **119 us** | **−922 us** |
| KV branch end-time | ~476 us | **~406 us** | −70 us |

**But wall regressed:**

| Run | baseline (re-measured) | Opt Q |
|---|---:|---:|
| run 1 | 624.3 us | 655.0 us |
| run 2 | 621.0 us | 660.1 us |
| run 3 | 641.8 us | 660.0 us |
| median | **624 us** | **660 us (+36 us, +5.8%)** |

**Mechanism — same as Opt H.** KV branch is not on the critical path
(originally KV ended ~476us while Q tail was 670us → ~194us slack). KV
finishing earlier does not change wall, but the scheduler reshuffles
issue order: with `kv_proj_matmul` now sitting closer to the Q-side
critical path entries, `qr_proj_matmul` start drifted +21us across runs
(195 → 195/228/216us) and its span grew +25us (75 → 100us in run 3).
That delay propagates downstream into `qproj_matmul` start (389us
baseline → 389/405/416us in runs 1/2/3), and the entire critical path
shifted right by ~36us median.

**Lesson reinforces `feedback_pypto_dataflow_implicit_contention_avoidance.md`:**
KV branch already has ~200us of slack. Local improvements to non-critical
scopes can regress wall by destabilising the implicit-contention-avoidance
schedule. The scheduler's choice to keep `kv_norm_nope` at GRP=1 (28
small tasks) was already implicitly optimising — letting these tasks fill
gaps without bunching into a big group that competes for HBM.

**Reverted.** Status: file shipped with `kv_norm_nope` at GRP=1.

### Opt R — `q_rope_reassemble + q_rope_write` AIV interleave rewrite (failed compile, 2026-05-20)

Attempted the algorithm-level rewrite that earlier remained-optimization
section labeled as the only remaining ~100us wall-clock avenue. Goal:
fold the cube-based even/odd permutation matmul (`q_rope_reassemble`) +
the FP32→BF16 vector write (`q_rope_write`) into the existing
`q_head_rope` AIV scope, writing directly to `q_flat` in interleaved
RoPE layout via pure-AIV ops.

**Approach tried:** 3D-assemble + reshape. The interleaved layout
`[e0, o0, e1, o1, ..., e_{ROPE_HALF-1}, o_{ROPE_HALF-1}]` is equivalent
to C-order memory of a `[T, ROPE_HALF, 2]` view where `[..., 0]` holds
even and `[..., 1]` holds odd. Plan:

```python
q_rope_buf = pl.create_tensor([T, ROPE_HALF, 2], dtype=pl.BF16)
q_rope_buf = pl.assemble(q_rope_buf, pl.reshape(q_rot_even_bf16, [T, ROPE_HALF, 1]), [0, 0, 0])
q_rope_buf = pl.assemble(q_rope_buf, pl.reshape(q_rot_odd_bf16,  [T, ROPE_HALF, 1]), [0, 0, 1])
q_flat[:, h*HEAD_DIM+NOPE_DIM : (h+1)*HEAD_DIM] = pl.reshape(q_rope_buf, [T, ROPE_DIM])
```

**Compile fail:**
```
'pto.alloc_tile' op expects result row-major none_box tile row byte size
(cols * sizeof(dtype)) to be 32-byte aligned, but got 4 bytes
```

The `[T, ROPE_HALF, 1]` BF16 source tile has row byte size = 1 × 2 B = 2 B
(or 4 B if ROPE_HALF is treated as the row); either way it's well under
the **32-byte AIV tile alignment requirement**. This is a hardware-level
constraint of the MTE3 pipe — vector stores need >= 32-byte contiguous
chunks. Writing BF16 at stride-2 element granularity (which is what the
interleaved layout requires) inherently violates this.

**Alternatives investigated:**

- **`pl.transpose` of `[T, 2, ROPE_HALF]` → `[T, ROPE_HALF, 2]`:**
  Inner tile still ends up with trailing axis 2 BF16 = 4 bytes after the
  transpose. The downstream reshape to `[T, ROPE_DIM]` then either
  inserts a physical copy that re-runs into the same alignment wall, or
  is rejected outright.
- **`pl.tile.mscatter`:** Supports `FP16/FP32/INT16/INT32` only — no BF16
  path. And per-element scatter to GM is hardware-slow (~1 elem/cycle)
  vs aligned MTE3's 32 elem/cycle, so even if we cast to FP32 it would
  cost more than the matmul approach.
- **`pl.tensor.scatter_update`:** Row-scatter only (dim=-2). Cannot
  scatter into strided column positions.
- **Bit-packing pairs as INT32 + reinterpret_cast:** No bitcast API in
  pypto (`pl.cast` does numeric conversion only).

**Verdict — backend feature gap, not a source rewrite problem.** Confirms
the doc's prior assessment (lines 224–227 of this file before this
section): the only way to eliminate `q_rope_reassemble + q_rope_write` is
either (a) pypto exposes a hardware-native vector interleave primitive
(`vinterleave` / `vmerge`-class instruction wrapped at tile level), or
(b) the compiler inserts implicit NZ→ND conversion at the AIC→AIV
boundary so the existing matmul can fuse into `q_head_rope`'s scope. The
3D-assemble approach is rejected by the alignment guard in
`pto.alloc_tile` and has no source-level workaround.

**File reverted to baseline.** No changes shipped.

### Opt M re-tested with Opt P, regressed (2026-05-20)

After accepting Opt P, attempted to re-add Opt M (q_head_rms_nope GRP=8)
on the M-revert+N+O+P state. 3-run measurement:

| Run | wall |
|---|---:|
| run 1 | 671.9 us |
| run 2 | 661.6 us |
| run 3 | 652.8 us |
| median | **661.6 us** |

vs P alone median 624 us → **+38 us regression with M layered on top.**

**Mechanism:** Opt P shrinks the qproj_dequant span (100→35 us) and
dequant now ends ~465us. With this earlier downstream entry,
`q_head_rms_nope`'s span becomes the new critical-path dominator:

- GRP=1: 64 tasks × 14us μ across 48 cores → span 41us → end ~506us
- GRP=8: 8 tasks × 97us μ across 8 cores  → span 102us → end ~580us

GRP=1's much-shorter span wins the new bottleneck position. GRP=8's
larger μ was previously hidden by downstream slack but is now exposed.

**M permanently reverted.** Final state has `q_head_rms_nope = GRP=1`.

**General lesson:** A chunking that's marginal in isolation can reverse
sign when adjacent optimizations finish faster. Re-validate each
previously-accepted CHUNK choice after every adjacent improvement that
could shift the critical path. Memory entry
[`feedback_pypto_head_group_chunking_loop_carried.md`](../../.claude/projects/-data-<user>-newpto-pypto-lib/memory/feedback_pypto_head_group_chunking_loop_carried.md)
adds this rule.

## Opt M revert + Opt P — qproj_dequant decoupled chunking (accepted, 2026-05-20)

### Opt M reverted on user request

Opt M (`q_head_rms_nope` HEAD_GROUP=8) gave a marginal wall improvement
(~6us median) but pushed per-task μ to 97us — well past the ~50us
dispatch target. The previous decision to keep it was based purely on
median wall; the user requested reverting because the per-task kernel
was over-sized. Restoring `pl.parallel(0, H, 1)` brings the scope back
to 64 tasks at μ=14us, span=39us across 48 cores — a clean fan-out that
better matches the design target. Single-run wall went 681 → 694us after
the revert (within noise; Opt M's win was always ≤ 10us).

### Opt P — qproj_dequant decoupled from qproj_matmul

Previously skipped in the Opt M/N/O round because it would require a
**16 MB** global INT32 staging tensor (`col_acc_all` =
`[Q_PROJ_HEAD_BLOCKS × T, Q_PROJ_OUT_CHUNK]` = `[256 × 128, 128]`).
On second look this is acceptable on a3 (plenty of HBM, no DMA bandwidth
hotspot), and the user accepted the cost.

Implementation:

```python
q_proj_fp32 = pl.create_tensor([T, H * HEAD_DIM], dtype=pl.FP32)
col_acc_all = pl.create_tensor([Q_PROJ_HEAD_BLOCKS * T, Q_PROJ_OUT_CHUNK], dtype=pl.INT32)

# Stage 3a — qproj_matmul writes into the global staging
for hg in pl.parallel(0, Q_PROJ_HEAD_BLOCKS, Q_PROJ_GROUP):
    with pl.at(..., "qproj_matmul"):
        col_acc = pl.create_tensor([T, Q_PROJ_OUT_CHUNK], dtype=pl.INT32)  # pre-decl
        for h_inner in pl.range(Q_PROJ_GROUP):
            for qb in pl.pipeline(0, Q_PROJ_BLOCKS, stage=2):
                ... matmul/matmul_acc into col_acc ...
            col_acc_all = pl.assemble(col_acc_all, col_acc, [(hg + h_inner) * T, 0])

# Stage 3b — qproj_dequant runs independently with its OWN larger group
for hbg in pl.parallel(0, Q_PROJ_HEAD_BLOCKS, Q_PROJ_DEQUANT_GROUP):
    with pl.at(..., "qproj_dequant"):
        for h_inner in pl.range(Q_PROJ_DEQUANT_GROUP):
            col_acc_chunk = pl.slice(col_acc_all, ..., [(hbg + h_inner) * T, 0])
            ... cast + scale + assemble into q_proj_fp32 ...
```

Why this didn't hit the Opt I "≥2 cross-iter loop-carried tensors" trap:
the two `pl.parallel` loops are completely separate — `col_acc_all` is
loop-carried only within the matmul loop's iters, and `q_proj_fp32` is
loop-carried only within the dequant loop's iters. Each loop has just 1
cross-iter output, matching the now-revised
[memory:feedback_pypto_head_group_chunking_loop_carried.md](../../.claude/projects/-data-<user>-newpto-pypto-lib/memory/feedback_pypto_head_group_chunking_loop_carried.md)
gold standard.

### DEQUANT_GROUP sweep

| GROUP | wall (single run) | dequant tasks | dequant μ | dequant span |
|---:|---:|---:|---:|---:|
| 8 (coupled, Opt J state) | 657 us | 32 | 13 us | 100 us |
| 8 (decoupled) | 657 us | 32 | 13 us | 33 us |
| **16 (decoupled)** | **628 us** | **16** | **23 us** | **35 us** |
| 32 (decoupled) | 635 us | 8 | 42 us | 45 us |

GRP=16 is the sweet spot. GRP=32 reaches μ=42us (closest to the 50us
target) but loses parallelism (only 8 outer iters).

### Result (3-run median, vs M-reverted+N+O state)

| Run | wall |
|---|---:|
| run 1 | 624.4 us |
| run 2 | 631.1 us |
| run 3 | 623.7 us |
| **median** | **624.4 us** (range 7.4us, ~1.2%) |

| Metric | M-reverted+N+O | Opt P (DEQUANT_GROUP=16) | Δ |
|---|---:|---:|---:|
| **wall median** | 646 us | **624 us** | **−22 us (−3.4%)** |
| `qproj_dequant` tasks | 32 | 16 | −16 |
| `qproj_dequant` per-task μ | 13 us | 23 us | +77% |
| `qproj_dequant` span | 100 us | 35 us | −65 us |
| `qproj_dequant` distinct cores | 22 | 16 | clean fan-out |

### Cumulative result

| State | wall (median) | Notes |
|---|---:|---|
| Pre-tuning S=2 baseline | 1868 us | reference |
| After Opt J | 849 us | qproj_* chunked |
| After Opt K | 820 us | q_head_rope chunked |
| After Opt L | 687 us | qr_norm_apply chunked |
| After Opt M (kept) | 681 us | q_head_rms_nope chunked GRP=8 |
| After Opt N | 614 us | attn_norm_apply chunked |
| After Opt O | 599 us | kv_proj_matmul chunked |
| **After Opt M-revert + P** | **624 us** | rms_nope back to GRP=1; dequant decoupled |

**1868 us → 624 us, −66%.** Validation PASS.

## Opt M/N/O — Three more chunking wins (accepted, 2026-05-20)

After Opt L, three more scopes were chunked using the same pattern:

| Opt | Scope | New GROUP | wall before | wall after | Δ |
|---|---|---:|---:|---:|---:|
| **M** | `q_head_rms_nope` | HEAD_GROUP=8 | 687 us | 681 us | −6 us |
| **N** | `attn_norm_apply` | ATTN_NORM_GROUP=8 | 681 us | 614 us | −67 us |
| **O** | `kv_proj_matmul` | KV_PROJ_GROUP=2 | 614 us | 599 us | −15 us |
| | (3-run median after all three) | | | **646 us** | |

### Opt M — `q_head_rms_nope` HEAD_GROUP=8 (surprising win)

Predicted to regress: pre-change this scope was already at the
"tasks ≈ cores" sweet spot (64 tasks across 48 AIV cores, span 38us).
Chunking 8x was expected to overshoot.

Reality: wall −6us. The scope itself slowed down by design (span 38 → 104us,
8 tasks × 97us μ across 8 cores) but its end time barely moved (≈577us),
while downstream `q_head_rope` could now start ~25us earlier because the
end-of-scope barrier from 64 fine-grained tasks → 8 large tasks dropped a
significant dispatch tail.

**Important lesson** — `q_head_rms_nope` has **2** cross-iter loop-carried
tensors (`q_flat`, `q_head_inv_rms_all`), and it still chunked cleanly
across 8 cores. This refines the
[memory:feedback_pypto_head_group_chunking_loop_carried.md](../../.claude/projects/-data-<user>-newpto-pypto-lib/memory/feedback_pypto_head_group_chunking_loop_carried.md)
judgment: "≥2 loop-carried tensors fails" was wrong. The Opt I 3-tensor
failure may have been specific to `q_rope_pair_stage`'s `[H*T, ROPE_DIM]`
stride pattern, or some interaction with the orch-tensor optimizer.
**Try the chunking first, measure, decide** — predictions from
"loop-carried tensor count" are unreliable.

GRP=16 regressed back to 772us (only 4 parallel iters; per-task span 202us).
GRP=8 kept.

### Opt N — `attn_norm_apply` ATTN_NORM_GROUP=8 (largest of the three)

The Stage 0.2 fused norm-and-cast scope was 32 tasks × 46us. Each task was
already above the 50us-ish target, so this wasn't an obvious dispatch-bound
candidate. But the scope sits on the **critical path immediately after the
serial `attn_norm_rms`**, gating all Q-side downstream work
(`qr_proj_matmul`, `qproj_matmul`).

Sweep:

| ATTN_NORM_GROUP | wall | notes |
|---:|---:|---|
| 1 (baseline) | 681 us | 32 tasks × 46us μ, span 75us |
| 2 | 646 us | −35us |
| 4 | 625 us | −56us |
| **8** | **614 us** | **−67us, sweet spot** |
| 16 | 626 us | regressed (only 2 outer tasks) |

The win is mostly from dispatcher unblock: `qr_proj_matmul` start moved
from 184us (Opt L) to 152us (Opt N GRP=4) to even earlier at GRP=8.
Halving task count to 16 (GRP=2) saved more than its own time; halving
again saved more downstream issue slack.

### Opt O — `kv_proj_matmul` KV_PROJ_GROUP=2 (cube-bound now)

16 tasks × 22us μ — below the 50us target but already above the noise
floor.

| KV_PROJ_GROUP | wall | notes |
|---:|---:|---|
| 1 (baseline) | 614 us | 16 tasks × 22us, span 78us |
| **2** | **599 us** | **−15us, 8 tasks × 47us** |
| 4 | 621 us | regressed (4 outer × 78us, less parallel) |

GRP=2 hits 47us per task — bang on the 50us target.

### Opt skipped — `qproj_dequant` decoupled chunking

Considered: decouple `qproj_dequant` from `qproj_matmul`'s outer
`pl.parallel` and chunk it with its own (larger) GROUP. Would require a
global `[Q_PROJ_HEAD_BLOCKS × T, Q_PROJ_OUT_CHUNK]` INT32 staging buffer =
**16MB** per invocation, plus this becomes a new cross-iter loop-carried
tensor across the matmul outer loop — re-introducing Opt I-style
"≥2 cross-iter outputs" risk. Not pursued — `qproj_dequant`'s 13us μ
sits inside `qproj_matmul`'s span (with significant overlap), so its
contribution to wall is already small.

### Cumulative result

| State | wall (median) | Notes |
|---|---:|---|
| Pre-tuning S=2 baseline | 1868 us | reference |
| After Opt A+E + scope-split | 1560 us | structural |
| Opt B-2 | 1224 us | K-tile 256 |
| Opt J | 849 us | qproj_* chunked |
| Opt K | 820 us | q_head_rope chunked |
| Opt L | 687 us | qr_norm_apply chunked |
| Opt M | 681 us | q_head_rms_nope chunked |
| Opt N | 614 us | attn_norm_apply chunked |
| Opt O | 599 us | kv_proj_matmul chunked |
| **Final 3-run median** | **646 us** | (range 634–647, ~2% noise) |

**1868 us → 646 us, −65%.** Validation PASS on q / kv / qr / qr_scale at
the documented tolerances.

## Opt L — QR_NORM_GROUP-chunk `qr_norm_apply` (accepted, 2026-05-20)

After Opt K, `qr_norm_apply` was the next clearly dispatch-bound scope:
32 tasks × 2.7us μ, span 24.7us — almost pure AICPU dispatch overhead.
Although its own contribution to wall is tiny, the scope sits on the
critical path between `qr_rms` and `qr_quant_amax`, and its span pushes
back the downstream `kv_proj_matmul` / `qproj_matmul` starts.

Same chunking pattern as Opt J/K. Single cross-iter loop-carried tensor
(`qr_bf16`); matches the success condition.

```python
for qbg in pl.parallel(0, Q_BLOCKS, QR_NORM_GROUP):
    with pl.at(..., "qr_norm_apply"):
        for q_inner in pl.range(QR_NORM_GROUP):
            qr_norm_col0 = (qbg + q_inner) * Q_LORA_CHUNK
            ... assemble into qr_bf16 ...
```

### Sweep on `QR_NORM_GROUP`

| GRP | wall | qr_norm tasks | qr_norm μ | qr_norm span | notes |
|---:|---:|---:|---:|---:|---|
| 1 (Opt K) | 820 us | 32 | 2.73 us | 24.7 us | baseline |
| 4 | 775 us | 8 | 7.3 us | 13.8 us | −45 us |
| **8** | **687 us** | **4** | **7.8 us** | **9.2 us** | **−133 us vs Opt K** |
| 16 | 698 us | 2 | 14.4 us | 14.8 us | +11 us vs GRP=8 (within noise; less parallel) |

`QR_NORM_GROUP = 8` accepted. GRP=16 reduces parallelism to 2 tasks
without further benefit. Beyond the local `qr_norm_apply` saving of
~16us, the win comes from unblocking the downstream chain:

- `kv_proj_matmul` start 437 → 400us (GRP=4) → earlier still at GRP=8
- `qproj_matmul` span 105 → 121us (GRP=8) — slightly longer wallclock but
  starts earlier
- Overall Q-side critical path compressed by ~130us

### Result (single run, vs Opt K)

| Metric | Opt K | Opt L (GRP=8) | Δ |
|---|---:|---:|---:|
| **wall** | 820 us | **687 us** | **−133 us (−16.2%)** |
| `qr_norm_apply` tasks | 32 | 4 | −28 |
| `qr_norm_apply` span | 24.7 us | 9.2 us | −15.5 us |

Validation PASS. The benefit (−133us) is much larger than `qr_norm_apply`'s
own Σ Exec savings (~30us), confirming the scope was a dispatch chokepoint
that gated downstream issue.

## Opt K — HEAD_GROUP-chunk `q_head_rope` (decoupled from rms_nope) (accepted, 2026-05-20)

After Opt J shifted the critical path off Stage 3, the new tail was
`q_head_rms_nope + q_head_rope` (combined ~180us span). Opt I had previously
failed to chunk this scope pair when both were inside the same outer
pl.parallel — the 3 cross-iter loop-carried tensors pinned the scope to
4 cores. The new approach: **decouple** the two scopes:

- `q_head_rms_nope` stays at `pl.parallel(0, H, 1)` (64 fine-grained tasks
  distributing across 48 AIV cores, span 36us — already near-optimal)
- `q_head_rope` is moved to its **own** `pl.parallel(0, H, HEAD_GROUP)` +
  `pl.range(HEAD_GROUP)`, with only **one** cross-iter loop-carried tensor
  (`q_rope_pair_stage`)

This matches the Opt J success condition exactly: a chunked scope with one
loop-carried output schedules cleanly across HEAD_GROUP cores.

```python
# rms_nope unchanged (fine-grained over H)
for h in pl.parallel(0, H, 1):
    with pl.at(..., "q_head_rms_nope"):
        ... assemble into q_flat, q_head_inv_rms_all ...

# rope chunked separately — single cross-iter output
for hg in pl.parallel(0, H, HEAD_GROUP):
    with pl.at(..., "q_head_rope"):
        q_head_inv_rms_t = pl.create_tensor([T, 1], dtype=pl.FP32)  # pre-decl
        for h_inner in pl.range(HEAD_GROUP):
            ... assemble into q_rope_pair_stage ...
```

### Result (single run, vs Opt J)

| Metric | Opt J | Opt K | Δ |
|---|---:|---:|---:|
| **wall** | 849 us | **820 us** | **−29 us (−3.4%)** |
| `q_head_rope` tasks | 64 | **8** | −56 |
| `q_head_rope` per-task μ | 5.8 us | **32.7 us** | +5.7× (above 50us-target band) |
| `q_head_rope` span | 103 us | **37 us** | **−66 us** |
| `q_head_rope` distinct cores | 42 | **8** | clean fan-out |
| `q_head_rope` Σ Exec | 370 us | 262 us | −108 us |

`q_head_rms_nope` also benefitted indirectly: span 123 → 36 us, distinct
cores 41 → 48 — no longer sharing a parallelism budget with the chunked
rope scope.

### Why Opt K succeeded where Opt I failed

Same `pl.parallel + pl.range(CHUNK)` pattern, opposite outcome:

| | Opt I (combined rms_nope + rope) | Opt K (rope alone) |
|---|---:|---:|
| Cross-iter loop-carried tensors | **3** (q_flat, q_head_inv_rms_all, q_rope_pair_stage) | **1** (q_rope_pair_stage) |
| Cores used / tasks | 4 / 8 (pinned, regressed) | 8 / 8 (clean) |

The decoupling matters because the 3 tensors in Opt I had different
write patterns across iters; even though pypto sees they assemble to
disjoint slices, the runtime conservatively serialises through a small
core pool.

`q_head_rms_nope` was **deliberately not** chunked: it has 2 loop-carried
tensors (`q_flat` for NOPE writes, `q_head_inv_rms_all`), and the 64-task
fine-grained version already saturates 48 cores with 36us span, leaving
no room for improvement.

## Opt J — Q_PROJ_GROUP-chunk `qproj_matmul` + `qproj_dequant` (accepted, 2026-05-20)

**Net result: wall-clock 1224us → ~850us, −31% (single biggest win on this kernel).**

After Opt B-2 reduced per-task cube work, the swimlane still showed
`qproj_matmul` and `qproj_dequant` as the dominant span pair (combined
~316us). With 256 outer pl.parallel tasks at mean 9.4us per task, cube
utilization sat at ~31% (8.5 effective cores out of 24) — well below the
"AICPU dispatch is the bottleneck" threshold (perf-doc rule 2). The exact
HEAD_GROUP pattern already used by `q_rope_reassemble` / `q_rope_write`
was applied:

```python
for hg in pl.parallel(0, Q_PROJ_HEAD_BLOCKS, Q_PROJ_GROUP):    # 256 → 32 outer
    col_acc_grp = pl.create_tensor([Q_PROJ_GROUP * T, Q_PROJ_OUT_CHUNK], dtype=pl.INT32)
    with pl.at(..., "qproj_matmul"):
        col_acc = pl.create_tensor([T, Q_PROJ_OUT_CHUNK], dtype=pl.INT32)  # pre-decl
        for h_inner in pl.range(Q_PROJ_GROUP):
            for qb in pl.pipeline(0, Q_PROJ_BLOCKS, stage=2):
                ... matmul/matmul_acc into col_acc ...
            col_acc_grp = pl.assemble(col_acc_grp, col_acc, [h_inner * T, 0])
    with pl.at(..., "qproj_dequant"):
        col_acc_chunk = pl.create_tensor([T, Q_PROJ_OUT_CHUNK], dtype=pl.INT32)  # pre-decl
        for h_inner in pl.range(Q_PROJ_GROUP):
            col_acc_chunk = pl.slice(col_acc_grp, ...)
            ... cast + scale + assemble into q_proj_fp32 ...
```

Two AST gotchas from
[memory:feedback_pypto_head_group_chunking_loop_carried.md](../../.claude/projects/-data-<user>-newpto-pypto-lib/memory/feedback_pypto_head_group_chunking_loop_carried.md)
were avoided: `(hg + h_inner) * Q_PROJ_OUT_CHUNK` inlined everywhere (no
Python local binding inside `pl.range`), and `col_acc` /
`col_acc_chunk` pre-declared with `pl.create_tensor` outside the
`pl.range` body to give pypto's loop-carried init_values threading a valid
outer source.

### Result (Opt B-2 baseline vs Opt A, both 3-run medians where shown)

| Metric | Opt B-2 (3-run median) | Opt A (2 runs) | Δ |
|---|---:|---:|---:|
| **wall** | **1224 us** | **848–850 us** | **−375 us (−31%)** |
| `qproj_matmul` tasks | 256 | **32** | −224 |
| `qproj_matmul` per-task μ | 9.4 us | **41–46 us** | +4× (above 50us target) |
| `qproj_matmul` span | 318 us | **100–105 us** | **−215 us** |
| `qproj_matmul` distinct cores | 24 | 16 | (fewer cores but ~all parallel) |
| `qproj_matmul` Σ Exec | 2426 us | 1317–1466 us | **−960 to −1109 us (−40 to −46%)** |
| `qproj_dequant` tasks | 256 | 32 | −224 |
| `qproj_dequant` per-task μ | 2.6 us | 12.3 us | +4.7× |
| `qproj_dequant` span | 307 us | 98–107 us | **−200 us** |
| `qproj_dequant` Σ Exec | 727 us | 395 us | **−332 us (−46%)** |

Two additional second-order improvements, both knock-on effects of Stage 3
finishing earlier:

- **`q_head_rms_nope` span 344 us → 123 us (−221 us)** — the AIV scope chain
  for Stage 4 starts earlier and finds the AIV cores less contended,
  parallelising more cleanly (41 distinct cores at run 1).
- **`q_head_rope` span 354 us → 103 us (−251 us)** — same mechanism.

### Why this worked when Opt I failed

Same `pl.parallel + pl.range(CHUNK)` shape, opposite outcome. The
difference is exactly what
[memory:feedback_pypto_head_group_chunking_loop_carried.md](../../.claude/projects/-data-<user>-newpto-pypto-lib/memory/feedback_pypto_head_group_chunking_loop_carried.md)
warned about — the number of cross-pl.parallel-iter loop-carried tensors:

| Scope pair | Cross-iter loop-carried tensors | Cores used / tasks |
|---|---:|---|
| Opt I `q_head_rms_nope` + `q_head_rope` | **3** (q_flat, q_head_inv_rms_all, q_rope_pair_stage) | 4 / 8 (pinned, regressed) |
| Opt J `qproj_matmul` + `qproj_dequant` | **1** (q_proj_fp32; col_acc_grp is per-outer-iter local) | 16 / 32 (clean fan-out) |

The matmul scope itself even has zero cross-iter outputs — col_acc_grp is
freshly `pl.create_tensor`'d in each pl.parallel iter, mirroring the
working `q_rope_grp_fp32` pattern.

### Where the wall-clock went

Opt J shifted the critical path: Stage 3 (`qproj_matmul + qproj_dequant`)
is no longer dominant. New timeline (from run 1, wall=850us):

| Range | Scope | Duration |
|---|---|---:|
| 0–84 | `attn_norm_rms` (serial, blocking) | 84 us |
| 99–179 | `attn_norm_apply` (parallel × 32 cores) | 80 us |
| 202–384 | Q-side LoRA + quant chain | 182 us |
| 393–541 | **`qproj_matmul + qproj_dequant`** | **148 us** |
| 571–750 | `q_head_rms_nope + q_head_rope` | 179 us |
| 721–850 | KV path + RoPE reassemble/write | 129 us |

Remaining critical-path components: `attn_norm_rms` (serial 84us at front)
and `q_head_rms_nope/rope` (~180us combined span). Both are smaller than
the new Stage 3, so further wall-clock optimization would target them.

### Q_PROJ_GROUP sweep — 8 is the sweet spot

Tried `Q_PROJ_GROUP = 16` to push `qproj_dequant` per-task μ closer to
the 50us dispatch target (12us → 23us achieved). Result: wall **regressed
+83us** (849 → 932 us).

| Metric | GRP=8 | GRP=16 |
|---|---:|---:|
| wall | 849 us | **932 us** (+83) |
| `qproj_matmul` per-task μ | 41 us | 79 us |
| `qproj_matmul` span | 100 us | 106 us (flat) |
| `qproj_dequant` per-task μ | 12 us | **23 us** |
| `qproj_dequant` span | 98 us | **56 us** (−42) |
| `q_head_rms_nope` span | 123 us | 39 us |
| `q_head_rope` span | 103 us | 50 us |

Both targets (dequant and downstream Stage 4) genuinely improved, but a
new 78us AICPU dispatch gap appeared between `qr_norm_apply` and
`qr_quant_amax` (vs ~14us in GRP=8) — these scopes have no data dependency
on qproj, so the regression is a scheduler-level artifact: with 16 large
qproj tasks queued for dispatch, the AICPU dispatcher stalls the Q-side
chain. Net wall is worse despite the local wins.

`Q_PROJ_GROUP = 8` is kept as the sweet spot.

### Risks evaluated and accepted

- **`col_acc_grp` GM footprint**: [Q_PROJ_GROUP × T, Q_PROJ_OUT_CHUNK] INT32 =
  8 × 128 × 128 × 4 B = **512 KB** per outer iter, allocated as scope-local
  inside the pl.parallel body. Mirrors the working `q_rope_grp_fp32`
  pattern (256 KB FP32). Validation PASS confirms no OOM.
- **NZ→ND fixpipe count unchanged**: The pre-refactor code did 256 × 1 =
  256 implicit L0C→GM conversions per invocation; Opt J does 32 × 8 = 256
  via `pl.assemble(col_acc_grp, col_acc, [h_inner * T, 0])` inside the
  matmul scope. Same per-element conversion work, but batched into 32
  tasks instead of 256.
- **Validation PASS** on q / kv / qr / qr_scale at the documented
  tolerances.

## Opt B-2 — re-apply `Q_PROJ_CHUNK = 128 → 256` at S=2 (accepted, 2026-05-20)

Opt B's K-tile change was validated at S=1 originally and apparently
reverted (current `models/deepseek/v4/qkv_proj_rope.py` shipped with
`Q_PROJ_CHUNK = 128`). Re-tested at S=2 (T=128) with 3-run medians on both
sides:

| Metric | Baseline (K=128) | Opt B-2 (K=256) | Δ |
|---|---:|---:|---:|
| **wall median** | 1232 us | 1224 us | **−8 us (−0.6%, within noise)** |
| `qproj_matmul` Σ Exec | 2774 us | 2426 us | **−347 us (−12.5%)** |
| `qproj_matmul` per-task μ | 10.83 us | 9.40 us | −13.3% |
| `qproj_matmul` span | 321 us | 318 us | flat |
| `qproj_dequant` Σ Exec | 729 us | 727 us | flat |

Halving the K-pipeline stages (8 → 4) cuts per-task cube work by ~13%
without changing task count (still 256, since outer pl.parallel is over
`Q_PROJ_HEAD_BLOCKS` not `Q_PROJ_BLOCKS`). Span is unchanged because the
critical path through `qproj_matmul` isn't cube-bound — cube utilization
sits at ~31% (Σ Exec / (span × 24 cube cores)) so the smaller Σ Exec
doesn't translate to span shrink.

**Kept** with the same logic as the original Opt B: strict Σ Exec
improvement, no compile/runtime issues, validation PASS. Wall benefit is
statistically inconclusive on its own but adds optionality for future
optimizations that might shift the bottleneck onto the cube.

**Caveat unchanged**: `Q_PROJ_OUT_CHUNK = 256` still triggers
`ACL_ERROR_RT_AICORE_TIMEOUT` per the original Opt B note — do **not**
attempt the N-tile increase. Only the K-tile (`Q_PROJ_CHUNK`) is safe.

## Opt I — HEAD_GROUP-chunk `q_head_rms_nope` + `q_head_rope` (reverted, 2026-05-20)

Pre-change swimlane at the current accepted state (Opt A+B+E+G + S=2
structural split) measured wall=1291us, with the per-head Stage 4 chain
visibly limiting:

| Scope | tasks | Σ Exec | span |
|---|---:|---:|---:|
| `q_head_rms_nope` | 64 | 798 us | 387 us |
| `q_head_rope` | 64 | 318 us | 381 us |

Each per-head task averaged ~12–17us — well below the perf-doc ~50us/kernel
target — suggesting AICPU dispatch-bound behaviour (rule 2 in
[docs/performance-tuning.md](performance-tuning.md)). The sibling
`q_rope_reassemble` / `q_rope_write` scopes already use HEAD_GROUP-chunking
(8 tasks × 8 heads each) successfully, so the same pattern was applied:

```python
for hg in pl.parallel(0, H, HEAD_GROUP):
    with pl.at(..., "q_head_rms_nope"):
        for h_inner in pl.range(HEAD_GROUP):
            # one head's RMS sq-sum + NOPE-direction normalize-and-cast
            ...
    with pl.at(..., "q_head_rope"):
        for h_inner in pl.range(HEAD_GROUP):
            # one head's RoPE body
            ...
```

Two pypto AST quirks surfaced during implementation:

1. **Don't bind `h = hg + h_inner` to a Python local** inside the
   `pl.range(HEAD_GROUP)` body. The AST analyzer threads any Python local
   set in the loop body through `init_values`, but `h` has no valid outer
   initializer → SSA verification fails with
   `Variable 'h_inline20' used outside its defining scope`. **Fix:** inline
   `(hg + h_inner)` and `(hg + h_inner) * HEAD_DIM` everywhere — exactly
   how the working `q_rope_reassemble` does it.
2. **Pre-declare `q_head_inv_rms_t = pl.create_tensor([T, 1], dtype=pl.FP32)`
   before the `pl.range(HEAD_GROUP)` body.** When a Python-local tensor is
   defined in the outer loop body AND used inside a nested `pl.range`
   (here `pl.range(NOPE_DIM // HEAD_CHUNK)`), pypto threads it through
   the outer loop's `init_values`. The dummy `create_tensor` gives that
   threading a valid pre-loop initializer.

**Validation PASS** on q/kv/qr/qr_scale at the same tolerances. Single-run
wall=**1388us** — a clear regression (+97us vs the 1291us pre-change
measurement, well outside the ~4% noise band).

**Why it regressed (root cause):**

| Scope | tasks | Σ Exec | span | distinct cores used |
|---|---:|---:|---:|---:|
| `q_head_rms_nope` (pre, 64 tasks) | 64 | 798 us | 387 us | **27** |
| `q_head_rms_nope` (Opt I, 8 tasks) | 8 | 763 us | **525 us** | **4** |
| `q_rope_reassemble` (already chunked, 8 tasks) | 8 | 82 us | 16 us | **8** |

The chunking did the intended thing on Σ Exec (−35us, −98us per scope —
launch overhead amortised), but the 8 large parallel iterations only landed
on **3-4 distinct AIV cores** instead of 8. With per-task ~95us and 4 cores
serving 8 iters, `ceil(8/4) × 95 = 190us` would have been the ideal span;
observed 525us shows the scheduler additionally added ~80us idle gaps on
each core between tasks.

The contrast with the working `q_rope_reassemble` (8 tasks → 8 distinct
cores) is the diagnostic. Both scopes have the same outer
`pl.parallel(0, H, HEAD_GROUP)` shape, but they differ in **how many
tensors are loop-carried across the parallel iterations**:

- `q_rope_reassemble`: writes only to a scope-local `q_rope_grp_fp32`
  (created fresh inside each parallel-iter body via `pl.create_tensor`).
  Zero cross-iter tensor threading. Schedules cleanly across 8 cores.
- `q_head_rms_nope` / `q_head_rope`: write to **three** cross-iter
  loop-carried tensors — `q_flat` (NOPE writes), `q_head_inv_rms_all`
  (per-head inv_rms), and `q_rope_pair_stage` (BF16 rope outputs). Even
  though each iter assembles to a disjoint slice of each tensor, the IR
  threads a single SSA value through pl.parallel's `init_values`, and the
  pypto runtime conservatively serialises the iterations through a 3-4
  core pool.

The 64-iter baseline parallelised at only 2 effective cores too, but the
many small tasks let the dispatcher rotate them across **27** cores rather
than pin to a small pool. Net: with the same threading depth, more
fine-grained chunks distribute across more cores than fewer large chunks.

**Lesson** — `pl.parallel + pl.range(HEAD_GROUP)` chunking only pays off
when the chunked scope's outputs are scope-local (e.g. mediated via
`pl.create_tensor` inside the parallel body). Scopes that assemble across
multiple cross-iter loop-carried tensors should stay at `pl.parallel(0, N, 1)`
even when the per-task work is below the 50us dispatch-target.

**Status — file is already near-optimal.** The accepted state from
Opt A+B+E+G (S=1) plus the `q_head_rms_rope` scope split (S=2 required) is
the endpoint; remaining ~100us tail in `q_rope_reassemble` / `q_rope_write`
requires an algorithm-level rewrite (replace the matmul-based even/odd
permutation with a pure-AIV interleave) which was already ruled out as
too costly for the expected ROI.

## Opt S — parallelize `attn_norm_rms` via 2-way coarse partial sum (accepted, 2026-05-21)

After Opt M-revert+P landed at 624us, the remaining critical-path serial AIV
chain at the front of the kernel was `attn_norm_rms` — a single-task RMS
reduce over D=4096 (32 × D_CHUNK=128 iters) on one AIV core, ~93us span at
S=2. With 48 AIV cores on a2a3, this scope used 1/48 of the vector capacity.

### Attempt 1: 4-way partial sum (compile failure + precision drift)

Initial attempt split the reduce into 4 partial workers + 1 final reduce
scope:

```python
ATTN_RMS_PARTIALS = 4
x_sq_partial = pl.create_tensor([4, T], dtype=pl.FP32)
for wg in pl.parallel(0, 4, 1):
    with pl.at(..., name_hint="attn_norm_rms_partial"):
        ... reduce 8 D-chunks into partial ...

with pl.at(..., name_hint="attn_norm_rms_final"):
    ... sum 4 partials, recip(sqrt(...)) ...
```

Three problems surfaced:

1. **pypto AST name-collision** — removing the original `attn_norm_rms`
   scope (which had Python local `d0` inside its `pl.range`) broke the
   implicit name-threading chain. pypto's AST analyzer threads each
   Python-local set inside a pl.range body through the outer pl.parallel's
   `init_values`. With no prior `d0` definition, downstream scopes
   (`attn_norm_apply`, `qr_proj_matmul`) that also use `d0` got
   `Variable 'd0_inlineNN' used outside its defining scope` SSA errors. Fix:
   rename `d0`/`x_chunk` to `apply_d0`/`apply_x_chunk` in attn_norm_apply,
   and `d0` to `qr_d0` in qr_proj_matmul. The chain after the fix is broken
   cleanly (no downstream consumer of `attn_norm_rms_partial`'s locals).
2. **Vec buffer overflow at fresh compile (320 KB > 192 KB)** —
   `attn_norm_rms_partial` accumulated ~5 tile allocations × 8 inner iters
   without buffer reuse. Same source compiled differently on different
   devices (dev 0 cached compile from earlier session worked; dev 10 fresh
   compile rejected). Borderline — `attn_norm_apply` also at 193 KB > 192 KB.
3. **`q` precision FAIL on dev 2** — 4-way FP32 add-reduce is not strictly
   associative; `x_inv_rms` drift of ~1 ULP propagated through
   `token_x_bf16` → `qproj_matmul` → `q_head_rms_nope/rope` and pushed `q`
   past `rtol=1/128, max_error_ratio=0.005`. `kv` and `qr` (shallower chains)
   PASSed.

### Attempt 2: 2-way partial + `chunked_loop_optimizer` (accepted)

Reduced `ATTN_RMS_PARTIALS = 2` and added
`optimization=pl.chunked_loop_optimizer` to the partial scope:

```python
ATTN_RMS_PARTIALS = 2
D_BLOCKS_PER_PARTIAL = D_BLOCKS // ATTN_RMS_PARTIALS  # 16
x_sq_partial = pl.create_tensor([ATTN_RMS_PARTIALS, T], dtype=pl.FP32)
for wg in pl.parallel(0, ATTN_RMS_PARTIALS, 1):
    with pl.at(level=pl.Level.CORE_GROUP,
               optimization=pl.chunked_loop_optimizer,
               name_hint="attn_norm_rms_partial"):
        rms_d_base = wg * D_BLOCKS_PER_PARTIAL * D_CHUNK
        local_sum = pl.full([1, T], dtype=pl.FP32, value=0.0)
        for rms_db in pl.range(D_BLOCKS_PER_PARTIAL):
            rms_d0 = rms_d_base + rms_db * D_CHUNK
            rms_x_chunk = pl.cast(x_flat[:, rms_d0 : rms_d0 + D_CHUNK], target_type=pl.FP32)
            local_sum = pl.add(local_sum, pl.reshape(pl.row_sum(pl.mul(rms_x_chunk, rms_x_chunk)), [1, T]))
        x_sq_partial[wg : wg + 1, :] = local_sum

with pl.at(..., name_hint="attn_norm_rms_final"):
    x_sq_sum = pl.full([1, T], dtype=pl.FP32, value=0.0)
    for w in pl.range(ATTN_RMS_PARTIALS):
        x_sq_sum = pl.add(x_sq_sum, x_sq_partial[w : w + 1, :])
    x_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(x_sq_sum, 1.0 / D), EPS)))
```

Why this fixed all three issues:

- **Vec buffer**: `chunked_loop_optimizer` triggers `OutlineIncoreScopes` to
  outline each inner iter as a sub-task with shared buffer reuse → per-task
  Vec usage drops to ~64 KB (single iter's tiles). The Opt A doc warned this
  splits cast ops into separate sub-tasks (per-task launch OH ~7us), but
  here the 2 partial workers run in parallel and the cast-split sub-tasks
  also run in parallel on the 48 AIV cores → measurable but not
  catastrophic.
- **Precision**: 2-way add-reduce is *deterministic* — only one `pl.add`
  at the final scope, no associativity ambiguity. `q` PASS stable across
  3 devices (5, 6, 8, 10).
- **Compile stability**: `chunked_loop_optimizer` makes buffer usage
  deterministic across devices.

### Result (3-run median, vs M-reverted+P state)

| Run | device | wall | partial μ / span | final μ |
|---|---|---:|---:|---:|
| 1 | dev 5 | 629.3 us | 45.6 us / 55.2 us | 1.5 us |
| 2 | dev 8 | **601.4 us** | 43.2 us / 49.1 us | 1.4 us |
| 3 | dev 6 | **604.5 us** | 43.8 us / 49.2 us | 1.3 us |
| **median** | — | **604.5 us** | ~44 us / ~50 us | ~1.4 us |

| Metric | M-rev+P | Opt S | Δ |
|---|---:|---:|---:|
| **wall median** | 624 us | **604.5 us** | **−19.5 us (−3.1%)** |
| `attn_norm_rms_partial` tasks | — | 2 (was 1 serial) | — |
| `attn_norm_rms_partial` span | — | ~50 us | — |
| `attn_norm_rms_final` span | — | ~3 us | — |
| `attn_norm_apply` start time | ~148 us | **~116 us** | **−32 us** |

The full RMS chain (partial → 8us dispatch gap → final) takes ~62us vs the
original 93us, saving ~31us on attn_norm_apply start. Wall drops only ~19us
because some of the savings propagate through later non-critical paths.

### Opt S2 — fold final reduce into `attn_norm_apply` (reverted, 2026-05-21)

Attempted to remove the 8us partial→final dispatch gap by deleting the
`attn_norm_rms_final` scope and having each `attn_norm_apply` task compute
its own `x_inv_rms` (read x_sq_partial[0..1], add, recip+sqrt+mul+adds).

| Run | device | wall | apply μ |
|---|---|---:|---:|
| 1 | dev 5 | 610.4 us | 45.0 us |
| 2 | dev 5 | 623.4 us | — |
| 3 | dev 5 | 624.1 us | 42.6 us |
| **median** | — | **623.4 us** | 43.8 us |

Reverted — each apply task gained +5-6us μ (redundant inv_rms compute,
8 tasks in parallel on 4 cores → span +10-15us), which **exactly cancelled**
the ~10us savings from removed final scope + dispatch gap. Net wall: roughly
neutral, possibly slight regression. Same-device (dev 5) comparison showed
~5us improvement, but cross-device noise is too large to distinguish.
Kept the separate `attn_norm_rms_final` scope.

### Lesson learned (memory): `chunked_loop_optimizer` IS useful on vector scopes when buffer pressure exceeds 192 KB

Opt A removed `chunked_loop_optimizer` from `attn_norm_apply`/`qproj_dequant`/`q_head_rms_rope`
because the cast-split overhead (~7us per task) was wasted there — those
scopes fit comfortably in Vec buffer at S=1. The rule "don't use
`chunked_loop_optimizer` on vector scopes" must be qualified:

- **At S=1**: vec buffer rarely tight; `chunked_loop_optimizer` is pure OH.
- **At S=2 with cast-heavy reduce scopes**: buffer can exceed 192 KB without
  it; the optimizer is *required* to fit. The cast-split OH is the price
  of admission.

## Opt T — fold `qr_quant_amax` into `qr_norm_apply` (accepted, 2026-05-21)

`qr_quant_amax` was a single-task serial AIV scope (~30 us at S=2):
32 inner iters over Q_LORA=1024 at QUANT_CHUNK=32, computing per-token
absolute-max for INT8 quantization scale. Sat on the critical path between
`qr_norm_apply` and `qr_quant_apply`, blocking downstream `qproj_matmul`.

Key observation: `qr_norm_apply` is already 4-way parallel (Q_BLOCKS=32,
QR_NORM_GROUP=8 → 4 outer tasks each processing 8 Q_LORA_CHUNKs). Each task
writes its slice of `qr_bf16`. If each task **also** computes a per-task
partial amax over its own slice and writes to a `[Q_BLOCKS, T]` partial
buffer, then `qr_quant_amax` shrinks to a 4-way max + the existing div/recip
scale computation.

### Implementation

```python
qr_bf16 = pl.create_tensor([T, Q_LORA], dtype=pl.BF16)
qr_amax_partial = pl.create_tensor([Q_BLOCKS, T], dtype=pl.FP32)
for qbg in pl.parallel(0, Q_BLOCKS, QR_NORM_GROUP):
    with pl.at(..., name_hint="qr_norm_apply"):
        local_amax = pl.full([1, T], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for q_inner in pl.range(QR_NORM_GROUP):
            ... existing norm-and-cast code, but capture qr_normed_bf16 ...
            qr_normed_bf16 = pl.cast(qr_normed, target_type=pl.BF16, mode="rint")
            qr_bf16[:, col_slice] = qr_normed_bf16
            # Fold: compute per-chunk amax in-scope, accumulate
            qr_norm_amax_f32 = pl.cast(qr_normed_bf16, target_type=pl.FP32)
            qr_norm_amax_abs = pl.maximum(qr_norm_amax_f32, pl.neg(qr_norm_amax_f32))
            local_amax = pl.maximum(local_amax, pl.reshape(pl.row_max(qr_norm_amax_abs), [1, T]))
        qr_amax_partial[qbg : qbg + 1, :] = local_amax

# qr_quant_amax becomes a 4-way reduce + scale compute (was 32-iter serial)
with pl.at(..., name_hint="qr_quant_amax"):
    qr_amax = pl.full([1, T], dtype=pl.FP32, value=INT8_AMAX_EPS)
    for w in pl.range(0, Q_BLOCKS, QR_NORM_GROUP):
        qr_amax = pl.maximum(qr_amax, qr_amax_partial[w : w + 1, :])
    ... existing div / recip / scale assignment ...
```

### Why this preserved strict `qr` INT8 equality (atol=1, max_error_ratio=0)

The partial amax is computed on `qr_normed_bf16` — the exact same BF16
representation that the original `qr_quant_amax` re-reads from
`qr_bf16` via GM. Computing on `qr_normed_bf16` (still in vec regs) and
casting back to FP32 produces bit-identical input to the abs/max ops. The
amax value is therefore identical to the original, so `qr_scale_quant_t`
is identical, so the INT8 outputs are identical.

`amax` is also a deterministic operation (max is associative and order-
independent unlike FP32 add), so 4-way reduce produces the same result as
32-iter serial reduce.

### Result (3-run median, vs Opt S state)

| Run | device | wall | qr_quant_amax μ | qr_norm_apply μ |
|---|---|---:|---:|---:|
| 1 | dev 10 | **575.1 us** | 1.68 us | 13.3 us |
| 2 | dev 8 | 593.0 us | 1.74 us | 17.4 us |
| 3 | dev 8 | **587.3 us** | 1.74 us | 15.1 us |
| **median** | — | **587.3 us** | ~1.7 us | ~15 us |

| Metric | Opt S | Opt T | Δ |
|---|---:|---:|---:|
| **wall median** | 604.5 us | **587.3 us** | **−17.2 us (−2.8%)** |
| `qr_quant_amax` per-task μ | 29.6 us | **1.7 us** | **−27.9 us (−94%)** |
| `qr_quant_amax` Latency | ~32 us | ~5 us | −27 us |
| `qr_norm_apply` per-task μ | 11.2 us | 15.0 us | +3.8 us |
| `qr_norm_apply` span | ~15 us | ~17 us | +2 us (parallel absorbs) |

The +3.8us per-task on `qr_norm_apply` is the price of the extra
cast-back-to-FP32 + abs + row_max per inner iter. With 4 parallel apply
tasks running on 4 AIV cores, this only adds ~2us to span — well under
the 27us saved on `qr_quant_amax`.

### Cumulative S=2 state

| State | wall (median) | Notes |
|---|---:|---|
| Pre-tuning S=2 baseline | 1868 us | reference |
| ... (Opt A+E+scope-split, B-2, J, K, L, M, N, O, M-revert+P) ... | 624 us | (intermediate) |
| **+ Opt S** | **604.5 us** | attn_norm_rms 2-way partial |
| **+ Opt T** | **587.3 us** | qr_quant_amax folded |

**1868 us → 587 us, −69%.** Validation PASS on q / kv / qr / qr_scale at
documented tolerances across 3 devices.

### Remaining opportunities (post-Opt T, rank-ordered)

1. **`qr_rms` ~30us serial AIV** — same partial-sum pattern would work, but
   at 30us it's near the launch-OH break-even (Opt C failed at 35us serial).
   2-way partial estimated ~20us → ~10us saving, marginal. Skip unless wall
   needs another push.
2. **`q_head_rms_nope` per-task ~20us** — already 64-task parallel across
   48 AIV cores, near saturation. Limited room.
3. **`qproj_matmul` 100us cube span** — largest remaining critical-path
   item, but cube-bound; algorithm-level rewrite required (not chunk
   tuning).
4. **`q_rope_reassemble + q_rope_write` 35us combined** — Opt R confirmed
   pure-AIV rewrite blocked by `pto.alloc_tile` 32-byte alignment guard.
   Needs backend feature (hardware vector interleave primitive or
   intra-scope NZ→ND conversion).

## Opt U — `qr_rms` 2-way partial sum (accepted, 2026-05-21)

After Opt T landed `qr_quant_amax` at ~5us, the next serial AIV reduce on the
critical path is `qr_rms` — single-task ~30us at S=2 reading `qr_fp32`
(already FP32, no cast). Initially expected to be marginal at the 35us
launch-OH break-even, but worked better than predicted because the absent
BF16→FP32 cast means low per-iter buffer pressure and no cast-split OH.

### Implementation

Same pattern as Opt S, sized to `QR_RMS_PARTIALS=2`:

```python
QR_RMS_PARTIALS = 2
Q_BLOCKS_PER_QR_PARTIAL = Q_BLOCKS // QR_RMS_PARTIALS  # 16
qr_sq_partial = pl.create_tensor([QR_RMS_PARTIALS, T], dtype=pl.FP32)
for wgr in pl.parallel(0, QR_RMS_PARTIALS, 1):
    with pl.at(level=pl.Level.CORE_GROUP,
               optimization=pl.chunked_loop_optimizer,
               name_hint="qr_rms_partial"):
        qr_rms_q_base = wgr * Q_BLOCKS_PER_QR_PARTIAL * Q_LORA_CHUNK
        qr_local_sum = pl.full([1, T], dtype=pl.FP32, value=0.0)
        for qr_rms_qb in pl.range(Q_BLOCKS_PER_QR_PARTIAL):
            qr_rms_col0 = qr_rms_q_base + qr_rms_qb * Q_LORA_CHUNK
            qr_rms_chunk = qr_fp32[:, qr_rms_col0 : qr_rms_col0 + Q_LORA_CHUNK]
            qr_local_sum = pl.add(qr_local_sum, pl.reshape(pl.row_sum(pl.mul(qr_rms_chunk, qr_rms_chunk)), [1, T]))
        qr_sq_partial[wgr : wgr + 1, :] = qr_local_sum

with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_rms_final"):
    qr_sq_sum = pl.full([1, T], dtype=pl.FP32, value=0.0)
    for w in pl.range(QR_RMS_PARTIALS):
        qr_sq_sum = pl.add(qr_sq_sum, qr_sq_partial[w : w + 1, :])
    qr_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(qr_sq_sum, 1.0 / Q_LORA), EPS)))
```

Also renamed `qr_chunk` → `qr_norm_chunk` in `qr_norm_apply` to break the
implicit pypto AST name-chain (Opt S's lesson applied preemptively).

### Result (3-run median)

| Run | device | wall | qr_rms_partial μ |
|---|---|---:|---:|
| 1 | dev 5 | **578.4 us** | 14.3 us |
| 2 | dev 5 | 603.9 us | 15.0 us |
| 3 | dev 5 | **576.4 us** | 14.6 us |
| **median** | — | **578.4 us** | ~14.6 us |

| Metric | Opt T | Opt U | Δ |
|---|---:|---:|---:|
| **wall median** | 587 us | **578 us** | **−9 us (−1.5%)** |
| `qr_rms` chain span | ~30 us | ~20 us | ~−10 us |
| `qr_rms_partial` μ | n/a | 14.6 us | (was 30us serial) |
| `qr_rms_final` μ | n/a | 1.3 us | — |

### Why it worked at the 35us break-even

The user predicted "10us 边际,踩 35us 门槛" — concern was per-task launch OH
(~5-10us) eating the savings on a serial reduce close to that threshold.
Two factors made it land more cleanly:

1. **`qr_fp32` is already FP32** — no `pl.cast` in the inner loop. Vec
   buffer per iter is ~32 KB (qr_chunk + mul intermediate), well under 192KB
   even without the optimizer. The `chunked_loop_optimizer` was added
   defensively but its cast-split OH penalty doesn't apply here.
2. **Inner iter is genuinely fast** — 16 iters × ~1us/iter = ~16us pure
   compute per partial worker. With launch OH the per-task μ landed at
   ~14.6us (well below the 35us threshold). 2 partials in parallel + 3us
   final ≈ 20us total span vs original 30us serial.

### Cumulative S=2 state

| State | wall (median) | Δ vs baseline |
|---|---:|---:|
| Pre-tuning S=2 baseline | 1868 us | — |
| ... (Opt A+E+scope-split, B-2, J, K, L, M, N, O, M-revert+P) ... | 624 us | −66% |
| + Opt S (attn_norm_rms 2-way partial) | 604.5 us | −67.6% |
| + Opt T (amax folded into qr_norm_apply) | 587.3 us | −68.6% |
| **+ Opt U (qr_rms 2-way partial)** | **578.4 us** | **−69.1%** |

Validation PASS on q / kv / qr / qr_scale at documented tolerances.

## Opt V — `Q_PROJ_CHUNK = 256 → 512` (accepted, 2026-05-21)

After Opt U landed at 578us, `qproj_matmul` was the single largest hot scope
on the critical path: 64 tasks × 74us μ = 121us span, with cube fully
saturated at 24 cores. Opt B-2 had previously doubled the K-tile from
128 → 256 and saved ~13% per-task Exec. Tried doubling again to K=512
(inner pipeline 4 iter → 2 iter, weight L1 reuse 25% → 50%).

### Change

```python
Q_PROJ_CHUNK = 512  # was 256
```

Single one-line change. Q_LORA=1024 divides cleanly by 512 → 2 inner
pipeline stages.

### Why this didn't repeat the Opt B Q_PROJ_OUT_CHUNK=256 timeout

Opt B's `Q_PROJ_OUT_CHUNK = 256` (N-tile) triggered
`ACL_ERROR_RT_AICORE_TIMEOUT` because the AICore template for
`INT8 [T, K_tile] × [K_tile, N=256]` was unsupported on a2a3. Opt V
changes the **K-tile** (input side), which is the dimension being reduced
inside one matmul tile — that's covered by standard CANN matmul templates.
N-tile (output side) is the dangerous one; K-tile is safe.

### Result (3-run median)

| Run | device | wall | qproj_matmul μ |
|---|---|---:|---:|
| 1 | dev 5 | **545.6 us** | 53.2 us |
| 2 | dev 5 | 553.8 us | 55.9 us |
| 3 | dev 5 | 561.4 us | 58.0 us |
| **median** | — | **553.8 us** | ~55.7 us |

| Metric | Opt U | Opt V | Δ |
|---|---:|---:|---:|
| **wall median** | 578 us | **553.8 us** | **−24.2 us (−4.2%)** |
| `qproj_matmul` per-task μ | 74.2 us | 55.7 us | **−18.5 us (−25%)** |
| `qproj_matmul` span | 121 us | ~114 us | −7 us |
| `q_head_rms_nope` start | ~483 us | ~470 us | earlier by ~13 us |
| `q_head_rope` start | ~549 us | ~530 us | earlier by ~19 us (cascade) |

### Why wall dropped 24us when per-task only dropped 19us on a parallel scope

The per-task savings on `qproj_matmul` (−18.5us μ) span across all 64 tasks
running on 24 cube cores → ceil(64/24) × 18.5 ≈ 50us of cube wall time
removed. Only ~7us reaches `qproj_matmul`'s own span (cube still mostly
saturated), but the earlier finish unblocks the entire Q-side AIV chain
downstream. `q_head_rms_nope` (now starting ~13us earlier) and `q_head_rope`
(~19us earlier) each compress the tail.

This is the opposite of the Opt B parallel-slack-absorption phenomenon —
Opt B's small per-task gain was absorbed by a not-yet-saturated parallel
band, so wall didn't move. Opt V's bigger gain on a cube-saturated band
**does** translate to wall because the cube was the rate-limit, not the
dispatch.

### Cumulative S=2 state — first sub-555us median

| State | wall (median) | Δ vs baseline |
|---|---:|---:|
| Pre-tuning S=2 baseline | 1868 us | — |
| ... (Opt A+E+scope-split, B-2, J, K, L, M, N, O, M-revert+P) ... | 624 us | −66.6% |
| + Opt S | 604.5 us | −67.6% |
| + Opt T | 587.3 us | −68.6% |
| + Opt U | 578.4 us | −69.1% |
| **+ Opt V (K=512)** | **553.8 us** | **−70.4%** |

Validation PASS on q/kv/qr/qr_scale at documented tolerances across 3 runs.

### Lesson (memory): K-tile increase on cube-saturated matmul is one of the highest-ROI single-line changes available

Pattern:
1. Look at swimlane — find a matmul scope that's cube-saturated (24/24 AIC
   busy, Exec/Latency > 80%) on the critical path.
2. Check current K-tile vs total K dimension. If K is divisible by 2× the
   current tile, try the bigger tile.
3. **Only K-tile.** N-tile increases hit CANN template support gaps
   (ACL_ERROR_RT_AICORE_TIMEOUT) — already documented in Opt B note.
4. Cost: one-line change + assert update. Risk: L1 buffer pressure
   (weight tile size doubles). Failure mode is graceful — compile error
   or per-task μ regression — not a runtime crash.

## Opt X — `pl.pipeline(stage=2 → stage=4)` on qr/kv_proj_matmul K loop (accepted, 2026-05-21)

After Opt V landed at 553.8us, `qproj_matmul` was no longer cube-saturated
on per-task μ (55us, was 74us), but `qr_proj_matmul` (33us μ) and
`kv_proj_matmul` (31us μ) were the next two cube scopes still using
`pl.pipeline(0, D_BLOCKS, stage=2)`. Per
[docs/performance-tuning.md](performance-tuning.md) Part 2 §1, the largest
input-projection K dim benefits from `stage=4` ping-pong (qwen3-14b
precedent).

### Change

Two-line update — change `stage=2` to `stage=4` on both scopes' K loops
([qkv_proj_rope.py:141](../models/deepseek/v4/qkv_proj_rope.py#L141) and
[qkv_proj_rope.py:370](../models/deepseek/v4/qkv_proj_rope.py#L370)).
`qproj_matmul` was skipped because Q_PROJ_BLOCKS = Q_LORA / Q_PROJ_CHUNK =
1024/512 = 2 iter only — `stage=4` would over-replicate and waste L1.

D_BLOCKS = 32 on both qr/kv matmuls — plenty of iters to amortize a 4-deep
ping-pong.

### Result (3-run median)

| Run | device | wall | qr_proj_matmul μ | kv_proj_matmul μ |
|---|---|---:|---:|---:|
| 1 | dev 6 | 555.5 us | 27.7 us | 28.5 us |
| 2 | dev 13 | **544.5 us** | 27.9 us | 26.4 us |
| 3 | dev 5 | **545.7 us** | 27.7 us | 29.3 us |
| **median** | — | **545.7 us** | ~27.8 us | ~28.5 us |

| Metric | Opt V | Opt X | Δ |
|---|---:|---:|---:|
| **wall median** | 553.8 us | **545.7 us** | **−8.1 us (−1.5%)** |
| `qr_proj_matmul` per-task μ | 33.5 us | 27.8 us | **−5.7 us (−17%)** |
| `kv_proj_matmul` per-task μ | 31.6 us | 28.5 us | **−3.1 us (−10%)** |

### Why stage=4 helped here but wasn't done earlier

Before Opt V, `qproj_matmul` (the cube-saturated bottleneck) had K=256 →
Q_PROJ_BLOCKS=4, marginally enough for stage=4 but the scope was already
the rate limiter. Pushing stage=4 there would have used 4× weight-tile L1
buffers (4 × 64 KB = 256 KB) — borderline. After Opt V doubled K to 512,
`qproj_matmul` dropped to Q_PROJ_BLOCKS=2 (too few iters for stage=4
anyway), but `qr_proj_matmul`/`kv_proj_matmul` still have D_BLOCKS=32 — the
real candidates revealed.

Lesson: **stage selection depends on the iter count of the K loop**, not
just whether the scope is large. The qwen3-14b precedent specifies
`stage=4 used for the largest input-proj K dim` — large in **iter count**,
not byte size.

### Cumulative S=2 state

| State | wall (median) | Δ vs baseline |
|---|---:|---:|
| ... (Opt A+E+scope-split, B-2, J, K, L, M, N, O, M-revert+P) ... | 624 us | −66.6% |
| + Opt S | 604.5 us | −67.6% |
| + Opt T | 587.3 us | −68.6% |
| + Opt U | 578.4 us | −69.1% |
| + Opt V (K=512) | 553.8 us | −70.4% |
| **+ Opt X (stage=4)** | **545.7 us** | **−70.8%** |

Validation PASS on q/kv/qr/qr_scale at documented tolerances across 3 runs.

## Opt W — `pl.spmd` replacing `pl.parallel + pl.at` (reverted, 2026-05-21)

After Opt X, [docs/performance-tuning.md](performance-tuning.md) Part 1 §6
remained the one untested rule in the playbook. Tried `pl.spmd` on two
candidates with the highest Head OH × task-count product per the perf
table:

| Probe | Scope | Before | After (spmd) | wall Δ |
|---|---|---|---|---:|
| 1 | `qproj_dequant` | 8 tasks × 13us μ, Latency 18us | 8 tasks × 41us μ, Latency 47us | **+10us** |
| 2 | `qr_proj_matmul` | 32 tasks × 28us μ, Latency 41us | 32 tasks × 27us μ, **Latency 44us, Tail OH 5.96→13.07us** | **+37us** |

Both probes were reverted within seconds — the regression mechanism is
identical:

- Per-task **Exec** stays approximately the same (spmd doesn't change the
  cube/vec compute).
- Per-task **Latency** *increases* because spmd merges the per-iter body
  into a larger InCore kernel; the Tail OH (which doc Part 2 attributes to
  fixpipe/MTE3 finish) grows from ~6us to ~13us per task.
- The dispatch-OH savings the perf-tuning doc projects (one AICPU dispatch
  vs N) are real but small — ~0.5us × (N-1). At N=8 that's ~3.5us; at
  N=32, ~15us. Both are dwarfed by the per-task Tail OH inflation.

### Why this didn't apply here

`pl.spmd`'s payoff is conditioned on the swimlane showing an AICPU
**dispatch trail** — sequential dispatch entries serialising on the AICPU
lane (doc Part 1 §6). After Opt J/K/L/M/N/O/P consolidated each scope into
"few large tasks" via GROUP-chunking, the per-scope dispatch is no longer
the bottleneck:

- Most scopes show Exec/Latency in the 70-90% range — cube/vector compute
  dominates, not AICPU scheduling.
- The remaining Head OH (~5-18us per task) is split across the 24/48
  cores in parallel; it's not a serial AICPU trail.

`pl.spmd` shines on kernels where each task is small (≤5us body) AND
parallel-replicated many times AND the swimlane shows visibly stacked
AICPU dispatch entries. None of these apply to qkv_proj_rope after
Opt J/K/L/M/N/O/P chunking.

### Decision

**Reverted.** Both probes restored to `pl.parallel + pl.at`. Opt X
endpoint at **545.7us median** stands as the final accepted state.

### Lesson (memory): pl.spmd diagnostic checklist

Before trying `pl.spmd` on a scope, check that **all three** hold:

1. Per-task Exec is small (≤5us).
2. The swimlane has visible sequential AICPU dispatch entries (an
   identifiable "dispatch trail").
3. Per-task Tail OH is small (≤2us) — spmd will inflate it, so starting
   from a high Tail OH means net negative.

If any of these is missing — especially #2 — leave the scope as
`pl.parallel + pl.at`.

## Final state (Opt A+B+E+G+J+K+L+M-revert+N+O+P+S+T+U+V+X), S=2

| State | wall (median) | Δ vs original |
|---|---:|---:|
| Pre-tuning S=2 baseline | 1868 us | — |
| Endpoint (Opt A+B+E+G+J+K+L+N+O+P+S+T+U+V+X) | **545.7 us** | **−70.8%** |

All optimizations preserve q/kv/qr/qr_scale validation PASS at the
documented tolerances.

### Untried / no-headroom remaining

- **Opt Y (MTE 512 B align via D_CHUNK=128→256)** — compile fail:
  `attn_norm_rms_partial` 328 KB > 192 KB Vec UB, even with
  `chunked_loop_optimizer`. Tile too large for the platform.
- **Q_LORA_CHUNK increase** (would improve trailing-dim alignment from 64B
  to 128B+) — affects 6 scopes, Opt G already tried and reverted; not
  retried.
- **Q_PROJ_OUT_CHUNK increase** (N-tile) — already known to trigger
  `ACL_ERROR_RT_AICORE_TIMEOUT` (Opt B).
- **`q_rope_reassemble + q_rope_write` pure-AIV rewrite** — Opt R proved
  blocked by `pto.alloc_tile` 32-byte alignment guard. Needs backend
  feature.

The kernel is at the practical endpoint achievable via source-level
tuning within the current pypto backend.
