# DSpark HCA decode tuning log

> 归档日期：2026-09-15。来源：`pypto-lib/local_archive/dspark-hca-decode-tuning-log.md`。
> 原文的实现、测量与计划保留其历史语境；阅读说明见[学习资料索引](../README.md)。

Where `decode_hca` spends its time, what has been tried, and what is left. Read
this before starting a new round on the DSpark decode attention path so the
dead ends below are not walked twice.

All figures are `models/deepseek_v4_flash_dspark/decode_hca.py -p a2a3 --tp 4`
on `pypto-lib` at `f069bc7` with the toolchain pinned to ptoas 0.57.

## Measuring it

The shape and the metric both have to be stated or the numbers are not
comparable.

| Knob | Value | Why it matters |
|---|---|---|
| `--start-pos` | 16 values | The list length sets `batch`; 16 x `S`(8) = 128 = `LOCAL_T`, the full designed shape. Fewer values shrinks the workload. |
| position value | 256 or 8192 | For HCA these measure the same: `visible_rows = (start + S) // 128` gives 2 vs 64, both of which floor `cmp_work_count` to 1. Measured difference is ~1.5%, inside the noise. |
| devices | four even dies | Dies pair over SIO as (0,1)(2,3)...; the even die of each pair is PCIe-attached and ~20-25% faster. Prefer four whole free packages so a neighbour's job cannot contend over the shared path. |
| metric | fastest-rank median | `[RUN] rank N: eff_us ... median=`. The headline `effective_us` is a per-round max across ranks and carries start skew. |
| rounds | 100 (harness default) | |

`--golden-data` requires `--start-pos` whenever TP != 1, so replay-based A/B
always passes it.

The box is shared. Two habits are load-bearing:

- **Interleave in ABBA order inside one allocation.** Running A then B in every
  round lets load drift masquerade as an effect.
- **Discard runs whose median is inflated while the min is not.** A kernel
  change cannot leave the fastest round untouched and double the median; that
  signature is a co-tenant. Several runs per session show it.

## Where the time goes

Level-4 chip swimlane, 16x256, fastest rank:

```
makespan 1044.7 us
  compute 862 us (82.5%)
  stall   183 us (17.5%)   data-wait 154 / front-gap 24 / core-wait 5
static CPM path 782 us over 28 tasks
```

Top of the observed path:

| task | compute us | % makespan |
|---|---:|---:|
| kv_score_proj | 232 | 22.2 |
| hca_stream_merge_pack_publish | 123 | 11.8 |
| hca_raw_attn | 92 | 8.8 |
| hca_gather_kv | 61 | 5.8 |
| o-projection tail + hc_post (14 hops) | 201 | 19.2 |

### The engine split is what actually constrains it

The chip has **24 AIC and 48 AIV cores**. Occupancy across the makespan is AIC
42.5% / AIV 37.6% -- neither engine is saturated overall. The kernel is
phase-serialised instead:

| window us | AIC% | AIV% | running |
|---|---:|---:|---|
| 196-326 | 98 | **0** | kv_score_proj, kv_proj_matmul, qproj_matmul |
| 522-588 | 86 | 93 | hca_cmp_qk_pv, hca_raw_attn |
| 653-783 | **0** | 100 -> 58 | hca_stream_merge_pack_publish |
| 979-1045 | **0** | 35 | hc_post, tp_o_rs_reduce |

Three floors follow:

| floor | value | what it takes to reach |
|---|---:|---|
| engine capacity | 444 us | AIC 10647 us / 24 cores |
| static CPM | 782 us | perfect scheduling, dependencies unchanged |
| measured | 1045 us | |

The 782 -> 444 span reads like dependency headroom but mostly is not. Every
attention step needs `kv_score_proj` + `kv_proj_matmul` + `qproj_matmul` done
first; that is 4933 us of cube work, so 24 AIC cores need **at least 205 us**.
They currently span ~261 us, i.e. already **79% packed**. Reordering them is
worth ~55 us at best, and they cannot move into the idle cube windows after
653 us because attention consumes their output at ~457 us.

**The real lever on this path is cube work volume, not scheduling.**

## Landed

- **#1155 -- coarsen the kv_score_proj grid.** One block per (token tile,
  output tile) meant a 512-token step ran 256 blocks of ~10 us and refetched
  each `[OUT_TILE, K_TILE]` weight tile once per token tile, 32 times over.
  Grouping `KV_SCORE_T_GROUP`(2) token tiles per block halves the weight
  traffic. Only the grid changes; the `[MM_B_TILE, OUT_TILE]` accumulator keeps
  its shape, so the cube tile stays row-compact. -4.0%, lower in all 3 pairs.

## Rejected, with the reason

Do not retry these without new evidence.

| change | result | why |
|---|---|---|
| `MM_B_TILE` 16 -> 64 in kv_score_proj | compile error | 16 rows is a fractal-pitch boundary; a narrowed non-compact accumulator trips `AccCompactValid`. Widen the grid, never the accumulator's row count. |
| `KV_SCORE_T_GROUP` 2 -> 4 | neutral | Same wall, but Static CPM rises 787 -> 912 us: it lengthens the dependency path for nothing. |
| kv_score_proj as persistent workers (48 or 32) | neutral | The kernel already uses all 24 AIC cores; the array is 89% busy in its window. Block granularity was never the constraint. |
| two-pass softmax in the merge kernel | +1.0% | `for stream_work in pl.range(cmp_work_count)` runs **once** -- `cmp_work_count` is 1 for any context under 16K. Hoisting the max out of a one-iteration loop only adds a scan. |
| `pl.reshape` to collapse the 16-row scatter in the merge kernel | golden FAIL | The style guide sanctions `pl.reshape` on a whole tile, not on a row slice of one. |
| drop the merge kernel's GM store/reload round trip | compile error | It is load-bearing: `pl.gather` (used for the RoPE pair swap) takes a TensorType, and the reload is what materialises the Vec tile as one. |
| fuse `tp_o_a_quant` into `tp_o_a` | not possible | Quantisation is per row over all `O_LORA`(1024) columns; `tp_o_a` produces `O_A_N_TILE`(128) columns per block. Holding a full row needs a `[128, 1024]` FP32 accumulator = 512 KB against a 128 KB L0C. |
| fuse `tp_o_b_dequant` into `tp_o_b` | not possible | Dequant reduces across `LOCAL_O_GROUPS`; `tp_o_b` runs inside `pl.parallel(LOCAL_O_GROUPS)`. The consumer aggregates the producer's parallel axis. |
| merge the `cp_token_allgather` hops | not possible | push / payload_wait / readback / readback_wait / retire are communication protocol phases; the waits block on remote ranks. |

## What is left

- **Cube volume.** The 4933 us of pre-attention matmul is the binding cost and
  is fixed by dtype and shape. Lower-precision weights or activations (INT8,
  FP4) are the only large lever, and they trade accuracy -- a model-side call.
- **Hop count.** The 183 us of stall is 28 path hops paying ~5.7 us each of
  dispatch tax, not one bad edge. Removing it means removing tasks, and the
  three obvious candidates above are all structurally blocked.
- `allow_early_resolve` on the o-projection tail measures -1.4%, inside the
  noise band but consistent in direction.

## Tooling notes

- The in-core simulator profiler cannot build a case for
  `hca_stream_merge_pack_publish`: its generator rejects the kernel's computed
  dynamic dimensions and wants a full PTOAS source checkout.
- Per-pipe cycle columns in an in-core instruction CSV **sum across concurrent
  pipes**, so a percentage of their total mixes parallel resources. Cycles
  attributed to `SET_FLAG` / `WAIT_FLAG` are mostly the blocked-at-issue time of
  the pipe they sit on, not the cost of the sync instruction.

## 2026-09-07 follow-up: request-contiguous HCA gather

This follow-up uses main `fa5bf7e` (including #1155 and #1159), not the older
`f069bc7` baseline above. The configuration is TP4, physical devices 4/6/8/10,
B16 per rank (64 requests total), S8, and context256. PyPTO f1bb086, simpler
15f5cbd, PTOAS 0.57, PTO ISA 96ba706, CANN 9.0.0.

| Independent attempt | Result | Decision |
|---|---|---|
| Guard a single 135-row physical gather instead of 8 x 16 + 7 x 1 row copies | Gather block median 47.07 -> 8.76 us; four-capture AICore-span median 889.47 -> 873.90 us | Retain for the requested profiled AICore end-to-end metric; unprofiled benchmark neutral |
| Skip full-window zeroing, keeping explicit invalid-row and partial-window initialization | AICore-span median 887.24 us; gather block median 48.80 us | No clear gain |
| Cast only 448 non-RoPE columns in merge | AICore-span median 901.86 us; merge block median 69.01 vs 67.60 us | Rejected |
| Runtime full-window branch skipping the raw-attention mask | Compile error: merged FP32 tile becomes column-major before row_sum | Excluded; independent small MIX repro recorded |
| Direct remote_store from the merge UB tile | Documented remote visibility differs from draining put | Not implemented |

The controlled unprofiled result for contiguous gather is **805.59 -> 804.46
us (-0.14%, neutral)**. Each value is the median of four fastest-rank medians;
each epoch contains 100 timed rounds after 5 warmups. One process/allocation
uses A B B A B A A B order and compiles each version once. Raw epoch medians:

- Main: 809.81, 793.02, 801.37, 813.50 us.
- Contiguous gather: 1422.27, 802.50, 806.43, 799.58 us.

No epoch was discarded. Excluding the paired first epoch still gives no win
(801.37 -> 802.50 us). A preliminary single CLI pair looked faster
(814.6 -> 792.4 us); do not use that pair as the final performance claim.

All retained full-layer captures and benchmark validation gates pass. Exact
standalone gather checks also cover fragmented, invalid-slot, short, and
growing-to-full windows. The whole sparse-attention simulator still stalls
on unmodified main, so use it for compilation only and use hardware for the
complete layer. Initial baseline benchmark/runtime_dir controls timed out;
subsequent CLI controls and live-compiled ABBA checks passed. Their cause is
unresolved; do not claim that persistent execution itself is broken.

The older GM reload restriction in the rejected-attempt table predates the
full-tile `pl.tile.gather` implementation now on main. It should not be confused
with the separate remote_store visibility restriction above.

No new PR was opened. Experimental code, all raw samples, and before/after
chip records are under `build_output/hca_work_reduction_20260907_2105/`;
`REPORT.md` records the conventions and limitations.

### Acceptance metric clarification

The user's target is the chip-swimlane AICore end-to-end duration. Contiguous
gather improves that primary metric: the four-capture median is 889.47 ->
873.90 us (-1.75%). The user reports viewer values of 888.98 -> 875.26 us
(-1.54%) for the compared traces; those viewer boundaries have not been
reconstructed by the analysis helper. Keep the helper measurements and viewer
readings separately labeled. The neutral unprofiled ABBA result is a separate
observation, not a reason to reject the measured profiled improvement. Retain
the candidate under the user's stated metric; the sub-600 us target is still
unmet.

### Fresh unprofiled recheck: task_20260907_233100_23307917338

The user requested another comparison with chip swimlane disabled and medians
as the statistic. Source, patch, frozen golden, devices, and pinned toolchain
are unchanged. Runtime flags were verified as chip_swimlane=0, PMU=0, and
dep_gen=False; neither fresh build produced chip records.

Four epochs per variant, 100 timed launches after 5 warmups each, in one
allocation/process with A B B A B A A B order:

| Epoch | Main median us | Contiguous gather median us |
|---|---:|---:|
| 1 | 825.27 | 818.83 |
| 2 | 805.25 | 800.58 |
| 3 | 788.95 | 797.11 |
| 4 | 799.69 | 814.34 |
| Median of epochs | 802.47 | 807.46 |

The new result is **802.47 -> 807.46 us (+0.62%)**:
two paired epochs are faster and two are slower, so this unprofiled recheck
does not establish a gain. All samples are retained; all eight correctness
gates pass. This is separate from the earlier positive profiled AICore-span
measurement. Do not replace the older raw results with this new series.

Report, submitted command, verified runtime flags, exact patch, and all raw
per-rank samples: `build_output/hca_gather_no_swimlane_20260908_063059/`.

### PR submission: #1170

The user accepted the profiled AICore-span improvement and requested a PR.
The subsequent single-pair unprofiled benchmark was cancelled while pending
(task_20260907_233902_6072075156), before any device execution or result.

PR: https://github.com/hw-native-sys/pypto-lib/pull/1170
Commit: `2512de230b1d7d13094941b684835391d7247032`. One commit, one kernel file.
The contiguous-gather patch exactly matches the measured candidate; upstream
main was still fa5bf7e at submission. All pre-commit hooks passed. The PR
records the four-capture profiled median 889.47 -> 873.90 us (-1.75%) and
explicitly retains both unprofiled series, which show no consistent gain.
No additional performance run was made for this submission.

## 2026-09-08 follow-up: raw mask bypass and UB direct publication

Baseline is `2512de2`: main `fa5bf7e` (including #1155 and #1159) plus the
contiguous-gather change submitted as PR #1170. Both new directions are
independent patches on that baseline, not stacked on each other. All runnable
comparisons use a2a3 TP4, B16 per rank (64 requests total), context256, S8,
localT128, and the same frozen input/golden. Pins remain PyPTO f1bb086, simpler
15f5cbd, PTOAS 0.57, PTO ISA 96ba706, CANN 9.0.0.

The user permits odd or even cards. Each completed comparison holds devices
1/3/5/7 in one allocator reservation. Each version is benchmarked once, with
5 warmups and 100 measured launches, chip_swimlane=0, PMU=0, dep_gen=False.
Report the lowest per-rank median of effective_us, with every measured sample
retained. This supersedes the earlier median-of-four-epochs preference for
these new experiments. One separate level-4 capture per version follows the
benchmarks; its AICore span includes internal gaps and starts at the first
real AICore task. It is not a repeated-capture median or a stability claim.

### First implementations

Experiment: `build_output/hca_raw_publish_20260908_065541/REPORT.md`.
Completed task: `task_20260908_001050_117243912457`, devices 1/3/5/7.
The earlier fixed-even reservation was cancelled while pending, without a run.

| Version | Unprofiled median us | Profiled AICore span us | Target block median us |
|---|---:|---:|---:|
| Baseline | 875.30 | 900.40 | Raw AIC 51.34; merge AIV 67.65 |
| Full raw head-loop branch | 892.73 | 890.36 | Raw AIC 49.06 |
| Per-token UB remote publication | 898.28 | 950.46 | Merge AIV 57.58 |

The raw full-window branch removes mask load/broadcast/bias/exp-mask work when
current_len==WIN. The partial-window branch preserves the original behavior.
The complete QK/softmax/PV head loop is duplicated so tiles never cross the
branch boundary. Both versions pass the raw-only simulator Torch reference
for full, short, growing-to-full, and empty windows, with bitwise-identical
m/l/o outputs. Full-layer hardware benchmark and profile gates pass. The raw
block is smaller, but the unprofiled median regresses 1.99%; do not claim a
benchmark improvement.

Per-token direct publication removes attention_grouped local GM stores and
subsequent put reloads, sending the two output-group tiles from UB instead.
The merge block shrinks, but both the layer median (+2.63%) and captured layer
span worsen. Each four-token pack now incurs eight publishing releases instead
of the original two. The follow-up below retains four-token transfers.

### Narrow raw branch: device assembly failure

Experiment: `build_output/hca_raw_publish_v2_20260908_0015/REPORT.md`.
Task `task_20260908_001640_14480954875`, devices 7/9/11/13, exits 1 during CCE
assembly of the raw candidate. Its baseline-only median is 883.94 us; there is
no paired candidate result and no remote-publication run in this task.

Moving row_sum and the BF16 cast inside the softmax branches lets PyPTO/PTOAS
codegen pass, and the raw simulator still passes exact baseline comparisons.
However, the branch-merged BF16 tile has ColMajor/RowMajor layout and TPUSH
tries to store it into an ND MIX FIFO. CCE fails with the PTO ISA source/dest
layout static assertion. Do not mistake compile-only or simulator success for
a successful hardware build. The existing local MIX-layout issue is updated
with a standalone device-free assembly reproducer:
`KNOWN_PYPTO_ISSUES/branch_bf16_matmul_fifo_layout.py`. The original FP32
row_sum repro remains valid. No compiler fix or new toolchain was used.

### Four-token UB publication

Experiment: `build_output/hca_publish_batch_20260908_0019/REPORT.md`.
Completed task: `task_20260908_001956_193507718226`, devices 1/3/5/7.

Assemble four token rows for each of the two output groups in two [4,4096]
BF16 UB tiles, then issue two remote stores per pack. Every row is written
before publication. This removes the local GM staging round trip while
preserving the original four-token transfer granularity. Same-block notify
and consumer wait/readback/retire phases are unchanged.

| Version | Unprofiled median us | Profiled AICore span us | Merge AIV block median us | Merge window us |
|---|---:|---:|---:|---:|
| Baseline | 1211.72 | 967.68 | 67.83 | 78.86 |
| Four-token UB publication | 880.90 | 887.56 | 63.13 | 76.32 |

**Qualification:** The measured pair is positive, but the nominal 27.30%
benchmark drop is not an established optimization gain. Both versions vary
substantially within the run. The selected baseline rank's consecutive
ten-sample medians range from 874.2 to 2714.3 us; the candidate's range from
851.6 to 4202.9 us. Every sample is retained, and the cause of the variation
is not isolated. Earlier baselines in other allocations are not substituted.
The captured target block shrinks 4.70 us (6.93%); the kernel's full execution
window shrinks 2.54 us. These are distinct from the single captured layer-span
improvement of 80.12 us. No stable whole-layer gain or sub-600 us result has
been established. Both benchmark gates and both profile gates pass all four
original full-layer output comparators.

### Correction to the previous remote-store deferral

The 2026-09-07 entry deferred direct remote_store because the bare operation
is non-draining. Inspection of the current pinned compiler provides new
evidence: InsertCommFence adds system.fence after publishing writes; the
backend emits peer-region cache handling and the generated C++ contains
TSTORE, DCCI, PIPE_ALL, and DSB_DDR before the existing TNOTIFY. Thus these
candidates use compiler-inserted releases, not a bare remote_store or a
manually assumed PIPE_ALL remote-DDR fence. Generated sequences are retained
in each remote candidate's release_sequence.txt. Hardware golden validation
passes both publication implementations. No hand-added barrier remains.

The root checkout and PR #1170 are unchanged. All optional patches, raw samples,
commands, source hashes, chip records, and failure logs remain in their isolated
experiment directories. Retain the four-token publication patch for review;
do not present these data as a proven stable 27.30% gain.

### User-requested PYPTO_BENCH_RAW=1 recheck

The prior publication comparison already enabled raw sample printing. At the
user's explicit request, a fresh single baseline/candidate pair was run with
PYPTO_BENCH=1, PYPTO_BENCH_RAW=1, 5 warmups and 100 measured launches.
Task `task_20260908_003655_350998321654` completed with exit 0 on devices
1/3/5/7. Source patches, pins, TP4/B16 per rank/context256/S8, and frozen golden
are unchanged. Chip swimlane, PMU, and dep-gen remained off throughout.

Fastest-rank medians: baseline **1291.38 us**, four-token UB publication
**885.96 us**, observed difference **-405.42 us (-31.39%)**. All four original
output comparators pass for both versions. Both raw rank lists contain all
100 measured launches and exactly match the recorded full-precision samples
up to the harness's one-decimal print rounding. No sample is excluded.

Timing variation remains substantial: selected baseline rank min/max are
821.68/10553.62 us, candidate 836.08/6770.60 us. The candidate's first five
consecutive ten-sample medians are 2933.4, 2920.0, 1395.9, 3129.6, 2866.8 us;
its last five are 878.8, 878.0, 868.7, 880.1, 878.8 us. The full observed 31.39%
drop cannot be attributed to the optimization without isolating this
variation. Do not combine this pair with previous allocations or present it
as a new chip AICore-span measurement.

Report, printed raw lines, all per-rank sample CSVs, exact submitted command,
source hashes, imported module paths, and verified runtime flags:
`build_output/hca_publish_bench_raw_20260908_003606/`. No PR or tracked source
was changed for this recheck.

### User-requested even-card raw benchmark

Even cards were initially occupied. Task `task_20260908_004125_394407112645`
queued for 4/6/8/10 and then completed with exit 0 after allocation. No other
job was interrupted. The baseline is 2512de2; the candidate contains exactly
the same four-token UB publication patch. Pins, frozen golden, TP4, B16/rank,
context256, S8, 5 warmups and 100 measured launches are unchanged.
PYPTO_BENCH_RAW=1 is verified, and profiling/PMU/dep-gen are off throughout.

Fastest-rank medians are **797.97 -> 784.81 us (-13.16 us, -1.65%)**.
All four output comparators pass for both versions. Both versions have all
4 x 100 samples; printed raw values match the stored samples to the harness's
one-decimal rounding. No samples were discarded and no chip capture was run.

The selected baseline rank's consecutive ten-sample medians range from
794.9 to 801.1 us; the candidate's range from 781.7 to 786.8 us. These are more
concentrated than the previous odd-card pair. Some outliers and other-rank
delays remain. This single pair supports a modest positive median result,
not a stable 31.39% improvement or a new profiled AICore-span claim.

Artifacts: `build_output/hca_publish_bench_raw_even_20260908_004125/REPORT.md`,
`raw_samples.log`, `results.json`, per-version sample CSVs, exact submit command,
verified revisions and runtime flags. The root checkout and PR are unchanged.

### PR #1170 publication commit

At the user's request, the exact even-card-tested four-token UB publication
patch was appended to PR #1170 as a separate commit `1f724c3b1ffaeea72a1160a1032ba2b7ef9de55f`.
The original gather commit 2512de2 remains unchanged. The branch contains two
commits and changes only decode_sparse_attn_hca.py and decode_hca.py. All
pre-commit hooks pass; the committed source hashes match the measured candidate.
No additional performance run was needed for this byte-identical submission.

The title is now `Perf: coalesce DSpark HCA KV gathers and output publication`.
The description preserves the gather-only chip and unprofiled results and
adds the incremental even-card median 797.97 -> 784.81 us (-1.65%), with the
single-benchmark 100/5 convention, raw samples enabled, instrumentation off,
TP4/B16 per rank/context256/S8, and frozen-golden validation. It explicitly
states that these comparisons do not measure a combined gain against main
or establish repeated-run stability. The noisy odd-card percentages are not
presented as optimization gains.

Push and remote PR title/body/head/two-commit scope were verified. Saved PR
metadata before/after, commit message, staged patch, and final PR text are in
`build_output/hca_publish_bench_raw_even_20260908_004125/`.

### One direct-model chip capture after the PR update

The user requested one swimlane capture of PR #1170. Fixed-even task
`task_20260908_012200_36938365100` remained pending and was cancelled before
execution. Under the user's prior odd/even-card authorization, replacement
`task_20260908_013801_417227913462` used allocator-selected devices 3/5/7/9
and completed with exit 0. Exactly one model invocation was captured, using
commit 1f724c3b, PYPTO_BENCH=0, chip level 4, TP4, B16 per rank, context256,
S8, and the unchanged frozen input/golden. No benchmark or baseline capture
was added. All four original output comparisons pass.

All four rank artifact sets pass level-4 row/dependency/name completeness
checks. First-real-AICore-start to last-AICore-end spans, including internal
gaps, are rank0 8145.78 us, rank1 976.38 us, rank2 1474.48 us, and rank3
3305.04 us. The selected shortest span is rank1/device5: **976.38 us**.
Its gather AIV block median is 9.33 us; raw AIC block median 59.14 us;
merge/publish AIV block median 55.68 us, with a full kernel window 69.46 us.
This is one profiled capture on a different card set, not a measurement of
speedup against the even-card benchmark or a sub-600 us result.

The copied chip records, dependencies, name maps, and merged viewer traces
are under `build_output/hca_pr1170_chip_20260908_012159/chip_swimlanes/`.
REPORT.md, summary.json, submitted commands, allocation metadata, and the
full validation log are retained in that experiment directory. No source or
PR description changed for this capture.

### Required even-card direct-model chip capture

The user explicitly corrected the card requirement: use even-numbered cards
for this capture. This requirement takes precedence over the older odd/even
permission; do not fall back to odd cards for this requested work.
Device6 was occupied, so task `task_20260908_015010_235522925642` reserved
4/8/10/12 and completed with exit0. The script and analysis both enforce the
exact even device set. One direct-model level-4 capture, PYPTO_BENCH=0,
commit1f724c3b, TP4, B16 per rank, context256, S8, and unchanged frozen golden.
All four original output comparators pass, and all four chip artifact sets
pass completeness and freshness validation.

AICore spans are rank0/device4 8343.86 us, rank1/device8 858.76 us,
rank2/device10 14932.60 us, and rank3/device12 11223.92 us. The selected
shortest first-real-AICore-start through last-AICore-end span is **858.76 us**,
including internal gaps. The selected merge AIV block median is60.83 us,
with a full kernel window76.40 us. This is one capture, with no new baseline;
do not interpret the difference from the preceding odd-card capture as a
measured optimization gain.

Use `build_output/hca_pr1170_chip_even_20260908_015010/` for this result.
The generic hca_pr1170_chip_current.txt pointer now selects this even-card
capture. Selected chip and merged viewer JSON files are copied with the
complete metadata under chip_swimlanes/dfx_outputs/rank1/d0/. No source or
PR description was changed for this capture.


### 2026-09-08: independent compressed QK/PV movement experiments

Artifacts: `build_output/hca_cmp_movement_20260908_021553/`.
Baseline is PR1170 head1f724c3. Three isolated patches: conditional neutral-O
initialization, direct Acc PV stores with contiguous [token,work,head,D]
partial-O layout, and a short-context32-row compressed tile with the original
128-row fallback. Both partial-O consumers are updated for direct stores.
The raw-attention tile and m/l producer stores are unchanged.

Final performance task `task_20260908_022714_299577620865` runs on even
cards4/6/8/10. TP4, B16/rank (64 total requests), context256, S8, frozen golden.
Exactly one 100/5 benchmark per version in one process/allocation, RAW1,
chip/PMU/dep-gen off while timing. All samples are retained and verified against
printed raw lists. One separate level4 capture per version follows using the
live compiled program. All four existing full-layer comparators pass each
version, both in the benchmark and the profile dispatch.

| Version | Lowest rank median us | Captured AICore span us | CMP AIC block median us |
|---|---:|---:|---:|
| Baseline | 794.93 | 848.34 | 88.87 |
| Skip neutral O for valid tokens | 782.39 | 865.28 | 82.91 |
| Direct Acc PV stores | 752.87 | 828.04 | 77.54 |
| Short compressed tile32 | 769.10 | 841.12 | 73.88 |

The direct-store candidate shows the largest observed benchmark reduction,
5.29%, and a lower captured layer span. Skip-zero's captured layer span rises
despite its lower benchmark median; do not call every metric positive. These
are independent changes, not a combined result or repeated-run stability claim.
The sub600us target is not reached. Generated direct-store PTO eliminates both
PV Acc-to-AIV pushes and concat, while retaining the QK-score/probability FIFO.

**Precision blocker discovered by strict boundary checks:** all eight original
extracted probes pass partial O, but fail m/l identically in the baseline and
all candidates. Cases include empty/negative lengths, invalid pages, partial
visibility, short/long paths, and two work items. Exposing m/l led to a reduced
42-line hardware reproducer, `KNOWN_PYPTO_ISSUES/strided_column_store.py`.
Task `task_20260908_024442_55780819287` on card3 fails 29/128 values at exact
zero tolerance. A ColMajor [16,1] store into ND [16,8] writes contiguously rather
than using destination row stride8. The same form exists in full-model m/l
stores. PTO ISA's one-column cross-layout exception routes it to the DN
contiguous transfer. The normal full-layer comparator passing does not clear
this stricter failure. Recorded as high severity in root KNOWN_PYPTO_ISSUES.md.
An attempted metadata consumer also exposes a separate assembly layout failure;
its build-only reproducer is `KNOWN_PYPTO_ISSUES/sliced_column_store_layout.py`.
No frontend workaround, toolchain edit, combined candidate, commit, or PR update
was made. Keep the patches pending until the precision gate is resolved.

The user explicitly allowed odd cards again during this turn. The performance
job had already started on evens and completed there; card3 was used only for
correctness diagnosis. Future card selection may use odd or even cards, while
before/after performance must share the same reserved set.

An earlier baseline allocation on0/2/12/14 failed its first dispatch (AICPU/AIV
error plus HDC disconnect and failed teardown), producing zero samples and no
candidate execution. It is archived under failed_allocation_0_2_12_14 and is
not used in any comparison. The existing simulator scheduler hang also
reproduces on the extracted unchanged CMP kernel; its reduced evidence was
added to the existing issue. The tracked root checkout and PR1170 are clean
and unchanged.


### PR1170: direct compressed-PV accumulator stores submitted

After the precision finding was disclosed, the user explicitly requested
pushing the selected performance commit to PR1170 and documenting its timing.
The exact measured direct_store patch was appended as separate commit
`fe06be84ab034f56f9c235f34477ce8b434dfe4c`. The original2512de2 and1f724c3 commits are unchanged;
the PR now has three commits. No skip-zero, short-tile, m/l workaround,
toolchain change, or generated artifact is included.

The committed two-file source hashes match the measured candidate exactly.
All pre-commit hooks pass. The existing PR merges cleanly with current main;
its measured base was preserved when appending the commit. Remote branch,
commit history, PR title/body, and clean worktree were verified.

The description preserves the earlier gather and publication results and
adds the incremental direct-PV-store comparison over1f724c3: one100/5 benchmark
per version on devices4/6/8/10, TP4, B16/rank, context256, S8, frozen golden,
RAW1, instrumentation off. Lowest per-rank median794.93 ->752.87us (-5.29%).
Separate single-capture chip span848.34 ->828.04us (-2.39%); CMP AIC block
median88.87 ->77.54us; kernel window92.34 ->80.42us. All samples are retained;
no combined gain against current main or repeated-run stability is claimed.
The original full-HCA comparisons pass, but the unchanged m/l row-stride
precision failure remains unresolved and is explicitly disclosed in the PR.
Submission does not clear that precision gate.

PR: https://github.com/hw-native-sys/pypto-lib/pull/1170
Commit/message/staged patch, local hook log, remote PR snapshots, and
verification: `build_output/hca_cmp_movement_20260908_021553/pr1170_direct_store_submission/`.


### 2026-09-08: PR1170 retest on merged PR1172

Artifacts: `build_output/hca_pr1172_retest_20260908_040247/`.
Exact base: merged PR1172 `0d39ad3c8452aec0f731b15cac8804865434e0ba`.
Three isolated worktrees: unchanged base; base plus2512de2/1f724c3;
base plus2512de2/1f724c3/fe06be8. All cherry-picks apply cleanly with no
additional source changes. PR1170 and the root checkout remain unchanged.

Task `task_20260908_040356_6081632630` completes exit0 on cards4/6/8/10.
Same pinned environment and frozen golden as the previous CMP study. TP4,
B16 per rank (64 requests total), context256, S8, localT128. Exactly one
100-round benchmark after5warmups per version, RAW1, chip/PMU/dep-gen off;
all samples retained and cross-checked with raw output. One separate level4
capture per version reuses its live compiled program. All four full-model
comparators pass in every benchmark and capture. All12 rank artifact sets
pass row-count, dependency, name, and level checks.

| Version | Lowest rank median us | Captured AICore span us |
|---|---:|---:|
| PR1172 base | 797.17 | 870.40 |
| Plus first two commits | 757.63 | 819.38 |
| Plus all three commits | 773.45 | 817.12 |

First two together lower benchmark median4.96%. The third commit increases
it15.82us (+2.09%) relative to first_two in this run, despite lowering CMP
AIC block median84.21->70.87us and slightly lowering chip span819.38->817.12us.
Do not carry forward the older-base third-commit5.29% gain to PR1172, claim
stable regression from this single benchmark, or equate a smaller kernel
with an unprofiled layer speedup. All chip selections are rank3/device10;
benchmark fastest-rank PIDs differ and all rank distributions are preserved.
Sub600us target is not reached. The previously disclosed m/l row-stride
precision defect is unchanged and was not retested or resolved here.

Reproduce using this experiment's submit.sh/env.sh/bench.py; inspect REPORT.md,
summary.json, raw_samples.log, and per-version chip_swimlane/dfx_outputs.


### 2026-09-08: diagnose third-commit retest regression without rerunning

Artifacts: `build_output/hca_pr1172_retest_20260908_040247/third_commit_diagnosis/`.
The official all100-sample lowest per-rank medians remain757.63->773.45us.
The chosen worker has5->31 samples above1000us. In all_three's final20
launches, selected PID108649 slows to median1605.32us while PID108661
is fastest in19/20 launches and has median753.93us. Diagnostic per-round
minimum medians are757.63->762.96us; this is NOT a replacement benchmark
statistic or a speedup claim. First80 selected-worker medians still differ
756.99->769.92us, so the tail alone does not explain all15.82us.

Both level4 rank3/device10 traces put CMP on the Observed critical path.
CMP block median84.21->70.87us, full window133.66->126.64us. Assigned-core
sequencing changes:8->12 of16 CMP AIC blocks run after raw attention.
Core23 CMP starts10.86us later but executes14.08us shorter, ending3.22us
earlier. First CMP producer-end/start gap12.06->15.34us and merge window
74.32->77.02us absorb much of the saving. Observed path contributions
664.92->635.66us and gaps154.46->181.46us reconcile819.38->817.12us.
These are single-capture facts, not a quantitative explanation of unprofiled
15.82us. The44 other generated kernel C++ bodies are identical after comment
removal; logical dependency pairs and task launch policies match.

The original benchmark retained effective_us arrays but discarded parsed
STRACE span inputs and did not retain PID-to-device mapping. Exact unprofiled
orch/sched/rank-start attribution is unavailable retrospectively. Do not
claim proven stable code regression, a global raw-attention dispatch blocker,
or a demonstrated new-base unprofiled benefit. A follow-up must retain full
spans and explicit worker mapping. No new device run, source edit, or PR edit.


### 2026-09-08: defer third commit, keep first two in PR1170, requested rerun

The user requested excluding the third commit and saving it for later, then
explicitly requested a fresh performance test. Commitfe06be8 is preserved on
local branch `backup/pr1170-third-commit-fe06be8-20260908_042354` and in
`local_archive/hca_pr1170_third_commit_20260908_042354/` as a format patch and
verified one-commit Git bundle. Its old/new-base experiments and diagnostic
artifacts remain intact. Restore later with `git cherry-pick fe06be8`.

PR1170 now contains exactly two commits rebased onto merged PR1172
0d39ad3c:9289afd9 (contiguous gather) andd781332c (four-token UB publication).
The third commit is absent. Remote rewrite used an explicit fe06be8 lease
and its resulting head/body/two-commit history were verified. The original
PR worktree is clean atd781332c; the root checkout is untouched. All
pre-commit hooks pass. No additional kernel changes were introduced.

Fresh task `task_20260908_042455_82620427939` completes exit0 on4/6/8/10.
Artifacts: `build_output/hca_pr1170_first_two_20260908_042354/`.
Same a2a3 TP4, B16/rank (64 requests), context256, S8, frozen golden and pins.
One benchmark per version,5warmups and100 measured launches, RAW1, chip/PMU/
dep-gen disabled. Baseline792.199us -> first_two757.0195us, a35.1795us
reduction (-4.44074%) in the lowest per-rank median. All samples retained;
all400/version checked against raw output. Both cache hits and all four
full-model comparators pass. Full STRACE text and parsed invocation spans
are retained after timing. No new chip capture was made in this rerun.

The PR title/body now describe only the two retained optimizations, list
both separate new-base comparisons797.17->757.63us and792.20->757.02us,
and retain the earlier same-source chip870.40->819.38us with explicit
separation from unprofiled timings. The unchanged m/l stride precision
issue remains disclosed. No direct-PV-store gain is claimed by the PR.


### 2026-09-09: skip zero initialization on the contiguous raw-gather branch

Base: main 216b497e4b7d8bb7b47e304e79fd59e01de2a2e9, including PR1170,
PR1172, and PR1183. Artifact root: build_output/hca_gather_skip_zero_20260909_025233/.
The candidate moves only the [135,512] BF16 zero fill into the fallback branch;
the fully checked contiguous branch overwrites its complete destination directly.

Successful task: task_20260908_200244_291966222357, physical devices3/5/7/9.
a2a3 TP4, B16 per rank (64 requests total), start-pos256, S8, localT128.
One benchmark per version, 5 warmups +100 timed rounds, PYPTO_BENCH_RAW=1,
chip_swimlane0, PMU0, dependency generation off. All samples retained.
A fresh baseline golden was saved; candidate input/output replay cache hits verified.

| Version | Lowest per-rank median us |
| --- | ---: |
| main216b497 | 884.35 |
| Skip zero fill on contiguous gather | 865.19 |

Observed change: -19.16us (-2.17%). This is one paired benchmark, not a stability
claim. Other rank distributions include large variation and tail spikes; complete
STRACE and all400 parsed invocations per version are retained. No chip swimlane
was collected in this experiment. Do not infer a measured per-kernel duration.

Both versions pass the four existing full-HCA comparators. Exact isolated gather
validation passes eight cases: contiguous, fragmented, invalid slot, short,
growing, empty, tail gap, invalid first slot. Full-model passing does not resolve
the previously recorded compressed m/l stride issue; that code is unchanged.

Generated C++ confirms baseline TEXPANDS+TSTORE before the guard, while candidate
emits both only in the fallback. With comments and generated inline ID suffixes
normalized,47/48 generated kernels are identical; only hca_gather_kv differs.
Each eligible request removes138240 bytes of logical initialization writes.

The first pending fixed-card reservation was cancelled without a run. A probe
callback signature error was corrected. Another attempt passed both L2 probes
but failed L3 worker initialization before timing; the fresh L3-only process
then succeeded. Keep single-device probes in a separate process from the
multi-card benchmark driver. These setup failures contributed no timed samples.

The patch is retained at candidate/source/ and candidate/source.patch beneath
the artifact root. No commit or PR was created. CMP neutral-O and explicit
AIC/AIV pipeline candidates were not tested by this experiment.


### 2026-09-09: publish gather skip-zero as PR1185

The measured gather-only patch is published as commit
1d35cdef3460aa4a5679b8a1bd71aa31b66dcd62 in
https://github.com/hw-native-sys/pypto-lib/pull/1185. The PR records the
profiling-off lowest-rank median 884.35 -> 865.19 us (-2.17%), one100/5
comparison on3/5/7/9, TP4,16requests/rank,start-pos256,S8,frozen golden.
Full repository pre-commit checks passed. Generated artifacts remain local.

Next experiment isolates conditional CMP partial-O initialization against
this PR head; it must show an incremental median reduction before submission.
It does not restore the archived direct-Acc-store patch from PR1170.


### 2026-09-09: conditional CMP output initialization after PR1185

Baseline is PR1185 gather-only commit1d35cde. The candidate keeps all m/l
initialization and QK/PV arithmetic, but guards neutral partial-O stores with
a per-token predicate equivalent to the existing valid-computation condition.
Existing m/l strided-column stores are unchanged; this does not resolve that issue.

Task task_20260908_202410_15499127829, same allocated cards3/5/7/9 for both
versions. TP4,16requests/rank (64total),start-pos256,S8,localT128. Chip0,
PMU0,dep-genFalse,PYPTO_BENCH=1,PYPTO_BENCH_RAW=1. One100-round benchmark
with5warmups per version; both replay the original gather experiment's
frozen inputs and outputs. All400samples per version match captured STRACE.

Lowest per-rank median: 855.6595 -> 864.79us, +9.1305us (+1.0671%).
No incremental gain: archive only, no new commit and no push. Do not treat
the earlier gather measurement's865.19us as this allocation's baseline.
There is substantial inter-rank variation; this is one comparison, not a
stability claim. No chip swimlane was captured.

Both variants pass all four existing full-model comparators and twelve exact
partial-O boundary cases. The probe uploads sentinel37 and uses zero queries
so outputs are exact sums of visible BF16 rows; it checks complete writes,
empty/invalid tokens and pages, and visibility boundaries across two work
items. It does not expose or validate the known m/l issue.

Codegen:46/48 kernels unchanged after comments/whitespace/inline-ID
normalization; only hca_cmp_qk_pv AIC/AIV files change. Conditional O stores
are emitted, removing16MiB/rank of logical writes for this workload; added
scalar checks are present. Without a kernel profile, do not assign a proven
cause to the end-to-end regression. Fewer writes did not yield a measured win.

Artifacts: build_output/hca_cmp_skip_zero_20260909_032102/
REPORT.md,decision.json,results.json,sample_verification.json,source patches,
full STRACE,codegen diffs,probe scripts,and reproduction commands retained.
PR1185 remains gather-only at1d35cde.


### 2026-09-09: whole-tile Q RoPE gather experiments after PR1185

Both candidates target qproj_dequant_rms_nope_rope against gather-only
PR1185 head1d35cde. The full8-token path's tensor gather currently lowers
to an8-row loop containing TGATHER and TMOV. The proposed indices add
row offsets once per work item and are reused across four heads.

- Explicit tile version converts full-path loads/stores to tile APIs and
  supplies reduction/high-precision-rsqrt scratch operands. The gather
  directly covers[8,64], removing the row loop and row copies.
- Flattened tensor version keeps the existing dequantization, reduction,
  high-precision rsqrt and stores; reshapes[8,64]to[1,512], gathers once,
  then reshapes back. Generated code has one full-width gather and copy
  per head, with no8-row gather loop.

Both compile successfully;47/48 generated model kernels match baseline
after comments/whitespace/inline-ID normalization. Only
aiv/qproj_dequant_rms_nope_rope.cpp changes.

Each candidate has a separate profiling-off comparison on the same
physical cards3/5/9/11 within its allocation. TP4,16requests/rank (64total),
start-pos256,S8,localT128; PYPTO_BENCH=1,PYPTO_BENCH_RAW=1,100rounds and
5warmups,chip0,PMU0,dep-genFalse. Every version benchmarks once per
comparison, replaying the same original gather experiment's input/output
fixture. All400samples per version match captured STRACE spans.

| Candidate | Baseline median us | Candidate median us | Difference |
| --- | ---: | ---: | ---: |
| Explicit tile gather | 845.9000 | 872.8105 | +3.18% |
| Flattened tensor gather | 858.4300 | 861.6095 | +0.37% |

The metric is the lowest per-rank effective_us median, not a mean or
cross-rank step latency. Both changes show no gain in these individual
comparisons; archive both and do not append a commit to PR1185. The small
0.37% increase is not a claim of repeatable regression. No chip profile
was captured and no per-kernel runtime improvement is asserted.

Validation: all four existing full-HCA comparators pass every benchmark
version. The extracted full Q-dequant kernel is separately checked with
16full tokens and a13-token call at output offset8 (full tile plus tail).
Random input accumulators, scales, angles, a zero head, identity and
row-specific permutations exercise the new full-tile indexing. Both
candidates match baseline BF16 outputs exactly, including the untouched
prefix sentinel37. Both baseline and candidates also pass Torch checks
(rtol0.01,atol0.02); the before/after equivalence gate has zero tolerance.
Baseline Q outputs are cached and reused after passing. This does not
resolve the unrelated existing compressed m/l strided-column-store issue.

Setup only: the first even-card reservation was cancelled while pending.
Two own-code tile-API mismatches were fixed before any model benchmark
(high-precision rsqrt needs an explicit tmp; Tile output uses pl.store).
Their logs and source states are archived under the explicit-tile root.
The first successful candidate benchmark is task_20260908_231347_3434461976;
the flattened comparison is task_20260908_231708_358202422591.

Artifacts:
- Explicit tile gather: build_output/hca_qproj_tile_gather_20260909_060727/
- Flattened tensor gather: build_output/hca_qproj_flat_gather_20260909_061625/

Each root retains REPORT.md,decision.json,all samples,STRACE,source patches,
Q validation outputs,generated code,and runnable reproduction scripts.
PR1185 and the user checkout remain unchanged.


### 2026-09-09: PR1185 gather check after merged PR1187

Baseline is merged main cec12680d61fba51199d0c82166cb641b3b194c1.
Candidate adds the exact gather-only patch from PR1185 head1d35cde; no
qproj or attention-pipeline edits are stacked. Two detached worktrees
preserve the root checkout.

Successful task task_20260908_233437_1537594128 on cards3/5/7/9. TP4,
16requests/rank (64total), start-pos256,S8,localT128. One benchmark per
version,100rounds/5warmups,PYPTO_BENCH=1,PYPTO_BENCH_RAW=1, chip0,PMU0,
dep-genFalse. Same frozen fixture; full four-output comparators PASS
for both variants. All400samples/version match captured device STRACE.

Lowest per-rank effective_us median:796.80 ->843.19us, +46.39us (+5.82%).
No incremental gain measured after1187; no new commit. This single
comparison does not establish a repeatable regression. Baseline rank
medians are4903.7000,4956.5800,1285.1005,796.8000us; candidate rank medians
are4711.0205,843.1900,4090.9100,2479.7295us. All samples are retained.
The selected rank changes between variants; no cross-rank maximum or
per-round rank-minimum is substituted for the agreed metric.

Only aiv/hca_gather_kv.cpp changes;47/48 generated kernels match after
normalization of comments, whitespace and inline identifiers. Reuse of
eight exact gather boundary tests is backed by identical gathered source
bodies, constants and pinned toolchain. Specs, input generators and
golden reference functions remain unchanged, making replay eligible.
No chip swimlane or per-kernel latency claim accompanies this test.

Initial task task_20260908_233151_1118725944 on0/2/12/14 failed baseline
first dispatch with S1:running-stalled,61/110completed,task4294967329,
HDCdisconnect/507901 and teardownexit139. It produced no benchmark
samples and did not execute candidate. Existing local runtime issue
updated; no source workaround. Retry uses a different authorized card
set for BOTH versions.

Artifacts: build_output/hca_1187_gather_20260909_063143/REPORT.md,
results.json,sample_verification.json,codegen_comparison.json,
fixture_eligibility.json,gather_validation_reuse.json,source.patch,
full STRACE and reproduction scripts. The initial failure is archived
in failed_allocation_0_2_12_14/. PR1185 description updated to distinguish
the old-main gain from the new-main result; branch/head unchanged.

PR1185 was closed at2026-09-09T07:25:22Z on the user's explicit request after the post1187 comparison showed no incremental gain. The branch and archived experiment remain available.


### 2026-09-09: PR1187 raw/CMP cause investigation with PMU and ablations

Compared216b497e withcec12680, frozen TP4 HCA fixture,16requests/rank,
start256,S8. Chip0,PMU event2,BENCH0: diagnostic task cycles, not an
unprofiled benchmark. Primary pair4/6/8/10; ablations6/8/10/14. Full
existing model comparators PASS for every completed dispatch. The known
old m/l stride issue is not cleared. All12completed captures/48rank CSVs
are retained, including two successful captures before a driver module-name
collision caused an abnormal exit. No new kernel commit or PR was made.

Old AIV subblock0 performs almost all useful vector work; subblock1 uses
empty tiles but maintains the pipe protocol. New split_aiv(2) shares the
heads. The64-head matmuls reduce measured AIC MTE1 busy cycles about47-48%
while Cube busy work falls about14-15%. Old Cube busy fractions are about
14-16%, not evidence that all remaining cycles are synchronization wait.
MTE2 activity increases in the full new implementation.

Using median of per-rank task-cycle medians for diagnostics: removing only
raw cross-query lookahead lengthens AIC tasks61269.5->84382.5cycles, all4ranks,
with unchanged MTE1 work. NewCMP with16workers instead of24 gives
74062->107130.5cycles, all4ranks, unchanged total MTE1 work. Context256 has
only one compressed attention block; a no-lookahead CMP test has mixed
rank directions and no demonstrated benefit from multi-block overlap.

Old CMP two-half PV plus concat forces PV results through AIV before final
stores. A full512-column PV at old32-head/16-worker settings removes that
path and duplicate probability consumption:154328->112720.5task cycles,
Cube busy essentially unchanged, MTE1 only~1.65%lower. A concat-only removal
also eliminates the PV AIV transfer, reducing AIV MTE2 busy work73.86%; its
small task-cycle reduction is not isolated from control-kernel variation.
These later ablations use separate reservations of the same cards; no
additive attribution to the full PR or end-to-end speedup is asserted.
Old raw width-only edit shows no improvement. Old raw full64-head tiling
with its original pipe exceeds the documented UB budget. All failed
variants, corrected diagnostic scripts, generated code, and limitations
are retained. No in-core simulator timing was collected or used.

Artifacts: build_output/hca_1187_cause_20260909_091306/REPORT.md,
all_counters.json,counter_summary.csv,lowering_evidence.json,source patches,
and exact run/submit scripts. Never infer a wait percentage from1-CubeBusy.


### 2026-09-14: merge coefficient normalization and pack rotation

Base: main d6f2920046df7a19c28a5079c096a53b7e600219. Artifact root:
`build_output/hca_merge_tuning_20260914_030144/`. Same pinned PyPTO f1bb086,
simpler15f5cbd, PTOAS0.57, PTO ISA96ba706, CANN9.0.0.

Two independent changes touch only hca_stream_merge_pack_publish:

- Normalize alpha/beta by the final denominator before multiplying the two
  large O tiles. This removes the full16x512 output division in favor of
  two1x16 coefficient divisions. The actual generated code has two TDIVs
  and no TROWEXPANDDIV; FP32 operation ordering changes.
- Rotate each worker's head-group destination between complete four-token
  packs. Each pack remains covered once, the transfer size and notification
  counts remain unchanged, and the sink is reloaded per pack. The original
  3-versus2 pack-count imbalance remains; this redistributes destinations.

Task task_20260913_200722_299021626714 passes on3/5/7/15. TP4,
16requests/rank (64total),context256,S8. One benchmark process per version,
5warmups+100samples, BENCH1,RAW1,chip0,PMU0. All variants replay the same
passing main input/golden snapshot and pass all four existing HCA comparators.
All400samples/version match the printed raw arrays; full-precision statistics
and full STRACE are retained. No samples were removed or benchmarks repeated.

| Version | Minimum per-rank median us | Change |
| --- | ---: | ---: |
| Baseline | 771.7605 | — |
| Coefficient normalization | 775.9695 | +4.2090us (+0.5454%) |
| Pack rotation | 788.7105 | +16.9500us (+2.1963%) |

Neither candidate established an unprofiled end-to-end improvement in this
comparison; do not present these as stable regressions either. The combined
candidate was prepared but not run because neither component improved the
baseline. No commit or PR was created and the root main checkout is unchanged.

The independent device probe passes256rows over eight float64-reference
cases (both branches, raw only, compressed only, empty, dominant sink,
dominant raw, dominant compressed, cancellation), at rtol3e-5/atol2e-6.
Exact rotation coverage passes128token counts at TP2/TP4. After stripping
comments/whitespace,45/46 emitted kernel C++ files match the baseline; only
the targeted merge/publish kernel differs. These checks do not clear the
pre-existing producer m/l store-stride issue.

Fresh level-4 captures completed on7/9/11/13 in task
`task_20260913_202344_398315229709`. One capture per version, BENCH0,PMU0,
the same frozen golden, all four model comparators PASS. Fastest-rank results:

| Version | Fastest rank/device | Layer AICore span us | Merge window us | Merge block min / median / max us |
| --- | --- | ---: | ---: | --- |
| Baseline | rank3/device13 | 793.94 | 73.04 | 31.02 /60.30 /72.88 |
| Coefficient normalization | rank3/device13 | 821.16 | 70.62 | 32.20 /61.30 /70.56 |
| Pack rotation | rank2/device11 | 847.84 | 67.38 | 42.96 /60.82 /67.26 |

Rotation makes the bars more even: duration range41.86->24.30us and merge
window73.04->67.38us. Nevertheless, neither the captured layer span nor the
unprofiled median improves. Do not count shorter or more uniform kernel bars
as a layer speedup. Single captures do not establish stable regressions or
the cause of timing changes in other kernels. Archive both patches; do not
promote either as an end-to-end performance commit from these measurements.

All12 rank folders include CPM_static.json and CPM_observed.json. Joined
physical task rows match both raw streams and dependency block/slot counts.
The full archive is `chip_swimlanes.tar.gz`; the three selected fastest ranks
are in `chip_swimlanes_fastest.tar.gz`. Benchmark numbers above are profiling
OFF on3/5/7/15 and must not be mixed with the profiling-ON numbers here.
The two fixed-card reservations were cancelled while pending and executed
no device work. No benchmark was repeated and no sample was discarded.


### 2026-09-14: even-card retest of merge normalization and pack rotation

User requested even cards. Task `task_20260913_204253_295192827247`
completed exit 0 on **4/6/8/10**, all benchmarks and profiles in the same
reservation. Artifact root: `build_output/hca_merge_even_20260914_034058/`.
Base remains main d6f29200; source patches, pinned dependencies, workload and
frozen input/golden are unchanged from the preceding experiment. All 46
emitted C++ kernels per version match the previous corresponding version
apart from comments/whitespace. No root-main edits or commit/PR were made.

One benchmark process per version, 5 warmups + 100 samples per rank,
BENCH1/RAW1/chip0/PMU0. Fastest-rank effective-time medians:

| Version | Median us | Change |
| --- | ---: | ---: |
| Baseline | 693.5400 | — |
| Coefficient normalization | 684.1195 | -9.4205 us (-1.3583%) |
| Pack rotation | 687.4200 | -6.1200 us (-0.8824%) |

All 400 samples per version were checked against raw arrays; all output
comparisons passed. These small gains belong to this even-card comparison.
The preceding odd-card results remain valid observations and showed no gain;
neither set proves a stable gain or regression. The combination was not run.

Separate single captures on the same even cards, BENCH0/chip4/PMU0:

| Version | Fastest rank/device | Layer AICore span us | Merge window us | Block min / median / max us |
| --- | --- | ---: | ---: | --- |
| Baseline | rank3/device10 | 731.70 | 74.54 | 37.62 / 64.82 / 74.42 |
| Coefficient normalization | rank3/device10 | 740.28 | 74.94 | 33.28 / 63.19 / 74.82 |
| Pack rotation | rank1/device6 | 730.16 | 76.32 | 46.56 / 65.10 / 74.78 |

Rotation narrows the duration range 36.80 -> 28.22 us but does not shorten
the merge window or longest block in this capture. The earlier odd-card
merge-window reduction did not reproduce here. Do not substitute profiling
spans for the unprofiled benchmark medians or infer causality from these
single captures. All profiles passed validation; all 12 rank folders have
CPM_static/CPM_observed and verified raw/joined/dependency row counts.
`chip_swimlanes_fastest.tar.gz` contains the three selected fastest ranks,
with complete critical-path files; `chip_swimlanes.tar.gz` retains all ranks.
Exact commands, environment, raw results and patches are linked in REPORT.md.


### 2026-09-14: combine coefficient normalization and pack rotation

Artifact root: `build_output/hca_merge_combined_even_20260914_035956/`.
Task `task_20260913_210032_49966718458` completed exit 0 on even cards
4/6/8/10. Reused the previously prepared combined worktree at main d6f29200;
verified it equals applying normalization to the rotation source. Same pinned
environment, TP4, 16 requests/rank, context256, S8 and saved passing golden.
No baseline or single-candidate measurement was repeated; those references
are the preceding even-card task, so this compares separate reservations.

One combined benchmark: 5 warmups + 100 samples/rank, BENCH1/RAW1/chip0/PMU0.
Fastest-rank effective median: **693.58 us**, versus retained main693.54,
normalize684.1195 and rotate687.42 us. Combined adds0.04us over main and
9.4605us (1.3829%) over normalization. The gains did not add; no combined
benefit was measured. This single comparison does not prove a stable regression.
All four output comparisons pass and all400samples match printed raw arrays.

One combined level-4 capture, BENCH0/PMU0: fastest rank3/device10 layer
span744.30us, merge window73.30us, block min/median/max47.06/61.74/72.36us.
The merge window is1.24us shorter than retained main but layer span is12.60us
longer. Do not attribute this difference to a specific wait or changed kernel
without further evidence. All capture validation checks pass. All4rank folders
include CPM_static/CPM_observed and agree with raw/joined/dependency row counts.

45/46 PTOAS-emitted C++ bodies match each reference after comments/whitespace
removal; only merge/publish differs. Runtime wrapper inline identifier numbers
also differ and are outside that body comparison. No kernel source was edited
in this follow-up. The combined patch remains an experiment, with no commit/PR
and no root-main change. Report, commands, raw logs, source patch and archives
are retained. `chip_swimlanes_fastest_comparison.tar.gz` contains the four
versions' fastest ranks with complete critical-path files.


### 2026-09-14: submit normalization only as PR #1212

User explicitly selected the normalize-only candidate for submission.
PR: https://github.com/hw-native-sys/pypto-lib/pull/1212
Commit: `21081e9ca97f65926f73ba9ff2eaac858f17409b`.
Branch: `perf/normalize-dspark-hca-merge-coefficients`.
PR workspace: `build_output/hca_normalize_pr_20260914_040939/source/`.

The PR adds one commit touching only the nine-line normalization diff in
`models/deepseek_v4_flash_dspark/decode_hca.py`. No rotation or combined change
is included. The source is byte-identical to the tested normalize candidate.
The PR base is upstream main3e29b7db (#1207), which adds the V4.1 model and
documentation; the HCA tree and golden harness match tested based6f29200.
Existing full-device validation and the arithmetic probe are reused, with
no new benchmark repetitions. All pre-commit hooks pass. Root main is unchanged.

The PR body states profiling OFF, even4/6/8/10, TP4,16requests/rank,
context256,S8,100samples/5warmups, RAW1 and fastest-rank effective median
693.54->684.12us (-1.36%) from one comparison. It also records the independent
profiling-ON span731.70->740.28us and explicitly says that span did not improve.
The earlier odd-card and combined experiments remain archived separately.
PR title/body, local hook log, metadata and the verified GitHub snapshot are
retained in the PR artifact directory. The PR is open and ready for review.


### 2026-09-14: fresh unprofiled HCA test of current PR #1212

User requested a new performance comparison with chip swimlanes disabled.
The current PR had acquired a second commit: head461f74ed contains both
normalization21081e9 and O-B publication workers24->32. The new comparison
therefore measures the full current PR against main3e29b7db, not normalization
in isolation. Snapshot, source patch and detached revisions are retained.

The initially submitted even-card task task_20260913_231731_184663920842
was cancelled while pending, with no device runs, after the user explicitly
authorized odd cards. Completed task task_20260913_233612_30705929161 used
3/5/7/9 for both versions in one reservation and exited0. Artifact root:
`build_output/hca_pr1212_bench_odd_20260914_063604/`.
The source worktrees were prepared under the earlier even-named artifact
directory and reused via symlinks; both actual device lists are3/5/7/9.

Same established HCA environment: PyPTO f1bb086, simpler15f5cbd, PTOAS0.57,
PTO ISA96ba706, CANN9.0.0; full pin chain reverified. TP4,
16requests/rank (64total), context256, S8, same passing frozen main golden.
One process per version,5warmups+100samples/rank, BENCH1/RAW1.
Chip, dep-gen, PMU, argument dump and scope stats were all checked off before
calling the unchanged model entry. Neither new built case contains a chip
swimlane record or dependency-capture spec. No profiling capture was run.

| Version | Revision | Fastest-rank effective median us |
| --- | --- | ---: |
| Main | 3e29b7db | 785.41 |
| Current PR #1212 | 461f74ed | 764.40 |

Reduction21.01us (2.6750%). Both output/golden cache hits and all four model
comparators pass. All400samples/version match printed raw arrays; no sample
was discarded and no extra group was collected. These are fresh before/after
measurements from one comparison, not substitutions of the earlier684.12us.
The full-PR gain cannot be assigned solely to normalization or O-B workers.
All four rank medians, full precision samples, STRACE, exact commands,
DFX configs and revisions are linked in REPORT.md. Source worktrees and root
main remain unchanged. No PR description or commit was modified by this test.


### 2026-09-14: queue the even-card comparison of PR #1212

User explicitly requested queueing after repeated availability checks.
Task `task_20260914_004328_303171426607` was submitted for4/6/8/10; status at check `pending`.
Artifacts: `build_output/hca_pr1212_bench_even_20260914_074258/`.
It compares fresh main3e29b7db and PR461f74ed benchmarks in the same
reservation, one process per version,5warmups/100samples,chip/dep-gen/PMU off.
The task automatically verifies environment/source pins, runs both versions,
validates all raw samples and writes REPORT.md plus completion.json on success.
The original frozen golden and detached source worktrees are reused; no source
or PR changes were made. No new performance result is claimed at queue time.
The submission is detached, without a wait-client timeout that would cancel
pending work. Maximum execution time after allocation is1800seconds.

## 2026-09-14: PR #1212 retest accepts any four even cards

The user relaxed the fixed 4/6/8/10 allocation to any four even cards from
0/2/4/6/8/10/12/14. Cancelled pending fixed task
`task_20260914_004328_303171426607` before execution. Submitted host-only
selector `task_20260914_005418_379087222539`; it requests one four-card device task
through task-submit when an eligible group is unreserved. Both versions
will run in that same allocation. No global queue policy was changed.

Artifacts: `build_output/hca_pr1212_any_even_20260914_074842`.
Main 3e29b7db vs current PR #1212 461f74ed; a2a3, TP4, context 256,
16 requests per rank, S=8, frozen golden, PYPTO_BENCH=1,
PYPTO_BENCH_RAW=1, profiling off, one process/version, 5 warmups and
100 samples/rank. Compare the fastest rank's median. The queued device
script verifies pins and automatically generates the comparison report.
No new performance result is claimed by this queue entry.

### 2026-09-14: recovered the even-card submission

The host selector failed at 08:20:39 UTC when its nested task-submit call
was rejected. task-submit explicitly prohibits nested submission inside a
daemon task. This was an error in the local orchestration script; the model
had not run. The captured stderr was lost by the original exception handler;
the rejection is confirmed by the task-submit entry guard and the daemon
forcing TASKQUEUE_INSIDE=1. Do not resubmit this nested selector.

At 08:31 UTC, seven even cards were unreserved. Submitted the prepared
comparison directly through task-submit on 2/4/6/8, task
`task_20260914_013119_224651818387`. Allocation and dependency preflight passed;
the baseline compilation started. Both variants remain in the same device
reservation, with profiling off. Recovery and failed-selector evidence are
preserved under `build_output/hca_pr1212_any_even_20260914_074842`.

### 2026-09-14: completed current PR #1212 on even cards 2/4/6/8

Task `task_20260914_013119_224651818387` completed both versions in the same
reservation. Main 3e29b7db fastest-rank median: 685.3495 us. Current PR
461f74ed fastest-rank median: 689.0995 us. Change: +3.7500 us (+0.5472%).
This single even-card comparison did not reproduce a benchmark gain.
It measures the full PR (normalization plus O-B publication workers 24 to 32),
not either commit in isolation. Do not attribute the difference to a kernel
without a separate controlled comparison.

Both versions passed all four HCA output comparators. Verified all 800 timed
samples against printed raw arrays; chip swimlane, dep-gen, PMU, argument dump
and scope stats were off. One process/version, 100 rounds after 5 warmups,
frozen golden, context 256, TP4, 16 requests per rank, S=8.
PyPTO f1bb086 / simpler 15f5cbd / PTOAS 0.57 / PTO ISA 96ba706 / CANN 9.0.0.
Report and full evidence: `build_output/hca_pr1212_any_even_20260914_074842`.
No source, commit, or PR description was changed by this retest.

## 2026-09-14: DSpark three-attention benchmark at context 128K

The user requested SWA, CSA and HCA performance based on main. Fetched and
pinned upstream main c3f0dea274f55d9648920f17c968e8564ae9fcdc in an isolated
worktree under `build_output/dspark_main_ctx128k_c3f0dea2_20260914_084509`.
Configuration: a2a3, TP4, 16 requests per rank (64/group), S8, every
start-position 131072; profiling off, raw benchmark samples on, one timed
process/case with 5 warmups and 100 rounds. Fresh 128K inputs and golden
outputs will be saved. Do not reuse the earlier 256-context fixture.

SWA/CSA sources remain main unchanged. HCA needs a two-line declaration/bind
fix in l3_decode_hca: cmp_kv uses CMP_BLOCK_NUM_DYN instead of a fixed 256
pages. The 128K fixture requires 2048 pages per card. The patch is archived;
there are no arithmetic, tiling or scheduling changes. HCA performance must
be qualified as main plus this shape fix. Host compile-only and full fixture
ABI check passed; compiled cmp_kv shape is [4,-1,32,1,512]. The device run
will reuse that compiled artifact if its allocated card IDs match.

Queued task `task_20260914_014855_373456229393` requests 2/4/6/8. Any four even
cards remain acceptable. All three cases will run in one reservation. The
script automatically saves per-rank samples, checks all DFX flags and actual
position IDs, and writes REPORT.md and benchmark_summary.json. No measured
128K result is claimed until correctness and benchmark verification pass.

The SWA and CSA 128K host compile-only/ABI checks also passed. All three
compiled artifacts are retained and can be reused by the device wrapper when
allocated device IDs match 2/4/6/8. These checks ran no NPU kernels and are not
performance or device-precision results. Task remains queued as of this entry.

### 2026-09-14: 128K device results on main

The client-side watcher cancelled the pending 2/4/6/8 task before execution
and submitted task `task_20260914_015737_416341411362` for available even
cards 0/4/6/8. All three entries ran once in the same reservation. Because
these device IDs differ from the host compile-only checks, each device run
compiled an artifact for its actual allocation.

Fastest-rank median effective time, profiling OFF:
- SWA: 567.2800 us, both output comparators PASS, all 400 samples verified.
- HCA: 1252.4790 us, all four output comparators PASS, all 400 samples verified.
- CSA: no benchmark result. Its six mapped compressor/cache outputs PASS,
  but x_out fails ratio_reldiff: worst rdiff=1.4 versus max_diff_hd=1.0
  (diff_thd=0.004, pct_thd=0.008). The harness stops before timed rounds.

Main revision c3f0dea2. HCA alone includes the documented two-line distributed
cache shape fix. TP4, 16 requests/rank, S8, start-pos131072 everywhere.
Fresh 128K input/golden data saved, no context-256 fixture reused. Successful
benchmarks use 5 warmups and 100 samples/rank; all DFX off. Pins match the
recorded f1bb086/15f5cbd/PTOAS0.57/ISA96ba706/CANN9.0.0 environment.

CSA's arithmetic and comparator are main unchanged. Its root cause is not
localized; no compiler/toolchain attribution is established. No tolerances
were relaxed, no timing was forced past validation, and no extra benchmark
was collected. The failing fixture is preserved only for precision repro,
not promoted as a passing reference. Failure details, complete logs, and
validated SWA/HCA samples are in the experiment directory's REPORT.md,
benchmark_summary.json, and csa/failure.json.

### 2026-09-14: CSA 128K precision failure localized

Evidence: `build_output/csa_ctx128k_precision_20260914_021540/REPORT.md`.
CSA main c3f0dea2, same environment and original failing inputs. Recomputed
CPU golden matches the saved expected output exactly. All even cards were
reserved; cancelled our pending even-card task and used allocated odd cards
3/7/11/13 for diagnosis. Ordinary replay, two partial-dump captures and the
later benchmark validation dispatch have identical device x_out tensors.

Cause: tiny HC/QR numerical differences cross BF16/INT8 rounding boundaries,
change the indexer query, and change Top-K membership. 31 of 512 queries have
different selected sets (72 entry replacements total); exactly those 31
queries contain the output threshold violations. On three worst queries,
CPU scoring with the captured device query reproduces the device's Top-K
set exactly. CPU Hadamard/quantization from the captured BF16 query also
reproduces the device INT8 query exactly. No sorter or compiler fault is
established.

Concrete worst-token example: rank0/token111, QR channel41, the golden
scaled value 57.5000076294 rounds to 58, while CPU quantization of the
captured device matmul gives 57.4999847412 and the actual device value 57.
This changes 594 indexer INT8 query elements and five selected KV entries.
For rank0/token2 and rank3/token66, captured HC gates reproduce BF16 mixed
rows exactly; coefficient differences of order 1e-8 cross BF16 rounding
boundaries and propagate to seven and ten selected KV replacements.

Original output gate: 17,207/8,388,608 bad elements (0.205123%, under the
0.8% budget), but two elements exceed the worst-rdiff cap. Worst actual
-0.0064224466 vs expected 0.0149355382; rdiff 1.3997168541. Supplying the
device Top-K indices to the otherwise original downstream CPU reference
reduces max absolute error 0.0228398442 -> 0.001906693 and produces zero
threshold violations. This is a diagnostic substitution, not a production
fix or proof that original validation passes. Tolerances remain unchanged.

At the user's explicit request to time the failing CSA before continuing
diagnosis, ran exactly one benchmark process on 3/7/11/13: profiling OFF,
PYPTO_BENCH=1/RAW=1, 5 warmups + 100 samples/rank, context131072, TP4,
16 requests/rank, S8. Fastest-rank median **5734.70 us**. Rank medians:
8403.9500, 7271.9495, 6862.6600, 5734.7000 us. Recorded the original
comparator failure before explicitly allowing timing to proceed; the JSON
result retains passed=false. Task task_20260914_022442_12592493605 completed.
Do not present this as a validated benchmark or compare it without qualification
to the earlier SWA/HCA measurements on a different card group.

Dump caution: under early resolve, a before-dispatch snapshot of a full
input tensor may contain unrelated rows still being written. Use the final
producer-completion dump for whole-tensor comparisons; the second capture
corrected that diagnostic artifact for QR. No production source changed;
only dump tags were added in the isolated build_output worktree. All three
device tasks completed and no queue reservation is left active.

### 2026-09-14: CSA 128K benchmark on even cards with precision gate bypass

User explicitly requested even-card timing while temporarily permitting the
known precision error. Main c3f0dea2, same compiled CSA and saved 128K inputs,
TP4, 16 requests/rank, S8, start-pos131072, all profiling off,
PYPTO_BENCH=1/RAW=1, five warmups and 100 samples/rank.

Completed on **0/4/6/8**, task `task_20260914_025614_252858217624`:
fastest-rank median **5615.38 us**. Rank medians: 8243.8610, 7548.8500,
6824.4605, 5615.3800 us. Verified all 400 samples against printed raw arrays,
the actual saved position IDs, and all five DFX options being off. Original
precision comparator still fails; its failure is recorded separately and
allowed to continue into timing. No production source or threshold changed.
The prior odd-card median was 5734.70 us; the difference is -119.32 us
(-2.08%), a device/run comparison rather than a code optimization.

Before this completed run, the pending 0/2/4/8 job was cancelled because card4
was allocated elsewhere. A 0/2/6/8 attempt then failed during its first
dispatch with device2 ACL507018 and subsequent mailbox cleanup errors; it
produced no validation or benchmark samples. That incident is preserved in
`failed_attempt_0268` and logged in KNOWN_PYPTO_ISSUES.md. The successful
0/4/6/8 run is a separate process; no samples were combined across attempts.

Report, source/fixture provenance, exact command and samples:
`build_output/csa_main_ctx128k_even_c3f0dea2_20260914_025158/REPORT.md`.
The successful task and observer are complete; no reservation remains active.


### 2026-09-14: DSpark TP1 batch16 context8192 baseline swimlanes

Artifact root: `build_output/dspark_tp1_b16_ctx8k_617e165f_20260914_184939`.
Main 617e165f, isolated checkout, same pinned f1bb086/15f5cbd/PTOAS0.57/
ISA96ba706/CANN9.0.0 environment. TP1 on even device6, 16 total requests,
8 decode tokens/request, all start positions8192 (128 active query tokens).
PYPTO_BENCH=0/RAW=0, chip swimlane perf level4, graph/timing separate.

Unmodified main fails all three entries at shared decode_o_proj_tp1
proj_a_mm with AccCompactValid. Apply the documented first-matmul peel,
explicit out_dtype=FP32, before accumulating remaining K tiles. This is a
compile prerequisite, not a measured optimization. Patch is archived as
`tp1_projection_compile_fix.patch`; root checkout is unchanged. Shape lists
must remain inline in pl.slice, and tensor-level matmul needs explicit FP32
for this accumulator (implicit dtype gave BF16).

Completed SWA and HCA captures, original precision checks PASS:
SWA AICore makespan606.72us (dispatch-to-finish610.90us), HCA722.14us
(dispatch-to-finish726.78us). These are profiled single-dispatch values,
not DFX-off benchmark medians. Their chip_swimlanes/{swa,hca} directories
include raw records, deps, names, merged trace, CPM_static/observed and full
critical-path reports. Validated all752 SWA and918 HCA physical rows against
both raw streams and dependency-declared block counts.

CSA compiled but default256MiB ring2 heap failed with
orch_error_code=2 HEAP_RING_DEADLOCK. Both depgen and timing failed and no
CSA timing from that attempt is usable. Own failed processes stalled during
HDC cleanup and were terminated after recording the classified failure;
HCA subsequently passed on the same allocated card. Generated ring2
allocations include a128MiB score arena plus large projection/indexer/attention
buffers. A retry uses ring_heap=(0,0,536870912,0); it first verifies the
capacity with a separate scope_stats replay, then captures level4 with
scope_stats OFF and original precision checks still enabled. Frozen8K
CSA data/build reused; no source-level heap workaround or tolerance change.

At this log entry, all even cards are held by an8-card allocation; CSA retry
is pending as task_20260914_185955_174166215804 on device6. It includes automatic
critical-path generation and REPORT.md/summary.json update after capture.
Consult those files and task status for completion rather than treating this
queued state as a CSA performance result.

Next candidate: TP1 proj_b_mm still computes512 padded rows for128 active
rows (32 blocks/group, eight groups). It is on both observed critical paths.
No before/after optimization benchmark yet; TP1 context128K remains unmeasured.


## 2026-09-14: TP1 8K / 128K tuning completed

The TP1 follow-up is complete; the preceding pending notes describe the
earlier capture stage. See [the Chinese results](dspark-hca-tp1-8k-128k-tuning-results-20260914.md)
and [the execution plan](dspark-hca-tp1-8k-128k-tuning-plan.md).

Four separate PRs retain active output-projection rows (#1226), grouped
compressed-KV gathers (#1227), TP1 K=128 compressed attention (#1228), and
TP1 UB merge/pack (#1229). On even device 12, the full combination changes
profiling-disabled HCA medians from 705.25 to 548.21 us at 8K and from
1197.98 to 753.38 us at 128K. Both sides include the required TP1 FP32
first-matmul compile fix. TP1, batch 16, eight decode tokens/request,
5 warmups and 100 in-process samples, frozen golden and original checks.

The standalone coefficient-normalization candidate is archived, not
submitted or stacked into the UB candidate: changes below 1% overlap the
sample variation. K64 is better than K128 for 8K in this sweep but worse
for 128K. All raw samples and unsuccessful attempts remain under
build_output/hca_tp1_tuning_20260914_195536. Do not reuse the older
TP4 work-count assumption or multiple-process best-run selection policy.

The published stack needs normal CI and merge handling: #1226/#1227 pass
the a2a3 job but hit existing PTOAS #1513 in a2a3sim; serving-dspark is
pending. #1228/#1229 use feature-branch bases to isolate their diffs and
need model CI after retargeting. No simulator test was suppressed.


## 2026-09-14: regroup TP1 tuning PRs without squashing commits

At the user's request, #1226 now contains the compile prerequisite,
active-row projection and grouped gather as three separate commits.
#1228 contains wider TP1 CMP tiles and UB merge/pack as two separate
commits, based on the combined #1226 branch. #1227 and #1229 are closed
as superseded. No commits were squashed and no performance was rerun: all
five stable patch IDs and both combined DSpark/golden source trees match
the previously measured variants. Full pre-commit checks pass on both.

The combined profiling-disabled medians are 701.46 -> 601.94 us (8K)
and 1197.17 -> 1038.99 us (128K) for #1226 on device 6; #1228 adds
613.96 -> 548.21 us (8K) and 1030.54 -> 753.38 us (128K) on device 12.
Commit mapping, before/after PR metadata and verification are retained
in build_output/hca_tp1_tuning_20260914_195536/regroup_prs.


## 2026-09-15: #1226 merged; #1228 rebased onto main

#1226 merged as f200ae1. Rebase #1228 onto that main with only its two
optimization commits: cdf74e6 (CMP tiles), 0373680 (UB merge/pack).
Its base is now main. Stable patch IDs and the sparse-HCA source match
the measured candidates. Main also contains #1212, so existing benchmark
figures remain explicitly attached to their original measurement base.
8K/128K compile-only checks and full pre-commit pass. No new device run
or benchmark. Evidence: build_output/hca_tp1_tuning_20260914_195536/
rebase_1228_20260915/.


## 2026-09-15: #1228 merged; subsequent CMP work is analysis only

#1228 merged into main as 1a9e487ce22a12c10e651b6b0f25fdf838068791.
The HCA sparse-attention file and golden tree match PR head 0373680.
The merged main also contains changes to decode_csa.py and decode_indexer.py.
No new performance measurement accompanies this status update; retain the
original measurement baselines. Future TP1 128K CMP work should start from this merged revision.
The user requested analysis before any changes: consider splitting the
current 5-6 queries per task into 2-3 while preserving the 128-row KV tile,
and investigate the late-starting CMP tasks separately. The 50-100 us
target concerns task granularity, not a proven whole-operator latency.
The existing 8K fast path has AIC tasks around 31-41 us in the historical
full-optimization capture; do not assume it needs the same split. No kernel
edits, new NPU tasks, or PR changes were made for this analysis.

## 2026-09-15: TP1 CMP query splitting measured; archive both candidates

After user authorization, test merged #1228 (`1a9e487c`) against two- and
three-query contiguous CMP blocks in isolated worktrees. Keep K=128, the
single-KV-tile fast path, and all per-query arithmetic. Give each logical
block independent transfer slots. The block counts are 24 -> 64 / 43 at
B16 S8 and context128K; transfer storage grows from12.41MiB to33.09/22.23MiB.

Same even device12 allocation, original pinned performance environment,
DFX-off, one process per configuration,5warmups/100samples, no sample
filtering.128K medians:747.64 ->743.389/745.49us (0.57%/0.29% lower).
8K medians:545.10 ->543.7695/544.94us. All six full-HCA validations and both
fragmented/invalid-page, mixed-history and query-tail boundary cases pass.
TP4 compile-only check passes; no TP4 performance measurement.

The q2 capture achieves AIC block durations59.54-75.66us (median68.14),
versus baseline140.98-197.02us (median162.09). However this does not reduce
total attention work. Profiled CMP windows348.14 ->212.82/251.34us and
HCA makespans882.58 ->772.98/796.48us reflect individual schedules; the
baseline capture has196.06us CMP AIC start spread. Do not substitute that
profiled gain for the small DFX-off median change. All five captures have
validated rows/block counts, both CPM files and full critical-path reports.

Archive both candidates without a performance commit/PR: the measured median
changes are small relative to the sample variation, and no stable win is
established. Artifacts and patches:build_output/hca_tp1_cmp_split_20260915_203011/.
Chinese report:dspark-hca-tp1-cmp-split-results-20260915.md.
Task task_20260915_203239_34667123538 completed exit0 and released device12.
