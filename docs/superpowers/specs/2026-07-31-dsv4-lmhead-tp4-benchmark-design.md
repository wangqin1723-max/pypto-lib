# DeepSeek V4 LM-Head TP4 Benchmark Design

## Objective

Add a projection-only DeepSeek V4 Flash LM-head benchmark on a branch based on
`dsv4-ascendc-decode-compare2`. Measure the TP4 path from already-normalized
hidden states to assembled FP32 logits and compare its latency with the supplied
AscendC trace.

The benchmark must exclude final RMSNorm and greedy sampling. It must preserve
the existing sampling-inclusive standalone fixture and must not change the
LM-head kernel algorithm.

## Reference Boundary

The reference artifact is `trace_view_a3_decode.json`. Within each captured
`ProfilerStep`, use the final decode model on NPU stream 240, model ID 47:

- Start: end of `RmsNorm`, task 59.
- End: end of `aclnnInplaceCopy_CastAiCore_Cast`, task 66.
- Exclude the later `aclnnArgMax_CastAiCore_Cast` event.

The three captured durations are:

| Profiler step | Duration (us) |
|---|---:|
| 1 | 779.316125 |
| 2 | 752.535125 |
| 3 | 780.2360625 |

The formal reference median is **779.316125 us**. Events from model ID 48 must
not be pooled into this statistic.

## Benchmark Workload

Use one four-rank TP group:

| Setting | Value |
|---|---:|
| Target | Ascend A2/A3 |
| TP world size | 4 |
| DP world size | 4 |
| Hidden size | 4096 |
| Vocabulary size | 129280 |
| Vocabulary shard per rank | 32320 |
| Active logit rows per owner | 8 |
| Gathered group rows | 32 |
| Weight dtype | BF16 |
| Logits dtype | FP32 |

The measured path includes:

```text
selected hidden publish
  -> TP hidden wait and gather
  -> local vocabulary-shard matmul
  -> owner logits publish
  -> TP logits wait and gather
  -> completion-counter clear
  -> FP32 logits
```

It excludes RMSNorm, HC head, greedy sampling, and ArgMax.

## Implementation Design

### Projection-only host entry

Add `l3_lm_head_projection` beside the existing `l3_lm_head` in
`models/deepseek/v4-flash/lm_head.py`.

The new host entry accepts only:

- `hidden_states`
- `lm_head_weight`
- `logits`
- `logit_row_indices`

It allocates the same hidden, logits, and completion-counter windows as the
existing fixture, then calls the existing `lm_head_test` entry on each rank.
It does not accept `sampled_ids` and does not call
`lm_head_with_sampling_test`.

### Tensor specifications

Extend `build_tensor_specs` with a keyword that controls whether
`sampled_ids` is present. The default remains sampling-inclusive to preserve
existing callers. Projection-only mode omits the sampled-ID output while
retaining the existing hidden, sharded weight, logits, and row-index fixtures.

`golden_lm_head` requires no algorithm change because it already writes sampled
IDs only when the tensor is present. Projection-only validation compares only
FP32 logits.

### CLI

Add `--projection-only` to the existing standalone CLI.

When it is set:

- select `l3_lm_head_projection`;
- build specs without `sampled_ids`;
- validate only `logits`.

Without the flag, preserve the current projection-plus-greedy-sampling behavior.
The formal workload uses `--tp 4 --dp 4 --num-tokens 8`.

## Error Handling and Contracts

Keep the current import-time and CLI checks for:

- supported TP and DP sizes;
- `DP_SIZE % TP_SIZE == 0`;
- CLI values matching import-time specialization;
- sufficient device IDs;
- valid active-token count.

Add a subprocess contract test specialized with `--tp 4 --dp 4` that verifies:

- TP4-derived tensor dimensions;
- `l3_lm_head_projection` exposes FP32 logits;
- the projection-only host signature has no `sampled_ids`;
- the existing sampling-inclusive host signature still has `sampled_ids`;
- projection-only tensor specs omit `sampled_ids`;
- default tensor specs retain `sampled_ids`.

## Verification

### Static and unit checks

Run:

```bash
python -m pytest tests/contract/test_deepseek_v4_flash_lm_head.py -v
python -m pytest tests/golden -q
python tests/lint/check_headers.py
python tests/lint/check_english_only.py
ruff check models/deepseek/v4-flash/lm_head.py \
  tests/contract/test_deepseek_v4_flash_lm_head.py
git diff --check
```

### Device correctness and structure

Compile and run projection-only TP4 on four A2/A3 devices. Validate FP32 logits
against `golden_lm_head`.

Collect one separate L2 swimlane run to confirm that the generated graph
contains:

- `lm_head_dispatch_push`
- `lm_head_dispatch_wait`
- `lm_head_dispatch_gather`
- `lm_head_matmul`
- `lm_head_combine_push`
- `lm_head_combine_wait`
- `lm_head_combine_gather`
- `lm_head_signal_clear`

The graph must not contain `lm_head_greedy_sample`.

### Formal timing

Use five discarded warmups and 100 measured rounds with L2 swimlane disabled:

```bash
PYPTO_BENCH=1 \
PYPTO_BENCH_WARMUP=5 \
PYPTO_BENCH_ROUNDS=100 \
PYPTO_BENCH_RAW=1 \
python models/deepseek/v4-flash/lm_head.py \
  -p a2a3 \
  --tp 4 \
  --dp 4 \
  --num-tokens 8 \
  --projection-only \
  -d "$TASK_DEVICE" \
  --enable-l2-swimlane 0
```

Report the slowest-rank effective-latency distribution, including median and
P90, plus:

```text
delta_us = pypto_median_us - 779.316125
ratio = pypto_median_us / 779.316125
speedup = 779.316125 / pypto_median_us
```

Keep all builds, logs, traces, and local summaries under ignored
`build_output/`.

## Acceptance Criteria

- The branch is based exactly on compare2 commit `a79c141`.
- TP4 projection-only mode validates FP32 logits on four devices.
- Projection-only mode contains no greedy-sampling task.
- Existing sampling-inclusive behavior remains available and covered.
- The formal run produces 100 parseable measured rounds after five warmups.
- The report compares the PyPTO median with the 779.316125 us reference using
  the same semantic start and end boundary.
- No generated artifact from `build_output/` is committed.

## Non-Goals

- Restoring LM-head into the 43-layer full-decode benchmark.
- Measuring HC head or final RMSNorm.
- Measuring greedy sampling or ArgMax.
- Retuning LM-head tiles, communication topology, or numerical behavior.
- Modifying the daily performance automation.
