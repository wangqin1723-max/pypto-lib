# DeepSeek V4-Flash W8A8 performance

> 归档日期：2026-09-15。来源：`pypto-lib/local_archive/20260817-main-cleanup/docs/models/deepseek_v4_flash_w8a8_performance.md`。
> 原文的实现、测量与计划保留其历史语境；阅读说明见[学习资料索引](../README.md)。

This page preserves the supplied operator-performance snapshot, records the
attention-only retest on the latest `main` branch, and records the even-card
retest for the branch that maintains the remaining MoE and decode performance
cases. It also records both the hybrid and PyPTO-pinned attention validations
of Simpler PR #1807 with PTO-ISA `83d01313`.

## Supplied snapshot

Configuration shown in the supplied table:

`TP=1, DP=EP=16, GBS=16×4, SeqLen=8K, MTP=1, EPLB`

| Operator / metric | Supplied PyPTO | AscendC | Supplied PyPTO / AscendC |
| --- | ---: | ---: | ---: |
| Attention CSA | 357 µs | 465 µs | 0.768 |
| Attention HCA | 261 µs | 307 µs | 0.850 |
| Attention SWA | 243 µs | 280 µs | 0.868 |
| MoE | 477 µs | 479 µs | 0.996 |
| Decode Main | 36.1 ms | 37.7 ms | 0.958 |
| Decode MTP | 1,162 µs | 1,268 µs | 0.916 |

The supplied snapshot does not identify its source revision, test time, exact
command, or aggregation statistic.

## Maintained MoE and decode branch

The following three performance cases are maintained on
`perf/dsv4-eplb-decode-logits-and-mtp-core`. Rebase that branch onto the latest
`main` before collecting a new result while keeping the branch name stable.
All three official performance measurements must use the fixed even-card set
`0,2,4,6,8,10,12,14`. Results from any other device set are sanity checks only
and must not replace or be compared directly with the official baselines.

| Performance case | Entry point | Branch runner / metric case |
| --- | --- | --- |
| MoE | `models/deepseek_v4_flash_mtp/moe.py` | Direct EP8 MoE benchmark |
| Decode Main | `models/deepseek_v4_flash_mtp/eplb_decode_logits.py` | `tools/run_dsv4_eplb_perf.sh --case decode-logits` (Compare3) |
| Decode MTP | `models/deepseek_v4_flash_mtp/eplb_mtp_core.py` | `tools/run_dsv4_eplb_perf.sh --case mtp-core` (Compare4) |

The latest-main attention retest below is separate from these branch-owned
cases.

## Maintained-branch even-card retest

The branch-owned cases were collected from 2026-08-12 13:38 to 13:46
(UTC+08:00) with the required ordered device set `0,2,4,6,8,10,12,14`. Each
result uses 5 warmup rounds followed by 100 measured rounds. Lower latency is
better.

| Performance case | Metric scope | Median | Comparison baseline | Difference | Selected logical rank / device | Validation |
| --- | --- | ---: | ---: | ---: | --- | --- |
| MoE | Minimum per-rank median | 481.9 µs | 477.0 µs (supplied snapshot) | +4.9 µs (+1.0%) | 7 / 14 | `x_next` PASS |
| Decode Main | Compare3 fastest-rank median | 35,813.4 µs | 36,144.680 µs | -331.280 µs (-0.917%) | 7 / 14 | Runtime PASS; numerical validation unavailable |
| Decode MTP | Compare4 fastest-rank compute-only median | 1,153.850 µs | 1,162.050 µs | -8.200 µs (-0.706%) | 7 / 14 | All four finite-output checks PASS |

The MoE entry point does not yet have a checked-in official metric parser. Its
value above is the minimum of the eight per-rank medians, consistent with the
operator-style EPLB reporting policy; it is not the generic max-rank
`effective_us` headline. Decode Main and Decode MTP were parsed by metric
contract `dsv4-eplb-v1`. The Decode MTP value contains only slot 0 compute
samples; the slot 1 cleanup median was 13.3 µs and is excluded.

Retest conditions:

- Source: `perf/dsv4-eplb-decode-logits-and-mtp-core` at
  `6bb518ae8d967e2d1e8f39d7a969e988133135b8`.
- Source freshness at collection time: `upstream/main` was `38c21b9`; the
  measured branch was 5 commits ahead and 8 commits behind it. This result is
  therefore the current branch snapshot, not a post-rebase result against that
  `upstream/main` revision.
- Platform: A2/A3 device with CANN 9.0.0.
- Workload: EP8, 16 experts per rank, 8 tokens, start position 8,192, and L2
  swimlane disabled. Decode Main and Decode MTP additionally used TP4.
- Routing: MoE used `--balanced-routing`; deterministic trace-hash routing for
  the decode cases is encoded in `eplb_fixture.py` and does not use a
  `--routing-mode` command-line argument.
- Toolchain: PyPTO `df3ac12`, simpler `3165cc8`, PTO ISA `83d0131`, and PTOAS
  0.57 on AArch64.
- Device health: card 0 reported healthy after its successful hot reset and
  passed a real-device single-card smoke test before this retest. All eight
  even-card allocations completed and released their locks normally.

The measured commands, run inside an existing eight-card allocation, were:

```bash
export PYPTO_BENCH=1 PYPTO_BENCH_RAW=1
export PYPTO_BENCH_WARMUP=5 PYPTO_BENCH_ROUNDS=100
export PYPTO_RUNTIME_LOG=error SIMPLER_DEVICE_STRACE_ENABLE=1
export PTO2_RING_TASK_WINDOW=262144 PTO2_RING_DEP_POOL=262144
export PTO2_RING_HEAP=2147483648

python models/deepseek_v4_flash_mtp/moe.py \
  -p a2a3 --ep 8 --experts-per-rank 16 \
  -d 0,2,4,6,8,10,12,14 \
  --layer-id 0 --num-tokens 8 --balanced-routing \
  --enable-l2-swimlane 0

tools/run_dsv4_eplb_perf.sh \
  --device 0,2,4,6,8,10,12,14 \
  --case all
```

The local raw artifacts were recorded under:

```text
build_output/dsv4_moe_ep8_perf/20260812T053703Z_even_after_reset/moe.log
build_output/dsv4_eplb_perf/20260812T054000Z_even_after_reset/results.tsv
build_output/dsv4_eplb_perf/20260812T054000Z_even_after_reset/rank-results.tsv
```

## Simpler PR #1807 with PTO-ISA 83d attention retest

The exact requested combination of Simpler PR #1807 at `a15632e3` and PTO-ISA
`83d01313` was collected from 2026-08-13 09:41:11 to 09:42:38 (UTC+08:00).
Each result used 5 warmup rounds followed by 100 measured rounds, and the
reported statistic is the `effective_us` median. Lower latency is better.

| Operator | Supplied PyPTO | Retest median | Difference from supplied | AscendC | Retest median / AscendC | Validation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Attention CSA | 357 µs | 346.7 µs | -10.3 µs (-2.9%) | 465 µs | 0.746 | `kv_cache` and `x_out` PASS |
| Attention HCA | 261 µs | 258.2 µs | -2.8 µs (-1.1%) | 307 µs | 0.841 | `kv_cache` and `x_out` PASS |
| Attention SWA | 243 µs | 244.6 µs | +1.6 µs (+0.7%) | 280 µs | 0.874 | `kv_cache` and `x_out` PASS |

Retest conditions:

- Task: `task_20260812_183029_395671115415`.
- Source: PyPTO-Lib `6741e824e50252e53ab4d70ed9d980a2317b592e`.
  The CSA, HCA, and SWA sources and Golden Harness were byte-identical to
  `main` at `77c78c99648bfcab6ed041c1f6a5a8f366fad205` at collection time.
- Platform: A2/A3 device with CANN 9.0.0, using even card 4.
- Workload: start position 8,192 with `B=4`, `S=2`, and `T=8`.
- L2 swimlane: disabled with `--enable-l2-swimlane 0`.
- Toolchain: PyPTO `36e53454`, Simpler `a15632e3` from PR #1807, PTO-ISA
  `83d01313`, and PTOAS 0.57 on AArch64.
- Runtime provenance: both A2/A3 runtime variants were rebuilt with the exact
  source stamp `a15632e3:pto-isa=83d01313` before the device smoke test.
- Validation: both `kv_cache` and `x_out` passed for all three operators.

This is an intentional hybrid validation. Simpler commit `a15632e3` normally
pins PTO-ISA `0cefc9a5`; only `pto_isa.pin` was changed to `83d01313` for this
experiment. PyPTO-Lib does not directly pin PTO-ISA: its selected PyPTO revision
chooses a Simpler submodule, whose `pto_isa.pin` selects PTO-ISA.

The measured commands, run inside one allocated even-card task, were:

```bash
export PYPTO_BENCH=1 PYPTO_BENCH_RAW=1
export PYPTO_BENCH_WARMUP=5 PYPTO_BENCH_ROUNDS=100
export PYPTO_RUNTIME_LOG=error

python models/deepseek_v4_flash_mtp/decode_csa.py \
  -p a2a3 -d 4 --start-pos 8192 --enable-l2-swimlane 0
python models/deepseek_v4_flash_mtp/decode_hca.py \
  -p a2a3 -d 4 --start-pos 8192 --enable-l2-swimlane 0
python models/deepseek_v4_flash_mtp/decode_swa.py \
  -p a2a3 -d 4 --start-pos 8192 --enable-l2-swimlane 0
```

## PyPTO-pinned Simpler PR #1807 backport retest

The corrected dependency-chain validation was collected from 2026-08-13
10:38:20 to 10:40:24 (UTC+08:00). It started from the Simpler revision selected
by the updated PyPTO checkout, applied only the five functional files from PR
#1807 as an adapted backport, and left `pto_isa.pin` unchanged. Each result used
5 warmup rounds followed by 100 measured rounds, and the reported statistic is
the `effective_us` median. Lower latency is better.

| Operator | Supplied PyPTO | Retest median | Difference from supplied | AscendC | Retest median / AscendC | Validation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Attention CSA | 357 µs | 346.6 µs | -10.4 µs (-2.9%) | 465 µs | 0.745 | `kv_cache` and `x_out` PASS |
| Attention HCA | 261 µs | 258.5 µs | -2.5 µs (-1.0%) | 307 µs | 0.842 | `kv_cache` and `x_out` PASS |
| Attention SWA | 243 µs | 243.1 µs | +0.1 µs (+0.0%) | 280 µs | 0.868 | `kv_cache` and `x_out` PASS |

Retest conditions:

- Task: `task_20260812_193813_384997828746`.
- Source: PyPTO-Lib `77c78c99648bfcab6ed041c1f6a5a8f366fad205`.
  The upstream branch advanced to `b7ee679` during the final audit, but that
  commit changed only MoE and prefill sources; the three measured decode
  attention entry points remained byte-identical.
- Platform: A2/A3 device with CANN 9.0.0, using even card 4.
- Workload: start position 8,192 with `B=4`, `S=2`, and `T=8`.
- L2 swimlane: disabled with `--enable-l2-swimlane 0`.
- Sampling: deterministic Python, NumPy, and Torch seed 1,807; 5 warmup rounds
  followed by 100 measured rounds with raw samples enabled.
- Toolchain: updated PyPTO `71020585`, Simpler base `3165cc89`, the five-file
  functional backport of Simpler PR #1807 commit `a15632e3`, PTO-ISA
  `83d01313`, and PTOAS 0.57 on AArch64.
- Backport identity: the binary diff SHA-256 was
  `7993add8c422efddd5ae2fa999673faa0380ec0137aad8a8faeabc4843c6720e`.
- Runtime provenance: all 27 CMake caches pointed into the isolated Simpler
  worktree. `pto_isa_build.json`, `pto_isa.pin`, and the managed PTO-ISA
  checkout all resolved to `83d01313`.
- Stream-reuse verification: the HCA probe reported `STREAM_COUNT=1` both
  before worker close and during close after multiple measured runs, confirming
  that PR #1807's same-image AICore stream reuse was active.
- Validation: both `kv_cache` and `x_out` passed for all three operators, and
  the task exited with status 0.

This is not a checkout of Simpler `a15632e3`. That commit is based on a newer
Simpler history whose PTO-ISA pin had already moved to `0cefc9a5`. Applying its
single-commit diff directly to `3165cc89` also produces conflicts. The retest
therefore used the adapted five-file functional backport on top of
`3165cc89`, preserving the dependency chain selected by PyPTO:

```text
PyPTO 71020585
  -> Simpler 3165cc89 + PR #1807 functional backport
    -> PTO-ISA 83d01313
```

The measured commands, run through a wrapper that set seed 1,807 before
executing each model, were equivalent to:

```bash
export PYPTO_BENCH=1 PYPTO_BENCH_RAW=1
export PYPTO_BENCH_WARMUP=5 PYPTO_BENCH_ROUNDS=100
export PYPTO_RUNTIME_LOG=error PYPTO_PERF_SEED=1807

python <deterministic-seed-wrapper> \
  models/deepseek_v4_flash_mtp/decode_csa.py \
  -p a2a3 -d 4 --start-pos 8192 --enable-l2-swimlane 0
python <deterministic-seed-wrapper> \
  models/deepseek_v4_flash_mtp/decode_hca.py \
  -p a2a3 -d 4 --start-pos 8192 --enable-l2-swimlane 0
python <deterministic-seed-wrapper> \
  models/deepseek_v4_flash_mtp/decode_swa.py \
  -p a2a3 -d 4 --start-pos 8192 --enable-l2-swimlane 0
```

The local raw logs and provenance metadata are under:

```text
.cache/pr1807-perf/20260813-updated-pypto-backport/
```

## Latest-main attention retest

Only CSA, HCA, and SWA are in scope for the `main` branch retest. MoE, Decode
Main, and Decode MTP are not included in this result set.

| Operator | Supplied PyPTO | Main median | Difference from supplied | AscendC | Main median / AscendC | Captured (UTC+08:00) | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Attention CSA | 357 µs | 356.6 µs | -0.1% | 465 µs | 0.767 | 2026-08-11 12:03:35–12:04:04 | PASS |
| Attention HCA | 261 µs | 271.0 µs | +3.8% | 307 µs | 0.883 | 2026-08-11 12:04:18–12:04:40 | PASS |
| Attention SWA | 243 µs | 255.5 µs | +5.1% | 280 µs | 0.913 | 2026-08-11 12:04:52–12:05:11 | PASS |

Lower latency and lower PyPTO-to-AscendC ratios are better.

## Main retest history

| Captured (UTC+08:00) | Main revision | PyPTO revision | CSA median | HCA median | SWA median | Result |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-08-11 09:25–09:27 | `d1cf017` | `009c368` | 358.6 µs | 270.3 µs | 255.6 µs | PASS |
| 2026-08-11 12:03–12:05 | `a92a983` | `016e69a` | 356.6 µs | 271.0 µs | 255.5 µs | PASS |

Relative to the preceding retest, the latest medians changed by -0.6% for
CSA, +0.3% for HCA, and less than 0.1% for SWA. The PyPTO revision also
changed, so these small differences should not be attributed solely to the
`pypto-lib` revision.

## Retest conditions

- Source: `main` at `a92a9830b9b311cc46017492aef5f68587814206`, verified
  against `upstream/main` before and after the tests.
- Platform: A2/A3 device with CANN 9.0.0.
- Workload: start position 8,192 with `B=4`, `S=2`, and `T=8`.
- Sampling: 5 warmup rounds followed by 100 measured rounds. The reported
  statistic is the `effective_us` median.
- Devices: CSA and HCA used even card 4; SWA used even card 2. Each test used a
  dedicated single-card allocation.
- L2 swimlane: disabled with `--enable-l2-swimlane 0`.
- Toolchain: PyPTO `016e69a`, simpler `3165cc8`, PTO ISA `83d0131`, and PTOAS
  0.54 on AArch64.
- Validation: both `kv_cache` and `x_out` passed for all three operators.
- Device health: card 0 reported an alarm during the preflight and was
  excluded. Cards 2 and 4 reported healthy and completed the tests normally.

The current entry points are:

```text
models/deepseek_v4_flash_mtp/decode_csa.py
models/deepseek_v4_flash_mtp/decode_hca.py
models/deepseek_v4_flash_mtp/decode_swa.py
```

Each was run with `-p a2a3`, `--start-pos 8192`, and
`--enable-l2-swimlane 0` under `PYPTO_BENCH=1`.
