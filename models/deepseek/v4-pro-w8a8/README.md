# DeepSeek V4 Pro W8A8 128K Bring-up Plan

![M3 HCA](https://img.shields.io/badge/M3_HCA-COMPLETE-2ea44f)
![M4 CSA](https://img.shields.io/badge/M4_CSA-COMPLETE-2ea44f)
![M5 Compute](https://img.shields.io/badge/M5_Compute-COMPLETE-2ea44f)
![M6 Forward](https://img.shields.io/badge/M6_Forward-COMPLETE-2ea44f)
![COMM EP2/4/8](https://img.shields.io/badge/COMM_EP2%2F4%2F8-CONTRACT_ONLY-f0883e)

> **Status key:** 🟢 complete · 🟡 partial · 🟠 contract-only/unmeasured ·
> 🟣 measured device evidence · 🔵 fixed contract

## Status and provenance

This directory is an implementation scaffold for the DeepSeek V4 Pro W8A8
decode benchmark tracked by GitHub issue 873.

- Seed directory: models/deepseek/v4-flash
- Seed commit: 2388850d82f40df3596b78e882a9903d255ae275
- Target backend: a2a3 on the current Ascend 910 server
- Initial state: the Python files were seeded as an exact tracked-file copy

Phases 1 through 4 now cover the immutable target contract, leaf/SWA path,
128K HCA, and full-history CSA/indexer path. Phase 5 enables the single-die
compute-only MoE proxy. Its physical COMM_EP2/4/8 side remains a host layout
contract with no distributed device measurement, so Milestone 5 is only
partially complete. Milestone 6 is complete for its explicitly bounded
single-die compute proxy: bounded four-layer goldens pass, every depth in the
static ladder completes 100 repeated device rounds, and the full 61-layer
resident graph fits and runs on one target die. Physical EP128 communication,
SWA, MTP, embedding, and the final model head remain outside this result.
Existing models/deepseek/v4-flash and models/deepseek/v4-pro files remain
unchanged.

Phase 1 completed with:

- B=2, S=4, start position 131072, and maximum sequence length 131584
- A frozen Pro-W8A8 preset while preserving the canonical 1M-position PRO
- Pure-Python main/MTP layer ownership validation
- Separate deployment-EP128 and physical COMM_EP2/4/8 shape contracts
- Hybrid logical and physical cache-capacity constants
- A CPU-only configuration contract test as the sole Phase 1 CI allowlist entry
- Directory-level exclusion from broad PR, simulator, and A2/A3 device sweeps

Phase 2 completed with:

- Host contracts for non-aliased searchable history and request-isolated rings
- Gate score routing over 384 experts with a 512-wide sort and exact tail IDs
- A single-die deployment-EP128 compute shape with three local experts and
  production receive capacity 1024
- Balanced, skewed, and tail routed workloads of [16, 16, 16], [48, 0, 0], and
  [17, 16, 15]
- A ratio-0 SWA path at positions 131072 and 131071 using an original-KV ring
  with six physical pages and no compressor or indexer work
- Frozen-data replay controls and level-4 L2 swimlane controls on every Phase 2
  executable

Phase 3 and Phase 4 completed with real-device metadata, HCA, compressor,
indexer, and composed CSA goldens at the required 16K, 128K, boundary, and
maximum-tail positions. Phase 5 compute-only completed with balanced and
skewed routed workloads; its level-4 trace identifies gate, shared-expert, and
routed-expert work independently. The separate communication entry remains
explicitly unmeasured and fails fast if device execution is requested.

The target-server W8 query projection uses head-major output-channel weights
with shape [H, HEAD_DIM, Q_LORA] and a transposed cube operand. The flat
[Q_LORA, H * HEAD_DIM] GM-to-Mat access used by the seed does not produce a
correct Pro-sized query projection on this server. The validated HCA and CSA
paths now use the head-major layout; future composed-model callers must retain
it.

The measured Phase 2 baselines use five discarded warmups and 100 frozen-data
rounds. L4 is the separately captured instrumented graph time; all values below
are microseconds. Measurements came from the uncommitted Phase 2 worktree based
on pypto-lib 3c3db6c07a97cc67382109dfefb2fc32e722e112.

| Program | Case | L4 | Min | Median | Mean | Max |
|---|---|---:|---:|---:|---:|---:|
| Gate | score route, tail experts | 74.98 | 50.6 | 67.0 | 65.5 | 72.3 |
| Shared expert | 8 rows | 111.82 | 118.2 | 132.1 | 131.4 | 142.5 |
| Routed expert | [16, 16, 16], capacity 1024 | 243.96 | 215.3 | 223.6 | 225.3 | 250.0 |
| SWA | B=2, S=4, position 131072 | 605.48 | 540.5 | 597.6 | 590.4 | 665.6 |

| Program | a2a3sim | a2a3 golden cases |
|---|---|---|
| Metadata | Host contract | ring, full history, boundaries, invalid pools |
| Gate | Pass | random and tail-expert |
| Shared expert | Pass | 8 rows |
| Routed expert | Pass | [16,16,16], [48,0,0], and [17,16,15] |
| SWA | Compile pass | positions 131072 and 131071 |

Gate, shared expert, and routed expert pass full a2a3sim goldens. The composed
SWA graph compiles on a2a3sim, while its full simulator run stalls in qk_pv;
the standalone sparse-attention components and both required a2a3 SWA
positions pass. The measurement environment pins PyPTO
7d743e8d35bfd45df2b09a08b6a79308fada1342, simpler
dccb8379080b43173744b9981de2542b3d025e19, and PTO ISA
83d01313d9bfc247c4b7c8bcf969d1019f0d106f.

Canonical project guidance remains in docs/. In particular, follow:

- docs/pypto-coding/pypto-coding-style.md
- docs/run-and-validate/compile-runtime-workflow.md
- docs/run-and-validate/golden-harness.md
- docs/run-and-validate/save-and-replay.md
- docs/debug-and-tune/performance-tuning.md

Do not commit generated build_output artifacts or saved machine-specific
runtime data.

## Objective

Measure the per-die compute cost of a DeepSeek V4 Pro main-model verification
step at a 128K context, then explain that wall time using standalone attention
and MoE component measurements.

The fixed target point is:

| Axis | Target |
|---|---:|
| Platform | a2a3 |
| Decode batch | 2 requests |
| Sequence rows per request | 4 |
| Total rows | 8 |
| Start position | 131072 |
| Maximum sequence length | 131584 |
| Main hidden layers | 61 |
| Main attention mix | 31 HCA, 30 CSA, 0 SWA |
| Learned MTP layers | 1 separate ratio-0/SWA layer |
| Routed experts | 384 global |
| Effective deployment EP | 128 |
| Local routed experts | 3 |
| Routed expert compute dtype | W8A8 |

The first row plus three draft rows in S=4 describes one target-model
verification batch. It does not generate three autoregressively dependent
drafts by itself.

## Measurement boundaries

### Main-model verification

The primary result is the directly measured wall time of decode_fwd at B=2,
S=4. Its explanatory reconstruction is:

~~~text
L_main_verify =
    31 * L_HCA
  + 30 * L_CSA
  + 61 * L_MoE_compute
  + R_main
~~~

R_main includes orchestration gaps, final head and normalization work, and
standalone-versus-composed scheduling differences. SWA has coefficient zero
in the main model.

### MTP draft generation

The separate MTP layer contains projection, SWA attention, MoE, head and
normalization work, plus serving-side packing, sampling, and cache management
when those operations are present in the selected runner.

A future draft-depth-three serving result must be measured as:

~~~text
L_cycle_D3 =
    L_main_verify(B=2, S=4)
  + L_verify_commit_handoff
  + sum(L_mtp_iteration[j], j=1..3)
  + L_control_gaps
~~~

Until a real three-iteration runner exists, three times a single-MTP latency
is an extrapolation rather than an end-to-end measurement.

### EP128 scope

Deployment sharding and physical communication world size are independent:

~~~text
GLOBAL_EXPERTS = 384
DEPLOYMENT_EP = 128
LOCAL_EXPERTS = GLOBAL_EXPERTS / DEPLOYMENT_EP = 3
COMM_EP = one of 2, 4, or 8 when communication is measured
~~~

The EP128 deliverable is a single-die compute proxy. It must not allocate a
128-rank leading tensor dimension or submit distributed peer operations.
Measured EP2/4/8 communication is reported separately and is never labelled
as measured EP128 communication.

For B=2, S=4, and top-k=6, the balanced routed load is:

~~~text
global routes                 = 128 * 8 * 6 = 6144
mean rows per global expert   = 6144 / 384 = 16
mean rows per local shard     = 3 * 16 = 48
worst recv capacity per expert = 128 * 8 = 1024
~~~

The production-layout benchmark keeps recv capacity 1024 and sets
recv_expert_count to [16, 16, 16]. The routed kernel must dispatch only one
16-row compute tile per expert for that balanced case. A compact capacity-16
run may be used as a control but is not the sole official result.

## Cache allocation contract

Logical addressability and physical residency are separate. Full-history
searchable caches require unique physical storage. Finite-horizon consumers
use request-isolated rings and may reuse only expired logical pages.

| Pool | Logical width per request | B=2 physical target | Policy |
|---|---:|---:|---|
| Original KV | 1028 pages | 6 pages for all alignments | Ring |
| Ratio-4 compressed KV | 257 pages | 514 pages | Full, non-aliased |
| Indexer KV | 257 pages | 514 pages | Full, non-aliased |
| Ratio-128 compressed KV | 9 pages | 18 pages | Full, non-aliased |
| HCA compressor state | 16448 state pages | 32 pages | Ring |
| CSA compressor state | 32896 state pages | 4 pages | Ring |
| CSA inner compressor state | 32896 state pages | 4 pages | Ring |

The exact start-position step touches two original-KV pages per request, or
four globally. Six globally covers every alignment of a 128-row window plus
four in-flight verification rows.

The current unified forward compressed-pool shape may use the ratio-4 capacity
for HCA layers as well. Splitting HCA and CSA physical pool shapes is an
optional memory optimization, not a correctness prerequisite.

Block-table validation must prove all of the following:

1. Different requests never share a simultaneously live physical page.
2. A ring never aliases two simultaneously live logical rows.
3. Every ratio-4 compressed and indexer logical page has unique physical
   storage for the full visible history.
4. Golden validation does not rely only on the same aliased mapping as the
   device kernel.
5. Physical cache contents use enough address diversity to avoid artificial
   L2 reuse in the 128K performance run.

## Milestone 1: configuration and CI isolation

### Work

- Add an immutable Pro-W8A8 kernel preset without changing the architectural
  PRO preset.
- Set B=2, S=4, start position 131072, and maximum sequence length 131584.
- Keep 384 global experts independent of the selected communication EP.
- Set routed expert storage and compute to INT8 W8A8 semantics.
- Derive the main layer mix from the first 61 compression ratios.
- Keep the trailing ratio-0 entry owned by the MTP layer.
- Port only the Pro dimension and tiling fixes needed by each kernel from
  models/deepseek/v4-pro; do not wholesale replace the maintained seed.
- Exclude this directory from broad model discovery while it is incomplete.
- Add an explicit, small CI allowlist as kernels become validated.

### Exit criteria

- Importing config does not import or initialize distributed runtime state.
- Assertions report 31 HCA, 30 CSA, and zero main-model SWA layers.
- Gate routing space remains exactly 384 for every supported COMM_EP value.
- No existing model directory is modified.

## Milestone 2: leaf compute and SWA bring-up

Bring up the smallest single-die programs before HCA, CSA, distributed MoE,
or the full forward.

### Order

1. Host metadata and block-table tests
2. gate.py
3. expert_shared.py
4. expert_routed.py
5. decode_attention_swa.py

### Required cases

Gate:

- 8 input rows
- 384 global experts
- score padding width 512
- top-k 6
- every output expert index in [0, 384)

Routed expert:

- 3 local experts
- recv capacity 1024
- balanced counts [16, 16, 16]
- skewed counts [48, 0, 0]
- zero-row coverage and tail counts [17, 16, 15]

SWA:

- B=2 and S=4 causal metadata
- target start position 131072
- a page-boundary correctness case
- request-isolated original-KV ring
- Pro hidden, head, and projection dimensions

### Validation sequence

For each executable:

1. Run import and tensor-shape assertions.
2. Compile and validate on a2a3sim when supported.
3. Run golden validation on one a2a3 die through task-submit.
4. Save the validated input and golden data once.
5. Replay the frozen data for repeated timing.
6. Capture a level-4 L2 swimlane.

### Exit criteria

- Every leaf program compiles and passes its golden comparison.
- The balanced routed trace executes one 16-row tile per local expert, not 64.
- SWA contains no compressed-cache or indexer work.
- Repeated device timings are stable enough to establish a baseline median.

## 🟢 Milestone 3: HCA at 128K (complete)

HCA is brought up before CSA because it has no learned indexer.

### Work

- Use a 1028-column original-KV logical block table backed by a safe ring.
- Use 18 non-aliased ratio-128 compressed pages for standalone HCA.
- Use a 16448-column HCA-state table backed by 32 ring pages.
- Validate metadata generated on both host and device paths.
- Preserve the fixed top-k contract while marking invalid compressed tail rows.

### Required positions

- 131072 for the official performance point, where the four-row chunk does
  not cross a ratio-128 compression boundary.
- 131071 for correctness coverage of state pooling and compressed-cache
  writeback at a ratio-128 boundary.

### Completion evidence

- Device-generated metadata passes on real a2a3 hardware for the Pro B=2/S=4
  target and maximum-tail fixtures.
- Full HCA golden validation passes at the official position 131072 and the
  ratio-128 boundary position 131071.
- The boundary replay and its metadata/output goldens prove state pooling and
  one compressed-cache writeback per affected request while preserving request
  isolation across the original-KV ring, 18 unique compressed pages, and 32
  HCA-state ring pages.
- The boundary level-4 L2 instrumented graph time is **834.26 us**. This is an
  instrumented trace time, not a repeated-run latency median.

### Exit criteria

- Both positions pass golden validation on a2a3.
- Host/device metadata tables and golden-validated cache outputs show no
  cross-request live alias.
- The boundary trace, metadata, and goldens together cover the expected window
  and compressed-tail reads.
- Boundary metadata and output validation prove exactly one ratio-128
  compression event per affected request; the matching trace records the
  combined compressor task.

## 🟢 Milestone 4: CSA and indexer at 128K (complete)

CSA is the highest-risk standalone operator and remains isolated until its
full-history path is correct.

### Work

- Validate B=2, S=4 at a 16K context with the full 32896-value score buffer.
- Replace the fixed 4096-value, two-half top-k merge with a hierarchy that
  supports a score length of 32896.
- Allocate 514 unique ratio-4 compressed pages and 514 unique indexer pages.
- Back main and inner compressor state tables with separate four-page rings.
- Ensure the target step writes logical compressed page 256 without aliasing
  it onto page zero.
- Keep indexer score generation proportional to the full visible ratio-4
  history while sparse attention remains bounded by top-k 1024.

### Completion evidence

- The real-device B=2/S=4 16K regression passes with the full 32896-score
  implementation enabled.
- The official 128K CSA case at position 131072 passes frozen-data golden
  validation on a2a3.
- The standalone indexer passes at the maximum-tail position 131580, including
  the final 128-value score group, with both scores and 1024 selected IDs
  matching the reference.
- The main and inner ratio-4 compressor leaf cases pass their boundary
  cache-writebacks. Ratio-4 compressed and indexer histories use 514 unique
  physical pages, and logical page 256 maps to physical pages 512 and 513
  rather than aliasing page zero.
- The target level-4 L2 instrumented graph time is **2374.68 us**.

> [!IMPORTANT]
> Trace task counts are not page counts: each token-level `score_mat` task loops
> over its visible pages internally. At position 131072 the four token rows see
> 256, 256, 256, and 257 pages per request. Full-history coverage is established
> by the visible-length contracts, unique 514-page tables, real-device
> score/top-k goldens, compressor writebacks, and the composed CSA trace
> together—not by interpreting the trace task count as 257.

### Exit criteria

- The 16K B=2/S=4 regression case passes with the full score path enabled.
- The 32896-score top-k matches the torch reference.
- The 128K orchestration passes golden validation from frozen data.
- The L2 trace is reconciled with metadata showing the 257-page maximum scan.
- No repeated physical page stands in for multiple visible history pages.

No 16K-to-128K CSA extrapolation is accepted as the final 128K number.

## 🟡 Milestone 5: EP128 compute proxy and communication contract (partial)

> [!WARNING]
> Completion of the single-die compute proxy does not complete this
> milestone's communication deliverable.

### Current status

- Compute-only: **COMPLETE**. The one-device entry exposes the gate-quantized
  handoff, uses gate [384, D], routed weights [3, ...], and recv
  [3, 1024, D], and passes balanced [16, 16, 16] and skewed [48, 0, 0]
  real-device goldens.
- The compute proxy uses stable unit-magnitude input rows while preserving the
  selected gate routing fixture. Random-input gate behavior remains covered by
  the standalone gate golden.
- The balanced level-4 L2 instrumented graph time is **276.76 us**; its trace
  separately identifies gate, shared-expert, and routed-expert task families.
- Communication: **CONTRACT-ONLY / UNMEASURED**. The host contract preserves
  384 global experts and defines COMM_EP2, COMM_EP4, and COMM_EP8 shapes, but
  no physical peer dispatch or combine kernel has been measured.
- The communication entry intentionally fails fast when execution is
  requested. No dispatch/combine latency table may be published until a real
  distributed path runs at each stated world size.

### Compute-only entry

The single-die compute entry is `moe_compute.py` and contains:

- Full 384-expert gate for eight local rows
- Shared expert for eight local rows
- Three routed experts with synthetic received rows
- Production recv stride and capacity
- Local packing or reduction only when its cost is separately identifiable
- No pld window, peer put, notify, wait, or DistributedConfig

The compute proxy may reproduce per-die shapes and scheduling but cannot claim
numerical equivalence to an EP128 distributed output. Most rows processed by a
real local expert shard originate on other ranks, while most routes for this
rank's own tokens terminate remotely.

### Communication entry

The current communication entry defines host-side layout contracts for physical
COMM_EP values 2, 4, and 8. A future separate distributed MoE path must report
dispatch and combine measurements with the actual world size and must not
mutate global expert count to keep a constant local expert count.

### Exit criteria

- Compute-only accepts exactly one device.
- Shapes are gate [384, D], routed weights [3, ...], and recv [3, 1024, D].
- Balanced and skewed routed workloads pass standalone goldens.
- L2 traces identify gate, shared, and routed compute independently.
- Communication tables identify COMM_EP explicitly.

## 🟢 Milestone 6: reduced-depth and full main forward (complete compute proxy)

> [!IMPORTANT]
> The complete 61-layer, 31-HCA/30-CSA single-die compute proxy was prepared
> with every resident compute-ABI spec on-device and completed 100 measured
> rounds on a target a2a3 die. This proves resident fit and repeated dispatch liveness for
> the stated compute boundary. Deep cases do not have a Torch golden, so this
> is not a claim of depth16/depth31/depth61 numerical correctness.

### Executable boundary and delivered graph

- `main_compute_manifest.py` defines the complete `hca1`, `csa1`, `depth2`,
  `depth4`, `depth16`, `depth31`, and `depth61` static ladder without
  materializing tensors.
- `decode_layer_compute.py` composes the validated HCA or CSA path with gate,
  shared-expert, and three-local-expert routed compute.
- `decode_fwd_compute.py` stacks independent per-layer weights and caches in a
  one-worker resident L3 program. Every layer owns its physical cache byte
  ranges and uses layer-local block IDs; there is no cross-layer circular
  alias.
- The ABI covers `x_hc` through `x_next`, all selected HCA/CSA layers, local
  deployment-EP128-shaped MoE compute, per-layer caches, and RoPE.
- The ABI excludes physical EP128 dispatch/combine communication, SWA, MTP,
  embedding lookup, final normalization, and the model/LM head. The result is
  neither EP128 end-to-end nor a complete serving-cycle latency.

Original KV and finite-horizon compressor state retain their validated,
request-isolated rings. Searchable compressed-KV and indexer caches are
allocated at their full declared production capacity and do not use circular
history aliasing.

### Correctness, fit, and repeat evidence

All cases use start position 131072, B=2, S=4, the balanced local-expert load,
and pinned PTOAS 0.54. Bounded `commit` goldens compare all 11 forward outputs;
deeper cases are explicitly fit/liveness-only.

| Case | Layer mix | Commit correctness or resident-fit evidence | 100-round overwrite |
|---|---:|---|---|
| `hca1` | 1 HCA | Golden pass | Pass |
| `csa1` | 1 CSA | Golden pass | Pass |
| `depth2` | 2 HCA | Golden pass | Pass |
| `depth4` | 3 HCA + 1 CSA | Golden pass | Pass |
| `depth16` | 9 HCA + 7 CSA | Resident fit/pass, no golden | Pass |
| `depth31` | 16 HCA + 15 CSA | Resident fit/pass, no golden | Pass |
| `depth61` | 31 HCA + 30 CSA | Resident fit/pass, no golden | Pass |

The depth4 golden canonicalizes each top-k row as expert-ID/weight pairs before
comparison. It therefore accepts harmless pair permutations without detaching
weights from their selected experts.

The static ladder includes shared embedding, final normalization/head,
LM-head, and RoPE assets. RoPE is part of the executable ABI; embedding and the
final model/LM-head assets are outside it. The shared allocation is
approximately 3.484 GiB, and metadata varies with the selected attention kinds.

| Ladder case | HCA | CSA | Static accounted bytes | Static accounted GiB |
|---|---:|---:|---:|---:|
| `hca1` | 1 | 0 | 4,426,231,636 | 4.122 |
| `csa1` | 0 | 1 | 4,534,786,588 | 4.223 |
| `depth2` | 2 | 0 | 5,111,448,620 | 4.760 |
| `depth4` | 3 | 1 | 6,587,466,476 | 6.135 |
| `depth16` | 9 | 7 | 15,421,774,604 | 14.363 |
| `depth31` | 16 | 15 | 26,518,737,844 | 24.697 |
| `depth61` | 31 | 30 | **48,604,508,164** | **45.266** |

> [!IMPORTANT]
> The 45.266 GiB `depth61` value is the full static-accounted footprint,
> including shared embedding, LM-head, and RoPE assets. Static accounting alone
> does not prove runtime fit; the successful resident device jobs above are the
> direct fit evidence.

The depth61 default uses a 14-GiB active ring-0 heap and 256 MiB for each
inactive ring. The one-dispatch scope-stat run observed a ring-0 heap high-water
of 12,593,982,464 bytes, 6,362 task entries, and 7,126 dependency entries. The
16,384-entry task window and 65,536-entry dependency pool remain above those
observed demands.

| Depth61 planning scope | Bytes | GiB |
|---|---:|---:|
| Full static manifest | 48,604,508,164 | 45.266 |
| Actual compute-ABI resident specs | 44,896,648,568 | 41.813 |
| Configured ring heaps | 15,837,691,904 | 14.750 |
| Runtime TMR shared memory | 325,583,744 | 0.303 |
| Pinned-runtime private-arena bound | 28,285,824 | 0.026 |
| Retained nonresident ABI staging | 24,591,360 | 0.023 |
| Compute ABI + known runtime + staging | 61,112,801,400 | 56.916 |
| Static manifest + known runtime + staging | 64,820,660,996 | 60.369 |
| Remaining nominal 64-GiB headroom | 3,898,815,740 | 3.631 |

Compiler code, device context, allocator fragmentation, and other unexposed
runtime overhead are not included. The 60.369-GiB row is a conservative
cross-boundary planning total; it is not a measured peak. The successful
depth61 device run, rather than either accounting row alone, establishes fit.

### Repeated effective timing

Each timing job discards five warmups and requires exactly 100 measured rounds,
one rank, and one dispatch per round. The harness rejects missing statistics,
flattened rounds, changing dispatch slots, incomplete grids, and non-positive
samples. Values are the runtime's Effective device window in milliseconds.

| Case | Min | Median | Mean | Max |
|---|---:|---:|---:|---:|
| `hca1` | 0.9630 | **1.0045** | 1.0281 | 1.1236 |
| `csa1` | 1.9113 | **2.5259** | 2.3695 | 2.6178 |
| `depth2` | 1.9918 | **2.0565** | 2.1052 | 2.2449 |
| `depth4` | 5.0205 | **5.8124** | 5.6978 | 5.9203 |
| `depth16` | 23.4219 | **23.7290** | 25.5988 | 28.3371 |
| `depth31` | 47.5022 | **56.8154** | 53.0156 | 57.4078 |
| `depth61` | 93.4279 | **112.7978** | 105.9753 | 113.8827 |

The raw samples have two recurring timing bands, so the table retains the full
min/median/mean/max context. Jobs used auto-assigned dies on the same target
server; the reconstruction below is explanatory and includes run/card and
composition effects in its residual.

The one-layer entries already contain attention plus the same compute-only MoE
proxy. They must not be combined with an additional `61 * L_MoE` term:

~~~text
L_layer_reconstruction = 31 * 1.0045 ms + 30 * 2.5259 ms
                       = 106.9165 ms
R_proxy                = 112.7978 ms - 106.9165 ms
                       = +5.8813 ms  (+5.214% of direct median)
~~~

The residual covers static-stack scheduling/overlap, cross-layer dependencies,
and run-to-run/card variation. It must not be relabelled as SWA, MTP, physical
EP128 communication, embedding, or final-head time because none of those
operations executes in this proxy.

### Cache policy for timing

`commit` is the one-shot correctness/smoke policy. Timed repeats require
`overwrite`: external inputs, position, and write mappings stay fixed; every
round performs full cache writes to the same slots, and resident cache/state is
not reset between rounds. This preserves store and scheduling work, but after
the first round it is a synthetic state-mutating hot repeat. It is not serving
sequence advancement, commit/rollback, or a golden-validated replay.

The completed validation ladder was:

~~~text
one HCA layer
one CSA layer
first two Pro layers
four layers including both attention kinds
16 layers
31 layers
61 layers
~~~

All seven depths complete the requested repeated grid with one resident worker;
the full graph fits one target die; the direct timing and reconstruction retain
an explicit residual; and every result is labelled as a main-model compute
proxy rather than EP128 end-to-end. These satisfy Milestone 6 within its stated
boundary. Milestone 7 owns the separate MTP-D3 serving cycle.

## Milestone 7: MTP-D3 serving cycle and reporting

MTP work begins only after the main-model B=2/S=4 verification path is stable.

### Work

- Validate one complete MTP iteration, including every operation inside the
  selected measurement boundary.
- Define cache commit and rollback behavior for accepted and rejected drafts.
- Implement three serial draft iterations using the one learned MTP layer.
- Feed the resulting four-row request layout into main-model verification.
- Measure the integrated wall time and compare it with the sum of three
  separately measured MTP iterations plus main verification.
- Normalize serving results by both cycle and accepted token count.

### Final report

Publish at least these tables:

1. Standalone operator latency at 16K and 128K
2. Weighted main-model reconstruction and measured decode_fwd wall time
3. Single-MTP and three-iteration draft latency
4. Full serving-cycle latency and accepted-token normalization
5. EP128 compute proxy and measured EP2/4/8 communication
6. Per-die HBM allocation by weights and cache family

Every row must identify platform, commit, B, S, context, active token count,
deployment EP, physical communication EP, cache policy, and whether the value
is measured or extrapolated.

### CI promotion

- Keep full 128K and 61-layer performance runs out of broad per-commit sweeps.
- Add small compile and golden cases for configuration, metadata, leaf
  operators, and representative HCA/CSA boundaries.
- Gate the dedicated device benchmark on the new directory's path.
- Preserve simulator coverage only for cases whose memory and instruction set
  are supported there.

## Benchmark procedure

Use the same sequence for every official device number:

1. Record the repository and dependency pins.
2. Generate inputs and torch golden once.
3. Replay identical frozen data for warmup and measured iterations.
4. Exclude compilation, input generation, host-to-device initialization, and
   output readback from kernel latency.
5. Report median, tail spread, and repetition count.
6. Capture the matching level-4 L2 swimlane.
7. Reconcile standalone components with the composed wall time.

Representative command forms are:

~~~bash
python models/deepseek/v4-pro-w8a8/gate.py -p a2a3sim --fixture tail-expert

task-submit --device auto --run \
  'python models/deepseek/v4-pro-w8a8/expert_routed.py -p a2a3 \
     -d $TASK_DEVICE --workload balanced --save-data --enable-l2-swimlane 4'

task-submit --device auto --run \
  'PYPTO_BENCH=1 python models/deepseek/v4-pro-w8a8/decode_attention_swa.py \
     -p a2a3 -d $TASK_DEVICE --start-pos 131072 \
     --golden-data build_output/SAVED_RUN/data'
~~~

Inspect each script's help before using a command; CLI support is added and
validated milestone by milestone.

## Stop conditions

- Do not report copied Flash behavior as Pro W8A8 behavior.
- Do not call a single-die compute proxy EP128 end-to-end.
- Do not use aliased ratio-4 or indexer history for an official latency.
- Do not count standalone SWA in both main-model and MTP totals.
- Do not report three times one MTP invocation as measured draft-depth-three.
- Do not silently reduce batch, context, or layer count after an OOM.
- Do not proceed from a failed golden solely because a performance trace was
  produced.

## Suggested change slices

Keep implementation reviewable in these slices:

1. Scaffold, configuration contract, metadata tests, and CI isolation
2. Gate, shared expert, routed expert, and SWA bring-up
3. HCA 128K cache and boundary support
4. CSA 128K indexer and full-history cache support
5. EP128 compute proxy and separate communication measurements
6. Reduced-depth and 61-layer main forward
7. MTP-D3 runner, benchmark report, and CI promotion

Each slice must leave its newly enabled entry points passing before the next
slice starts.
