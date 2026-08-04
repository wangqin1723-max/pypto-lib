# DeepSeek V4 Pro W8A8 128K Bring-up Plan

## Status and provenance

This directory is an implementation scaffold for the DeepSeek V4 Pro W8A8
decode benchmark tracked by GitHub issue 873.

- Seed directory: models/deepseek/v4-flash
- Seed commit: 2388850d82f40df3596b78e882a9903d255ae275
- Target backend: a2a3 on the current Ascend 910 server
- Initial state: the Python files were seeded as an exact tracked-file copy

Phase 1 now defines the immutable Pro-W8A8 target contract in config.py and
isolates this directory from broad kernel discovery. The copied executable
kernels still select the Flash preset until each one is explicitly enabled,
validated, and allowlisted in a later milestone. They must not yet be reported
as Pro W8A8 results. Existing models/deepseek/v4-flash and
models/deepseek/v4-pro files remain unchanged.

Phase 1 completed with:

- B=2, S=4, start position 131072, and maximum sequence length 131584
- A frozen Pro-W8A8 preset while preserving the canonical 1M-position PRO
- Pure-Python main/MTP layer ownership validation
- Separate deployment-EP128 and measured COMM_EP2/4/8 shape contracts
- Hybrid logical and physical cache-capacity constants
- A CPU-only configuration contract test as the sole CI allowlist entry
- Directory-level exclusion from broad PR, simulator, and A2/A3 device sweeps

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
- zero and tail-row coverage

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

## Milestone 3: HCA at 128K

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

### Exit criteria

- Both positions pass golden validation on a2a3.
- Cache address traces show no cross-request live alias.
- The target trace contains the expected window and compressed-tail reads.
- The boundary trace executes exactly one ratio-128 compression event per
  affected request.

## Milestone 4: CSA and indexer at 128K

CSA is the highest-risk standalone operator and remains isolated until its
full-history path is correct.

### Work

- First validate B=2, S=4 at the existing 16K score length.
- Replace the fixed 4096-value, two-half top-k merge with a hierarchy that
  supports a score length of 32896.
- Allocate 514 unique ratio-4 compressed pages and 514 unique indexer pages.
- Back main and inner compressor state tables with separate four-page rings.
- Ensure the target step writes logical compressed page 256 without aliasing
  it onto page zero.
- Keep indexer score generation proportional to the full visible ratio-4
  history while sparse attention remains bounded by top-k 1024.

### Exit criteria

- The 16K B=2/S=4 transition case passes before changing score length.
- The 32896-score top-k matches the torch reference.
- The 128K orchestration passes golden validation from frozen data.
- The L2 trace shows all 257 visible pages per request being scanned.
- No repeated physical page stands in for multiple visible history pages.

No 16K-to-128K CSA extrapolation is accepted as the final 128K number.

## Milestone 5: EP128 compute proxy and measured communication

### Compute-only entry

Add a clearly named single-die entry, preferably moe_compute.py, containing:

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

Retain a separate distributed MoE path for physical COMM_EP values 2, 4, and
8. Report dispatch and combine measurements with the actual world size. Do
not mutate global expert count to keep a constant local expert count.

### Exit criteria

- Compute-only accepts exactly one device.
- Shapes are gate [384, D], routed weights [3, ...], and recv [3, 1024, D].
- Balanced and skewed routed workloads pass standalone goldens.
- L2 traces identify gate, shared, and routed compute independently.
- Communication tables identify COMM_EP explicitly.

## Milestone 6: reduced-depth and full main forward

Do not start with 61 layers. Scale in this order:

~~~text
one HCA layer
one CSA layer
first two Pro layers
four layers including both attention kinds
16 layers
31 layers
61 layers
~~~

### Work

- Compose the validated attention paths with the compute-only MoE proxy.
- Keep each layer's physical cache pool independent; block tables map request
  logical pages within a layer rather than separating layers inside one pool.
- Reuse persistent weights and caches across timed iterations.
- Build a static HBM manifest before each depth increase.
- Measure main verification only; do not silently include a standalone MTP
  layer or add SWA with a main-layer coefficient.
- Keep reduced-depth results explicitly labelled when they are used for
  diagnosis or fallback extrapolation.

### Exit criteria

- Each depth runs repeatedly without allocation growth or stale signal state.
- The 61-layer case fits one target die, or an evidenced memory blocker and a
  documented reduced-depth fallback are produced.
- The measured wall time and operator reconstruction include an explicit
  residual instead of forcing additive equality.
- The result is labelled main-model compute proxy, not EP128 end-to-end.

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
python models/deepseek/v4-pro-w8a8/gate.py -p a2a3sim --compile-only

task-submit --device auto --run \
  'python models/deepseek/v4-pro-w8a8/gate.py -p a2a3 -d $TASK_DEVICE'

task-submit --device auto --run \
  'python models/deepseek/v4-pro-w8a8/decode_attention_swa.py \
     -p a2a3 -d $TASK_DEVICE --start-pos 131072 --enable-l2-swimlane'
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
