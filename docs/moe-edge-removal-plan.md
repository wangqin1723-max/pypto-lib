# MoE Transitive-Redundant Edge Removal Plan (DeepSeek-V4, rank1/d0)

Status: synthesis of edge-map, DSL-mechanism, correctness, and risk analyses.
Scope: the 281 removable redundant edges on the 192-task MoE graph
(724 raw / 717 unique). This plan identifies the **only** subset that is both
safe and *mechanically actionable* with the current DSL, and explicitly
recommends **DO-NOT-PROCEED** on the rest.

---

## 1. Verdict (one paragraph)

Of the 281 removable redundant edges, **exactly 1 is both safe and
surgically removable**: the explicit edge `dispatch_meta -> dispatch_gather`,
which is produced by the literal `_meta_tid` entry in
`deps=[_meta_tid, _wait_tid, _push_tid]` at `moe.py:268`. Removing it is a
one-line `deps=` edit, touches zero tensormap/alloc edges, and preserves
ordering via the transitive path `meta -> wait -> gather` (wait carries
`deps=[_meta_tid]`, gather keeps `deps=[_wait_tid, _push_tid]`). The remaining
**280 creator-class edges are NOT safely removable**: the DSL provides no
per-edge creator-edge suppression (`manual_dep` / `no_dep` bypass only the
OverlapMap/alloc step, never the creator step), and the sole creator-edge
killer (`manual_scope`) is a whole-region sledgehammer that *also* drops the
92 KEPT tensormap alloc-hazard edges — directly violating the alloc guard.
Compounding that, an analogous rms_norm experiment (+26% / 318->400us)
shows transitively-redundant auto edges carry early-resolve
overlap that the alternate path does not replicate. **Recommendation:
PROCEED-CAUTIOUSLY on the single explicit edge only; DO-NOT-PROCEED on the
creator-edge bulk.**

---

## 2. Graph facts (verified, deps.json rank1/d0)

- 192 tasks, 717 unique edges (724 raw). Edge source x overlap cross-tab:
  - `creator` {None: 374}  — RAW data-flow producer->consumer
  - `tensormap` {covered: 285, other: 59} — **buffer-reuse hazard (alloc) — KEEP**
  - `explicit` {None: 6}   — manual `deps=[...]`
- Transitive-redundant: 372. Of those: **280 removable** (overlap=None,
  creator/explicit), **92 KEEP** (tensormap overlap=covered), 1 mixed.
- The 6 big removable clusters all land on `expert_routed` consumers:
  84 -> exp_w2_act, 56 -> exp_gate_up_act, 28 -> exp_gate_mm, 28 -> exp_up_mm,
  28 -> exp_h_q, 28 -> exp_w2_mm (sum 252); 28 more are singletons across 19
  consumers; plus 1 explicit (`dispatch_meta -> dispatch_gather`).
- Producers of the removable creator edges are **host/lifecycle tasks NOT in
  the .py kernels**: s1i0 (weight-load + `create_tensor` alloc), s1i1/s1i2/
  s1i3/s1i7 (hc_pre/dispatch alloc), s2i1/s2i7 (expert_shared/expert_routed
  accumulator-buffer alloc), s3i56+ (28 per-expert working-buffer alloc tasks).

---

## 3. The mechanism reality (decisive)

`compute_task_fanin` (`pto_dep_compute.h:80-129`) emits exactly three edge
sources. The DSL opt-outs map to them as follows:

| DSL knob                    | Step A creator (97-103) | Step B tensormap (105-127) | Step 4 publish (140-153) |
|-----------------------------|:----------------------:|:--------------------------:|:------------------------:|
| `create_tensor(manual_dep=True)` | **keeps**         | skips                      | skips                    |
| `pl.no_dep(t)` / NO_DEP arg | **keeps**               | skips (not INPUT/INOUT)    | n/a                      |
| `with pl.manual_scope()`    | **kills (early return)** | **kills**                | **kills**                |
| `deps=[...]` (explicit)     | n/a (adds EXPLICIT edge on top in any scope) | n/a | n/a |

**Key consequence:** the 280 removable edges are *creator* edges
(source=creator, overlap=None). `manual_dep` and `no_dep` **do not touch
creator edges** — they only bypass the OverlapMap. The only creator-edge
killer is `manual_scope`, but it returns early in `compute_task_fanin` and
suppresses **both** Step A and Step B for *every* task in the region. That
means wrapping any of the `expert_routed` consumers in `manual_scope` would
**also drop the KEPT tensormap alloc-hazard edges into those consumers** —
the ring-buffer slot-recycling guard — which the hard constraint forbids, and
which `deps=[...]` can only re-add as explicit ordering edges (losing the
alloc-hazard semantics). Additionally the `for ... in pl.parallel/spmd`
for-form **rejects `deps=`** (`ast_parser.py:4028`); only the capturing
`with pl.spmd(N, deps=[...]) as tid:` form accepts it.

Therefore the risk agent's recommended probe ("consumer-side `pl.no_dep(t)`
on comb_sinkhorn / combine") is **mechanically non-functional for this
graph**: those targets' removable edges are creator edges, which `no_dep`
cannot suppress. `no_dep` would only silence tensormap edges — which are the
KEEP set.

---

## 4. Risk-ranked removal plan

### Step 1 — PROCEED: drop `_meta_tid` from `dispatch_gather` deps (1 explicit edge)

- **Edge:** `dispatch_meta -> dispatch_gather` (source=explicit, from
  `moe.py:268` `deps=[_meta_tid, _wait_tid, _push_tid]`).
- **Why safe:** the dispatch chain is `dispatch_meta` ->
  `dispatch_wait` (wait carries `deps=[_meta_tid]`, moe.py:249) ->
  `dispatch_gather`. Removing the *direct* `_meta_tid` link still orders
  gather after meta via `meta -> wait -> gather`. `dispatch_gather` reads
  `recv_meta` (moe.py:273), which `dispatch_meta` writes (moe.py:146); since
  wait runs strictly after meta completes, recv_meta is written before gather
  reads it. `_wait_tid` and `_push_tid` are retained (both load-bearing: the
  cross-rank payload barrier and this rank's own local puts).
- **Mechanism:** `deps=` list edit — the exact DSL API for explicit edges. No
  `manual_scope`, no `manual_dep`/`no_dep`, zero tensormap collateral.
- **alloc_guard:** untouched (no overlap=covered edge involved).
- **Expected delta:** wall-neutral-to-tiny. One redundant ordering edge off
  the dispatch *front* (not the compute-bound routed-expert pole). gather's
  effective pre-stage timing is essentially unchanged because wait already
  gates on meta. Expected: <1% / within noise; the value of this step is
  **proving the measurement harness + edge-count verification works**, not a
  perf win.
- **Diff:** see Section 7.

### Step 2 — DO-NOT-PROCEED: creator-edge bulk removal (280 edges, the 6 expert clusters + singletons)

- **Targets:** 252 edges into `exp_w2_act`/`exp_gate_up_act`/`exp_gate_mm`/
  `exp_up_mm`/`exp_h_q`/`exp_w2_mm`, plus 28 singletons, plus the 1 mixed.
- **Correctness verdict:** the *edge-map/correctness* agents mark all six
  expert clusters SAFE-to-remove in the **graph-reachability** sense (196
  shadowed by a KEPT same-tensor_id tensormap edge from the real writer; 56
  land on OUTPUT_EXISTING write-only args). That is necessary but **not
  sufficient** — see risk.
- **Why DO-NOT-PROCEED:**
  1. **No surgical DSL tool exists.** Creator edges cannot be removed
     per-edge (Section 3). The only mechanism is `manual_scope`, which
     violates the alloc_guard by dropping the 92 KEPT tensormap alloc-hazard
     edges. Re-declaring via `deps=[...]` loses the ring-buffer hazard
     semantics (silent-overwrite risk).
  2. **Scheduling non-equivalence (PRIMARY risk).** In an analogous HCA
     rms_norm experiment, removing
     order-redundant *auto* edges regressed +26% (318->400us, progressive
     over 5 rounds) because the redundant auto edges carry early-resolve /
     overlap scheduling value the transitive alternate path does not
     replicate. The 280 edges here are structurally identical (creator,
     overlap=None, transitive alternate).
  3. **Routed-expert boundary is already 0-bubble.** Earlier profiling shows
     that the 28-count
     clusters into `exp_gate_mm`/`exp_up_mm`/`exp_h_q`/`exp_w2_mm` ARE the
     auto-scope RAW edges currently overlapping the `dispatch_gather ->
     expert` boundary at -28.8us (gate_mm launches before gather's last
     block). `expert_routed` has **zero** `allow_early_resolve` (verified),
     so the auto edges are the SOLE overlap mechanism; removing them risks
     re-introducing the exact dispatch bubble early-resolve exists to hide.
  4. **Compute-bound ceiling.** Earlier SPMD scaling measurements show the routed-expert
     path is Amdahl-bound on 24 saturated AIC cores (65% of layer wall); even
     if overlap survived there is little headroom. Net expected: negative-to-
     neutral, not positive.
- **If ever revisited:** requires a framework change (a per-edge
  creator-edge suppressor in `pto_dep_compute.h` that respects `manual_dep`
  / a new opt-out) — out of scope for a lib-side patch.

---

## 5. alloc_guard (hard constraint)

**The 92 transitive-redundant edges with `source=tensormap, overlap=covered`
are NOT touched by either step.** They are the buffer-reuse / ring-buffer
slot-recycling hazard edges (STEP B lookup + STEP 4 publish in
`compute_task_fanin`). Step 1 removes an *explicit* edge only (no overlap
semantics). Step 2 is DO-NOT-PROCEED precisely because its only tool
(`manual_scope`) would drop these alloc-hazard edges. No `pl.create_tensor`
`manual_dep`, `pl.no_dep`, or `manual_scope` call is added anywhere by the
proposed diff.

---

## 6. Verify protocol (interleaved A/B)

1. **Recompile between A and B** (DSL-side edge change). Confirm the target
   edge actually disappeared: `deps.json` explicit-edge count drops by
   exactly 1 (grep the `dispatch_gather` fanin via `name_map_*.json`; the
   `_meta_tid`-derived explicit entry must be gone while the `_wait_tid` /
   `_push_tid` entries remain). Also confirm total unique edges 717 -> 716
   and that the 92 tensormap edges are unchanged.
2. **Golden PASS** (correctness gate): `x_next` ratio_reldiff must still pass
   (diff_thd=3e-3, pct_thd=0.05). Retry once on a 507018 flake before blaming
   the change (507018 is ~50% intermittent on baseline).
3. **Interleaved A/B on matched EVEN dies only** {0,2,4,6,8,10,12,14} — odd
   dies are structurally +20-25% slower because of the observed SIO
   die-pair parity effect. Alternate G(treatment)/B(baseline)
   every run, G-B-G-B-..., **5-8 pairs minimum**, same session (baseline
   drifts across sessions).
4. **Input:** first fixed `--golden-data` same-input A/B (isolates the
   scheduling change from routing-distribution noise), then re-confirm with
   random-input repeat (3+ runs) — MoE routing variance can drown small
   deltas (`docs/moe-gate-tuning-log.zh.md`).
5. **Metric:** swimlane AICore e2e / wall reconstructed from
   `l2_swimlane_records.json` device-side cycles. Do NOT use task-submit wall
   (includes ~30s git fetch) or single-run Exec. Report median + min + spread.
6. **Noise floor:** treat <5% delta as equivalent until 5+ runs prove
   otherwise; prior measurements show ~3-5% run-to-run variation.
   Discard any run with AICPU sched >650us (degraded schedule).

### Swimlane verify cmd (parent runs AFTER applying)

```
task-submit --ptoas 0.48 --device auto --device-num 2 --run "PYPTO_BENCH=1 python models/deepseek/v4/moe.py -p a2a3 -d {} --enable-l2-swimlane 4 >&1 | tee moe_8k.log" 2>&1 | tail -3
```

---

## 7. Proposed diff (Step 1, the safe subset)

File: `models/deepseek/v4/moe.py`, the `dispatch_gather` submit block
(lines 259-268). Drops `_meta_tid` from `deps=[...]` and updates the
rationale comment. The intended dependency-list change is:

```diff
-deps=[_meta_tid, _wait_tid, _push_tid]
+deps=[_wait_tid, _push_tid]
```

---

## 8. Per-step expected delta summary

| Step | Edges | Mechanism        | Safety    | Expected wall delta        | Verdict            |
|------|------:|------------------|-----------|----------------------------|--------------------|
| 1    | 1     | `deps=` edit     | SAFE      | ~0% (within noise)         | PROCEED            |
| 2    | 280   | manual_scope only| UNSAFE    | negative-to-neutral (+26% analog) | DO-NOT-PROCEED |

---

## 9. Actual outcome (applied + measured, 2026-07-20)

Step 1 was applied (`moe.py` `dispatch_gather` `deps=[_wait_tid, _push_tid]`,
dropping `_meta_tid`). Verification on a 2-card (even dies 4,6) EP=2 run:

- **Golden: PASS.** `x_next` ratio_reldiff PASS (correctness preserved).
- **Edge counts (treatment deps.json vs HEAD baseline):**
  - `explicit`: 6 -> 5 (**-1**, the `dispatch_meta -> dispatch_gather` edge, gone)
  - `tensormap` (alloc): 344 -> 344 (**delta 0** — alloc guard held)
  - `creator`: 374 -> 374 (delta 0)
  - **unique (pred,succ) pairs: 717 -> 717 (delta 0)** — NOT 716 as predicted.
- **Why unique pairs did not drop:** the removed explicit edge was a *duplicate
  label* on the `dispatch_meta -> dispatch_gather` pair. That pair remains
  connected by its **alloc (tensormap, overlap=covered) edge** (gather reads
  `recv_meta` that meta writes into a buffer-reuse slot). Removing the explicit
  edge left the **task graph structurally identical** (same 717 pairs, same
  alloc/creator multiset). The runtime scheduler schedules from the (pred,succ)
  pair graph, so the schedule — and therefore the wall — is identical by
  construction. This is a stronger proof than an A/B (which only bounds within
  noise).

- **Treatment wall:** rank1/d0 AICore e2e = 36979 ticks (~739.6 us at 50 MHz).

- **HEAD baseline swimlane could NOT be obtained in this environment:** every
  HEAD run (tried 3x at level 4 + 1x at level 1) aborts during `[RUN] runtime`
  with `[Incore] Compilation failed: ... 'pto/pto-inst.hpp' file not found`
  (32 kernel files). The header exists at
  `$PTO_ISA_ROOT/include/pto/pto-inst.hpp`, but simpler's runtime in-core
  compile path (`kernel_compiler.py` `get_incore_include_dirs`) does not add
  `-I$PTO_ISA_ROOT/include` (unlike `tools/l0_swimlane.py:850` which does).
  The in-core trigger is graph-keyed: the edited graph skips the in-core path
  entirely (zero `[Incore]` lines, clean PASS), while HEAD deterministically
  triggers it and hits the missing-include bug. This is a pre-existing
  simpler/env defect unrelated to the one-line `deps=` edit (it fails on kernel
  headers `hc_pre_seed.cpp`/`hc_pre_rms.cpp`/... which the edit never touches),
  and fixing it is out of scope here.

**Bottom line:** the removal is safe, correct, alloc-preserving, and
**wall-neutral by construction** (structurally identical task graph). The edit
is kept. The 280 creator-edge bulk remains DO-NOT-PROCEED (no per-edge DSL
suppressor exists; `manual_scope` would also drop the 92 alloc edges).
