# HCA early-dispatch 移植调优日志（2026-07-15）

> 目标：把 Qwen3 `pr-691` / `#762`（commit `99a037d`）的 **early-dispatch + task_dummy 屏障**优化移植到 HCA decode attention（`models/deepseek/v4/decode_attention_hca.py` → kernel `decode_sparse_attn_hca.py`）。
> 结论（**已用 interleaved A/B 修正**）：**`hca_gather_kv` 打 `allow_early_resolve=True` 是真收益**——把 HCA decode 稳定在 **~308–314us**（σ≈5），baseline 不稳定（309–443us，中位 ~385）。`merge_norm` flag **中性**（`merge_norm→proj_a` 是并行组核争用，flag 不动 proj_a；但不害）。之前记的"+14.9% 失败"是**对比了一个运气好的 306us baseline**（真 baseline ~370-443）的假象——单跑对比在这个 ~25% 噪声的共享 device 上**无意义**，必须 interleaved A/B。
> 关键资产：已验证的 **baseline 关键路径**（16 任务，bubble 预算 96.5us = wall 的 32%）；改动在 **main 工作区（未提交）**：gather_kv(真收益) + merge_norm(中性，留)。

> 路径约定：以下命令运行前，按本机布局初始化一次；conda 环境位置在激活后由
> `$CONDA_PREFIX` 提供。

```bash
export WORKSPACE=/path/to/workspace
export PYPTO_LIB_ROOT="$WORKSPACE/pypto-lib"
export PTO_ISA_ROOT="$WORKSPACE/pto-isa"
export HCA_WORKTREE="$WORKSPACE/hca-early-dispatch"
export PR691_WORKTREE="$WORKSPACE/pr-691-study"
export PTOAS_ROOT=/path/to/ptoas-bin
```

---

## 0. 背景机制（Qwen3 那套 early-dispatch 设计）

来源：`pr-691`（分支 `perf/qwen3-decode-early-dispatch`，Little-oil fork，4 commit）+ `#762`（commit `99a037d`，"remove redundant SPMD task-id dependencies"）。本仓库 worktree 副本：`$PR691_WORKTREE`（分支 `study/pr-691`）。

**两个机制 + 关键任务视角**：

1. **`allow_early_resolve=True`（early dispatch）**：producer-side 调度 hint。打在任务 T 上 = "T 完成时，T 的**消费者**可被 pre-stage 到空闲核，T 一完成 doorbell 释放"。
   - **AND 规则**：T 自己被 pre-stage ⟺ T 的**所有生产者**都打了 flag（或已 complete）。
   - flag 影响**下游**；自己提不提前看**上游**。两件事独立。
   - 消费者的 `dispatch_fanin`（已 flag+完成的生产者数）要够到 `fanin_actual_count`（生产者总数）才会 pre-stage。

2. **`pl.system.task_dummy(deps=[])`（dummy 屏障）**：未 flag 的空 barrier，不依赖上层。塞进某任务 deps → 该任务有了一个 unflagged 生产者 → 按 AND 规则**不 pre-stage**。= "fanin 断路器"，用来**按住**会抢窗口的非关键任务。

3. **必须结合关键任务看**：dispatch window 是竞争资源。先定关键任务（Qwen3 里是 `q_proj`，fa 最长前驱），给它通机制①（生产者全 flag → 独占窗口），给竞争者挂机制②（按住）。**反面教材：到处打 flag → 大家都 pre-stage 互相抢 → 关键任务没优待。**

`#762` 的增量：q/k/v_proj/dcr_xgamma/x_gamma0 都是单 SPMD dispatch（单 TASK_ID）后，删掉 per-tile task-id 数组（`rope_dep_tids[71]` 等）和重复 dep 边，让关键路径对调度器更清晰。

---

## 1. HCA baseline 关键路径（**已验证，可检查**）

数据源（baseline，clean main `99a037d`，单跑）：
`build_output/_jit_attention_hca_test_20260714_230224/dfx_outputs/merged_swimlane_20260714_230308.json`

**WALL = 306.1us。** 关键链 16 任务，bubble（正 gap）合计 **96.5us = wall 的 32%** = dispatch-latency 预算。每条边的 gating producer 用 fanin + end-time 核过：

| # | task | dur(us) | [start→end] | gap | gating producer (ends) |
|---|------|---------|-------------|-----|------------------------|
| 0 | hc_pre_seed | 3.0 | 1.4→4.5 | | (root) |
| 1 | hc_pre_linear | 11.6 | 14.9→26.5 | **+10.4 bubble** | r1t6 hc_pre_seed (4.5) |
| 2 | split_pre_post | 4.8 | 29→33.8 | +2.5 | r1t7 hc_pre_linear |
| 3 | mix_x | 6.2 | 36.3→42.5 | +2.4 | r1t9 split_pre_post |
| 4 | rms_norm | 10.6 | 44.7→55.4 | +2.2 | r1t11 mix_x |
| 5 | qr_proj_matmul | 19.3 | 58.9→78.2 | +3.6 | r1t13 rms_norm (55.4) |
| 6 | qr_rms_norm_quant | 18.5 | 69.2→87.7 | −9.0 *pipeline* | r1t20 qr_proj_matmul |
| 7 | **qproj_matmul** | **45.4** | 85→130.3 | −2.8 *pipeline* | r1t21 qr_rms_quant |
| 8 | **qproj_dequant_rms_nope_rope** | **46.0** | 95.3→141.4 | −35 *pipeline* | r1t22 qproj_matmul |
| 9 | qk_pv_aic | 14.6 | 150.6→165.2 | **+9.2 bubble** | r1t23 qproj_dequant (141.4) |
| 10 | merge_norm | 17.1 | 173.7→190.8 | **+8.5 bubble** | r1t34 qk_pv (165.2) |
| 11 | proj_a_mm | 22.0 | 234.7→256.7 | **+43.9 bubble** | r1t36 merge_norm (190.8) |
| 12 | quant | 20.9 | 244.6→265.5 | −12 *pipeline* | r2t3 proj_a_mm |
| 13 | proj_b_mm | 13.4 | 275.9→289.3 | +10.4 | r2t4 quant (265.5) |
| 14 | proj_b_act | 20.4 | 275→295.4 | −14 *pipeline* | r2t5 proj_b_mm（gather 全部 O_GROUPS）|
| 15 | hc_post | 7.4 | 298.7→306.1 | +3.3 | r1t37 proj_b_act (295.4) |

> **SPMD span 口径**：每个 dispatch 的 [start→end] = 首块 start → 末块 end。负 gap = 块级流水（B 的块在 A 末块结束前就启动），不是 bug。所以"各任务 span 之和"（~315us）> wall（306us）。

### 两个 compute 大 pole
`qproj_matmul`（45）+ `qproj_dequant`（46），但**流水重叠 35us**（qproj_dequant 比 qproj_matmul 末块早 35us 启动）→ 有效串行 ~56us，不是 91us。

### 四个真 bubble（单生产者 gap）
- **merge_norm → proj_a_mm：43.9us** ← 最大，但是**异常**（见下）
- hc_pre_seed → hc_pre_linear：10.4us
- qproj_dequant → qk_pv：9.2us（注意：qk_pv 还等 `hca_gather_kv`，后者 ends 139.7，只比 qproj_dequant 早 1.7us —— 两者共同 gate）
- qk_pv → merge_norm：8.5us

### ⚠ 关键判断：merge_norm→proj_a 的 44us 是**核争用**，不是 fanin 延迟
验证显示 **proj_a_mm 的唯一生产者是 merge_norm**，却等了 44us。若是 dispatch-fanin 延迟，给 merge_norm 打 flag 应该能削（已试）—— **结果更差**。合理解释：`proj_a_mm` 是 **`O_GROUPS` 路并行 dispatch**，44us 是并行 proj_a 实例之间的**核争用**，不是生产者侧 fanin gap。给 merge_norm 打 flag → 所有实例同时抢核 → 争用更重。

**结构差异**（Qwen3 vs HCA）：Qwen3 的关键消费者 fa 由**单一**关键 q_proj 供养；HCA 的 proj_a 是**并行 reduction**，wall 取决于最慢那路。early-dispatch 不是这里的对的工具。

---

## 2. 已做实验 + 负结果

**改动**（`models/deepseek/v4/decode_sparse_attn_hca.py`，2 行）：
```python
# line 215
with pl.spmd(T * GATHER_SEGS, name_hint="hca_gather_kv", allow_early_resolve=True) as gather_tid:
# line 353
with pl.spmd(T * (H // H_TILE), name_hint="merge_norm", allow_early_resolve=True) as merge_tid:
```
思路：这俩是已 flag 的 `qk_pv`(deps=[gather_tid]) / `proj_a_mm`(deps=[merge_tid]) 的**未 flag 生产者**，按 AND 规则补全生产者集 → 让消费者 pre-stage。

**结果**（opt swimlane `build_output/_jit_attention_hca_test_20260714_234838/dfx_outputs/merged_swimlane_20260714_234858.json`）：
- golden **PASS**（kv_cache + x_out，20.88s）。
- WALL：306.1 → **351.7us（+14.9%）**。critical-path 时长反而 −57us（路径变了，wall 被新的尾巴拖长）。
- `merge_norm→proj_a_mm` bubble：44 → **52us（更大）**；`gather_kv→qk_pv` bubble 缩小（gather_kv 这个 flag 可能有正作用）。
- **未做多跑确认**（被打断）。共享设备的既有测量显示 perf 常有 3-5% 噪声；单跑 +14.9% 且目标 bubble 反而变大，强烈指向真负，但严谨起见仍需 3 跑 A/B。

---

## 3. 当前代码状态

改动现在在**两处**（内容相同）：
1. **main 工作区**（`$PYPTO_LIB_ROOT`）：`models/deepseek/v4/decode_sparse_attn_hca.py` **未提交**改动（在 `main` 上）。`git diff` 见 2 行 flag。
2. **worktree**（`$HCA_WORKTREE`）：分支 `perf/dsv4-hca-early-dispatch`（基于 `99a037d`），已提交。含实验 swimlane 日志。

**revert main 工作区**（若决定不要）：
```bash
git -C "$PYPTO_LIB_ROOT" checkout -- models/deepseek/v4/decode_sparse_attn_hca.py
```
**删 worktree + 分支**：
```bash
git -C "$PYPTO_LIB_ROOT" worktree remove "$HCA_WORKTREE"
git -C "$PYPTO_LIB_ROOT" branch -D perf/dsv4-hca-early-dispatch
```
**pr-691 学习 worktree**（如不再需要也可删）：`$PR691_WORKTREE`，分支 `study/pr-691`。

---

## 2b. 修正：interleaved A/B（§2 的"+14.9%"是假象）

§2 的负结果是把 opt(352us) 对比了一个**运气好的 306us baseline**——而真 baseline 在这个共享 device 上是**双峰/极不稳定**的（同代码 306→443us）。**单跑对比无意义**，必须 interleaved A/B（G/B 交替，抵消 device drift）。

**方法**：每对 = flag-on 跑一次 → `git stash`（关 flag）→ baseline 跑一次 → `git stash pop`。device drift 对相邻 G/B 影响≈相等，差值即 flag 净效果。

**结果（两轮 interleaved A/B，共 6 对）**：

| config | wall (us) | median | 稳定性 |
|---|---|---|---|
| **gather_kv flag** | 303, 308, 327 | ~308 | 好 |
| **gather_kv + merge_norm** | 305, 309, 314 | ~309 | 最稳 (σ≈5) |
| **baseline (无 flag)** | 309, 372, 380, 443, 377, 390 | **~385** | 不稳，冲到 443 |

- **gather_kv**：6 对里 **G < B 全部成立**，从不冲高。把 decode 锁定在快档 ~308-314。
- **merge_norm**：full config ≈ gather_kv-only（~309），**中性**（proj_a 是并行组核争用，flag 不动它；但不害）。
- **机制（直接观测）**：gather_kv 补全 qk_pv 生产者集 → qk_pv pre-stage → 调度稳定。`qproj_dequant` block span **双峰**：baseline staggered ~48us，flag 后 concurrent ~8.5us——flag 消除了 baseline 的病态调度态。

**教训**：①共享 NPU device 有 ~25% run-to-run 噪声，单跑对比会骗人（306 vs 370 同代码）；②interleaved A/B 是这里唯一靠谱测法；③`qproj_dequant` span 双峰 = 调度有 good/bad 两态，early-dispatch 的真实作用是**锁定 good 态**（稳定性收益 > 名义 bubble 收益）。

**最终改动（main 未提交）**：gather_kv（真收益）+ merge_norm（中性，按最终实验配置保留）。golden PASS。

---

## 4. 复现 / 运行

**baseline 运行命令**（device，task-submit）：
```bash
conda activate wq3
export PATH="$PTOAS_ROOT:$PATH"
export PTO2_RING_TASK_WINDOW=131072
export PTO2_RING_DEP_POOL=131072
export PTO2_RING_HEAP=4294967296
cd "$PYPTO_LIB_ROOT"   # 或 "$HCA_WORKTREE"
task-submit --device auto --max-time 1800 \
  --run "python models/deepseek/v4/decode_attention_hca.py -p a2a3 --start-pos 8192 --enable-l2-swimlane > decode_attention_hca_8k.log 2>&1" \
  2>&1 | tail -3
```
- `task-submit` 应能从 `$PATH` 解析；conda 安装位置不限，激活后的 `wq3` 环境目录为 `$CONDA_PREFIX`。
- 产物 swimlane：`build_output/_jit_attention_hca_test_<ts>/dfx_outputs/merged_swimlane_<ts>.json`（+ `deps.json`、`l2_swimlane_records.json`、`name_map_*.json`）。
- 一次 run ~30s（compile+exec+swimlane，swimlane 会跑 2 次：1 抓 dep 图、2 抓干净 timing）。

**关键路径分析脚本**（附录 A，`/tmp/hca_critpath.py` 也有一份，但 /tmp 易失）。

---

## 5. 下一步候选方向（未验证）

1. **多跑确认负结果**：opt vs baseline 各 3 跑，同 session，确认 +14.9% 是真负还是噪声。
2. **隔离实验**：撤 `merge_norm` 的 flag，只留 `gather_kv`（gather_kv→qk_pv bubble 缩小了，可能净正）。
3. **proj_a 核争用另想思路**（不是 early-dispatch）：
   - proj_a_mm 是 `O_GROUPS` 路 `pl.parallel`，44us bubble 是实例间核争用。可查：O_GROUPS 多少路？核够不够？能否扩 SPMD 宽度 / 改调度？
   - 需重点排查两类已知机制：in-place read/write 可能把 `pl.parallel` 串行化到少量核心，过细任务也可能被 per-task launch 开销吞掉收益。
4. **真单生产者 bubble 用 early-dispatch**（可能有效，区别于 proj_a）：
   - `hc_pre_seed → hc_pre_linear`（10.4us）：hc_pre_linear 在 `hc_pre.py:468`，未 flag。它是 hc_pre_seed 的唯一下游。flag hc_pre_linear → 让下游 pre-stage。
   - `qproj_dequant → qk_pv`（9.2us）/ `qk_pv → merge_norm`（8.5us）：qk_pv/merge_norm 的下游是否补全生产者集。
   - 注意：`hc_pre.py` 是 prefill/decode 共用文件，flag 要确认不影响 prefill（allow_early_resolve 是"无害 hint"，但需测）。
5. **dummy barrier 的潜在用途**：目前 HCA 路径串行主导（critical path ~占满 wall），没看到明显"非关键任务抢 qproj 窗口"。但若多跑发现 qproj 启动时有竞争（kv_score_proj / build_valid 抢核），可考虑挂 dummy 按住。

---

## 6. 参考

- **pr-691 worktree**：`$PR691_WORKTREE`（分支 `study/pr-691`，4 commit，最关键 `883c1e3` = early-dispatch infra + critical-path 重构 + fa_fused 三阶段融合）。
- **#762 commit**：`99a037d`（本仓库 main HEAD）= remove redundant SPMD task-id dependencies。
- **调度与 codegen 约束**：`allow_early_resolve` 遵循本日志 §0 的 AND 规则；多输出 unrolled scope 打 flag 曾触发括号失衡的 ptoas codegen 问题，但 HCA 的 merge_norm 是单输出 SPMD，不受影响。
- **相关文件**：
  - 入口：`models/deepseek/v4/decode_attention_hca.py`（`@pl.jit.inline attention_hca` 串联 hc_pre→rms_norm→qkv_proj_rope→compressor→sparse_attn_hca→proj_a/proj_b→hc_post）。
  - kernel：`models/deepseek/v4/decode_sparse_attn_hca.py`（gather_kv:215、qk_pv:278、merge_norm:353、proj_a_mm:442、quant:462、proj_b_mm:491、proj_b_act:522）。
  - q 投影链：`models/deepseek/v4/qkv_proj_rope.py`（qr_proj_matmul:169、qr_rms_norm_quant:187、qproj_matmul:227、qproj_dequant:249 —— 均已 flag）。
  - hc_pre：`models/deepseek/v4/hc_pre.py`（hc_pre_seed:463、hc_pre_linear:468 —— 未 flag）。
- **前序 HCA 调优**：`docs/hca-qkpv-tuning-log.zh.md`（qk_pv 静态拆分 −23%，已提交 `0e9adbc`）。

---

## 附录 A：关键路径分析脚本

存为 `hca_critpath.py`，`python3 hca_critpath.py`（改 `f` 路径指向目标 swimlane）。输出关键链 + 每个 bubbled link 的全部生产者 end-time，便于核对 gating。

```python
import json, re
f = "build_output/_jit_attention_hca_test_20260714_230224/dfx_outputs/merged_swimlane_20260714_230308.json"
ev = json.load(open(f))["traceEvents"]
X = [e for e in ev if e.get("ph") == "X"]
TID = re.compile(r"\((r\dt\d+)\)")
BASE = re.compile(r"\(r\dt\d+\)$")
TIDLIST = re.compile(r"r\dt\d+")  # fanin/fanout-hint 里的裸 id（无括号）

disp = {}
for e in X:
    m = TID.search(e["name"])
    t = m.group(1) if m else e["name"]
    a = e.get("args", {})
    fi = set(TIDLIST.findall(a.get("fanin-hint", ""))) if a.get("fanin-hint") else set()
    base = BASE.sub("", e["name"])
    d = disp.get(t)
    if d is None:
        d = {"base": base, "s": 1e18, "e": 0.0, "fi": set()}
        disp[t] = d
    d["s"] = min(d["s"], e["ts"])
    d["e"] = max(d["e"], e["ts"] + e.get("dur", 0))
    d["fi"] |= fi
for d in disp.values():
    d["w"] = d["e"] - d["s"]

leaf = max(disp, key=lambda t: disp[t]["e"])
path = []
cur = leaf
while True:
    path.append(cur)
    preds = [p for p in disp[cur]["fi"] if p in disp]
    if not preds:
        break
    cur = max(preds, key=lambda p: disp[p]["e"])
path.reverse()

wall = max(d["e"] for d in disp.values()) - min(d["s"] for d in disp.values())
posgap = sum(max(0.0, disp[path[i]]["s"] - disp[path[i - 1]]["e"]) for i in range(1, len(path)))
print("WALL=%.1fus  bubble-total=%.1fus (%.0f%%)" % (wall, posgap, posgap / wall * 100))
prev = None
for i, t in enumerate(path):
    d = disp[t]
    g = (d["s"] - prev) if prev is not None else 0.0
    tag = ("BUBBLE +%0.1f" % g) if g > 3 else (("pipeline %+0.1f" % g) if g < 0 else "")
    preds = [p for p in d["fi"] if p in disp]
    gate = max(preds, key=lambda p: disp[p]["e"]) if preds else None
    gd = ("  gated by %s(%s, ends %.1f)" % (gate, disp[gate]["base"], disp[gate]["e"])) if gate else ""
    print("%2d %7s %5.1fus [%6.1f->%6.1f]  %-34s%14s%s" % (i, t, d["w"], d["s"], d["e"], d["base"], tag, gd))
    prev = d["e"]
print("\n=== bubbled links: ALL producers + end times ===")
for i in range(1, len(path)):
    g = disp[path[i]]["s"] - disp[path[i - 1]]["e"]
    if g > 3:
        d = disp[path[i]]
        preds = sorted([p for p in d["fi"] if p in disp], key=lambda p: -disp[p]["e"])
        print("\n  %s %s starts %.1f. producers:" % (path[i], d["base"], d["s"]))
        for p in preds[:6]:
            print("     %7s ends %6.1f  %s" % (p, disp[p]["e"], disp[p]["base"]))
```
