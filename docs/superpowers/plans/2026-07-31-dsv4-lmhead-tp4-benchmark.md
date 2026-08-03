# DeepSeek V4 LM-Head TP4 Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and measure a projection-only DeepSeek V4 Flash LM-head TP4 fixture that produces assembled FP32 logits without greedy sampling, then compare its latency with the 779.316125 us AscendC trace median.

**Architecture:** Reuse the existing `lm_head_test` projection kernel and add a sibling distributed host entry that omits `sampled_ids`. Keep the current sampling-inclusive entry as the default, and select the new path only through `--projection-only`; validate the import-time TP4 specialization and both fixture contracts in an isolated subprocess.

**Tech Stack:** Python 3.10, PyPTO JIT/host DSL, PyTorch golden fixtures, pytest, Ruff, Ascend A2/A3 runtime, `task-submit`, L2 swimlane tracing.

## Global Constraints

- Base the branch exactly on `upstream/dsv4-ascendc-decode-compare2` commit `a79c141c30851e4adaaa5925f99db597fbb6082e`.
- Use one TP4 group: `--tp 4 --dp 4 --num-tokens 8`.
- Keep hidden size 4096, vocabulary size 129280, vocabulary shard size 32320, gathered group rows 32, BF16 weights, and FP32 logits.
- Measure from already-normalized hidden states through TP dispatch, local shard matmul, TP combine, and completion-counter clear.
- Exclude final RMSNorm, HC head, greedy sampling, and ArgMax.
- Do not change LM-head tiling, collective topology, numerical behavior, full-decode integration, or daily automation.
- Preserve the existing projection-plus-greedy-sampling standalone behavior as the default.
- Use `trace_view_a3_decode.json` model ID 47 only; compare against its `RmsNorm` task 59 end to `aclnnInplaceCopy_CastAiCore_Cast` task 66 end median of `779.316125 us`.
- Do not pool model ID 48 or include the later `aclnnArgMax_CastAiCore_Cast`.
- Keep generated builds, logs, traces, and summaries under ignored `build_output/`; do not commit them.
- Run repository Python checks in the `wq3` Conda environment.

---

### Task 1: Add the TP4 projection-only fixture contract

**Files:**
- Create: `tests/contract/test_deepseek_v4_flash_lm_head.py`
- Test: `tests/contract/test_deepseek_v4_flash_lm_head.py`

**Interfaces:**
- Consumes: import-time specialization through `sys.argv = [lm_head.py, "--tp", "4", "--dp", "4"]`
- Produces: contract assertions for `l3_lm_head_projection`, `build_tensor_specs(num_tokens, *, with_sampling)`, the preserved `l3_lm_head`, and the `--projection-only` CLI option

- [ ] **Step 1: Write the subprocess contract test**

Create `tests/contract/test_deepseek_v4_flash_lm_head.py` with this complete content:

```python
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "deepseek" / "v4-flash"
_LM_HEAD_PATH = _MODEL_DIR / "lm_head.py"
_PROBE_PREFIX = "LM_HEAD_CONTRACT="
_CLI_PROBE_PREFIX = "LM_HEAD_CLI="
_PROBE = f"""
import importlib.util
import inspect
import json
import sys
from pathlib import Path

from pypto.jit.decorator import _classify_params, _get_func_def

path = Path({str(_LM_HEAD_PATH)!r})
sys.argv = [str(path), "--tp", "4", "--dp", "4"]
spec = importlib.util.spec_from_file_location("dsv4_lm_head_contract_probe", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def function_contract(function):
    if function is None:
        return None
    signature = inspect.signature(function._func)
    out_parameters, inout_parameters, _, _, _ = _classify_params(
        _get_func_def(function._func)
    )
    return {{
        "parameters": list(signature.parameters),
        "out_parameters": out_parameters,
        "inout_parameters": inout_parameters,
        "annotations": {{
            name: {{
                "shape": list(parameter.annotation.shape),
                "dtype": str(parameter.annotation.dtype),
            }}
            for name, parameter in signature.parameters.items()
        }},
    }}


build_parameters = inspect.signature(module.build_tensor_specs).parameters
supports_with_sampling = "with_sampling" in build_parameters
projection_specs = None
if supports_with_sampling:
    projection_specs = module.build_tensor_specs(8, with_sampling=False)

payload = {{
    "dimensions": {{
        "tp_size": module.TP_SIZE,
        "dp_size": module.DP_SIZE,
        "hidden_size": module.D,
        "vocab_size": module.VOCAB,
        "vocab_per_tp": module.VOCAB_PER_TP,
        "max_logit_rows": module.MAX_LOGIT_ROWS,
        "group_logit_rows": module.GROUP_LOGIT_ROWS,
    }},
    "projection_signature": function_contract(
        getattr(module, "l3_lm_head_projection", None)
    ),
    "sampling_signature": function_contract(module.l3_lm_head),
    "supports_with_sampling": supports_with_sampling,
    "default_specs": [
        {{
            "name": tensor_spec.name,
            "shape": list(tensor_spec.shape),
            "dtype": str(tensor_spec.dtype),
            "is_output": tensor_spec.is_output,
        }}
        for tensor_spec in module.build_tensor_specs(8)
    ],
    "projection_specs": (
        [
            {{
                "name": tensor_spec.name,
                "shape": list(tensor_spec.shape),
                "dtype": str(tensor_spec.dtype),
                "is_output": tensor_spec.is_output,
            }}
            for tensor_spec in projection_specs
        ]
        if projection_specs is not None
        else None
    ),
}}
print({_PROBE_PREFIX!r} + json.dumps(payload, sort_keys=True))
"""
_CLI_PROBE = """
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import golden

path = Path(sys.argv[1])
captured = {}


def capture_run_jit(**kwargs):
    mode = "projection" if "--projection-only" in sys.argv else "default"
    captured[mode] = {
        "function": kwargs["fn"]._func.__name__,
        "specs": [tensor_spec.name for tensor_spec in kwargs["specs"]],
        "comparators": sorted(kwargs["compare_fn"]),
    }
    return SimpleNamespace(passed=True, error=None)


golden.run_jit = capture_run_jit
for extra_args in ([], ["--projection-only"]):
    sys.argv = [
        str(path),
        "--tp",
        "4",
        "--dp",
        "4",
        "--num-tokens",
        "8",
        "-d",
        "0,1,2,3",
        *extra_args,
    ]
    runpy.run_path(str(path), run_name="__main__")

print("LM_HEAD_CLI=" + json.dumps(captured, sort_keys=True))
"""


def _probe_environment() -> dict[str, str]:
    environment = os.environ.copy()
    python_path = [
        str(_MODEL_DIR),
        str(_REPO_ROOT),
    ]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    return environment


@pytest.fixture(scope="module")
def lm_head_contract() -> dict:
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=_REPO_ROOT,
        env=_probe_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    contract_line = next(
        line
        for line in completed.stdout.splitlines()
        if line.startswith(_PROBE_PREFIX)
    )
    return json.loads(contract_line.removeprefix(_PROBE_PREFIX))


def _probe_cli_selection() -> dict:
    completed = subprocess.run(
        [sys.executable, "-c", _CLI_PROBE, str(_LM_HEAD_PATH)],
        cwd=_REPO_ROOT,
        env=_probe_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    contract_line = next(
        line
        for line in completed.stdout.splitlines()
        if line.startswith(_CLI_PROBE_PREFIX)
    )
    return json.loads(contract_line.removeprefix(_CLI_PROBE_PREFIX))


def test_tp4_projection_fixture_exposes_only_fp32_logits(lm_head_contract) -> None:
    assert lm_head_contract["dimensions"] == {
        "tp_size": 4,
        "dp_size": 4,
        "hidden_size": 4096,
        "vocab_size": 129280,
        "vocab_per_tp": 32320,
        "max_logit_rows": 8,
        "group_logit_rows": 32,
    }
    assert lm_head_contract["projection_signature"] == {
        "parameters": [
            "hidden_states",
            "lm_head_weight",
            "logits",
            "logit_row_indices",
        ],
        "out_parameters": ["logits"],
        "inout_parameters": [],
        "annotations": {
            "hidden_states": {
                "shape": [4, 16, 4096],
                "dtype": "bfloat16",
            },
            "lm_head_weight": {
                "shape": [4, 32320, 4096],
                "dtype": "bfloat16",
            },
            "logits": {
                "shape": [4, 8, 129280],
                "dtype": "fp32",
            },
            "logit_row_indices": {
                "shape": [4, 8],
                "dtype": "int32",
            },
        },
    }
    assert lm_head_contract["sampling_signature"] == {
        "parameters": [
            "hidden_states",
            "lm_head_weight",
            "logits",
            "sampled_ids",
            "logit_row_indices",
        ],
        "out_parameters": ["logits", "sampled_ids"],
        "inout_parameters": [],
        "annotations": {
            "hidden_states": {
                "shape": [4, 16, 4096],
                "dtype": "bfloat16",
            },
            "lm_head_weight": {
                "shape": [4, 32320, 4096],
                "dtype": "bfloat16",
            },
            "logits": {
                "shape": [4, 8, 129280],
                "dtype": "fp32",
            },
            "sampled_ids": {
                "shape": [4, 8, 8],
                "dtype": "int32",
            },
            "logit_row_indices": {
                "shape": [4, 8],
                "dtype": "int32",
            },
        },
    }


def test_projection_specs_omit_sampling_without_changing_default(
    lm_head_contract,
) -> None:
    assert lm_head_contract["supports_with_sampling"] is True
    assert [spec["name"] for spec in lm_head_contract["default_specs"]] == [
        "hidden_states",
        "lm_head_weight",
        "logits",
        "sampled_ids",
        "logit_row_indices",
    ]
    assert [spec["name"] for spec in lm_head_contract["projection_specs"]] == [
        "hidden_states",
        "lm_head_weight",
        "logits",
        "logit_row_indices",
    ]
    projection_logits = lm_head_contract["projection_specs"][2]
    assert projection_logits == {
        "name": "logits",
        "shape": [4, 8, 129280],
        "dtype": "torch.float32",
        "is_output": True,
    }


def test_cli_selects_projection_and_preserves_sampling_default() -> None:
    selections = _probe_cli_selection()

    assert selections == {
        "default": {
            "function": "l3_lm_head",
            "specs": [
                "hidden_states",
                "lm_head_weight",
                "logits",
                "sampled_ids",
                "logit_row_indices",
            ],
            "comparators": ["logits", "sampled_ids"],
        },
        "projection": {
            "function": "l3_lm_head_projection",
            "specs": [
                "hidden_states",
                "lm_head_weight",
                "logits",
                "logit_row_indices",
            ],
            "comparators": ["logits"],
        },
    }
```

The production mutations caught by these tests are: removing the projection
host entry, losing its compiler-visible `pl.Out` direction, accidentally adding
`sampled_ids` to it, changing its FP32 output shape or dtype, changing the full
legacy host ABI, removing sampling from the default CLI path, ignoring
`with_sampling=False`, or routing `--projection-only` to the wrong fixture,
specs, or comparators.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
conda run --no-capture-output -n wq3 \
  python -m pytest tests/contract/test_deepseek_v4_flash_lm_head.py -v
```

Expected: the fixture/spec assertions fail because `l3_lm_head_projection` and
`with_sampling` are missing, and the CLI probe rejects `--projection-only`.
The base module import must succeed; fix test setup if it errors for any
unrelated reason.

---

### Task 2: Implement projection-only host and CLI selection

**Files:**
- Modify: `models/deepseek/v4-flash/lm_head.py:470-579`
- Modify: `models/deepseek/v4-flash/lm_head.py:624-656`
- Test: `tests/contract/test_deepseek_v4_flash_lm_head.py`

**Interfaces:**
- Consumes: existing `lm_head_test(hidden_states, lm_head_weight, logit_row_indices, logits, hidden_window, hidden_done, logits_window, logits_done, group_base, tp_rank, done_epoch)`
- Produces: `l3_lm_head_projection(hidden_states, lm_head_weight, logits, logit_row_indices)` and `build_tensor_specs(num_tokens=TEST_TOKENS, *, with_sampling=True)`

- [ ] **Step 1: Add the minimal projection-only host entry**

Insert this function immediately before the existing `l3_lm_head`:

```python
@pl.jit.host
def l3_lm_head_projection(
    hidden_states: pl.Tensor[[DP_SIZE, TEST_TOKENS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[DP_SIZE, VOCAB_PER_TP, D], pl.BF16],
    logits: pl.Out[pl.Tensor[[DP_SIZE, MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    logit_row_indices: pl.Tensor[[DP_SIZE, MAX_LOGIT_ROWS], pl.INT32],
):
    hidden_window_buf = pld.alloc_window_buffer(GROUP_LOGIT_ROWS * D * 2)
    logits_window_buf = pld.alloc_window_buffer(MAX_LOGIT_ROWS * VOCAB * 4)
    hidden_done_buf = pld.alloc_window_buffer(TP_SIZE * 4)
    logits_done_buf = pld.alloc_window_buffer(TP_SIZE * 4)

    for r in pl.range(pld.world_size()):
        hidden_window = pld.window(
            hidden_window_buf,
            [GROUP_LOGIT_ROWS, D],
            dtype=pl.BF16,
        )
        hidden_done = pld.window(
            hidden_done_buf,
            [TP_SIZE, 1],
            dtype=pl.INT32,
        )
        logits_window = pld.window(
            logits_window_buf,
            [MAX_LOGIT_ROWS, VOCAB],
            dtype=pl.FP32,
        )
        logits_done = pld.window(
            logits_done_buf,
            [TP_SIZE, 1],
            dtype=pl.INT32,
        )
        lm_head_test(
            hidden_states[r],
            lm_head_weight[r],
            logit_row_indices[r],
            logits[r],
            hidden_window,
            hidden_done,
            logits_window,
            logits_done,
            r // TP_SIZE * TP_SIZE,
            r % TP_SIZE,
            DONE_VALUE,
            device=r,
        )
```

This deliberately duplicates only the host window setup; it does not alter the
projection kernel, tiling constants, or collectives.

- [ ] **Step 2: Make sampled tensor creation optional while preserving default order**

Change the signature to:

```python
def build_tensor_specs(num_tokens=TEST_TOKENS, *, with_sampling=True):
```

Replace the direct `return [...]` with a local `specs` list that contains
`hidden_states`, `lm_head_weight`, and `logits` in the current order. Then append
the existing sampled-ID spec only when requested:

```python
    if with_sampling:
        specs.append(
            TensorSpec(
                "sampled_ids",
                [DP_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD],
                torch.int32,
                is_output=True,
            )
        )
```

Finally append the existing `logit_row_indices` spec and return:

```python
    specs.append(
        TensorSpec(
            "logit_row_indices",
            [DP_SIZE, MAX_LOGIT_ROWS],
            torch.int32,
            init_value=init_logit_row_indices,
        )
    )
    return specs
```

The default call must retain the exact current order:
`hidden_states`, `lm_head_weight`, `logits`, `sampled_ids`,
`logit_row_indices`. The `with_sampling=False` order must be:
`hidden_states`, `lm_head_weight`, `logits`, `logit_row_indices`.

- [ ] **Step 3: Add CLI selection without changing the default**

Add this argument beside `--num-tokens`:

```python
    parser.add_argument(
        "--projection-only",
        action="store_true",
        help="Benchmark LM-head projection without greedy sampling",
    )
```

Replace the unconditional fixture selection with:

```python
    golden_fn = golden_lm_head
    if args.projection_only:
        fn = l3_lm_head_projection
        specs = build_tensor_specs(args.num_tokens, with_sampling=False)
        compare_fn = {
            "logits": compare_logits,
        }
    else:
        fn = l3_lm_head
        specs = build_tensor_specs(args.num_tokens)
        compare_fn = {
            "logits": compare_logits,
            "sampled_ids": compare_sampled_ids,
        }
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
conda run --no-capture-output -n wq3 \
  python -m pytest tests/contract/test_deepseek_v4_flash_lm_head.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Mutation-check output direction and CLI routing**

Temporarily change the projection host's logits annotation from:

```python
logits: pl.Out[pl.Tensor[[DP_SIZE, MAX_LOGIT_ROWS, VOCAB], pl.FP32]]
```

to:

```python
logits: pl.Tensor[[DP_SIZE, MAX_LOGIT_ROWS, VOCAB], pl.FP32]
```

Run:

```bash
conda run --no-capture-output -n wq3 \
  python -m pytest \
    tests/contract/test_deepseek_v4_flash_lm_head.py::test_tp4_projection_fixture_exposes_only_fp32_logits \
    -v
```

Expected: FAIL because compiler classification reports no `logits` output.
Restore the `pl.Out[...]` annotation immediately.

Then temporarily route the projection branch to:

```python
fn = l3_lm_head
```

Run:

```bash
conda run --no-capture-output -n wq3 \
  python -m pytest \
    tests/contract/test_deepseek_v4_flash_lm_head.py::test_cli_selects_projection_and_preserves_sampling_default \
    -v
```

Expected: FAIL because the projection mode captured `l3_lm_head`. Restore
`fn = l3_lm_head_projection`, then rerun the full focused file and require
`3 passed`.

- [ ] **Step 6: Run static and repository regression checks**

Run:

```bash
conda run --no-capture-output -n wq3 \
  python -m pytest tests/golden -q
conda run --no-capture-output -n wq3 \
  python tests/lint/check_headers.py
conda run --no-capture-output -n wq3 \
  python tests/lint/check_english_only.py
conda run --no-capture-output -n wq3 \
  ruff check models/deepseek/v4-flash/lm_head.py \
    tests/contract/test_deepseek_v4_flash_lm_head.py
git diff --check
```

Expected: 169 golden tests pass; both lint scripts, Ruff, and
`git diff --check` exit successfully.

- [ ] **Step 7: Commit the independently testable implementation**

Use the repository `git-commit` workflow: review the exact diff, run
`conda run --no-capture-output -n wq3 pre-commit run --all-files`, stage only
the plan, contract test, and `lm_head.py`, then commit:

```bash
git add \
  docs/superpowers/plans/2026-07-31-dsv4-lmhead-tp4-benchmark.md \
  models/deepseek/v4-flash/lm_head.py \
  tests/contract/test_deepseek_v4_flash_lm_head.py
git commit -m "Add: projection-only DSV4 LM-head benchmark"
```

---

### Task 3: Verify TP4 correctness and generated task boundaries

**Files:**
- Generate only: `build_output/` runtime, build, and L2 swimlane artifacts
- Modify: none

**Interfaces:**
- Consumes: `l3_lm_head_projection`, the `--projection-only` CLI mode, four A2/A3 devices, and `golden_lm_head`
- Produces: one four-device correctness result and one L2 task-name audit

- [ ] **Step 1: Discover the current task-submit launch syntax**

Run:

```bash
task-submit --help
sed -n '1,240p' tools/run_dsv4_daily_perf.sh
```

Use the repository runner's current `task-submit` options. Do not invoke
`npu-smi` directly, because device access is brokered by `task-submit`.

- [ ] **Step 2: Run four-device projection-only correctness**

Inside the submitted job, activate the `wq3` environment, source
`/usr/local/Ascend/cann-9.0.0/set_env.sh`, set
`PTOAS_ROOT=/usr/local/bin/ptoas-bin`, derive `PTO_ISA_ROOT` from the primary
checkout's sibling `pto-isa`, change to this worktree, and run:

```bash
python models/deepseek/v4-flash/lm_head.py \
  -p a2a3 \
  --tp 4 \
  --dp 4 \
  --num-tokens 8 \
  --projection-only \
  -d "$TASK_DEVICE" \
  --enable-l2-swimlane 0
```

Capture the submitted job output under
`build_output/lmhead_tp4_projection_correctness.log`. Expected: compilation,
four-rank execution, and FP32 logits comparison all succeed.

- [ ] **Step 3: Collect one L2 swimlane run**

Run the same four-device command once with:

```bash
--enable-l2-swimlane 1
```

Keep its output and trace under `build_output/lmhead_tp4_projection_swimlane/`.

- [ ] **Step 4: Audit generated task names**

Search the generated L2 JSON:

```bash
rg -o \
  'lm_head_(dispatch_(push|wait|gather)|matmul|combine_(push|wait|gather)|signal_clear|greedy_sample)' \
  build_output/lmhead_tp4_projection_swimlane \
  | sort -u
```

Expected task set:

```text
lm_head_combine_gather
lm_head_combine_push
lm_head_combine_wait
lm_head_dispatch_gather
lm_head_dispatch_push
lm_head_dispatch_wait
lm_head_matmul
lm_head_signal_clear
```

`lm_head_greedy_sample` must be absent. Do not commit generated files.

---

### Task 4: Run formal timing and compare with the trace median

**Files:**
- Generate only: `build_output/lmhead_tp4_projection_benchmark/`
- Modify: none

**Interfaces:**
- Consumes: verified four-device projection-only fixture and formal AscendC median `779.316125 us`
- Produces: 100 measured rounds after five warmups, slowest-rank effective median/P90, delta, ratio, and speedup

- [ ] **Step 1: Run the formal four-device benchmark**

Submit a fresh four-device job with L2 swimlane disabled:

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

Save the job output under
`build_output/lmhead_tp4_projection_benchmark/benchmark.log`.

- [ ] **Step 2: Verify benchmark completeness**

Use the benchmark summary and raw round records to verify exactly five discarded
warmups and 100 parseable measured rounds for every rank. Stop and diagnose if
any rank has fewer than 100 measured rounds.

- [ ] **Step 3: Calculate the semantic comparison**

Take the slowest-rank effective-latency distribution from the 100 measured
rounds and report its median and P90 in microseconds. Compute:

```python
reference_median_us = 779.316125
delta_us = pypto_median_us - reference_median_us
ratio = pypto_median_us / reference_median_us
speedup = reference_median_us / pypto_median_us
```

State explicitly that the PyPTO measurement starts with already-normalized
hidden states and ends at assembled FP32 logits, while the trace comparison
uses model ID 47 from `RmsNorm` task 59 end to model-output Cast task 66 end.
Do not include greedy sampling, ArgMax, or model ID 48.

- [ ] **Step 4: Re-run completion verification**

Before reporting completion, freshly run:

```bash
conda run --no-capture-output -n wq3 \
  python -m pytest tests/contract/test_deepseek_v4_flash_lm_head.py -v
conda run --no-capture-output -n wq3 \
  python -m pytest tests/golden -q
conda run --no-capture-output -n wq3 \
  pre-commit run --all-files
git diff --check
git status --short --branch
```

Confirm that only intended source, test, and plan commits are on the feature
branch, and that no `build_output/` artifact is tracked.
