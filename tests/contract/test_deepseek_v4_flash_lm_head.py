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
