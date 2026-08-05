# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import ast
import runpy
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "deepseek" / "v4-pro-w8a8"
_CONFIG_PATH = _MODEL_DIR / "config.py"
_ENTRY_PATH = _MODEL_DIR / "moe_compute.py"


def _load_config() -> dict[str, Any]:
    return runpy.run_path(str(_CONFIG_PATH))


def _parse() -> ast.Module:
    return ast.parse(_ENTRY_PATH.read_text(encoding="utf-8"), filename=str(_ENTRY_PATH))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} was not found")


def _calls(function: ast.FunctionDef, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


def _argument_calls(tree: ast.Module) -> dict[str, ast.Call]:
    calls = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                calls[argument.value] = node
    return calls


def _keyword(call: ast.Call, name: str) -> ast.AST:
    return next(keyword.value for keyword in call.keywords if keyword.arg == name)


def test_compute_entry_uses_exact_ep128_single_die_shapes() -> None:
    config = _load_config()
    model = config["ACTIVE"]

    assert config["MOE_TOKENS"] == 8
    assert model.hidden_size == 7168
    assert model.moe_intermediate_size == 3072
    assert model.n_routed_experts == 384
    assert model.num_experts_per_tok == 6
    assert config["MOE_LOCAL_EXPERTS"] == 3
    assert config["MOE_DEPLOYMENT_RECV_MAX"] == 1024

    function = _function(_parse(), "moe_compute")
    annotations = {argument.arg: ast.unparse(argument.annotation) for argument in function.args.args}
    assert annotations["gate_w"] == "pl.Tensor[[N_EXPERTS, D], pl.FP32]"
    assert annotations["recv_x"] == "pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8]"
    assert annotations["recv_expert_count"] == "pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32]"
    assert annotations["routed_w1"] == "pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8]"
    assert annotations["routed_w2"] == "pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8]"
    assert annotations["x_norm_i8"] == "pl.Out[pl.Tensor[[T, D], pl.INT8]]"
    assert annotations["x_norm_scale"] == "pl.Out[pl.Tensor[[T, 1], pl.FP32]]"
    assert annotations["indices"] == "pl.Out[pl.Tensor[[T, TOPK], pl.INT32]]"
    assert annotations["weights"] == "pl.Out[pl.Tensor[[T, TOPK], pl.FP32]]"
    assert annotations["sh"] == "pl.Out[pl.Tensor[[T, D], pl.BF16]]"
    assert annotations["recv_y"] == "pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16]]"
    assert annotations["layer_id"] == "pl.Scalar[pl.INT32]"
    assert annotations["num_tokens"] == "pl.Scalar[pl.INT32]"
    assert all("N_RANKS" not in annotation and "DEPLOYMENT_EP" not in annotation for annotation in annotations.values())


def test_compute_graph_composes_gate_shared_and_synthetic_routed_inputs() -> None:
    function = _function(_parse(), "moe_compute")
    gate_call, = _calls(function, "gate")
    shared_call, = _calls(function, "expert_shared")
    routed_call, = _calls(function, "expert_routed")

    gate_names = [argument.id for argument in gate_call.args if isinstance(argument, ast.Name)]
    shared_names = [argument.id for argument in shared_call.args if isinstance(argument, ast.Name)]
    routed_names = [argument.id for argument in routed_call.args if isinstance(argument, ast.Name)]
    assert gate_names[-4:] == ["x_norm_i8", "x_norm_scale", "indices", "weights"]
    assert shared_names[:2] == ["x_norm_i8", "x_norm_scale"]
    assert shared_names[-1] == "sh"
    assert routed_names[:4] == ["recv_x", "recv_scale_dq", "recv_weights", "recv_expert_count"]
    assert routed_names[-1] == "recv_y"

    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    assert len(returns) == 1
    assert ast.unparse(returns[0].value) == "(x_norm_i8, x_norm_scale, indices, weights, sh, recv_y)"


def test_compute_entry_has_no_distributed_surface() -> None:
    tree = _parse()
    forbidden_modules = {
        "pypto.language.distributed",
        "pypto.ir.distributed_compiled_program",
    }
    forbidden_calls = {
        "alloc_window_buffer",
        "window",
        "remote_store",
        "put",
        "notify",
        "wait",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name not in forbidden_modules for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden_modules
            assert all(alias.name != "DistributedConfig" for alias in node.names)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden_calls
        if isinstance(node, ast.Name):
            assert node.id not in {"pld", "DistributedConfig"}


def test_compute_cli_accepts_one_device_and_exposes_replay_controls() -> None:
    tree = _parse()
    arguments = _argument_calls(tree)
    required = {
        "--device",
        "--workload",
        "--fixture",
        "--compile-only",
        "--save-data",
        "--golden-data",
        "--enable-l2-swimlane",
    }
    assert required <= arguments.keys()
    assert "--ep" not in arguments

    device_type = _keyword(arguments["--device"], "type")
    assert isinstance(device_type, ast.Name) and device_type.id == "int"
    l2_choices = ast.literal_eval(_keyword(arguments["--enable-l2-swimlane"], "choices"))
    assert 4 in l2_choices

    run_jit_call, = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_jit"
    ]
    run_keywords = {keyword.arg: keyword.value for keyword in run_jit_call.keywords}
    assert ast.unparse(run_keywords["compile_only"]) == "args.compile_only"
    assert ast.unparse(run_keywords["save_data"]) == "args.save_data"
    assert ast.unparse(run_keywords["golden_data"]) == "args.golden_data"
    assert "DistributedConfig" not in ast.unparse(run_keywords["compile_cfg"])
    assert "device_id=args.device" in ast.unparse(run_keywords["runtime_cfg"])


def test_compute_fixtures_reuse_all_leaf_workloads() -> None:
    tree = _parse()
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert {"GATE_FIXTURES", "ROUTED_WORKLOAD_COUNTS"} <= imports

    builder = _function(tree, "build_tensor_specs")
    builder_source = ast.unparse(builder)
    assert "build_gate_tensor_specs" in builder_source
    assert "build_shared_tensor_specs" in builder_source
    assert "build_routed_tensor_specs" in builder_source
    assert "workload=workload" in builder_source
    assert "torch.ones(T, D)" in builder_source
