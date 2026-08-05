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

import pytest
import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "deepseek" / "v4-pro-w8a8"
_CONFIG_PATH = _MODEL_DIR / "config.py"
_ENTRY_PATH = _MODEL_DIR / "decode_layer_compute.py"

_FFN_PARAMETER_NAMES = (
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "norm_w",
    "gate_w",
    "gate_bias",
    "tid2eid",
    "input_ids",
    "recv_x",
    "recv_scale_dq",
    "recv_weights",
    "recv_expert_count",
    "routed_w1",
    "routed_w1_scale",
    "routed_w3",
    "routed_w3_scale",
    "routed_w2",
    "routed_w2_scale",
    "shared_w1",
    "shared_w1_scale",
    "shared_w3",
    "shared_w3_scale",
    "shared_w2",
    "shared_w2_scale",
    "route_to_recv",
    "indices",
    "weights",
    "x_next",
)


def _load_config() -> dict[str, Any]:
    return runpy.run_path(str(_CONFIG_PATH))


def _parse() -> ast.Module:
    return ast.parse(_ENTRY_PATH.read_text(encoding="utf-8"), filename=str(_ENTRY_PATH))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} was not found")


def _assignment_value(tree: ast.Module, name: str) -> ast.AST:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return node.value
    raise AssertionError(f"assignment {name} was not found")


def _call(function: ast.FunctionDef, name: str) -> ast.Call:
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]
    (call,) = calls
    return call


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


def _annotations(function: ast.FunctionDef) -> dict[str, str]:
    return {argument.arg: ast.unparse(argument.annotation) for argument in function.args.args}


def test_one_die_entries_keep_hca_and_csa_physical_pools_separate() -> None:
    config = _load_config()
    assert config["DECODE_BATCH"] == 2
    assert config["DECODE_SEQ"] == 4
    assert config["KV_HCA_MAX_BLOCKS"] == 9
    assert config["DECODE_BATCH"] * config["KV_HCA_MAX_BLOCKS"] == 18
    assert config["KV_CMP_MAX_BLOCKS"] == 257
    assert config["DECODE_CMP_BLOCK_NUM"] == 514

    tree = _parse()
    hca = _annotations(_function(tree, "decode_layer_compute_hca_test"))
    csa = _annotations(_function(tree, "decode_layer_compute_csa_test"))
    assert "HCA_CMP_BLOCK_NUM_DYN" in hca["cmp_kv"]
    assert "CSA_CMP_BLOCK_NUM" in csa["cmp_kv"]
    assert hca["cmp_block_table"] == "pl.Tensor[[B, HCA_CMP_MAX_BLOCKS], pl.INT32]"
    assert csa["cmp_block_table"] == "pl.Tensor[[B, CSA_CMP_MAX_BLOCKS], pl.INT32]"


@pytest.mark.parametrize(
    ("body_name", "entry_name", "layer_constant"),
    [
        ("decode_layer_compute_hca", "decode_layer_compute_hca_test", "HCA_LAYER_ID"),
        ("decode_layer_compute_csa", "decode_layer_compute_csa_test", "CSA_LAYER_ID"),
    ],
)
def test_reusable_inline_body_has_a_thin_fixed_layer_entry(
    body_name: str,
    entry_name: str,
    layer_constant: str,
) -> None:
    tree = _parse()
    body = _function(tree, body_name)
    entry = _function(tree, entry_name)
    assert any(ast.unparse(decorator) == "pl.jit.inline" for decorator in body.decorator_list)
    assert any(ast.unparse(decorator) == "pl.jit(auto_scope=False)" for decorator in entry.decorator_list)

    body_parameters = [argument.arg for argument in body.args.args]
    entry_parameters = [argument.arg for argument in entry.args.args]
    assert body_parameters == [*entry_parameters, "layer_id"]

    body_call = _call(entry, body_name)
    assert [
        argument.id for argument in body_call.args[:-1] if isinstance(argument, ast.Name)
    ] == entry_parameters
    assert ast.unparse(body_call.args[-1]) == f"pl.const({layer_constant}, pl.INT32)"
    assert ast.unparse(entry.body[-1].value) == "(indices, weights, x_next)"


def test_inline_layer_apis_accept_manifest_layer_ids() -> None:
    tree = _parse()
    for name in ("decode_layer_compute_hca", "decode_layer_compute_csa"):
        annotations = _annotations(_function(tree, name))
        assert annotations["layer_id"] == "pl.Scalar[pl.INT32]"

    assert ast.literal_eval(_assignment_value(tree, "HCA_LAYER_ID")) == 0
    assert ast.literal_eval(_assignment_value(tree, "CSA_LAYER_ID")) == 2


def test_layer_entries_expose_head_major_q_and_all_mutable_state() -> None:
    tree = _parse()
    hca = _annotations(_function(tree, "decode_layer_compute_hca_test"))
    csa = _annotations(_function(tree, "decode_layer_compute_csa_test"))
    assert hca["wq_b"] == "pl.Tensor[[H, HEAD_DIM, Q_LORA], pl.INT8]"
    assert csa["wq_b"] == "pl.Tensor[[H, HEAD_DIM, Q_LORA], pl.INT8]"

    for name in ("compress_state", "kv_cache", "cmp_kv"):
        assert hca[name].startswith("pl.InOut[")
    for name in (
        "compress_state",
        "inner_compress_state",
        "kv_cache",
        "cmp_kv",
        "idx_kv_cache",
        "idx_kv_scale",
    ):
        assert csa[name].startswith("pl.InOut[")
    for entry_annotations in (hca, csa):
        assert entry_annotations["indices"] == "pl.Out[pl.Tensor[[T, TOPK], pl.INT32]]"
        assert entry_annotations["weights"] == "pl.Out[pl.Tensor[[T, TOPK], pl.FP32]]"
        assert entry_annotations["x_next"] == "pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]]"


def test_spec_manifests_follow_the_executable_parameter_order() -> None:
    tree = _parse()
    hca_attention = tuple(ast.literal_eval(_assignment_value(tree, "HCA_ATTENTION_SPEC_NAMES")))
    csa_attention = tuple(ast.literal_eval(_assignment_value(tree, "CSA_ATTENTION_SPEC_NAMES")))
    hca_parameters = [argument.arg for argument in _function(tree, "decode_layer_compute_hca_test").args.args]
    csa_parameters = [argument.arg for argument in _function(tree, "decode_layer_compute_csa_test").args.args]
    assert hca_parameters == [*hca_attention, *_FFN_PARAMETER_NAMES]
    assert csa_parameters == [*csa_attention, *_FFN_PARAMETER_NAMES]

    hca_builder = ast.unparse(_function(tree, "build_hca_tensor_specs"))
    csa_builder = ast.unparse(_function(tree, "build_csa_tensor_specs"))
    assert "build_hca_attention_tensor_specs(start_pos)" in hca_builder
    assert "build_csa_attention_tensor_specs(start_pos)" in csa_builder
    assert "name in HCA_MUTABLE_NAMES" in hca_builder
    assert "name in CSA_MUTABLE_NAMES" in csa_builder


def test_layer_body_orders_attention_before_the_ffn_proxy() -> None:
    tree = _parse()
    for body_name, attention_name in (
        ("decode_layer_compute_hca", "attention_hca"),
        ("decode_layer_compute_csa", "attention_csa"),
    ):
        body = _function(tree, body_name)
        attention_call = _call(body, attention_name)
        ffn_call = _call(body, "ffn_compute_proxy")
        assert attention_call.lineno < ffn_call.lineno
        assert ast.unparse(attention_call.args[-1]) == "x_attn"
        assert ast.unparse(ffn_call.args[0]) == "x_attn"
        assert ast.unparse(ffn_call.args[-1]) == "layer_id"


def test_ffn_proxy_composes_all_compute_leaves_and_prevents_routed_dce() -> None:
    tree = _parse()
    proxy = _function(tree, "ffn_compute_proxy")
    ordered_names = (
        "hc_pre",
        "gate",
        "expert_shared",
        "expert_routed",
        "local_proxy_combine",
        "hc_post",
    )
    line_numbers = [_call(proxy, name).lineno for name in ordered_names]
    assert line_numbers == sorted(line_numbers)
    assert ast.unparse(_call(proxy, "expert_routed").args[-1]) == "recv_y"
    assert [ast.unparse(argument) for argument in _call(proxy, "local_proxy_combine").args] == [
        "sh",
        "recv_y",
        "route_to_recv",
        "ffn_out",
    ]
    assert [ast.unparse(argument) for argument in _call(proxy, "hc_post").args] == [
        "ffn_out",
        "x_attn",
        "post",
        "comb",
        "x_next",
    ]


def test_local_proxy_combine_consumes_six_host_mapped_rows_per_token() -> None:
    combine = _function(_parse(), "local_proxy_combine")
    source = ast.unparse(combine)
    assert "pl.reshape(recv_y, [N_LOCAL_EXPERTS * RECV_MAX, D])" in source
    assert "for token in pl.spmd(T, name_hint='local_proxy_combine')" in source
    assert "for route in pl.range(TOPK)" in source
    assert "pl.read(route_to_recv, [token, route])" in source
    assert "recv_y_flat[recv_row:recv_row + 1, :]" in source
    assert "acc = pl.add(acc, routed_row_fp32)" in source
    assert "target_type=pl.BF16" in source


@pytest.mark.parametrize("workload", ["balanced", "skewed", "tail"])
def test_host_route_map_is_a_permutation_of_all_valid_rows(workload: str) -> None:
    tree = _parse()
    function = _function(tree, "build_route_to_recv")
    namespace = {
        "ROUTED_WORKLOAD_COUNTS": {
            "balanced": (16, 16, 16),
            "skewed": (48, 0, 0),
            "tail": (17, 16, 15),
        },
        "RECV_MAX": 1024,
        "T": 8,
        "TOPK": 6,
    }
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, filename=str(_ENTRY_PATH), mode="exec"), namespace)
    route_map = namespace["build_route_to_recv"](workload)

    expected = []
    for expert, count in enumerate(namespace["ROUTED_WORKLOAD_COUNTS"][workload]):
        expected.extend(expert * namespace["RECV_MAX"] + row for row in range(count))
    assert route_map.shape == (8, 6)
    assert route_map.dtype == torch.int32
    assert sorted(route_map.reshape(-1).tolist()) == sorted(expected)
    assert len(set(route_map.reshape(-1).tolist())) == 48


def test_goldens_are_sequential_and_honor_optional_manifest_layer_id() -> None:
    tree = _parse()
    for name, attention_name, default_name in (
        ("golden_decode_layer_compute_hca", "golden_attention_hca", "HCA_LAYER_ID"),
        ("golden_decode_layer_compute_csa", "golden_attention_csa", "CSA_LAYER_ID"),
    ):
        golden = _function(tree, name)
        source = ast.unparse(golden)
        assert f"{attention_name}(attention_tensors)" in source
        assert "attention_tensors['x_out'] = x_attn" in source
        assert f"tensors.get('layer_id', {default_name})" in source
        assert "_golden_ffn_compute_proxy(tensors, x_attn, layer_id)" in source

    ffn_golden = ast.unparse(_function(tree, "_golden_ffn_compute_proxy"))
    ordered = (
        "golden_hc_pre",
        "golden_gate_core",
        "golden_expert_shared",
        "golden_expert_routed",
        "golden_hc_post",
    )
    assert [ffn_golden.index(name) for name in ordered] == sorted(ffn_golden.index(name) for name in ordered)


def test_layer_compute_surface_is_explicitly_not_distributed_ep128() -> None:
    tree = _parse()
    module_doc = ast.get_docstring(tree)
    assert module_doc is not None
    assert "synthetic" in module_doc.lower()
    assert "not an ep128" in module_doc.lower()

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
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden_calls


def test_cli_exposes_single_device_simulator_replay_and_swimlane_controls() -> None:
    tree = _parse()
    arguments = _argument_calls(tree)
    required = {
        "--platform",
        "--device",
        "--kind",
        "--start-pos",
        "--workload",
        "--enable-l2-swimlane",
        "--compile-only",
        "--save-data",
        "--golden-data",
    }
    assert required <= arguments.keys()
    assert "--ep" not in arguments
    assert ast.literal_eval(_keyword(arguments["--platform"], "choices")) == (
        "a2a3sim",
        "a2a3",
    )
    assert ast.literal_eval(_keyword(arguments["--enable-l2-swimlane"], "choices")) == (
        0,
        1,
        2,
        4,
    )
    run_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_jit"
    )
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in run_call.keywords}
    assert keywords["compile_only"] == "args.compile_only"
    assert keywords["save_data"] == "args.save_data"
    assert keywords["golden_data"] == "args.golden_data"
    assert "device_id=args.device" in keywords["runtime_cfg"]
    assert "enable_l2_swimlane=args.enable_l2_swimlane" in keywords["runtime_cfg"]
