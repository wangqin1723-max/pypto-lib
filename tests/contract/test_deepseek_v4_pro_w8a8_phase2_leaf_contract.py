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

_PHASE2_PRIMARY_ENTRIES = (
    "gate.py",
    "expert_shared.py",
    "expert_routed.py",
    "decode_attention_swa.py",
)
_PHASE2_TRANSITIVE_DEPENDENCIES = (
    "decode_sparse_attn_swa.py",
    "qkv_proj_rope.py",
    "rmsnorm.py",
    "hc_pre.py",
    "hc_post.py",
    "decode_metadata.py",
)
_REQUIRED_CLI_FLAGS = {
    "--compile-only",
    "--save-data",
    "--golden-data",
    "--enable-l2-swimlane",
}


def _load_config() -> dict[str, Any]:
    return runpy.run_path(str(_CONFIG_PATH))


def _parse(filename: str) -> ast.Module:
    path = _MODEL_DIR / filename
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _literal_assignment(filename: str, name: str) -> Any:
    for node in _parse(filename).body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{filename} does not define a literal {name}")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} was not found")


def _config_imports(tree: ast.Module) -> set[tuple[str, str | None]]:
    return {
        (alias.name, alias.asname)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "config"
        for alias in node.names
    }


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    comparison = node.test
    if len(comparison.ops) != 1 or not isinstance(comparison.ops[0], ast.Eq):
        return False
    operands = (comparison.left, *comparison.comparators)
    names = {operand.id for operand in operands if isinstance(operand, ast.Name)}
    values = {operand.value for operand in operands if isinstance(operand, ast.Constant)}
    return "__name__" in names and "__main__" in values


def _argument_calls(tree: ast.Module) -> dict[str, ast.Call]:
    calls: dict[str, ast.Call] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                calls[argument.value] = node
    return calls


def _is_args_attribute(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == name
        and isinstance(node.value, ast.Name)
        and node.value.id == "args"
    )


def _assert_cli_contract(filename: str) -> None:
    tree = _parse(filename)
    assert any(_is_main_guard(node) for node in tree.body), f"{filename} is not executable"

    arguments = _argument_calls(tree)
    assert _REQUIRED_CLI_FLAGS <= arguments.keys(), filename

    l2_call = arguments["--enable-l2-swimlane"]
    l2_keywords = {keyword.arg: keyword.value for keyword in l2_call.keywords}
    assert "choices" in l2_keywords, f"{filename} does not enumerate L2 levels"
    assert 4 in ast.literal_eval(l2_keywords["choices"]), f"{filename} does not expose L2 level 4"

    run_jit_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "run_jit")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "run_jit")
        )
    ]
    assert run_jit_calls, f"{filename} does not call run_jit"
    for call in run_jit_calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        for keyword in ("compile_only", "save_data", "golden_data"):
            assert keyword in keywords, f"{filename} does not forward {keyword}"
            assert _is_args_attribute(keywords[keyword], keyword), filename

    assert any(
        _is_args_attribute(node, "enable_l2_swimlane") for node in ast.walk(tree)
    ), f"{filename} does not forward the selected L2 level"


def test_gate_full_expert_sort_and_tail_fixture_contract() -> None:
    config = _load_config()
    model = config["ACTIVE"]
    assert config["MOE_TOKENS"] == 8
    assert model.hidden_size == 7168
    assert model.n_routed_experts == 384
    assert model.num_experts_per_tok == 6

    assert _literal_assignment("gate.py", "GATE_D_TILE") == 512
    assert _literal_assignment("gate.py", "SCORE_PAD") == 512

    tree = _parse("gate.py")
    merge_block_lengths = [
        ast.literal_eval(next(keyword.value for keyword in node.keywords if keyword.arg == "block_len"))
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "mrgsort"
        and any(keyword.arg == "block_len" for keyword in node.keywords)
    ]
    # sort32 leaves sixteen 64-value runs. Each mrgsort is four-way, so these
    # two stages produce four 256-value runs and then one 512-value run.
    assert merge_block_lengths == [64, 256]

    assert _literal_assignment("gate.py", "GATE_FIXTURES") == ("random", "tail-expert")
    fixture_source = ast.unparse(_function(tree, "build_tensor_specs"))
    assert "torch.zeros(N_EXPERTS, D)" in fixture_source
    assert "bias[N_EXPERTS - TOPK:] = torch.arange(1, TOPK + 1" in fixture_source
    assert "torch.arange(N_EXPERTS - 1, N_EXPERTS - TOPK - 1, -1" in fixture_source
    assert tuple(range(model.n_routed_experts - 1, model.n_routed_experts - 7, -1)) == (
        383,
        382,
        381,
        380,
        379,
        378,
    )


def test_shared_expert_uses_pro_shapes_and_tiling() -> None:
    config = _load_config()
    model = config["ACTIVE"]
    expected_tiling = {
        "K_TILE": 512,
        "INTER_K": 512,
        "MM_INTER_TILE": 256,
        "ACT_INTER_TILE": 1024,
        "D_OUT_TILE": 256,
        "QUANT_TILE": 1024,
        "D_OUT_TILE_ACT": 512,
        "W2_ACT_INNER": 14,
    }
    actual_tiling = {
        name: _literal_assignment("expert_shared.py", name) for name in expected_tiling
    }
    assert actual_tiling == expected_tiling
    assert model.hidden_size == expected_tiling["W2_ACT_INNER"] * expected_tiling["D_OUT_TILE_ACT"]
    assert model.moe_intermediate_size % expected_tiling["QUANT_TILE"] == 0
    assert model.moe_intermediate_size % expected_tiling["ACT_INTER_TILE"] == 0


def test_routed_expert_ep128_capacity_and_workloads() -> None:
    config = _load_config()
    assert config["DEPLOYMENT_EP"] == 128
    assert config["MOE_LOCAL_EXPERTS"] == 3
    assert config["MOE_DEPLOYMENT_RECV_MAX"] == 1024
    assert config["MOE_BALANCED_ROWS_PER_EXPERT"] == 16
    assert config["MOE_BALANCED_ROWS_PER_SHARD"] == 48

    workloads = _literal_assignment("expert_routed.py", "ROUTED_WORKLOAD_COUNTS")
    assert workloads == {
        "balanced": (16, 16, 16),
        "skewed": (48, 0, 0),
        "tail": (17, 16, 15),
    }
    assert all(len(counts) == config["MOE_LOCAL_EXPERTS"] for counts in workloads.values())
    assert all(sum(counts) == config["MOE_BALANCED_ROWS_PER_SHARD"] for counts in workloads.values())
    assert all(max(counts) <= config["MOE_DEPLOYMENT_RECV_MAX"] for counts in workloads.values())


def test_phase2_entries_and_dependencies_bind_active_model() -> None:
    for filename in _PHASE2_PRIMARY_ENTRIES + _PHASE2_TRANSITIVE_DEPENDENCIES:
        imports = _config_imports(_parse(filename))
        assert ("ACTIVE", "M") in imports, f"{filename} does not bind ACTIVE as M"
        assert all(name != "FLASH" for name, _ in imports), f"{filename} still binds FLASH"


def test_swa_dependencies_keep_required_pro_tiles() -> None:
    config = _load_config()
    assert _literal_assignment("hc_pre.py", "RMS_K_CHUNK") == 256
    assert _literal_assignment("qkv_proj_rope.py", "Q_PROJ_TILE") == 256
    assert _literal_assignment("qkv_proj_rope.py", "QPROJ_MM_N_TILE") == 512
    assert _literal_assignment("qkv_proj_rope.py", "QPROJ_MM_N_TILE") == config["ACTIVE"].head_dim
    assert _literal_assignment("qkv_proj_rope.py", "Q_ROPE_H_TILE") == 4
    assert _literal_assignment("qkv_proj_rope.py", "KV_K_TILE") == 128

    function = _function(_parse("qkv_proj_rope.py"), "qkv_proj_rope")
    wq_b = next(argument for argument in function.args.args if argument.arg == "wq_b")
    assert ast.unparse(wq_b.annotation) == "pl.Tensor[[H, HEAD_DIM, Q_LORA], pl.INT8]"

    qproj_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"matmul", "matmul_acc"}
        and any(isinstance(argument, ast.Name) and argument.id == "wq_chunk" for argument in node.args)
    ]
    assert qproj_calls
    for call in qproj_calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert ast.literal_eval(keywords["b_trans"]) is True


def test_primary_phase2_clis_expose_compute_only_and_replay_controls() -> None:
    for filename in _PHASE2_PRIMARY_ENTRIES:
        _assert_cli_contract(filename)
