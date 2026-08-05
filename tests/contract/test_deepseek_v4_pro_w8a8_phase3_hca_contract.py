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
import importlib.util
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "deepseek" / "v4-pro-w8a8"
_CONFIG_PATH = _MODEL_DIR / "config.py"
_HCA_ENTRIES = (
    "decode_attention_hca.py",
    "decode_compressor_ratio128.py",
    "decode_sparse_attn_hca.py",
)
_REQUIRED_CLI_FLAGS = {
    "--compile-only",
    "--save-data",
    "--golden-data",
    "--enable-l2-swimlane",
}


def _load_config() -> dict[str, Any]:
    return runpy.run_path(str(_CONFIG_PATH))


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_host_metadata() -> tuple[ModuleType, ModuleType]:
    old_config = sys.modules.get("config")
    try:
        config = _load_module("config", _CONFIG_PATH)
        metadata = _load_module("_deepseek_v4_pro_w8a8_phase3_metadata", _MODEL_DIR / "decode_metadata.py")
    finally:
        if old_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = old_config
    return config, metadata


def _parse(filename: str) -> ast.Module:
    path = _MODEL_DIR / filename
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _source(filename: str) -> str:
    return (_MODEL_DIR / filename).read_text(encoding="utf-8")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} was not found")


def _assignment(tree: ast.Module, name: str) -> ast.AST:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return node.value
    raise AssertionError(f"assignment {name} was not found")


def _literal_assignment(filename: str, name: str) -> Any:
    return ast.literal_eval(_assignment(_parse(filename), name))


def _config_imports(tree: ast.Module) -> set[tuple[str, str | None]]:
    return {
        (alias.name, alias.asname)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "config"
        for alias in node.names
    }


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
    arguments = _argument_calls(tree)
    assert _REQUIRED_CLI_FLAGS <= arguments.keys(), filename

    l2_keywords = {keyword.arg: keyword.value for keyword in arguments["--enable-l2-swimlane"].keywords}
    assert 4 in ast.literal_eval(l2_keywords["choices"]), filename
    assert ast.literal_eval(l2_keywords["const"]) == 4, filename

    run_jit_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "run_jit")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "run_jit")
        )
    ]
    assert run_jit_calls, filename
    for call in run_jit_calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        for keyword in ("compile_only", "save_data", "golden_data"):
            assert keyword in keywords, f"{filename} does not forward {keyword}"
            assert _is_args_attribute(keywords[keyword], keyword), filename


def test_hca_entries_bind_active_and_use_standalone_pools() -> None:
    config = _load_config()
    assert config["DECODE_BATCH"] == 2
    assert config["KV_HCA_MAX_BLOCKS"] == 9
    assert config["HCA_STATE_PHYSICAL_BLOCKS"] == 32
    assert config["HCA_STATE_TABLE_MAX_BLOCKS"] == 16448

    for filename in _HCA_ENTRIES:
        tree = _parse(filename)
        imports = _config_imports(tree)
        assert ("ACTIVE", "M") in imports, filename
        assert all(name != "FLASH" for name, _ in imports), filename
        assert ("KV_HCA_MAX_BLOCKS", None) in imports, filename
        assert ast.unparse(_assignment(tree, "CMP_MAX_BLOCKS")) == "KV_HCA_MAX_BLOCKS"
        assert ast.unparse(_assignment(tree, "CMP_BLOCK_NUM")) == "B * CMP_MAX_BLOCKS"

    for filename in ("decode_attention_hca.py", "decode_compressor_ratio128.py"):
        tree = _parse(filename)
        imports = _config_imports(tree)
        assert ("HCA_STATE_PHYSICAL_BLOCKS", None) in imports
        assert ("HCA_STATE_TABLE_MAX_BLOCKS", None) in imports
        assert ast.unparse(_assignment(tree, "COMPRESS_STATE_PHYSICAL_BLOCKS")) == (
            "HCA_STATE_PHYSICAL_BLOCKS"
        )
        assert ast.unparse(_assignment(tree, "COMPRESS_STATE_MAX_BLOCKS")) == "HCA_STATE_TABLE_MAX_BLOCKS"


def test_hca_query_projection_uses_head_major_weights() -> None:
    attention_tree = _parse("decode_attention_hca.py")
    for function_name in ("attention_hca", "attention_hca_test"):
        function = _function(attention_tree, function_name)
        wq_b = next(argument for argument in function.args.args if argument.arg == "wq_b")
        assert ast.unparse(wq_b.annotation) == "pl.Tensor[[H, HEAD_DIM, Q_LORA], pl.INT8]"

    fixture_source = ast.unparse(_function(attention_tree, "build_tensor_specs"))
    assert "torch.randn(H, HEAD_DIM, Q_LORA)" in fixture_source
    assert "amax(dim=-1)" in fixture_source
    assert "wq_b_scale.reshape(H * HEAD_DIM)" in fixture_source
    assert "TensorSpec('wq_b', [H, HEAD_DIM, Q_LORA]" in fixture_source
    assert "TensorSpec('compress_state'" in fixture_source and "is_output=True" in fixture_source
    assert "TensorSpec('cmp_kv'" in fixture_source and "is_output=True" in fixture_source


def test_hca_cache_addresses_at_target_and_boundary() -> None:
    config, metadata = _load_host_metadata()
    batch = config.DECODE_BATCH
    seq = config.DECODE_SEQ

    ori_table = metadata.block_table(
        batch=batch,
        table_blocks=config.KV_ORI_TABLE_MAX_BLOCKS,
        physical_blocks=config.ORI_KV_BLOCK_NUM,
    )
    cmp_table = metadata.block_table(
        batch=batch,
        table_blocks=config.KV_HCA_MAX_BLOCKS,
        physical_blocks=batch * config.KV_HCA_MAX_BLOCKS,
        permuted=True,
    )
    state_table = metadata.block_table(
        batch=batch,
        table_blocks=config.HCA_STATE_TABLE_MAX_BLOCKS,
        physical_blocks=config.HCA_STATE_PHYSICAL_BLOCKS,
    )

    assert ori_table.shape == (2, 1028)
    assert torch.unique(ori_table).numel() == 6
    assert cmp_table.shape == (2, 9)
    assert torch.unique(cmp_table).numel() == 18
    assert state_table.shape == (2, 16448)
    assert set(state_table[0].tolist()).isdisjoint(set(state_table[1].tolist()))

    for start_pos, expected_events in ((131072, 0), (131071, 1)):
        starts = torch.full((batch,), start_pos, dtype=torch.int32)
        positions = metadata.position_ids_from_starts(starts, seq=seq)
        ori_slots = metadata.ori_slot_mapping(positions, ori_table, block_size=config.BLOCK_SIZE)
        cmp_slots = metadata.compressed_slot_mapping(
            positions,
            cmp_table,
            compress_ratio=config.HCA_COMPRESS_RATIO,
            block_size=config.BLOCK_SIZE,
        )
        state_slots = metadata.state_slot_mapping(
            positions,
            state_table,
            state_block_size=config.C128_COMPRESSOR_BLOCK_SIZE,
        )
        window_indices, window_lens = metadata.swa_indices_and_lens(
            positions,
            ori_table,
            block_size=config.BLOCK_SIZE,
            window=config.ACTIVE.sliding_window,
        )

        assert (cmp_slots >= 0).sum(dim=1).tolist() == [expected_events, expected_events]
        assert set(ori_slots[0].tolist()).isdisjoint(set(ori_slots[1].tolist()))
        assert set(state_slots[0].tolist()).isdisjoint(set(state_slots[1].tolist()))
        assert set(cmp_slots[0][cmp_slots[0] >= 0].tolist()).isdisjoint(
            set(cmp_slots[1][cmp_slots[1] >= 0].tolist())
        )
        for token in range(config.DECODE_TOKENS):
            live = window_indices[token, : int(window_lens[token])]
            assert torch.unique(live).numel() == live.numel()

    boundary_oldest = torch.tensor([[131071 - 127], [131071 - 127]], dtype=torch.int32)
    boundary_future = torch.tensor([[131072], [131072]], dtype=torch.int32)
    oldest_slots = metadata.state_slot_mapping(
        boundary_oldest,
        state_table,
        state_block_size=config.C128_COMPRESSOR_BLOCK_SIZE,
    )
    future_slots = metadata.state_slot_mapping(
        boundary_future,
        state_table,
        state_block_size=config.C128_COMPRESSOR_BLOCK_SIZE,
    )
    assert torch.equal(oldest_slots, future_slots)


def test_boundary_pooling_precedes_ring_reuse() -> None:
    compressor_source = _source("decode_compressor_ratio128.py")
    prefix_write = compressor_source.index("prefix_state_i64")
    pool_read = compressor_source.index("softmax_score_state")
    tail_write = compressor_source.index("tail_state_i64")
    assert prefix_write < pool_read < tail_write

    golden_source = ast.unparse(_function(_parse("decode_compressor_ratio128.py"), "golden_compressor"))
    prefix_write = golden_source.index("for s in range(state_prefix)")
    pool_read = golden_source.index("for pos in range(compress_pos - ratio + 1, compress_pos + 1)")
    tail_write = golden_source.index("for s in range(state_prefix, S)")
    assert prefix_write < pool_read < tail_write
    assert "kv_zero_row = pl.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0)" in compressor_source


def test_hca_fixed_tail_marks_invalid_rows() -> None:
    config = _load_config()
    max_seq_len = config["ACTIVE"].max_position_embeddings
    capacity = max_seq_len // config["HCA_COMPRESS_RATIO"]
    padded_tail = ((capacity + 127) // 128) * 128
    assert capacity == 1028
    assert padded_tail == 1152
    assert config["ACTIVE"].sliding_window + padded_tail == 1280
    assert padded_tail // config["BLOCK_SIZE"] == config["KV_HCA_MAX_BLOCKS"]

    for start_pos in (131072, 131071):
        kv_seq_len = start_pos + config["DECODE_SEQ"]
        valid_rows = [
            min(capacity, (position + 1) // 128, kv_seq_len // 128)
            for position in range(start_pos, start_pos + config["DECODE_SEQ"])
        ]
        assert valid_rows == [1024] * config["DECODE_SEQ"]
        assert [padded_tail - valid for valid in valid_rows] == [128] * config["DECODE_SEQ"]

    sparse_source = _source("decode_sparse_attn_hca.py")
    attention_source = _source("decode_attention_hca.py")
    assert "CMP_CAPACITY = MAX_SEQ_LEN // DEFAULT_COMPRESS_RATIO" in sparse_source
    assert "CMP_TOPK = ((CMP_CAPACITY + ATTN_K_TILE - 1) // ATTN_K_TILE) * ATTN_K_TILE" in sparse_source
    assert "M.index_topk" not in sparse_source
    assert "HCA_TOPK_LIMIT = COMPRESS_TOPK" in attention_source
    assert "pl.cast(-1, pl.INT32)" in attention_source


def test_hca_keeps_pro_compatible_tiles() -> None:
    config = _load_config()
    model = config["ACTIVE"]
    assert _literal_assignment("decode_compressor_ratio128.py", "K_TILE") == 512
    assert _literal_assignment("decode_compressor_ratio128.py", "OUT_TILE") == 64
    assert _literal_assignment("decode_compressor_ratio128.py", "POOL_HEAD_TILE") == 128
    assert _literal_assignment("decode_sparse_attn_hca.py", "QK_M_TILE") == 32
    assert _literal_assignment("decode_sparse_attn_hca.py", "ATTN_K_TILE") == 128
    assert _literal_assignment("decode_sparse_attn_hca.py", "PROJ_B_D_CHUNK") == 512
    assert model.hidden_size % 512 == 0
    assert model.num_attention_heads % 32 == 0
    assert model.head_dim % 128 == 0


def test_hca_clis_expose_compile_replay_and_level4_controls() -> None:
    for filename in _HCA_ENTRIES:
        _assert_cli_contract(filename)

    for filename in ("decode_attention_hca.py", "decode_compressor_ratio128.py"):
        assert "--start-pos" in _argument_calls(_parse(filename))
