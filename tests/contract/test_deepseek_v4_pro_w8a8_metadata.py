# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "deepseek" / "v4-pro-w8a8"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def model_modules() -> tuple[ModuleType, ModuleType]:
    old_config = sys.modules.get("config")
    try:
        config = _load_module("config", _MODEL_DIR / "config.py")
        metadata = _load_module("_deepseek_v4_pro_w8a8_metadata", _MODEL_DIR / "decode_metadata.py")
    finally:
        if old_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = old_config
    return config, metadata


def test_full_history_tables_are_non_aliased(model_modules: tuple[ModuleType, ModuleType]) -> None:
    config, metadata = model_modules
    table_cases = (
        (config.KV_CMP_MAX_BLOCKS, config.CMP_KV_BLOCK_NUM),
        (config.IDX_CACHE_MAX_BLOCKS, config.IDX_KV_BLOCK_NUM),
        (config.KV_HCA_MAX_BLOCKS, config.DECODE_BATCH * config.KV_HCA_MAX_BLOCKS),
    )

    for logical_width, physical_blocks in table_cases:
        table = metadata.block_table(
            batch=config.DECODE_BATCH,
            table_blocks=logical_width,
            physical_blocks=physical_blocks,
            permuted=True,
        )
        assert table.shape == (config.DECODE_BATCH, logical_width)
        assert int(table.min()) == 0
        assert int(table.max()) == physical_blocks - 1
        assert torch.unique(table).numel() == table.numel()


def test_original_kv_ring_is_request_isolated(model_modules: tuple[ModuleType, ModuleType]) -> None:
    config, metadata = model_modules
    table = metadata.block_table(
        batch=config.DECODE_BATCH,
        table_blocks=config.KV_ORI_TABLE_MAX_BLOCKS,
        physical_blocks=config.ORI_KV_BLOCK_NUM,
    )

    request_pages = [set(table[b].tolist()) for b in range(config.DECODE_BATCH)]
    assert request_pages == [{0, 2, 4}, {1, 3, 5}]
    assert request_pages[0].isdisjoint(request_pages[1])

    for start in (0, 127, 128, 131071, config.DECODE_START_POS):
        starts = torch.full((config.DECODE_BATCH,), start, dtype=torch.int32)
        positions = metadata.position_ids_from_starts(starts, seq=config.DECODE_SEQ)
        slots = metadata.paged_slot_mapping(positions, table, block_size=config.BLOCK_SIZE)
        indices, lens = metadata.swa_indices_and_lens(
            positions,
            table,
            block_size=config.BLOCK_SIZE,
            window=config.ACTIVE.sliding_window,
        )

        assert bool((slots >= 0).all())
        assert int(slots.max()) < config.ORI_KV_BLOCK_NUM * config.BLOCK_SIZE
        for token in range(config.DECODE_TOKENS):
            valid = indices[token, : int(lens[token])]
            assert bool((valid >= 0).all())
            assert torch.unique(valid).numel() == valid.numel()

        request_zero = indices[: config.DECODE_SEQ]
        request_one = indices[config.DECODE_SEQ :]
        assert set(request_zero[request_zero >= 0].tolist()).isdisjoint(
            set(request_one[request_one >= 0].tolist())
        )


def test_state_rings_reuse_only_after_request_capacity(model_modules: tuple[ModuleType, ModuleType]) -> None:
    config, metadata = model_modules
    state_cases = (
        (config.HCA_STATE_TABLE_MAX_BLOCKS, config.HCA_STATE_PHYSICAL_BLOCKS),
        (config.CSA_STATE_TABLE_MAX_BLOCKS, config.CSA_STATE_PHYSICAL_BLOCKS),
        (config.CSA_INNER_STATE_TABLE_MAX_BLOCKS, config.CSA_INNER_STATE_PHYSICAL_BLOCKS),
    )

    for logical_width, physical_blocks in state_cases:
        table = metadata.block_table(
            batch=config.DECODE_BATCH,
            table_blocks=logical_width,
            physical_blocks=physical_blocks,
        )
        request_capacity = physical_blocks // config.DECODE_BATCH
        assert set(table[0].tolist()).isdisjoint(set(table[1].tolist()))
        for request in range(config.DECODE_BATCH):
            for start in (0, 1, logical_width - request_capacity):
                live_pages = table[request, start : start + request_capacity]
                assert torch.unique(live_pages).numel() == live_pages.numel()
            assert int(table[request, 0]) == int(table[request, request_capacity])


def test_block_table_rejects_unsafe_global_pools(model_modules: tuple[ModuleType, ModuleType]) -> None:
    _, metadata = model_modules

    with pytest.raises(ValueError, match="multiple of batch"):
        metadata.block_table(batch=2, table_blocks=8, physical_blocks=5)


def test_swa_entry_uses_only_the_original_kv_pool() -> None:
    source = (_MODEL_DIR / "decode_attention_swa.py").read_text(encoding="utf-8")
    code = compile(source, str(_MODEL_DIR / "decode_attention_swa.py"), "exec", dont_inherit=True)
    imported_names = set(code.co_names)

    assert "decode_compressor_ratio4" not in imported_names
    assert "decode_compressor_ratio128" not in imported_names
    assert "decode_indexer" not in imported_names
    assert "physical_blocks=ORI_BLOCK_NUM" in source
