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
import math
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
        metadata = _load_module("_v4_pro_w8a8_phase4_metadata", _MODEL_DIR / "decode_metadata.py")
    finally:
        if old_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = old_config
    return config, metadata


def test_full_csa_tables_reach_page_256_without_alias(
    model_modules: tuple[ModuleType, ModuleType],
) -> None:
    config, metadata = model_modules
    starts = torch.full((config.DECODE_BATCH,), config.DECODE_START_POS, dtype=torch.int32)
    positions = metadata.position_ids_from_starts(starts, seq=config.DECODE_SEQ)

    for physical_blocks in (config.CMP_KV_BLOCK_NUM, config.IDX_KV_BLOCK_NUM):
        table = metadata.block_table(
            batch=config.DECODE_BATCH,
            table_blocks=config.KV_CMP_MAX_BLOCKS,
            physical_blocks=physical_blocks,
        )
        assert torch.unique(table).numel() == config.DECODE_BATCH * config.KV_CMP_MAX_BLOCKS
        assert table[:, 0].tolist() == [0, 1]
        assert table[:, 256].tolist() == [512, 513]

        slots = metadata.compressed_slot_mapping(
            positions,
            table,
            compress_ratio=config.CSA_COMPRESS_RATIO,
            block_size=config.BLOCK_SIZE,
        )
        assert slots[:, :3].eq(-1).all()
        assert slots[:, 3].tolist() == [512 * config.BLOCK_SIZE, 513 * config.BLOCK_SIZE]


def test_target_and_full_tail_visible_lengths() -> None:
    def visible_lengths(start: int, seq: int = 4) -> list[int]:
        committed = (start + seq) // 4
        return [min(committed, (start + token + 1) // 4, 32896) for token in range(seq)]

    target = visible_lengths(131072)
    assert target == [32768, 32768, 32768, 32769]
    assert [math.ceil(length / 128) for length in target] == [256, 256, 256, 257]
    assert sum(math.ceil(length / 128) for length in target) == 1025

    full_tail = visible_lengths(131580)
    assert full_tail == [32895, 32895, 32895, 32896]
    assert full_tail[-1] - 16 * 2048 == 128


def _state_row(position: int) -> int:
    logical_page, intra = divmod(position, 4)
    return (logical_page % 2) * 4 + intra


@pytest.mark.parametrize("phase", [0, 1, 2, 3])
def test_two_page_state_ring_pools_before_post_boundary_writes(phase: int) -> None:
    start = 131072 + phase
    boundary_s = 3 - phase
    boundary_end = start + boundary_s
    current_start = boundary_end - 3
    previous_start = current_start - 4
    pool_positions = list(range(previous_start, boundary_end + 1))

    state: dict[int, int] = {}
    for position in pool_positions:
        if position < start:
            state[_state_row(position)] = position
    incoming = list(range(start, start + 4))

    for token, position in enumerate(incoming):
        if token <= boundary_s:
            state[_state_row(position)] = position
    assert [state[_state_row(position)] for position in pool_positions] == pool_positions

    for token, position in enumerate(incoming):
        if token > boundary_s:
            state[_state_row(position)] = position

    if phase:
        naive = {key: value for key, value in state.items()}
        for position in pool_positions:
            if position < start:
                naive[_state_row(position)] = position
        for position in incoming:
            naive[_state_row(position)] = position
        assert [naive[_state_row(position)] for position in pool_positions] != pool_positions


def test_hierarchical_topk_is_exact_with_tail_candidates() -> None:
    score_len = 32896
    group_width = 2048
    keep = 1024
    scores = torch.arange(score_len, dtype=torch.float32) * -1.0
    for group in range(16):
        scores[group * group_width + group] = 10000.0 + group
    scores[-128:] = torch.arange(128, dtype=torch.float32) + 20000.0

    candidates: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, score_len, group_width):
        end = min(start + group_width, score_len)
        count = min(keep, end - start)
        values, local_ids = torch.topk(scores[start:end], count)
        if count < keep:
            values = torch.cat([values, torch.full((keep - count,), float("-inf"))])
            local_ids = torch.cat([local_ids, torch.zeros(keep - count, dtype=torch.int64)])
        candidates.append((values, local_ids + start))
    candidates.append((torch.full((keep,), float("-inf")), torch.zeros(keep, dtype=torch.int64)))

    while len(candidates) > 1:
        if len(candidates) % 2:
            candidates.append((torch.full((keep,), float("-inf")), torch.zeros(keep, dtype=torch.int64)))
        merged: list[tuple[torch.Tensor, torch.Tensor]] = []
        for left in range(0, len(candidates), 2):
            values = torch.cat([candidates[left][0], candidates[left + 1][0]])
            ids = torch.cat([candidates[left][1], candidates[left + 1][1]])
            top_values, order = torch.topk(values, keep)
            merged.append((top_values, ids[order]))
        candidates = merged

    expected_values, expected_ids = torch.topk(scores, keep)
    actual_values, actual_ids = candidates[0]
    assert torch.equal(actual_values, expected_values)
    assert torch.equal(actual_ids, expected_ids)
    assert bool((actual_ids >= score_len - 128).any())


def test_device_sources_freeze_full_capacity_and_compact_topk_abi() -> None:
    indexer = (_MODEL_DIR / "decode_indexer.py").read_text(encoding="utf-8")
    sparse = (_MODEL_DIR / "decode_sparse_attn.py").read_text(encoding="utf-8")
    main_compressor = (_MODEL_DIR / "decode_compressor_ratio4.py").read_text(encoding="utf-8")
    inner_compressor = (_MODEL_DIR / "decode_indexer_compressor.py").read_text(encoding="utf-8")
    metadata_device = (_MODEL_DIR / "decode_metadata_device.py").read_text(encoding="utf-8")

    assert "assert SCORE_LEN == 32896" in indexer
    assert "TOPK_REAL_GROUPS == 17" in indexer
    assert "topk_idxs: pl.Tensor[[B, S, IDX_TOPK]" in indexer
    assert "idx_topk: pl.Tensor[[T, IDX_TOPK]" in sparse
    assert "if s_sc <= boundary_s:" in main_compressor
    assert "if s_sc > boundary_s:" in main_compressor
    assert "if s_sc <= boundary_s:" in inner_compressor
    assert "if s_sc > boundary_s:" in inner_compressor
    assert "ACTIVE as M" in metadata_device
    assert "HCA_STATE_TABLE_MAX_BLOCKS" in metadata_device
    assert "CSA_STATE_TABLE_MAX_BLOCKS" in metadata_device
