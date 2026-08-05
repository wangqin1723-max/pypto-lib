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
import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "deepseek" / "v4-pro-w8a8"
_CONFIG_PATH = _MODEL_DIR / "config.py"
_MANIFEST_PATH = _MODEL_DIR / "main_compute_manifest.py"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def manifest_module() -> ModuleType:
    old_config = sys.modules.get("config")
    try:
        _load_module("config", _CONFIG_PATH)
        manifest = _load_module("_v4_pro_w8a8_phase6_manifest", _MANIFEST_PATH)
    finally:
        if old_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = old_config
    return manifest


def test_manifest_import_and_accounting_are_tensor_free(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _MANIFEST_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.partition(".")[0])
    assert imported_roots == {"__future__", "config", "dataclasses", "math", "typing"}

    forbidden_names = {"Tensor", "TensorSpec", "torch", "pypto", "pl", "zeros", "ones", "randn"}
    assert not {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} & forbidden_names

    old_config = sys.modules.get("config")
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        assert name.partition(".")[0] not in {"golden", "pypto", "torch"}
        return original_import(name, globals, locals, fromlist, level)

    try:
        monkeypatch.setattr(builtins, "__import__", guarded_import)
        _load_module("config", _CONFIG_PATH)
        loaded = _load_module("_v4_pro_w8a8_phase6_manifest_pure", _MANIFEST_PATH)
        full = loaded.build_ladder_manifest("depth61")
        assert len(full.layer_ids) == 61
    finally:
        if old_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = old_config


def test_depth_ladder_uses_exact_pro_main_layer_ids(manifest_module: ModuleType) -> None:
    ladder = dict(manifest_module.DEPTH_LADDER_LAYER_IDS)
    assert ladder == {
        "hca1": (0,),
        "csa1": (2,),
        "depth2": tuple(range(2)),
        "depth4": tuple(range(4)),
        "depth16": tuple(range(16)),
        "depth31": tuple(range(31)),
        "depth61": tuple(range(61)),
    }

    expected_counts = {
        "hca1": (1, 0),
        "csa1": (0, 1),
        "depth2": (2, 0),
        "depth4": (3, 1),
        "depth16": (9, 7),
        "depth31": (16, 15),
        "depth61": (31, 30),
    }
    for name, counts in expected_counts.items():
        manifest = manifest_module.build_ladder_manifest(name)
        assert (len(manifest.hca_layer_ids), len(manifest.csa_layer_ids)) == counts
        assert not set(manifest.hca_layer_ids) & set(manifest.csa_layer_ids)
        assert manifest.hca_layer_ids + manifest.csa_layer_ids != ()


def test_exact_weight_cache_and_shared_totals(manifest_module: ModuleType) -> None:
    expected = {
        "hca1": (681_022_680, 4_194_304, 139_880, 685_216_984, 4_426_091_756),
        "csa1": (716_367_320, 77_005_824, 538_672, 793_373_144, 4_534_247_916),
        "depth2": (1_362_045_360, 8_388_608, 139_880, 1_370_433_968, 5_111_308_740),
        "depth4": (2_756_332_640, 89_588_736, 670_328, 2_845_921_376, 6_586_796_148),
        "depth16": (11_103_440_000, 576_789_504, 670_328, 11_680_229_504, 15_421_104_276),
        "depth31": (21_554_996_520, 1_222_196_224, 670_328, 22_777_192_744, 26_518_067_516),
        "depth61": (42_422_764_920, 2_440_198_144, 670_328, 44_862_963_064, 48_603_837_836),
    }

    for name, (weights, caches, metadata, layer_total, model_total) in expected.items():
        manifest = manifest_module.build_ladder_manifest(name)
        assert manifest.weight_bytes == weights
        assert manifest.cache_bytes == caches
        assert manifest.shared_asset_bytes == 3_740_874_772
        assert manifest.shared_metadata_bytes == metadata
        assert manifest.layer_bytes == layer_total
        assert manifest.model_bytes_before_metadata == model_total
        assert manifest.accounted_bytes == model_total + metadata


def test_cache_pools_have_independent_layer_ranges_and_local_tables(manifest_module: ModuleType) -> None:
    manifest = manifest_module.build_ladder_manifest("depth4")
    assert manifest.cache_ranges_are_independent
    assert len(manifest.cache_pools) == 3 * 3 + 6

    ranges = [pool.allocation.byte_range for pool in manifest.cache_pools]
    assert len({(byte_range.start, byte_range.end) for byte_range in ranges}) == len(ranges)
    for index, left in enumerate(ranges):
        assert all(not left.overlaps(right) for right in ranges[index + 1 :])

    expected = {
        "original_kv": ((6, 128, 1, 512), "bf16", 6, 1028, "block_table"),
        "hca_compressed_kv": ((18, 128, 1, 512), "bf16", 18, 9, "hca_cmp_block_table"),
        "hca_compressor_state": ((32, 8, 1024), "fp32", 32, 16448, "hca_state_block_table"),
        "csa_compressed_kv": ((514, 128, 1, 512), "bf16", 514, 257, "csa_cmp_block_table"),
        "csa_index_kv": ((514, 128, 1, 128), "int8", 514, 257, "csa_idx_block_table"),
        "csa_index_scale": ((514, 128, 1, 1), "fp32", 514, 257, "csa_idx_block_table"),
        "csa_compressor_state": ((4, 4, 2048), "fp32", 4, 32896, "csa_state_block_table"),
        "csa_inner_compressor_state": (
            (4, 4, 512),
            "fp32",
            4,
            32896,
            "csa_inner_state_block_table",
        ),
    }
    for pool in manifest.cache_pools:
        shape, dtype, capacity, logical_width, table_name = expected[pool.family]
        assert pool.allocation.shape == shape
        assert pool.allocation.dtype == dtype
        assert pool.physical_blocks == capacity
        assert pool.logical_table_blocks == logical_width
        assert pool.block_table_name == table_name
        assert pool.block_ids_are_local((0, capacity - 1))
        assert not pool.block_ids_are_local((-1,))
        assert not pool.block_ids_are_local((capacity,))

    original_ranges = [pool.allocation.byte_range for pool in manifest.cache_pools if pool.family == "original_kv"]
    assert len(original_ranges) == 4
    assert len({(byte_range.start, byte_range.end) for byte_range in original_ranges}) == 4


def test_hash_tables_exist_only_for_the_first_three_model_layers(manifest_module: ModuleType) -> None:
    full = manifest_module.build_ladder_manifest("depth61")
    hash_allocations = [allocation for allocation in full.allocations if allocation.name.endswith(".tid2eid")]
    assert [allocation.layer_id for allocation in hash_allocations] == [0, 1, 2]
    assert all(allocation.shape == (129280, 6) for allocation in hash_allocations)
    assert all(allocation.nbytes == 3_102_720 for allocation in hash_allocations)


@pytest.mark.parametrize("layer_ids", [(), (0, 0), (-1,), (61,), (True,)])
def test_manifest_rejects_non_main_or_ambiguous_layer_sets(
    manifest_module: ModuleType,
    layer_ids: tuple[int, ...],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        manifest_module.build_hbm_manifest(layer_ids)


def test_full_manifest_is_shape_only_and_below_one_die_static_budget(manifest_module: ModuleType) -> None:
    full = manifest_module.build_ladder_manifest("depth61")
    assert all(isinstance(allocation.shape, tuple) for allocation in full.allocations)
    assert all(isinstance(dimension, int) for allocation in full.allocations for dimension in allocation.shape)
    assert full.accounted_bytes == 48_604_508_164
    assert full.accounted_bytes < 64 * manifest_module.GIB
