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
import runpy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "deepseek" / "v4-pro-w8a8"
_CONFIG_PATH = _MODEL_DIR / "config.py"


def _load_config() -> dict[str, object]:
    return runpy.run_path(str(_CONFIG_PATH))


def test_config_import_is_pure_python(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _CONFIG_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.partition(".")[0])
    assert imported_roots == {"dataclasses", "typing"}

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        assert name.partition(".")[0] not in {"golden", "pypto", "torch"}
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    _load_config()


def test_target_preset_is_independent_and_immutable() -> None:
    config = _load_config()
    pro = config["PRO"]
    preset = config["PRO_W8A8"]
    target = config["TARGET"]

    assert config["ACTIVE"] is preset
    assert preset is not pro
    assert pro.name == "pro"
    assert pro.max_position_embeddings == 1048576
    assert preset.name == "pro-w8a8"
    assert preset.max_position_embeddings == 131584
    assert preset.max_batch_size == 2
    assert preset.expert_dtype is None
    assert target.model is preset
    assert target.routed_activation_dtype == "int8"
    assert target.routed_weight_dtype == "int8"

    with pytest.raises(FrozenInstanceError):
        preset.name = "mutated"
    with pytest.raises(FrozenInstanceError):
        target.batch = 1


def test_decode_shape_and_layer_ownership() -> None:
    config = _load_config()
    schedule = config["PRO_W8A8_LAYER_SCHEDULE"]

    assert config["DECODE_BATCH"] == 2
    assert config["DECODE_SEQ"] == 4
    assert config["DECODE_TOKENS"] == 8
    assert config["MTP_DRAFT_DEPTH"] == 3
    assert config["DECODE_START_POS"] == 131072
    assert config["DECODE_MAX_SEQ_LEN"] == 131584

    assert len(schedule.main) == 61
    assert schedule.main == config["EXPECTED_PRO_MAIN_RATIOS"]
    assert config["MAIN_COMPRESS_RATIOS"] is schedule.main
    assert config["MTP_COMPRESS_RATIOS"] is schedule.mtp
    assert schedule.main_hca_layers == 31
    assert schedule.main_csa_layers == 30
    assert schedule.main_swa_layers == 0
    assert config["MAIN_HCA_LAYERS"] == 31
    assert config["MAIN_CSA_LAYERS"] == 30
    assert config["MAIN_SWA_LAYERS"] == 0
    assert schedule.mtp == (config["SWA_COMPRESS_RATIO"],)

    invalid = replace(
        config["PRO_W8A8"],
        compress_ratios=config["PRO_W8A8"].compress_ratios[:-1],
    )
    with pytest.raises(ValueError, match="main-plus-MTP"):
        config["split_layer_ratios"](invalid)


def test_deployment_and_communication_ep_are_independent() -> None:
    config = _load_config()
    deployment = config["DEPLOYMENT_LAYOUT"]
    communication = config["COMMUNICATION_LAYOUTS"]

    assert config["MOE_GLOBAL_EXPERTS"] == 384
    assert config["DEPLOYMENT_EP"] == 128
    assert config["COMM_EP_DEFAULT"] == 2
    assert deployment.world_size == 128
    assert deployment.global_experts == 384
    assert deployment.local_experts == 3
    assert deployment.recv_capacity == 1024
    assert config["MOE_BALANCED_ROWS_PER_EXPERT"] == 16
    assert config["MOE_BALANCED_ROWS_PER_SHARD"] == 48

    assert tuple(layout.world_size for layout in communication) == (2, 4, 8)
    assert all(layout.global_experts == 384 for layout in communication)
    assert tuple(layout.local_experts for layout in communication) == (192, 96, 48)
    assert tuple(layout.recv_capacity for layout in communication) == (16, 32, 64)


def test_128k_cache_contract() -> None:
    config = _load_config()

    assert config["KV_ORI_TABLE_MAX_BLOCKS"] == 1028
    assert config["KV_CMP_MAX_BLOCKS"] == 257
    assert config["KV_HCA_MAX_BLOCKS"] == 9
    assert config["IDX_CACHE_MAX_BLOCKS"] == 257
    assert config["ORI_KV_BLOCK_NUM"] == 6
    assert config["CMP_KV_BLOCK_NUM"] == 514
    assert config["IDX_KV_BLOCK_NUM"] == 514
    assert config["HCA_STATE_PHYSICAL_BLOCKS"] == 32
    assert config["CSA_STATE_PHYSICAL_BLOCKS"] == 4
    assert config["CSA_INNER_STATE_PHYSICAL_BLOCKS"] == 4
    assert config["HCA_STATE_TABLE_MAX_BLOCKS"] == 16448
    assert config["CSA_STATE_TABLE_MAX_BLOCKS"] == 32896
    assert config["CSA_INNER_STATE_TABLE_MAX_BLOCKS"] == 32896


def test_unvalidated_model_tree_is_excluded_from_broad_ci(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    detector = runpy.run_path(str(_REPO_ROOT / ".github" / "scripts" / "detect_changes.py"))
    changed_config = "models/deepseek/v4-pro-w8a8/config.py"

    assert detector["select_runnable"]([changed_config]) == []

    daily_ci = (_REPO_ROOT / ".github" / "workflows" / "daily_ci.yml").read_text(encoding="utf-8")
    exclusion = "! -path 'models/deepseek/v4-pro-w8a8/*'"
    assert daily_ci.count(exclusion) == 2

    pull_request_ci = (_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    contract_path = "tests/contract/test_deepseek_v4_pro_w8a8_bringup_contract.py"
    assert contract_path in pull_request_ci
