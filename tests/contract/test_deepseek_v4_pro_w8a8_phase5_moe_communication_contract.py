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
import sys
from pathlib import Path
from types import ModuleType

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "deepseek" / "v4-pro-w8a8"
_CONFIG_PATH = _MODEL_DIR / "config.py"
_ENTRY_PATH = _MODEL_DIR / "moe_communication.py"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_communication() -> ModuleType:
    old_config = sys.modules.get("config")
    try:
        _load_module("config", _CONFIG_PATH)
        return _load_module("_deepseek_v4_pro_w8a8_moe_communication", _ENTRY_PATH)
    finally:
        if old_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = old_config


def test_communication_contract_separates_ep128_from_unvalidated_comm_ep() -> None:
    communication = _load_communication()
    source = _ENTRY_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_ENTRY_PATH))

    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert "moe_compute" not in source
    assert "pypto.language.distributed" not in imported_modules
    assert "pypto.ir.distributed_compiled_program" not in imported_modules
    assert "DistributedConfig" not in source

    expected_local_experts = {2: 192, 4: 96, 8: 48}
    expected_recv_capacity = {2: 16, 4: 32, 8: 64}
    for comm_ep in (2, 4, 8):
        contract = communication.communication_contract(comm_ep)
        assert contract.deployment_ep == 128
        assert contract.deployment_local_experts == 3
        assert contract.deployment_recv_capacity_per_expert == 1024
        assert contract.communication_ep == comm_ep
        assert contract.global_experts == 384
        assert contract.local_experts == expected_local_experts[comm_ep]
        assert contract.recv_capacity_per_expert == expected_recv_capacity[comm_ep]
        assert contract.tokens_per_rank == 8
        assert contract.topk == 6
        assert contract.routes_per_rank == 48
        assert contract.dispatch_logical_row_bytes == 7180
        assert contract.dispatch_window_row_bytes == 7232
        assert contract.combine_row_bytes == 14336
        assert contract.count_metadata_bytes_per_rank == 1536

        routes = []
        for source_rank in range(comm_ep):
            token_routes = []
            for token in range(contract.tokens_per_rank):
                selected_experts = []
                for topk_slot in range(contract.topk):
                    destination_rank = (source_rank + token + topk_slot) % comm_ep
                    local_expert = (token * contract.topk + topk_slot) % contract.local_experts
                    selected_experts.append(
                        destination_rank * contract.local_experts + local_expert
                    )
                token_routes.append(selected_experts)
            routes.append(token_routes)

        plan = communication.build_all_to_all_plan(routes, comm_ep)
        assert len(plan.routes) == comm_ep * contract.routes_per_rank
        assert all(sum(send_row) == contract.routes_per_rank for send_row in plan.send_counts)
        assert sum(sum(send_row) for send_row in plan.send_counts) == len(plan.routes)
        assert sum(sum(expert_row) for expert_row in plan.recv_expert_counts) == len(plan.routes)
        assert all(
            count <= contract.recv_capacity_per_expert
            for expert_row in plan.recv_expert_counts
            for count in expert_row
        )
        assert 0 < plan.remote_routes < len(plan.routes)
        assert plan.remote_dispatch_window_bytes == (
            plan.remote_routes * contract.dispatch_window_row_bytes
        )
        assert plan.remote_combine_bytes == plan.remote_routes * contract.combine_row_bytes

        for route in plan.routes:
            assert communication.expert_destination(route.global_expert, comm_ep) == (
                route.destination_rank,
                route.local_expert,
            )
            assert route.return_route == route.source_token * contract.topk + route.topk_slot

        report = contract.report_row()
        assert report["deployment_ep"] == 128
        assert report["deployment_local_experts"] == 3
        assert report["deployment_recv_capacity_per_expert"] == 1024
        assert report["communication_ep"] == comm_ep
        assert report["communication_local_experts"] == expected_local_experts[comm_ep]
        assert report["communication_recv_capacity_per_expert"] == expected_recv_capacity[comm_ep]
        assert report["measurement_status"] == "unvalidated-contract-only"
        assert report["hardware_execution_available"] is False

    non_integral_routes = [
        [list(selected_experts) for selected_experts in token_routes]
        for token_routes in routes
    ]
    non_integral_routes[0][0][0] = 10.9
    with pytest.raises(ValueError, match="expert routes must be integers"):
        communication.build_all_to_all_plan(non_integral_routes, 8)

    with pytest.raises(ValueError, match="COMM_EP must be one of"):
        communication.communication_contract(128)
    with pytest.raises(
        communication.DistributedCommunicationUnavailable,
        match="No dedicated distributed kernel has been implemented or validated",
    ):
        communication.execute_distributed_communication(
            comm_ep=8,
            device_ids=tuple(range(8)),
        )
