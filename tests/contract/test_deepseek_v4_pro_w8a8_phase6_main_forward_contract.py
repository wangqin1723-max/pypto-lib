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
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENTRY_PATH = _REPO_ROOT / "models" / "deepseek" / "v4-pro-w8a8" / "decode_fwd_compute.py"

_CASES = ("hca1", "csa1", "depth2", "depth4", "depth16", "depth31", "depth61")
_GOLDEN_CASES = ("hca1", "csa1", "depth2", "depth4")
_CACHE_NAMES = {
    "kv_cache",
    "hca_cmp_kv",
    "hca_compress_state",
    "csa_cmp_kv",
    "csa_idx_kv_cache",
    "csa_idx_kv_scale",
    "csa_compress_state",
    "csa_inner_compress_state",
}
_SLOT_MAPPING_NAMES = {
    "ori_slot_mapping",
    "hca_cmp_slot_mapping",
    "hca_state_slot_mapping",
    "csa_cmp_slot_mapping",
    "csa_idx_slot_mapping",
    "csa_state_slot_mapping",
    "csa_inner_state_slot_mapping",
}


def _parse() -> ast.Module:
    return ast.parse(_ENTRY_PATH.read_text(encoding="utf-8"), filename=str(_ENTRY_PATH))


def _functions(tree: ast.Module, name: str) -> list[ast.FunctionDef]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    functions = _functions(tree, name)
    assert len(functions) == 1, f"expected one {name}, found {len(functions)}"
    return functions[0]


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


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _tuple_assignment(tree: ast.Module, name: str) -> tuple[str, ...]:
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"assignment {name} was not found")


def _assignment(tree: ast.Module, name: str) -> ast.Assign:
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            return node
    raise AssertionError(f"assignment {name} was not found")


def _ring_heap_constants(tree: ast.Module, static_case: str):
    names = (
        "RING0_HEAP_BYTES_BY_CASE",
        "INACTIVE_RING_HEAP_BYTES",
        "DEFAULT_RING_HEAP_BYTES",
    )
    namespace = {"GIB": 1 << 30, "STATIC_CASE": static_case}
    module = ast.fix_missing_locations(
        ast.Module(body=[_assignment(tree, name) for name in names], type_ignores=[])
    )
    exec(compile(module, str(_ENTRY_PATH), "exec"), namespace)
    return namespace


def _cache_policy_factory(tree: ast.Module):
    namespace = {"CACHE_POLICIES": ("commit", "overwrite")}
    function = _function(tree, "_apply_cache_policy")
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(_ENTRY_PATH), "exec"), namespace)
    return namespace["_apply_cache_policy"]


def _benchmark_result_error_factory(tree: ast.Module):
    namespace = {}
    function = _function(tree, "_benchmark_result_error")
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(_ENTRY_PATH), "exec"), namespace)
    return namespace["_benchmark_result_error"]


def _valid_benchmark_stats(rounds: int = 2):
    dispatches = [object() for _ in range(rounds)]
    effective_us = [30.0 + index for index in range(rounds)]
    return SimpleNamespace(
        fallback_flattened=False,
        unstable_dispatch_slots=False,
        rounds=rounds,
        device_wall_us=[10.0 + index for index in range(rounds)],
        host_wall_us=[20.0 + index for index in range(rounds)],
        rounds_dispatches=[{101: [dispatch]} for dispatch in dispatches],
        invocations=dispatches,
        per_round=lambda metric: effective_us if metric == "effective" else [],
    )


def _ring_resource_functions(tree: ast.Module):
    namespace = {
        "RUNTIME_RING_COUNT": 4,
        "RING_HEAP_MIN_BYTES": 1024,
        "RUNTIME_MEMORY_ALIGNMENT_BYTES": 64,
        "RUNTIME_SHARED_MEMORY_HEADER_BYTES": 896,
        "RUNTIME_TASK_SLOT_SEGMENT_BYTES": (40, 4864, 64),
        "PINNED_RUNTIME_ARENA_FIXED_BYTES": 20_413_312,
        "PINNED_RUNTIME_ARENA_BYTES_PER_RING_ENTRY": 24,
        "PINNED_RUNTIME_ARENA_ALIGNMENT_SLACK_BYTES": 8192,
    }
    functions = [
        _function(tree, name)
        for name in (
            "_parse_ring_values",
            "_parse_ring_heap_bytes",
            "_align_runtime_bytes",
            "_runtime_shared_memory_bytes",
            "_runtime_private_arena_bytes",
        )
    ]
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, str(_ENTRY_PATH), "exec"), namespace)
    return namespace


def _route_pair_compare_factory(tree: ast.Module):
    namespace = {}
    function = _function(tree, "_unordered_route_pair_compare")
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(_ENTRY_PATH), "exec"), namespace)
    return namespace["_unordered_route_pair_compare"]


def test_import_time_case_specialization_uses_manifest_layer_ids() -> None:
    tree = _parse()
    source = ast.unparse(tree)
    arguments = _argument_calls(tree)

    case_call = arguments["--case"]
    assert ast.unparse(_keyword(case_call, "choices")) == "CASES"
    assert _tuple_assignment(tree, "CASES") == _CASES
    assert ast.literal_eval(_keyword(case_call, "default")) == "hca1"
    assert "parse_known_args" in source
    assert "build_ladder_manifest(STATIC_CASE)" in source
    assert "LAYER_IDS = STATIC_MANIFEST.layer_ids" in source
    assert "HCA_LAYER_IDS = STATIC_MANIFEST.hca_layer_ids" in source
    assert "CSA_LAYER_IDS = STATIC_MANIFEST.csa_layer_ids" in source

    canonical_source = ast.unparse(_function(tree, "_canonicalize_runtime_ring_env"))
    assert "','.join((str(value) for value in values))" in canonical_source
    canonical_call_line = next(
        node.lineno
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and _call_name(node.value) == "_canonicalize_runtime_ring_env"
    )
    pypto_import_line = min(
        node.lineno
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and "pypto" in ast.unparse(node)
    )
    assert canonical_call_line < pypto_import_line

    first_jit_line = min(
        function.lineno
        for function in ast.walk(tree)
        if isinstance(function, ast.FunctionDef)
        and any("jit" in ast.unparse(decorator) for decorator in function.decorator_list)
    )
    specialization_calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "build_ladder_manifest"
    ]
    assert specialization_calls and min(specialization_calls) < first_jit_line


def test_forward_has_no_distributed_or_serving_only_compute_surface() -> None:
    tree = _parse()
    forbidden_modules = {"pypto.language.distributed"}
    forbidden_calls = {
        "alloc_window_buffer",
        "window",
        "remote_store",
        "put",
        "notify",
        "wait",
        "attention_swa",
        "lookup_embedding",
        "lm_head",
    }
    forbidden_names = {"pld", "DistributedTensor"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name not in forbidden_modules for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden_modules
        elif isinstance(node, ast.Call):
            assert _call_name(node) not in forbidden_calls
        elif isinstance(node, ast.Name):
            assert node.id not in forbidden_names

    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not {
        "decode_attention_swa",
        "decode_mtp",
        "decode_fwd_mtp",
        "lookup_embedding",
        "lm_head",
    } & imported_modules


def test_one_worker_host_dispatches_one_child_on_device_zero() -> None:
    tree = _parse()
    child = _function(tree, "decode_fwd_compute")
    host = _function(tree, "l3_decode_fwd_compute")

    assert any(ast.unparse(decorator) == "pl.jit(auto_scope=False)" for decorator in child.decorator_list)
    assert [ast.unparse(decorator) for decorator in host.decorator_list] == ["pl.jit.host"]
    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(host))

    dispatches = [
        node
        for node in ast.walk(host)
        if isinstance(node, ast.Call) and _call_name(node) == "decode_fwd_compute"
    ]
    assert len(dispatches) == 1
    device = _keyword(dispatches[0], "device")
    assert isinstance(device, ast.Constant) and device.value == 0

    config_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "DistributedConfig"
    ]
    assert len(config_calls) == 1
    device_ids = _keyword(config_calls[0], "device_ids")
    assert ast.unparse(device_ids) == "[args.device]"
    assert ast.literal_eval(_keyword(config_calls[0], "num_sub_workers")) == 0


def test_cache_abi_is_layer_local_kind_stacked_and_resident() -> None:
    tree = _parse()
    child = _function(tree, "decode_fwd_compute")
    host = _function(tree, "l3_decode_fwd_compute")
    child_annotations = {argument.arg: ast.unparse(argument.annotation) for argument in child.args.args}
    host_annotations = {argument.arg: ast.unparse(argument.annotation) for argument in host.args.args}

    assert child_annotations["kv_cache"].startswith("pl.InOut[")
    assert "[NUM_LAYERS, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM]" in child_annotations["kv_cache"]
    assert "[HCA_STORAGE_COUNT, HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM]" in child_annotations["hca_cmp_kv"]
    assert "[CSA_STORAGE_COUNT, CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM]" in child_annotations["csa_cmp_kv"]
    assert "[CSA_STORAGE_COUNT, CSA_IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM]" in child_annotations[
        "csa_idx_kv_cache"
    ]
    assert child_annotations["hca_cmp_block_table"] == (
        "pl.Tensor[[B, HCA_CMP_MAX_BLOCKS], pl.INT32]"
    )
    for name in _CACHE_NAMES:
        assert child_annotations[name].startswith("pl.InOut[")
        assert host_annotations[name].startswith("pl.InOut[")

    builder_source = ast.unparse(_function(tree, "build_tensor_specs"))
    assert "resident=0" in builder_source
    assert "is_output=True" in builder_source
    assert "RESIDENT_CACHE_NAMES" in builder_source
    assert "RESIDENT_WEIGHT_NAMES" in builder_source
    assert "block_ids_are_local" in ast.unparse(_function(tree, "_preflight_manifest"))
    assert "active_hca_layers" in ast.unparse(_function(tree, "_print_runtime_manifest"))
    assert "inactive_placeholder_bytes" in ast.unparse(_function(tree, "_print_runtime_manifest"))


def test_layer_weights_are_stacked_and_wq_b_is_head_major() -> None:
    child = _function(_parse(), "decode_fwd_compute")
    annotations = {argument.arg: ast.unparse(argument.annotation) for argument in child.args.args}

    assert annotations["wq_b"] == "pl.Tensor[[NUM_LAYERS, H, HEAD_DIM, Q_LORA], pl.INT8]"
    assert annotations["attn_norm_w"] == "pl.Tensor[[NUM_LAYERS, D], pl.BF16]"
    assert annotations["gate_w"] == "pl.Tensor[[NUM_LAYERS, N_EXPERTS, D], pl.FP32]"
    assert annotations["routed_w1"] == (
        "pl.Tensor[[NUM_LAYERS, N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8]"
    )

    builder_source = ast.unparse(_function(_parse(), "build_tensor_specs"))
    assert "_stack_layer_specs" in builder_source
    assert "layer_ids=LAYER_IDS" in builder_source
    stack_source = ast.unparse(_function(_parse(), "_stack_layer_specs"))
    assert "spec.create_tensor().reshape(shape)" in stack_source


def test_overwrite_policy_preserves_every_cache_write_mapping() -> None:
    tree = _parse()
    source = ast.unparse(tree)
    policy_source = ast.unparse(_function(tree, "_apply_cache_policy"))
    apply_cache_policy = _cache_policy_factory(tree)
    specs = [object() for _ in _SLOT_MAPPING_NAMES]

    assert "CACHE_POLICIES = ('commit', 'overwrite')" in source
    for cache_policy in ("commit", "overwrite"):
        result = apply_cache_policy(specs, cache_policy)
        assert result is specs
        assert all(actual is expected for actual, expected in zip(result, specs, strict=True))
    with pytest.raises(ValueError, match="unknown cache policy"):
        apply_cache_policy(specs, "no-write")

    assert "NO_WRITE_SLOT_MAPPING_NAMES" not in source
    assert "torch.full(shape, -1" not in policy_source
    for name in _SLOT_MAPPING_NAMES:
        assert name in source
    assert "PYPTO_BENCH" in source
    assert "cache_policy != 'overwrite'" in source
    assert "benchmark_value = os.environ.get('PYPTO_BENCH', '').strip()" in source
    assert "benchmark_enabled = benchmark_value not in ('', '0', 'false', 'False')" in source


def test_benchmark_integrity_helper_rejects_unusable_or_incomplete_results() -> None:
    benchmark_result_error = _benchmark_result_error_factory(_parse())

    assert benchmark_result_error(_valid_benchmark_stats()) is None
    assert "no benchmark statistics" in benchmark_result_error(None)

    fallback = _valid_benchmark_stats()
    fallback.fallback_flattened = True
    assert "flattened" in benchmark_result_error(fallback)

    unstable = _valid_benchmark_stats()
    unstable.unstable_dispatch_slots = True
    assert "dispatch slots changed" in benchmark_result_error(unstable)

    invalid_rounds = _valid_benchmark_stats()
    invalid_rounds.rounds = 0
    assert "invalid measured-round count" in benchmark_result_error(invalid_rounds)

    incomplete = _valid_benchmark_stats()
    incomplete.device_wall_us.pop()
    assert "device wall has 1 samples for 2 rounds" in benchmark_result_error(incomplete)

    changed_rank = _valid_benchmark_stats()
    changed_rank.rounds_dispatches[1] = {202: changed_rank.rounds_dispatches[1][101]}
    assert "changed the participating rank set" in benchmark_result_error(changed_rank)

    extra_dispatch = _valid_benchmark_stats()
    extra_dispatch.rounds_dispatches[0][101].append(object())
    assert "has 2 dispatches; expected 1" in benchmark_result_error(extra_dispatch)

    missing_invocation = _valid_benchmark_stats()
    missing_invocation.invocations.pop()
    assert "flat invocation list has 1 entries" in benchmark_result_error(missing_invocation)

    incomplete_effective = _valid_benchmark_stats()
    incomplete_effective.per_round = lambda metric: [30.0]
    assert "effective timing has 1 samples for 2 rounds" in benchmark_result_error(
        incomplete_effective
    )

    nonfinite = _valid_benchmark_stats()
    nonfinite.host_wall_us[1] = math.nan
    assert "host wall sample 1 is invalid" in benchmark_result_error(nonfinite)

    zero_effective = _valid_benchmark_stats()
    zero_effective.per_round = lambda metric: [30.0, 0.0]
    assert "effective sample 1 is invalid" in benchmark_result_error(zero_effective)


def test_benchmark_mode_is_hard_gated_on_overwrite_and_complete_stats() -> None:
    tree = _parse()
    source = ast.unparse(tree)

    assert "if benchmark_enabled and args.compile_only:" in source
    assert "PYPTO_BENCH cannot be combined with --compile-only" in source
    assert "if benchmark_enabled and args.cache_policy != 'overwrite':" in source
    assert "PYPTO_BENCH requires --cache-policy overwrite" in source
    assert "if not benchmark_enabled and args.cache_policy == 'overwrite':" in source
    assert "--cache-policy overwrite requires PYPTO_BENCH" in source
    assert "benchmark_error = _benchmark_result_error(result.bench)" in source
    assert "if benchmark_error is not None:" in source
    assert "benchmark rejected" in source

    run_call_line = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "run_jit"
    )
    integrity_call_line = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "_benchmark_result_error"
    )
    assert run_call_line < integrity_call_line


def test_multilayer_schedule_is_static_and_runtime_ring_is_explicit() -> None:
    tree = _parse()
    source = ast.unparse(tree)
    child_source = ast.unparse(_function(tree, "decode_fwd_compute"))
    manifest_source = ast.unparse(_function(tree, "_print_runtime_manifest"))

    assert "for layer_index in pl.range(NUM_LAYERS)" in child_source
    assert "init_values" not in child_source
    assert "pl.yield_" not in child_source
    assert "hidden: pl.Tensor" in child_source
    assert "hidden = decode_layer_compute_hca(" in child_source
    assert "hidden = decode_layer_compute_csa(" in child_source
    assert "hidden = hidden_next" not in child_source
    assert "if layer_index == NUM_LAYERS - 1:" in child_source
    assert "hidden_next = x_out" in child_source
    assert "pl.assemble(x_out" not in child_source
    assert (
        "os.environ.setdefault('PTO2_RING_HEAP', "
        "','.join((str(value) for value in DEFAULT_RING_HEAP_BYTES)))"
        in source
    )
    assert "os.environ.setdefault('PTO2_RING_TASK_WINDOW', str(RING_TASK_WINDOW))" in source
    assert "os.environ.setdefault('PTO2_RING_DEP_POOL', str(RING_DEP_POOL))" in source
    assert "RING_TASK_WINDOW = 16384" in source
    assert "RING_DEP_POOL = 65536" in source
    preflight_source = ast.unparse(_function(tree, "_preflight_manifest"))
    assert "_known_runtime_resource_bytes()" in preflight_source
    resource_source = ast.unparse(_function(tree, "_known_runtime_resource_bytes"))
    assert "_effective_ring_heap_bytes()" in resource_source
    assert "_runtime_shared_memory_bytes(task_windows)" in resource_source
    assert "_runtime_private_arena_bytes(task_windows, dep_pools)" in resource_source
    specs_preflight_source = ast.unparse(_function(tree, "_preflight_specs"))
    assert "_retained_nonresident_temp_bytes(specs)" in specs_preflight_source
    assert "_preflight_specs(specs)" in source
    assert "runtime_ring_heap_per_ring_bytes" in manifest_source
    assert "runtime_ring_heap_aggregate_bytes" in manifest_source
    assert "runtime_ring_task_window_per_ring_entries" in manifest_source
    assert "runtime_ring_dep_pool_per_ring_entries" in manifest_source
    assert "runtime_ring_shared_memory_bytes" in manifest_source
    assert "runtime_private_arena_bytes" in manifest_source
    assert "runtime_private_arena_default_exact_bytes" in manifest_source
    assert "runtime_private_arena_accounting=pinned-runtime-derived-upper-bound" in manifest_source
    assert "retained_nonresident_temp_bytes" in manifest_source
    assert "static_accounted_plus_known_runtime_and_temp_bytes" in manifest_source
    assert "static_accounted_plus_known_runtime_bytes" in manifest_source
    assert "runtime_private_arena_in_known_allocation_bytes=True" in manifest_source
    assert "runtime_unexposed_overhead_in_known_allocation_bytes=False" in manifest_source
    assert "runtime_ring_resources_in_resident_spec_bytes=False" in manifest_source


def test_default_ring_heaps_are_case_specific_and_inactive_rings_are_bounded() -> None:
    tree = _parse()
    expected_ring0_heaps = {
        "hca1": 2 * (1 << 30),
        "csa1": 2 * (1 << 30),
        "depth2": 2 * (1 << 30),
        "depth4": 2 * (1 << 30),
        "depth16": 4 * (1 << 30),
        "depth31": 8 * (1 << 30),
        "depth61": 14 * (1 << 30),
    }
    inactive_heap = 256 * (1 << 20)

    for static_case, expected_ring0_heap in expected_ring0_heaps.items():
        constants = _ring_heap_constants(tree, static_case)
        assert constants["RING0_HEAP_BYTES_BY_CASE"] == expected_ring0_heaps
        assert constants["INACTIVE_RING_HEAP_BYTES"] == inactive_heap
        assert constants["DEFAULT_RING_HEAP_BYTES"] == (
            expected_ring0_heap,
            inactive_heap,
            inactive_heap,
            inactive_heap,
        )


def test_ring_resource_accounting_matches_four_ring_runtime_semantics() -> None:
    functions = _ring_resource_functions(_parse())
    parse_ring_values = functions["_parse_ring_values"]
    parse_ring_heap = functions["_parse_ring_heap_bytes"]
    runtime_shared_memory_bytes = functions["_runtime_shared_memory_bytes"]
    runtime_private_arena_bytes = functions["_runtime_private_arena_bytes"]

    assert parse_ring_heap("2147483648") == 4 * 2147483648
    assert parse_ring_heap("1024") == 4 * 1024
    assert parse_ring_heap("1024,2048,3072,4096") == 10240
    assert parse_ring_values("TASK", "16384", 4, (1 << 31) - 1, True) == (
        16384,
        16384,
        16384,
        16384,
    )
    assert parse_ring_values("DEP", "4,8,12,16", 4, (1 << 31) - 1) == (
        4,
        8,
        12,
        16,
    )
    assert runtime_shared_memory_bytes((16384,) * 4) == 325_583_744
    assert runtime_shared_memory_bytes((262144,) * 4) == 5_209_326_464
    assert runtime_shared_memory_bytes((4, 8, 16, 32)) == 299_008
    assert runtime_private_arena_bytes((16384,) * 4, (16384,) * 4) == 23_567_232
    assert runtime_private_arena_bytes((262144,) * 4, (262144,) * 4) == 70_753_152
    assert runtime_private_arena_bytes((16384,) * 4, (65536,) * 4) == 28_285_824
    default_hca1_ring_heaps = 2 * (1 << 30) + 3 * 256 * (1 << 20)
    assert default_hca1_ring_heaps + 325_583_744 + 28_285_824 == 3_306_659_584
    assert (
        48_604_508_164 + default_hca1_ring_heaps + 325_583_744 + 28_285_824
        == 51_911_167_748
    )

    for invalid in (
        "",
        "1023",
        " 1024 ",
        "1_024",
        "+1024",
        "1024,2048",
        "1024,2048,3072,4096,5120",
    ):
        with pytest.raises(ValueError, match="invalid PTO2_RING_HEAP"):
            parse_ring_heap(invalid)
    with pytest.raises(ValueError, match="invalid TASK"):
        parse_ring_values("TASK", "12", 4, (1 << 31) - 1, True)


def test_cli_bounds_goldens_and_labels_deeper_smoke() -> None:
    tree = _parse()
    arguments = _argument_calls(tree)
    required = {
        "--case",
        "--platform",
        "--device",
        "--start-pos",
        "--workload",
        "--cache-policy",
        "--compile-only",
        "--save-data",
        "--golden-data",
        "--enable-l2-swimlane",
        "--enable-scope-stats",
        "--enable-dep-gen",
        "--dump-passes",
        "--print-manifest",
    }
    assert required <= arguments.keys()
    assert "--ep" not in arguments and "--tp" not in arguments
    assert ast.unparse(_keyword(arguments["--cache-policy"], "choices")) == "CACHE_POLICIES"
    assert _tuple_assignment(tree, "CACHE_POLICIES") == ("commit", "overwrite")
    assert ast.literal_eval(_keyword(arguments["--enable-dep-gen"], "action")) == "store_true"

    source = ast.unparse(tree)
    assert "GOLDEN_CASES = ('hca1', 'csa1', 'depth2', 'depth4')" in source
    assert "NO_GOLDEN_CASES = ('depth16', 'depth31', 'depth61')" in source
    assert "validation skipped" in source
    assert "no golden" in source.lower()
    assert "main-model compute proxy" in source
    assert "EP128 end-to-end" in source

    run_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "run_jit"
    ]
    assert len(run_calls) == 1
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in run_calls[0].keywords}
    assert keywords["fn"] == "l3_decode_fwd_compute"
    assert keywords["compile_only"] == "args.compile_only"
    assert keywords["save_data"] == "args.save_data"
    assert keywords["golden_data"] == "golden_data"
    assert keywords["golden_fn"] == "golden_fn"
    assert "distributed_config=distributed_config" in keywords["compile_cfg"]
    assert "enable_scope_stats=args.enable_scope_stats" in keywords["runtime_cfg"]
    assert "enable_dep_gen=args.enable_dep_gen" in keywords["runtime_cfg"]
    assert "enable_dep_gen=True" not in keywords["runtime_cfg"]


def test_routing_audit_outputs_keep_gate_results_live() -> None:
    tree = _parse()
    child = _function(tree, "decode_fwd_compute")
    host = _function(tree, "l3_decode_fwd_compute")
    for function in (child, host):
        annotations = {argument.arg: ast.unparse(argument.annotation) for argument in function.args.args}
        assert annotations["layer_indices"] == (
            "pl.Out[pl.Tensor[[NUM_LAYERS, T, TOPK], pl.INT32]]"
        )
        assert annotations["layer_weights"] == (
            "pl.Out[pl.Tensor[[NUM_LAYERS, T, TOPK], pl.FP32]]"
        )

    child_source = ast.unparse(child)
    assert "layer_indices_ranked = pl.slice(layer_indices" in child_source
    assert "layer_weights_ranked = pl.slice(layer_weights" in child_source
    assert "pl.reshape(layer_indices_ranked" in child_source
    assert "pl.reshape(layer_weights_ranked" in child_source
    assert child_source.count("layer_indices_view") >= 3
    assert child_source.count("layer_weights_view") >= 3
    assert "indices = pl.create_tensor" not in child_source
    assert "weights = pl.create_tensor" not in child_source

    builder_source = ast.unparse(_function(tree, "build_tensor_specs"))
    assert "'layer_indices'" in builder_source
    assert "'layer_weights'" in builder_source
    main_source = ast.unparse(tree)
    assert "'layer_indices': route_pair_compare" in main_source
    assert "'layer_weights': route_pair_compare" in main_source
    route_compare_source = ast.unparse(_function(tree, "_unordered_route_pair_compare"))
    assert "actual_indices.sort(dim=-1)" in route_compare_source
    assert "expected_indices.sort(dim=-1)" in route_compare_source
    assert "actual_weights.gather(-1, actual_order)" in route_compare_source
    assert "expected_weights.gather(-1, expected_order)" in route_compare_source
    assert "strict_state_compare = ratio_allclose(atol=0.001, rtol=0.001)" in main_source
    assert (
        "hca_accumulated_state_compare = ratio_reldiff(diff_thd=0.02, pct_thd=0.005)"
        in main_source
    )
    assert (
        "csa_accumulated_state_compare = ratio_reldiff(diff_thd=0.02, pct_thd=0.02)"
        in main_source
    )
    assert "'hca_compress_state': hca_state_compare" in main_source
    assert "'csa_compress_state': csa_state_compare" in main_source
    assert "'csa_inner_compress_state': csa_state_compare" in main_source

    selection_source = ast.unparse(_function(tree, "_state_slice_comparators"))
    assert "storage_count != max(1, len(layer_ids))" in selection_source
    assert "if not layer_ids:" in selection_source
    assert "return (strict_compare,) * storage_count" in selection_source
    assert "LAYER_IDS.index(layer_id) == 0" in selection_source
    assert "else accumulated_compare" in selection_source
    assert "HCA_LAYER_IDS" in main_source
    assert "HCA_STORAGE_COUNT" in main_source
    assert "CSA_LAYER_IDS" in main_source
    assert "CSA_STORAGE_COUNT" in main_source

    slice_source = ast.unparse(_function(tree, "_per_first_axis_slice"))
    assert "compares = tuple(compares)" in slice_source
    assert "actual.shape[0] != len(compares)" in slice_source
    assert "for (slice_index, compare) in enumerate(compares)" in slice_source
    assert "slice[{slice_index}] ({compare.__name__}) failed" in slice_source


def test_unordered_route_pair_compare_preserves_expert_weight_association() -> None:
    import torch

    def exact_weight_compare(actual, expected, **_kwargs):
        passed = torch.equal(actual, expected)
        return passed, "    canonical weights differ"

    compare = _route_pair_compare_factory(_parse())(
        "layer_indices",
        "layer_weights",
        exact_weight_compare,
    )
    expected_outputs = {
        "layer_indices": torch.tensor([[[1, 2, 3]]], dtype=torch.int32),
        "layer_weights": torch.tensor([[[10.0, 20.0, 30.0]]]),
    }
    permuted_outputs = {
        "layer_indices": torch.tensor([[[2, 1, 3]]], dtype=torch.int32),
        "layer_weights": torch.tensor([[[20.0, 10.0, 30.0]]]),
    }
    compare_kwargs = {
        "actual_outputs": permuted_outputs,
        "expected_outputs": expected_outputs,
        "inputs": {},
        "rtol": 0.0,
        "atol": 0.0,
    }

    passed, _ = compare(
        permuted_outputs["layer_indices"],
        expected_outputs["layer_indices"],
        **compare_kwargs,
    )
    assert passed

    wrong_expert_outputs = dict(permuted_outputs)
    wrong_expert_outputs["layer_indices"] = torch.tensor([[[2, 4, 3]]], dtype=torch.int32)
    passed, _ = compare(
        wrong_expert_outputs["layer_indices"],
        expected_outputs["layer_indices"],
        **{**compare_kwargs, "actual_outputs": wrong_expert_outputs},
    )
    assert not passed

    detached_weight_outputs = dict(permuted_outputs)
    detached_weight_outputs["layer_weights"] = torch.tensor([[[10.0, 20.0, 30.0]]])
    passed, _ = compare(
        detached_weight_outputs["layer_weights"],
        expected_outputs["layer_weights"],
        **{**compare_kwargs, "actual_outputs": detached_weight_outputs},
    )
    assert not passed
