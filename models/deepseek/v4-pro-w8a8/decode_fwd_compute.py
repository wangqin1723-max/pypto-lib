# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Static-depth DeepSeek-V4 Pro W8A8 main-model compute proxy."""

from __future__ import annotations

import argparse
import os

from main_compute_manifest import GIB, build_ladder_manifest


CASES = ("hca1", "csa1", "depth2", "depth4", "depth16", "depth31", "depth61")
GOLDEN_CASES = ("hca1", "csa1", "depth2", "depth4")
NO_GOLDEN_CASES = ("depth16", "depth31", "depth61")
CACHE_POLICIES = ("commit", "overwrite")

_STATIC_PARSER = argparse.ArgumentParser(add_help=False)
_STATIC_PARSER.add_argument("--case", choices=CASES, default="hca1")
_STATIC_ARGS, _ = _STATIC_PARSER.parse_known_args()
STATIC_CASE = _STATIC_ARGS.case
STATIC_MANIFEST = build_ladder_manifest(STATIC_CASE)
LAYER_IDS = STATIC_MANIFEST.layer_ids
HCA_LAYER_IDS = STATIC_MANIFEST.hca_layer_ids
CSA_LAYER_IDS = STATIC_MANIFEST.csa_layer_ids
NUM_LAYERS = len(LAYER_IDS)
HCA_LAYER_COUNT = len(HCA_LAYER_IDS)
CSA_LAYER_COUNT = len(CSA_LAYER_IDS)
HCA_STORAGE_COUNT = max(1, HCA_LAYER_COUNT)
CSA_STORAGE_COUNT = max(1, CSA_LAYER_COUNT)
HASH_LAYER_IDS = tuple(layer_id for layer_id in LAYER_IDS if layer_id < 3)
HASH_LAYER_COUNT = len(HASH_LAYER_IDS)

RUNTIME_RING_COUNT = 4
RING0_HEAP_BYTES_BY_CASE = {
    "hca1": 2 * GIB,
    "csa1": 2 * GIB,
    "depth2": 2 * GIB,
    "depth4": 2 * GIB,
    "depth16": 4 * GIB,
    "depth31": 8 * GIB,
    "depth61": 14 * GIB,
}
INACTIVE_RING_HEAP_BYTES = 256 * 1024 * 1024
DEFAULT_RING_HEAP_BYTES = (
    RING0_HEAP_BYTES_BY_CASE[STATIC_CASE],
    INACTIVE_RING_HEAP_BYTES,
    INACTIVE_RING_HEAP_BYTES,
    INACTIVE_RING_HEAP_BYTES,
)
RING_HEAP_MIN_BYTES = 1024
RING_TASK_WINDOW = 16384
RING_DEP_POOL = 65536
RING_ENTRY_MIN = 4
RING_ENTRY_MAX = (1 << 31) - 1
RUNTIME_MEMORY_ALIGNMENT_BYTES = 64
RUNTIME_SHARED_MEMORY_HEADER_BYTES = 896
RUNTIME_TASK_SLOT_SEGMENT_BYTES = (40, 4864, 64)
PINNED_RUNTIME_ARENA_FIXED_BYTES = 20_413_312
PINNED_RUNTIME_ARENA_BYTES_PER_RING_ENTRY = 24
PINNED_RUNTIME_ARENA_ALIGNMENT_SLACK_BYTES = 8192
PINNED_RUNTIME_DEFAULT_ARENA_EXACT_BYTES = 28_277_632
RUNTIME_RETAINED_TEMP_ALIGNMENT_BYTES = 1024
os.environ.setdefault(
    "PTO2_RING_HEAP",
    ",".join(str(value) for value in DEFAULT_RING_HEAP_BYTES),
)
os.environ.setdefault("PTO2_RING_TASK_WINDOW", str(RING_TASK_WINDOW))
os.environ.setdefault("PTO2_RING_DEP_POOL", str(RING_DEP_POOL))


def _parse_ring_values(
    variable_name,
    raw_value,
    minimum,
    maximum=None,
    require_power_of_two=False,
):
    """Parse one broadcast value or four explicit runtime ring values."""

    raw_values = raw_value.split(",")
    if len(raw_values) == 1:
        raw_values *= RUNTIME_RING_COUNT
    elif len(raw_values) != RUNTIME_RING_COUNT:
        raise ValueError(f"invalid {variable_name}={raw_value!r}")
    if any(not value or not value.isascii() or not value.isdecimal() for value in raw_values):
        raise ValueError(f"invalid {variable_name}={raw_value!r}")
    per_ring_values = tuple(int(value) for value in raw_values)
    if any(
        value < minimum
        or (maximum is not None and value > maximum)
        or (require_power_of_two and value & (value - 1))
        for value in per_ring_values
    ):
        raise ValueError(f"invalid {variable_name}={raw_value!r}")
    return per_ring_values


def _parse_ring_heap_bytes(raw_value):
    """Return the aggregate heap bytes for the runtime's four rings."""

    return sum(
        _parse_ring_values(
            "PTO2_RING_HEAP",
            raw_value,
            RING_HEAP_MIN_BYTES,
        )
    )


def _effective_ring_heap_values():
    """Return the effective per-ring output-heap sizes."""

    return _parse_ring_values(
        "PTO2_RING_HEAP",
        os.environ["PTO2_RING_HEAP"],
        RING_HEAP_MIN_BYTES,
    )


def _effective_ring_heap_bytes():
    """Return the configured aggregate runtime ring-heap footprint."""

    return sum(_effective_ring_heap_values())


def _effective_ring_task_windows():
    """Return the effective per-ring task-window entry counts."""

    return _parse_ring_values(
        "PTO2_RING_TASK_WINDOW",
        os.environ["PTO2_RING_TASK_WINDOW"],
        RING_ENTRY_MIN,
        maximum=RING_ENTRY_MAX,
        require_power_of_two=True,
    )


def _effective_ring_dep_pools():
    """Return the effective per-ring dependency-pool entry counts."""

    return _parse_ring_values(
        "PTO2_RING_DEP_POOL",
        os.environ["PTO2_RING_DEP_POOL"],
        RING_ENTRY_MIN,
        maximum=RING_ENTRY_MAX,
    )


def _align_runtime_bytes(value):
    """Align one runtime allocation segment to the pinned ABI boundary."""

    alignment = RUNTIME_MEMORY_ALIGNMENT_BYTES
    return (value + alignment - 1) // alignment * alignment


def _runtime_shared_memory_bytes(task_windows=None):
    """Return the exact pinned-runtime TMR shared-memory allocation size."""

    if task_windows is None:
        task_windows = _effective_ring_task_windows()
    total_bytes = _align_runtime_bytes(RUNTIME_SHARED_MEMORY_HEADER_BYTES)
    for task_window in task_windows:
        for segment_bytes in RUNTIME_TASK_SLOT_SEGMENT_BYTES:
            total_bytes += _align_runtime_bytes(task_window * segment_bytes)
    return total_bytes


def _runtime_private_arena_bytes(task_windows=None, dep_pools=None):
    """Return a conservative pinned-runtime private arena allocation bound."""

    if task_windows is None:
        task_windows = _effective_ring_task_windows()
    if dep_pools is None:
        dep_pools = _effective_ring_dep_pools()
    variable_entries = sum(task_windows) + sum(dep_pools)
    return (
        PINNED_RUNTIME_ARENA_FIXED_BYTES
        + PINNED_RUNTIME_ARENA_BYTES_PER_RING_ENTRY * variable_entries
        + PINNED_RUNTIME_ARENA_ALIGNMENT_SLACK_BYTES
    )


def _known_runtime_resource_bytes():
    """Return heap, TMR shared-memory, and pinned private-arena bytes."""

    task_windows = _effective_ring_task_windows()
    dep_pools = _effective_ring_dep_pools()
    return (
        _effective_ring_heap_bytes()
        + _runtime_shared_memory_bytes(task_windows)
        + _runtime_private_arena_bytes(task_windows, dep_pools)
    )


def _canonicalize_runtime_ring_env():
    """Canonicalize validated runtime ring values before importing PyPTO."""

    values_by_name = {
        "PTO2_RING_HEAP": _effective_ring_heap_values(),
        "PTO2_RING_TASK_WINDOW": _effective_ring_task_windows(),
        "PTO2_RING_DEP_POOL": _effective_ring_dep_pools(),
    }
    for variable_name, values in values_by_name.items():
        os.environ[variable_name] = ",".join(str(value) for value in values)


_canonicalize_runtime_ring_env()


import pypto.language as pl  # noqa: E402
from golden import TensorSpec  # noqa: E402
from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: E402

from config import (  # noqa: E402
    C128_COMPRESSOR_BLOCK_SIZE,
    DECODE_START_POS,
    HCA_STATE_PHYSICAL_BLOCKS,
    ORI_KV_BLOCK_NUM,
)
from decode_layer_compute import (  # noqa: E402
    B,
    BLOCK_SIZE,
    CSA_CMP_BLOCK_NUM,
    CSA_CMP_MAX_BLOCKS,
    CSA_COMPRESS_RATIO,
    CSA_IDX_CACHE_BLOCK_NUM,
    CSA_IDX_CACHE_MAX_BLOCKS,
    CSA_IDX_HEAD_DIM,
    CSA_IDX_N_HEADS,
    CSA_INNER_OUT_DIM,
    CSA_INNER_STATE_BLOCK_NUM,
    CSA_INNER_STATE_BLOCK_SIZE,
    CSA_INNER_STATE_DIM,
    CSA_INNER_STATE_MAX_BLOCKS,
    CSA_MAIN_OUT_DIM,
    CSA_MAIN_STATE_BLOCK_NUM,
    CSA_MAIN_STATE_BLOCK_SIZE,
    CSA_MAIN_STATE_DIM,
    CSA_MAIN_STATE_MAX_BLOCKS,
    D,
    H,
    HC_DIM,
    HC_MULT,
    HEAD_DIM,
    HCA_COMPRESS_RATIO,
    HCA_CMP_MAX_BLOCKS,
    HCA_MAIN_OUT_DIM,
    HCA_STATE_DIM,
    HCA_STATE_MAX_BLOCKS,
    MAX_SEQ_LEN,
    MIX_HC,
    MOE_INTER,
    N_EXPERTS,
    N_LOCAL_EXPERTS,
    O_GROUPS,
    O_GROUP_IN,
    O_LORA,
    Q_LORA,
    RECV_MAX,
    ROPE_HEAD_DIM,
    ROUTED_WORKLOAD_COUNTS,
    T,
    TOPK,
    VOCAB,
    WIN,
    build_csa_tensor_specs,
    build_hca_tensor_specs,
    decode_layer_compute_csa,
    decode_layer_compute_hca,
    golden_decode_layer_compute_csa,
    golden_decode_layer_compute_hca,
)
from main_compute_manifest import HCA_COMPRESSED_BLOCKS  # noqa: E402


ORI_BLOCK_NUM = ORI_KV_BLOCK_NUM
HCA_CMP_BLOCK_NUM = HCA_COMPRESSED_BLOCKS
HCA_MAIN_STATE_BLOCK_NUM = HCA_STATE_PHYSICAL_BLOCKS

COMMON_ATTENTION_WEIGHT_NAMES = (
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "attn_norm_w",
    "wq_a",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "gamma_cq",
    "gamma_ckv",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
)
COMMON_FFN_WEIGHT_NAMES = (
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "norm_w",
    "gate_w",
    "gate_bias",
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
)
HCA_WEIGHT_NAMES = ("cmp_wkv", "cmp_wgate", "cmp_ape", "cmp_norm_w")
CSA_WEIGHT_NAMES = (
    "cmp_wkv",
    "cmp_wgate",
    "cmp_ape",
    "cmp_norm_w",
    "idx_wq_b",
    "idx_wq_b_scale",
    "weights_proj",
    "hadamard_idx",
    "inner_wkv",
    "inner_wgate",
    "inner_ape",
    "inner_norm_w",
)
RESIDENT_WEIGHT_NAMES = frozenset(
    (
        *COMMON_ATTENTION_WEIGHT_NAMES,
        *COMMON_FFN_WEIGHT_NAMES,
        *(f"hca_{name}" for name in HCA_WEIGHT_NAMES),
        *(f"csa_{name}" for name in CSA_WEIGHT_NAMES),
        "tid2eid",
        "freqs_cos",
        "freqs_sin",
    )
)
RESIDENT_CACHE_NAMES = frozenset(
    {
        "kv_cache",
        "hca_cmp_kv",
        "hca_compress_state",
        "csa_cmp_kv",
        "csa_idx_kv_cache",
        "csa_idx_kv_scale",
        "csa_compress_state",
        "csa_inner_compress_state",
    }
)
def _preflight_manifest():
    """Validate the selected layer schedule and layer-local cache capacities."""

    expected = build_ladder_manifest(STATIC_CASE)
    if expected.layer_ids != LAYER_IDS:
        raise ValueError("the selected main-layer IDs changed after static specialization")
    if not expected.cache_ranges_are_independent:
        raise ValueError("main-layer cache byte ranges overlap")
    known_allocation_bytes = expected.accounted_bytes + _known_runtime_resource_bytes()
    if known_allocation_bytes >= 64 * GIB:
        raise ValueError(
            "the static main-model allocation plus known runtime ring allocations "
            "exceeds one 64-GiB die"
        )

    expected_capacities = {
        "original_kv": ORI_BLOCK_NUM,
        "hca_compressed_kv": HCA_CMP_BLOCK_NUM,
        "hca_compressor_state": HCA_MAIN_STATE_BLOCK_NUM,
        "csa_compressed_kv": CSA_CMP_BLOCK_NUM,
        "csa_index_kv": CSA_IDX_CACHE_BLOCK_NUM,
        "csa_index_scale": CSA_IDX_CACHE_BLOCK_NUM,
        "csa_compressor_state": CSA_MAIN_STATE_BLOCK_NUM,
        "csa_inner_compressor_state": CSA_INNER_STATE_BLOCK_NUM,
    }
    for pool in expected.cache_pools:
        capacity = expected_capacities[pool.family]
        if pool.physical_blocks != capacity:
            raise ValueError(f"{pool.family} capacity changed from {capacity} to {pool.physical_blocks}")
        if not pool.block_ids_are_local((0, capacity - 1)):
            raise ValueError(f"{pool.family} block IDs are not layer-local")


_preflight_manifest()


def _print_manifest():
    """Print the selected shape-only HBM manifest."""

    manifest = STATIC_MANIFEST
    print(f"case={STATIC_CASE}")
    print(f"layer_ids={manifest.layer_ids}")
    print(f"hca_layer_ids={manifest.hca_layer_ids}")
    print(f"csa_layer_ids={manifest.csa_layer_ids}")
    print(f"weights_bytes={manifest.weight_bytes}")
    print(f"caches_bytes={manifest.cache_bytes}")
    print(f"shared_assets_bytes={manifest.shared_asset_bytes}")
    print(f"shared_metadata_bytes={manifest.shared_metadata_bytes}")
    print(f"accounted_bytes={manifest.accounted_bytes}")


@pl.jit(auto_scope=False)
def decode_fwd_compute(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[NUM_LAYERS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[NUM_LAYERS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[NUM_LAYERS, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[NUM_LAYERS, D], pl.BF16],
    wq_a: pl.Tensor[[NUM_LAYERS, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[NUM_LAYERS, H, HEAD_DIM, Q_LORA], pl.INT8],
    wq_b_scale: pl.Tensor[[NUM_LAYERS, H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[NUM_LAYERS, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[NUM_LAYERS, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[NUM_LAYERS, HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[
        pl.Tensor[[NUM_LAYERS, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    attn_sink: pl.Tensor[[NUM_LAYERS, H], pl.FP32],
    wo_a: pl.Tensor[[NUM_LAYERS, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[NUM_LAYERS, D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[NUM_LAYERS, D], pl.FP32],
    hca_cmp_wkv: pl.Tensor[[HCA_STORAGE_COUNT, HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_STORAGE_COUNT, HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[
        [HCA_STORAGE_COUNT, HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    hca_cmp_norm_w: pl.Tensor[[HCA_STORAGE_COUNT, HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[
        pl.Tensor[
            [
                HCA_STORAGE_COUNT,
                HCA_MAIN_STATE_BLOCK_NUM,
                C128_COMPRESSOR_BLOCK_SIZE,
                HCA_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    hca_compress_state_block_table: pl.Tensor[[B, HCA_STATE_MAX_BLOCKS], pl.INT32],
    hca_cmp_kv: pl.InOut[
        pl.Tensor[
            [HCA_STORAGE_COUNT, HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    hca_cmp_block_table: pl.Tensor[[B, HCA_CMP_MAX_BLOCKS], pl.INT32],
    hca_cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_cmp_wkv: pl.Tensor[[CSA_STORAGE_COUNT, CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_STORAGE_COUNT, CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[
        [CSA_STORAGE_COUNT, CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    csa_cmp_norm_w: pl.Tensor[[CSA_STORAGE_COUNT, HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[
        pl.Tensor[
            [
                CSA_STORAGE_COUNT,
                CSA_MAIN_STATE_BLOCK_NUM,
                CSA_MAIN_STATE_BLOCK_SIZE,
                CSA_MAIN_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    csa_compress_state_block_table: pl.Tensor[
        [B, CSA_MAIN_STATE_MAX_BLOCKS],
        pl.INT32,
    ],
    csa_idx_wq_b: pl.Tensor[
        [CSA_STORAGE_COUNT, Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM],
        pl.INT8,
    ],
    csa_idx_wq_b_scale: pl.Tensor[
        [CSA_STORAGE_COUNT, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM],
        pl.FP32,
    ],
    csa_weights_proj: pl.Tensor[[CSA_STORAGE_COUNT, D, CSA_IDX_N_HEADS], pl.BF16],
    csa_hadamard_idx: pl.Tensor[
        [CSA_STORAGE_COUNT, CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM],
        pl.BF16,
    ],
    csa_inner_wkv: pl.Tensor[[CSA_STORAGE_COUNT, CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[CSA_STORAGE_COUNT, CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[
        [CSA_STORAGE_COUNT, CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM],
        pl.FP32,
    ],
    csa_inner_norm_w: pl.Tensor[[CSA_STORAGE_COUNT, CSA_IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[
        pl.Tensor[
            [
                CSA_STORAGE_COUNT,
                CSA_INNER_STATE_BLOCK_NUM,
                CSA_INNER_STATE_BLOCK_SIZE,
                CSA_INNER_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    csa_inner_compress_state_block_table: pl.Tensor[
        [B, CSA_INNER_STATE_MAX_BLOCKS],
        pl.INT32,
    ],
    csa_cmp_kv: pl.InOut[
        pl.Tensor[
            [CSA_STORAGE_COUNT, CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    csa_cmp_block_table: pl.Tensor[[B, CSA_CMP_MAX_BLOCKS], pl.INT32],
    csa_idx_kv_cache: pl.InOut[
        pl.Tensor[
            [
                CSA_STORAGE_COUNT,
                CSA_IDX_CACHE_BLOCK_NUM,
                BLOCK_SIZE,
                1,
                CSA_IDX_HEAD_DIM,
            ],
            pl.INT8,
        ]
    ],
    csa_idx_kv_scale: pl.InOut[
        pl.Tensor[
            [CSA_STORAGE_COUNT, CSA_IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1],
            pl.FP32,
        ]
    ],
    csa_idx_block_table: pl.Tensor[[B, CSA_IDX_CACHE_MAX_BLOCKS], pl.INT32],
    csa_cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    window_swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    hc_ffn_fn: pl.Tensor[[NUM_LAYERS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[NUM_LAYERS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[NUM_LAYERS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[NUM_LAYERS, D], pl.BF16],
    gate_w: pl.Tensor[[NUM_LAYERS, N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[NUM_LAYERS, N_EXPERTS], pl.FP32],
    tid2eid: pl.Tensor[[HASH_LAYER_COUNT, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1: pl.Tensor[
        [NUM_LAYERS, N_LOCAL_EXPERTS, MOE_INTER, D],
        pl.INT8,
    ],
    routed_w1_scale: pl.Tensor[
        [NUM_LAYERS, N_LOCAL_EXPERTS, MOE_INTER],
        pl.FP32,
    ],
    routed_w3: pl.Tensor[
        [NUM_LAYERS, N_LOCAL_EXPERTS, MOE_INTER, D],
        pl.INT8,
    ],
    routed_w3_scale: pl.Tensor[
        [NUM_LAYERS, N_LOCAL_EXPERTS, MOE_INTER],
        pl.FP32,
    ],
    routed_w2: pl.Tensor[
        [NUM_LAYERS, N_LOCAL_EXPERTS, D, MOE_INTER],
        pl.INT8,
    ],
    routed_w2_scale: pl.Tensor[[NUM_LAYERS, N_LOCAL_EXPERTS, D], pl.FP32],
    shared_w1: pl.Tensor[[NUM_LAYERS, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[NUM_LAYERS, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[NUM_LAYERS, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[NUM_LAYERS, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[NUM_LAYERS, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[NUM_LAYERS, D], pl.FP32],
    route_to_recv: pl.Tensor[[T, TOPK], pl.INT32],
    layer_ids: pl.Tensor[[NUM_LAYERS], pl.INT32],
    hca_stack_indices: pl.Tensor[[NUM_LAYERS], pl.INT32],
    csa_stack_indices: pl.Tensor[[NUM_LAYERS], pl.INT32],
    hash_stack_indices: pl.Tensor[[NUM_LAYERS], pl.INT32],
    layer_indices: pl.Out[pl.Tensor[[NUM_LAYERS, T, TOPK], pl.INT32]],
    layer_weights: pl.Out[pl.Tensor[[NUM_LAYERS, T, TOPK], pl.FP32]],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
):
    hidden: pl.Tensor[[T, HC_MULT, D], pl.FP32] = x_hc
    for layer_index in pl.range(NUM_LAYERS):
        hidden_next = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
        if layer_index == NUM_LAYERS - 1:
            hidden_next = x_out
        layer_id = pl.read(layer_ids, [layer_index])
        hca_index_i32 = pl.read(hca_stack_indices, [layer_index])
        csa_index_i32 = pl.read(csa_stack_indices, [layer_index])
        hash_index_i32 = pl.read(hash_stack_indices, [layer_index])
        hash_index = pl.cast(hash_index_i32, pl.INDEX)

        hc_attn_fn_ranked = pl.slice(
            hc_attn_fn,
            [1, MIX_HC, HC_DIM],
            [layer_index, 0, 0],
        )
        hc_attn_fn_layer: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.reshape(
            hc_attn_fn_ranked,
            [MIX_HC, HC_DIM],
        )
        hc_attn_scale_ranked = pl.slice(hc_attn_scale, [1, 3], [layer_index, 0])
        hc_attn_scale_layer: pl.Tensor[[3], pl.FP32] = pl.reshape(
            hc_attn_scale_ranked,
            [3],
        )
        hc_attn_base_ranked = pl.slice(
            hc_attn_base,
            [1, MIX_HC],
            [layer_index, 0],
        )
        hc_attn_base_layer: pl.Tensor[[MIX_HC], pl.FP32] = pl.reshape(
            hc_attn_base_ranked,
            [MIX_HC],
        )
        attn_norm_w_ranked = pl.slice(attn_norm_w, [1, D], [layer_index, 0])
        attn_norm_w_layer: pl.Tensor[[D], pl.BF16] = pl.reshape(
            attn_norm_w_ranked,
            [D],
        )
        wq_a_ranked = pl.slice(wq_a, [1, D, Q_LORA], [layer_index, 0, 0])
        wq_a_layer: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.reshape(
            wq_a_ranked,
            [D, Q_LORA],
        )
        wq_b_ranked = pl.slice(
            wq_b,
            [1, H, HEAD_DIM, Q_LORA],
            [layer_index, 0, 0, 0],
        )
        wq_b_layer: pl.Tensor[[H, HEAD_DIM, Q_LORA], pl.INT8] = pl.reshape(
            wq_b_ranked,
            [H, HEAD_DIM, Q_LORA],
        )
        wq_b_scale_ranked = pl.slice(
            wq_b_scale,
            [1, H * HEAD_DIM],
            [layer_index, 0],
        )
        wq_b_scale_layer: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.reshape(
            wq_b_scale_ranked,
            [H * HEAD_DIM],
        )
        wkv_ranked = pl.slice(wkv, [1, D, HEAD_DIM], [layer_index, 0, 0])
        wkv_layer: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.reshape(
            wkv_ranked,
            [D, HEAD_DIM],
        )
        gamma_cq_ranked = pl.slice(gamma_cq, [1, Q_LORA], [layer_index, 0])
        gamma_cq_layer: pl.Tensor[[Q_LORA], pl.BF16] = pl.reshape(
            gamma_cq_ranked,
            [Q_LORA],
        )
        gamma_ckv_ranked = pl.slice(gamma_ckv, [1, HEAD_DIM], [layer_index, 0])
        gamma_ckv_layer: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.reshape(
            gamma_ckv_ranked,
            [HEAD_DIM],
        )
        kv_cache_ranked = pl.slice(
            kv_cache,
            [1, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            [layer_index, 0, 0, 0, 0],
        )
        kv_cache_layer: pl.Tensor[
            [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            pl.BF16,
        ] = pl.reshape(
            kv_cache_ranked,
            [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
        )
        attn_sink_ranked = pl.slice(attn_sink, [1, H], [layer_index, 0])
        attn_sink_layer: pl.Tensor[[H], pl.FP32] = pl.reshape(
            attn_sink_ranked,
            [H],
        )
        wo_a_ranked = pl.slice(
            wo_a,
            [1, O_GROUPS, O_LORA, O_GROUP_IN],
            [layer_index, 0, 0, 0],
        )
        wo_a_layer: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.reshape(
            wo_a_ranked,
            [O_GROUPS, O_LORA, O_GROUP_IN],
        )
        wo_b_ranked = pl.slice(
            wo_b,
            [1, D, O_GROUPS * O_LORA],
            [layer_index, 0, 0],
        )
        wo_b_layer: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.reshape(
            wo_b_ranked,
            [D, O_GROUPS * O_LORA],
        )
        wo_b_scale_ranked = pl.slice(wo_b_scale, [1, D], [layer_index, 0])
        wo_b_scale_layer: pl.Tensor[[D], pl.FP32] = pl.reshape(
            wo_b_scale_ranked,
            [D],
        )
        hc_ffn_fn_ranked = pl.slice(
            hc_ffn_fn,
            [1, MIX_HC, HC_DIM],
            [layer_index, 0, 0],
        )
        hc_ffn_fn_layer: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.reshape(
            hc_ffn_fn_ranked,
            [MIX_HC, HC_DIM],
        )
        hc_ffn_scale_ranked = pl.slice(hc_ffn_scale, [1, 3], [layer_index, 0])
        hc_ffn_scale_layer: pl.Tensor[[3], pl.FP32] = pl.reshape(
            hc_ffn_scale_ranked,
            [3],
        )
        hc_ffn_base_ranked = pl.slice(
            hc_ffn_base,
            [1, MIX_HC],
            [layer_index, 0],
        )
        hc_ffn_base_layer: pl.Tensor[[MIX_HC], pl.FP32] = pl.reshape(
            hc_ffn_base_ranked,
            [MIX_HC],
        )
        norm_w_ranked = pl.slice(norm_w, [1, D], [layer_index, 0])
        norm_w_layer: pl.Tensor[[D], pl.BF16] = pl.reshape(norm_w_ranked, [D])
        gate_w_ranked = pl.slice(
            gate_w,
            [1, N_EXPERTS, D],
            [layer_index, 0, 0],
        )
        gate_w_layer: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.reshape(
            gate_w_ranked,
            [N_EXPERTS, D],
        )
        gate_bias_ranked = pl.slice(
            gate_bias,
            [1, N_EXPERTS],
            [layer_index, 0],
        )
        gate_bias_layer: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.reshape(
            gate_bias_ranked,
            [N_EXPERTS],
        )
        tid2eid_ranked = pl.slice(
            tid2eid,
            [1, VOCAB, TOPK],
            [hash_index, 0, 0],
        )
        tid2eid_layer: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.reshape(
            tid2eid_ranked,
            [VOCAB, TOPK],
        )
        routed_w1_ranked = pl.slice(
            routed_w1,
            [1, N_LOCAL_EXPERTS, MOE_INTER, D],
            [layer_index, 0, 0, 0],
        )
        routed_w1_layer: pl.Tensor[
            [N_LOCAL_EXPERTS, MOE_INTER, D],
            pl.INT8,
        ] = pl.reshape(
            routed_w1_ranked,
            [N_LOCAL_EXPERTS, MOE_INTER, D],
        )
        routed_w1_scale_ranked = pl.slice(
            routed_w1_scale,
            [1, N_LOCAL_EXPERTS, MOE_INTER],
            [layer_index, 0, 0],
        )
        routed_w1_scale_layer: pl.Tensor[
            [N_LOCAL_EXPERTS, MOE_INTER],
            pl.FP32,
        ] = pl.reshape(
            routed_w1_scale_ranked,
            [N_LOCAL_EXPERTS, MOE_INTER],
        )
        routed_w3_ranked = pl.slice(
            routed_w3,
            [1, N_LOCAL_EXPERTS, MOE_INTER, D],
            [layer_index, 0, 0, 0],
        )
        routed_w3_layer: pl.Tensor[
            [N_LOCAL_EXPERTS, MOE_INTER, D],
            pl.INT8,
        ] = pl.reshape(
            routed_w3_ranked,
            [N_LOCAL_EXPERTS, MOE_INTER, D],
        )
        routed_w3_scale_ranked = pl.slice(
            routed_w3_scale,
            [1, N_LOCAL_EXPERTS, MOE_INTER],
            [layer_index, 0, 0],
        )
        routed_w3_scale_layer: pl.Tensor[
            [N_LOCAL_EXPERTS, MOE_INTER],
            pl.FP32,
        ] = pl.reshape(
            routed_w3_scale_ranked,
            [N_LOCAL_EXPERTS, MOE_INTER],
        )
        routed_w2_ranked = pl.slice(
            routed_w2,
            [1, N_LOCAL_EXPERTS, D, MOE_INTER],
            [layer_index, 0, 0, 0],
        )
        routed_w2_layer: pl.Tensor[
            [N_LOCAL_EXPERTS, D, MOE_INTER],
            pl.INT8,
        ] = pl.reshape(
            routed_w2_ranked,
            [N_LOCAL_EXPERTS, D, MOE_INTER],
        )
        routed_w2_scale_ranked = pl.slice(
            routed_w2_scale,
            [1, N_LOCAL_EXPERTS, D],
            [layer_index, 0, 0],
        )
        routed_w2_scale_layer: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32] = pl.reshape(
            routed_w2_scale_ranked,
            [N_LOCAL_EXPERTS, D],
        )
        shared_w1_ranked = pl.slice(
            shared_w1,
            [1, MOE_INTER, D],
            [layer_index, 0, 0],
        )
        shared_w1_layer: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.reshape(
            shared_w1_ranked,
            [MOE_INTER, D],
        )
        shared_w1_scale_ranked = pl.slice(
            shared_w1_scale,
            [1, MOE_INTER],
            [layer_index, 0],
        )
        shared_w1_scale_layer: pl.Tensor[[MOE_INTER], pl.FP32] = pl.reshape(
            shared_w1_scale_ranked,
            [MOE_INTER],
        )
        shared_w3_ranked = pl.slice(
            shared_w3,
            [1, MOE_INTER, D],
            [layer_index, 0, 0],
        )
        shared_w3_layer: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.reshape(
            shared_w3_ranked,
            [MOE_INTER, D],
        )
        shared_w3_scale_ranked = pl.slice(
            shared_w3_scale,
            [1, MOE_INTER],
            [layer_index, 0],
        )
        shared_w3_scale_layer: pl.Tensor[[MOE_INTER], pl.FP32] = pl.reshape(
            shared_w3_scale_ranked,
            [MOE_INTER],
        )
        shared_w2_ranked = pl.slice(
            shared_w2,
            [1, D, MOE_INTER],
            [layer_index, 0, 0],
        )
        shared_w2_layer: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.reshape(
            shared_w2_ranked,
            [D, MOE_INTER],
        )
        shared_w2_scale_ranked = pl.slice(
            shared_w2_scale,
            [1, D],
            [layer_index, 0],
        )
        shared_w2_scale_layer: pl.Tensor[[D], pl.FP32] = pl.reshape(
            shared_w2_scale_ranked,
            [D],
        )
        layer_indices_ranked = pl.slice(
            layer_indices,
            [1, T, TOPK],
            [layer_index, 0, 0],
        )
        layer_indices_view: pl.Tensor[[T, TOPK], pl.INT32] = pl.reshape(
            layer_indices_ranked,
            [T, TOPK],
        )
        layer_weights_ranked = pl.slice(
            layer_weights,
            [1, T, TOPK],
            [layer_index, 0, 0],
        )
        layer_weights_view: pl.Tensor[[T, TOPK], pl.FP32] = pl.reshape(
            layer_weights_ranked,
            [T, TOPK],
        )
        if hca_index_i32 >= 0:
            hca_index = pl.cast(hca_index_i32, pl.INDEX)
            hca_cmp_wkv_ranked = pl.slice(
                hca_cmp_wkv,
                [1, HCA_MAIN_OUT_DIM, D],
                [hca_index, 0, 0],
            )
            hca_cmp_wkv_layer: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16] = pl.reshape(
                hca_cmp_wkv_ranked,
                [HCA_MAIN_OUT_DIM, D],
            )
            hca_cmp_wgate_ranked = pl.slice(
                hca_cmp_wgate,
                [1, HCA_MAIN_OUT_DIM, D],
                [hca_index, 0, 0],
            )
            hca_cmp_wgate_layer: pl.Tensor[
                [HCA_MAIN_OUT_DIM, D],
                pl.BF16,
            ] = pl.reshape(
                hca_cmp_wgate_ranked,
                [HCA_MAIN_OUT_DIM, D],
            )
            hca_cmp_ape_ranked = pl.slice(
                hca_cmp_ape,
                [1, HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM],
                [hca_index, 0, 0],
            )
            hca_cmp_ape_layer: pl.Tensor[
                [HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM],
                pl.FP32,
            ] = pl.reshape(
                hca_cmp_ape_ranked,
                [HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM],
            )
            hca_cmp_norm_w_ranked = pl.slice(
                hca_cmp_norm_w,
                [1, HEAD_DIM],
                [hca_index, 0],
            )
            hca_cmp_norm_w_layer: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.reshape(
                hca_cmp_norm_w_ranked,
                [HEAD_DIM],
            )
            hca_compress_state_ranked = pl.slice(
                hca_compress_state,
                [
                    1,
                    HCA_MAIN_STATE_BLOCK_NUM,
                    C128_COMPRESSOR_BLOCK_SIZE,
                    HCA_STATE_DIM,
                ],
                [hca_index, 0, 0, 0],
            )
            hca_compress_state_layer: pl.Tensor[
                [
                    HCA_MAIN_STATE_BLOCK_NUM,
                    C128_COMPRESSOR_BLOCK_SIZE,
                    HCA_STATE_DIM,
                ],
                pl.FP32,
            ] = pl.reshape(
                hca_compress_state_ranked,
                [
                    HCA_MAIN_STATE_BLOCK_NUM,
                    C128_COMPRESSOR_BLOCK_SIZE,
                    HCA_STATE_DIM,
                ],
            )
            hca_cmp_kv_ranked = pl.slice(
                hca_cmp_kv,
                [1, HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
                [hca_index, 0, 0, 0, 0],
            )
            hca_cmp_kv_layer: pl.Tensor[
                [HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
                pl.BF16,
            ] = pl.reshape(
                hca_cmp_kv_ranked,
                [HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            )
            hidden = decode_layer_compute_hca(
                hidden,
                hc_attn_fn_layer,
                hc_attn_scale_layer,
                hc_attn_base_layer,
                attn_norm_w_layer,
                wq_a_layer,
                wq_b_layer,
                wq_b_scale_layer,
                wkv_layer,
                gamma_cq_layer,
                gamma_ckv_layer,
                freqs_cos,
                freqs_sin,
                hca_cmp_wkv_layer,
                hca_cmp_wgate_layer,
                hca_cmp_ape_layer,
                hca_cmp_norm_w_layer,
                hca_compress_state_layer,
                hca_compress_state_block_table,
                kv_cache_layer,
                hca_cmp_kv_layer,
                hca_cmp_block_table,
                ori_slot_mapping,
                window_swa_indices,
                window_swa_lens,
                hca_cmp_slot_mapping,
                hca_state_slot_mapping,
                position_ids,
                kv_seq_lens,
                attn_sink_layer,
                wo_a_layer,
                wo_b_layer,
                wo_b_scale_layer,
                hc_ffn_fn_layer,
                hc_ffn_scale_layer,
                hc_ffn_base_layer,
                norm_w_layer,
                gate_w_layer,
                gate_bias_layer,
                tid2eid_layer,
                input_ids,
                recv_x,
                recv_scale_dq,
                recv_weights,
                recv_expert_count,
                routed_w1_layer,
                routed_w1_scale_layer,
                routed_w3_layer,
                routed_w3_scale_layer,
                routed_w2_layer,
                routed_w2_scale_layer,
                shared_w1_layer,
                shared_w1_scale_layer,
                shared_w3_layer,
                shared_w3_scale_layer,
                shared_w2_layer,
                shared_w2_scale_layer,
                route_to_recv,
                layer_indices_view,
                layer_weights_view,
                hidden_next,
                layer_id,
            )
        else:
            csa_index = pl.cast(csa_index_i32, pl.INDEX)
            csa_cmp_wkv_ranked = pl.slice(
                csa_cmp_wkv,
                [1, CSA_MAIN_OUT_DIM, D],
                [csa_index, 0, 0],
            )
            csa_cmp_wkv_layer: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.reshape(
                csa_cmp_wkv_ranked,
                [CSA_MAIN_OUT_DIM, D],
            )
            csa_cmp_wgate_ranked = pl.slice(
                csa_cmp_wgate,
                [1, CSA_MAIN_OUT_DIM, D],
                [csa_index, 0, 0],
            )
            csa_cmp_wgate_layer: pl.Tensor[
                [CSA_MAIN_OUT_DIM, D],
                pl.BF16,
            ] = pl.reshape(
                csa_cmp_wgate_ranked,
                [CSA_MAIN_OUT_DIM, D],
            )
            csa_cmp_ape_ranked = pl.slice(
                csa_cmp_ape,
                [1, CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM],
                [csa_index, 0, 0],
            )
            csa_cmp_ape_layer: pl.Tensor[
                [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM],
                pl.FP32,
            ] = pl.reshape(
                csa_cmp_ape_ranked,
                [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM],
            )
            csa_cmp_norm_w_ranked = pl.slice(
                csa_cmp_norm_w,
                [1, HEAD_DIM],
                [csa_index, 0],
            )
            csa_cmp_norm_w_layer: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.reshape(
                csa_cmp_norm_w_ranked,
                [HEAD_DIM],
            )
            csa_compress_state_ranked = pl.slice(
                csa_compress_state,
                [
                    1,
                    CSA_MAIN_STATE_BLOCK_NUM,
                    CSA_MAIN_STATE_BLOCK_SIZE,
                    CSA_MAIN_STATE_DIM,
                ],
                [csa_index, 0, 0, 0],
            )
            csa_compress_state_layer: pl.Tensor[
                [
                    CSA_MAIN_STATE_BLOCK_NUM,
                    CSA_MAIN_STATE_BLOCK_SIZE,
                    CSA_MAIN_STATE_DIM,
                ],
                pl.FP32,
            ] = pl.reshape(
                csa_compress_state_ranked,
                [
                    CSA_MAIN_STATE_BLOCK_NUM,
                    CSA_MAIN_STATE_BLOCK_SIZE,
                    CSA_MAIN_STATE_DIM,
                ],
            )
            csa_idx_wq_b_ranked = pl.slice(
                csa_idx_wq_b,
                [1, Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM],
                [csa_index, 0, 0],
            )
            csa_idx_wq_b_layer: pl.Tensor[
                [Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM],
                pl.INT8,
            ] = pl.reshape(
                csa_idx_wq_b_ranked,
                [Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM],
            )
            csa_idx_wq_b_scale_ranked = pl.slice(
                csa_idx_wq_b_scale,
                [1, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM],
                [csa_index, 0],
            )
            csa_idx_wq_b_scale_layer: pl.Tensor[
                [CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM],
                pl.FP32,
            ] = pl.reshape(
                csa_idx_wq_b_scale_ranked,
                [CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM],
            )
            csa_weights_proj_ranked = pl.slice(
                csa_weights_proj,
                [1, D, CSA_IDX_N_HEADS],
                [csa_index, 0, 0],
            )
            csa_weights_proj_layer: pl.Tensor[
                [D, CSA_IDX_N_HEADS],
                pl.BF16,
            ] = pl.reshape(
                csa_weights_proj_ranked,
                [D, CSA_IDX_N_HEADS],
            )
            csa_hadamard_idx_ranked = pl.slice(
                csa_hadamard_idx,
                [1, CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM],
                [csa_index, 0, 0],
            )
            csa_hadamard_idx_layer: pl.Tensor[
                [CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM],
                pl.BF16,
            ] = pl.reshape(
                csa_hadamard_idx_ranked,
                [CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM],
            )
            csa_inner_wkv_ranked = pl.slice(
                csa_inner_wkv,
                [1, CSA_INNER_OUT_DIM, D],
                [csa_index, 0, 0],
            )
            csa_inner_wkv_layer: pl.Tensor[
                [CSA_INNER_OUT_DIM, D],
                pl.BF16,
            ] = pl.reshape(
                csa_inner_wkv_ranked,
                [CSA_INNER_OUT_DIM, D],
            )
            csa_inner_wgate_ranked = pl.slice(
                csa_inner_wgate,
                [1, CSA_INNER_OUT_DIM, D],
                [csa_index, 0, 0],
            )
            csa_inner_wgate_layer: pl.Tensor[
                [CSA_INNER_OUT_DIM, D],
                pl.BF16,
            ] = pl.reshape(
                csa_inner_wgate_ranked,
                [CSA_INNER_OUT_DIM, D],
            )
            csa_inner_ape_ranked = pl.slice(
                csa_inner_ape,
                [1, CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM],
                [csa_index, 0, 0],
            )
            csa_inner_ape_layer: pl.Tensor[
                [CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM],
                pl.FP32,
            ] = pl.reshape(
                csa_inner_ape_ranked,
                [CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM],
            )
            csa_inner_norm_w_ranked = pl.slice(
                csa_inner_norm_w,
                [1, CSA_IDX_HEAD_DIM],
                [csa_index, 0],
            )
            csa_inner_norm_w_layer: pl.Tensor[
                [CSA_IDX_HEAD_DIM],
                pl.BF16,
            ] = pl.reshape(
                csa_inner_norm_w_ranked,
                [CSA_IDX_HEAD_DIM],
            )
            csa_inner_compress_state_ranked = pl.slice(
                csa_inner_compress_state,
                [
                    1,
                    CSA_INNER_STATE_BLOCK_NUM,
                    CSA_INNER_STATE_BLOCK_SIZE,
                    CSA_INNER_STATE_DIM,
                ],
                [csa_index, 0, 0, 0],
            )
            csa_inner_compress_state_layer: pl.Tensor[
                [
                    CSA_INNER_STATE_BLOCK_NUM,
                    CSA_INNER_STATE_BLOCK_SIZE,
                    CSA_INNER_STATE_DIM,
                ],
                pl.FP32,
            ] = pl.reshape(
                csa_inner_compress_state_ranked,
                [
                    CSA_INNER_STATE_BLOCK_NUM,
                    CSA_INNER_STATE_BLOCK_SIZE,
                    CSA_INNER_STATE_DIM,
                ],
            )
            csa_cmp_kv_ranked = pl.slice(
                csa_cmp_kv,
                [1, CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
                [csa_index, 0, 0, 0, 0],
            )
            csa_cmp_kv_layer: pl.Tensor[
                [CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
                pl.BF16,
            ] = pl.reshape(
                csa_cmp_kv_ranked,
                [CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            )
            csa_idx_kv_cache_ranked = pl.slice(
                csa_idx_kv_cache,
                [1, CSA_IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM],
                [csa_index, 0, 0, 0, 0],
            )
            csa_idx_kv_cache_layer: pl.Tensor[
                [CSA_IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM],
                pl.INT8,
            ] = pl.reshape(
                csa_idx_kv_cache_ranked,
                [CSA_IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM],
            )
            csa_idx_kv_scale_ranked = pl.slice(
                csa_idx_kv_scale,
                [1, CSA_IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1],
                [csa_index, 0, 0, 0, 0],
            )
            csa_idx_kv_scale_layer: pl.Tensor[
                [CSA_IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1],
                pl.FP32,
            ] = pl.reshape(
                csa_idx_kv_scale_ranked,
                [CSA_IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1],
            )
            hidden = decode_layer_compute_csa(
                hidden,
                hc_attn_fn_layer,
                hc_attn_scale_layer,
                hc_attn_base_layer,
                attn_norm_w_layer,
                wq_a_layer,
                wq_b_layer,
                wq_b_scale_layer,
                wkv_layer,
                gamma_cq_layer,
                gamma_ckv_layer,
                freqs_cos,
                freqs_sin,
                csa_cmp_wkv_layer,
                csa_cmp_wgate_layer,
                csa_cmp_ape_layer,
                csa_cmp_norm_w_layer,
                csa_compress_state_layer,
                csa_compress_state_block_table,
                csa_idx_wq_b_layer,
                csa_idx_wq_b_scale_layer,
                csa_weights_proj_layer,
                csa_hadamard_idx_layer,
                csa_inner_wkv_layer,
                csa_inner_wgate_layer,
                csa_inner_ape_layer,
                csa_inner_norm_w_layer,
                csa_inner_compress_state_layer,
                csa_inner_compress_state_block_table,
                kv_cache_layer,
                csa_cmp_kv_layer,
                csa_cmp_block_table,
                csa_idx_kv_cache_layer,
                csa_idx_kv_scale_layer,
                csa_idx_block_table,
                ori_slot_mapping,
                window_swa_indices,
                window_swa_lens,
                csa_cmp_slot_mapping,
                csa_idx_slot_mapping,
                csa_state_slot_mapping,
                csa_inner_state_slot_mapping,
                position_ids,
                kv_seq_lens,
                attn_sink_layer,
                wo_a_layer,
                wo_b_layer,
                wo_b_scale_layer,
                hc_ffn_fn_layer,
                hc_ffn_scale_layer,
                hc_ffn_base_layer,
                norm_w_layer,
                gate_w_layer,
                gate_bias_layer,
                tid2eid_layer,
                input_ids,
                recv_x,
                recv_scale_dq,
                recv_weights,
                recv_expert_count,
                routed_w1_layer,
                routed_w1_scale_layer,
                routed_w3_layer,
                routed_w3_scale_layer,
                routed_w2_layer,
                routed_w2_scale_layer,
                shared_w1_layer,
                shared_w1_scale_layer,
                shared_w3_layer,
                shared_w3_scale_layer,
                shared_w2_layer,
                shared_w2_scale_layer,
                route_to_recv,
                layer_indices_view,
                layer_weights_view,
                hidden_next,
                layer_id,
            )
    return x_out


@pl.jit.host
def l3_decode_fwd_compute(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[NUM_LAYERS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[NUM_LAYERS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[NUM_LAYERS, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[NUM_LAYERS, D], pl.BF16],
    wq_a: pl.Tensor[[NUM_LAYERS, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[NUM_LAYERS, H, HEAD_DIM, Q_LORA], pl.INT8],
    wq_b_scale: pl.Tensor[[NUM_LAYERS, H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[NUM_LAYERS, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[NUM_LAYERS, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[NUM_LAYERS, HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[
        pl.Tensor[[NUM_LAYERS, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    attn_sink: pl.Tensor[[NUM_LAYERS, H], pl.FP32],
    wo_a: pl.Tensor[[NUM_LAYERS, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[NUM_LAYERS, D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[NUM_LAYERS, D], pl.FP32],
    hca_cmp_wkv: pl.Tensor[[HCA_STORAGE_COUNT, HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_STORAGE_COUNT, HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[
        [HCA_STORAGE_COUNT, HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    hca_cmp_norm_w: pl.Tensor[[HCA_STORAGE_COUNT, HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[
        pl.Tensor[
            [
                HCA_STORAGE_COUNT,
                HCA_MAIN_STATE_BLOCK_NUM,
                C128_COMPRESSOR_BLOCK_SIZE,
                HCA_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    hca_compress_state_block_table: pl.Tensor[[B, HCA_STATE_MAX_BLOCKS], pl.INT32],
    hca_cmp_kv: pl.InOut[
        pl.Tensor[
            [HCA_STORAGE_COUNT, HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    hca_cmp_block_table: pl.Tensor[[B, HCA_CMP_MAX_BLOCKS], pl.INT32],
    hca_cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_cmp_wkv: pl.Tensor[[CSA_STORAGE_COUNT, CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_STORAGE_COUNT, CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[
        [CSA_STORAGE_COUNT, CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    csa_cmp_norm_w: pl.Tensor[[CSA_STORAGE_COUNT, HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[
        pl.Tensor[
            [
                CSA_STORAGE_COUNT,
                CSA_MAIN_STATE_BLOCK_NUM,
                CSA_MAIN_STATE_BLOCK_SIZE,
                CSA_MAIN_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    csa_compress_state_block_table: pl.Tensor[
        [B, CSA_MAIN_STATE_MAX_BLOCKS],
        pl.INT32,
    ],
    csa_idx_wq_b: pl.Tensor[
        [CSA_STORAGE_COUNT, Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM],
        pl.INT8,
    ],
    csa_idx_wq_b_scale: pl.Tensor[
        [CSA_STORAGE_COUNT, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM],
        pl.FP32,
    ],
    csa_weights_proj: pl.Tensor[[CSA_STORAGE_COUNT, D, CSA_IDX_N_HEADS], pl.BF16],
    csa_hadamard_idx: pl.Tensor[
        [CSA_STORAGE_COUNT, CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM],
        pl.BF16,
    ],
    csa_inner_wkv: pl.Tensor[[CSA_STORAGE_COUNT, CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[CSA_STORAGE_COUNT, CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[
        [CSA_STORAGE_COUNT, CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM],
        pl.FP32,
    ],
    csa_inner_norm_w: pl.Tensor[[CSA_STORAGE_COUNT, CSA_IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[
        pl.Tensor[
            [
                CSA_STORAGE_COUNT,
                CSA_INNER_STATE_BLOCK_NUM,
                CSA_INNER_STATE_BLOCK_SIZE,
                CSA_INNER_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    csa_inner_compress_state_block_table: pl.Tensor[
        [B, CSA_INNER_STATE_MAX_BLOCKS],
        pl.INT32,
    ],
    csa_cmp_kv: pl.InOut[
        pl.Tensor[
            [CSA_STORAGE_COUNT, CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    csa_cmp_block_table: pl.Tensor[[B, CSA_CMP_MAX_BLOCKS], pl.INT32],
    csa_idx_kv_cache: pl.InOut[
        pl.Tensor[
            [
                CSA_STORAGE_COUNT,
                CSA_IDX_CACHE_BLOCK_NUM,
                BLOCK_SIZE,
                1,
                CSA_IDX_HEAD_DIM,
            ],
            pl.INT8,
        ]
    ],
    csa_idx_kv_scale: pl.InOut[
        pl.Tensor[
            [CSA_STORAGE_COUNT, CSA_IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1],
            pl.FP32,
        ]
    ],
    csa_idx_block_table: pl.Tensor[[B, CSA_IDX_CACHE_MAX_BLOCKS], pl.INT32],
    csa_cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    window_swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    hc_ffn_fn: pl.Tensor[[NUM_LAYERS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[NUM_LAYERS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[NUM_LAYERS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[NUM_LAYERS, D], pl.BF16],
    gate_w: pl.Tensor[[NUM_LAYERS, N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[NUM_LAYERS, N_EXPERTS], pl.FP32],
    tid2eid: pl.Tensor[[HASH_LAYER_COUNT, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1: pl.Tensor[
        [NUM_LAYERS, N_LOCAL_EXPERTS, MOE_INTER, D],
        pl.INT8,
    ],
    routed_w1_scale: pl.Tensor[
        [NUM_LAYERS, N_LOCAL_EXPERTS, MOE_INTER],
        pl.FP32,
    ],
    routed_w3: pl.Tensor[
        [NUM_LAYERS, N_LOCAL_EXPERTS, MOE_INTER, D],
        pl.INT8,
    ],
    routed_w3_scale: pl.Tensor[
        [NUM_LAYERS, N_LOCAL_EXPERTS, MOE_INTER],
        pl.FP32,
    ],
    routed_w2: pl.Tensor[
        [NUM_LAYERS, N_LOCAL_EXPERTS, D, MOE_INTER],
        pl.INT8,
    ],
    routed_w2_scale: pl.Tensor[[NUM_LAYERS, N_LOCAL_EXPERTS, D], pl.FP32],
    shared_w1: pl.Tensor[[NUM_LAYERS, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[NUM_LAYERS, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[NUM_LAYERS, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[NUM_LAYERS, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[NUM_LAYERS, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[NUM_LAYERS, D], pl.FP32],
    route_to_recv: pl.Tensor[[T, TOPK], pl.INT32],
    layer_ids: pl.Tensor[[NUM_LAYERS], pl.INT32],
    hca_stack_indices: pl.Tensor[[NUM_LAYERS], pl.INT32],
    csa_stack_indices: pl.Tensor[[NUM_LAYERS], pl.INT32],
    hash_stack_indices: pl.Tensor[[NUM_LAYERS], pl.INT32],
    layer_indices: pl.Out[pl.Tensor[[NUM_LAYERS, T, TOPK], pl.INT32]],
    layer_weights: pl.Out[pl.Tensor[[NUM_LAYERS, T, TOPK], pl.FP32]],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
):
    decode_fwd_compute(
        x_hc,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_norm_w,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        gamma_cq,
        gamma_ckv,
        freqs_cos,
        freqs_sin,
        kv_cache,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        hca_cmp_wkv,
        hca_cmp_wgate,
        hca_cmp_ape,
        hca_cmp_norm_w,
        hca_compress_state,
        hca_compress_state_block_table,
        hca_cmp_kv,
        hca_cmp_block_table,
        hca_cmp_slot_mapping,
        hca_state_slot_mapping,
        csa_cmp_wkv,
        csa_cmp_wgate,
        csa_cmp_ape,
        csa_cmp_norm_w,
        csa_compress_state,
        csa_compress_state_block_table,
        csa_idx_wq_b,
        csa_idx_wq_b_scale,
        csa_weights_proj,
        csa_hadamard_idx,
        csa_inner_wkv,
        csa_inner_wgate,
        csa_inner_ape,
        csa_inner_norm_w,
        csa_inner_compress_state,
        csa_inner_compress_state_block_table,
        csa_cmp_kv,
        csa_cmp_block_table,
        csa_idx_kv_cache,
        csa_idx_kv_scale,
        csa_idx_block_table,
        csa_cmp_slot_mapping,
        csa_idx_slot_mapping,
        csa_state_slot_mapping,
        csa_inner_state_slot_mapping,
        ori_slot_mapping,
        window_swa_indices,
        window_swa_lens,
        position_ids,
        kv_seq_lens,
        hc_ffn_fn,
        hc_ffn_scale,
        hc_ffn_base,
        norm_w,
        gate_w,
        gate_bias,
        tid2eid,
        input_ids,
        recv_x,
        recv_scale_dq,
        recv_weights,
        recv_expert_count,
        routed_w1,
        routed_w1_scale,
        routed_w3,
        routed_w3_scale,
        routed_w2,
        routed_w2_scale,
        shared_w1,
        shared_w1_scale,
        shared_w3,
        shared_w3_scale,
        shared_w2,
        shared_w2_scale,
        route_to_recv,
        layer_ids,
        hca_stack_indices,
        csa_stack_indices,
        hash_stack_indices,
        layer_indices,
        layer_weights,
        x_out,
        device=0,
    )


def _clone_spec(spec, name=None, *, resident=None, is_output=False, init_value=None):
    """Clone one leaf spec without materializing its tensor."""

    return TensorSpec(
        name or spec.name,
        list(spec.shape),
        spec.dtype,
        init_value=spec.init_value if init_value is None else init_value,
        is_output=is_output,
        resident=resident,
    )


def _stack_layer_specs(name, specs, *, resident=0, is_output=False):
    """Materialize independent first-axis storage for one spec per selected layer."""

    import torch

    if not specs:
        raise ValueError(f"cannot build an empty layer stack for {name}")
    shape = list(specs[0].shape)
    dtype = specs[0].dtype
    if any(list(spec.shape) != shape or spec.dtype != dtype for spec in specs):
        raise ValueError(f"incompatible leaf specs in {name} layer stack")

    def init_value():
        return torch.stack([spec.create_tensor().reshape(shape) for spec in specs])

    return TensorSpec(
        name,
        [len(specs), *shape],
        dtype,
        init_value=init_value,
        is_output=is_output,
        resident=resident,
    )


def _placeholder_spec(name, shape, dtype, *, is_output=False):
    """Build one positive-shape inactive-kind ABI placeholder."""

    return TensorSpec(
        name,
        list(shape),
        dtype,
        init_value=0,
        is_output=is_output,
        resident=0,
    )


def _specs_by_name(specs):
    return {spec.name: spec for spec in specs if isinstance(spec, TensorSpec)}


def _active_layer_sources(start_pos, workload):
    hca_specs = None
    csa_specs = None
    if HCA_LAYER_COUNT:
        hca_specs = _specs_by_name(build_hca_tensor_specs(start_pos, workload))
    if CSA_LAYER_COUNT:
        csa_specs = _specs_by_name(build_csa_tensor_specs(start_pos, workload))

    sources = []
    for layer_id in LAYER_IDS:
        if layer_id in HCA_LAYER_IDS:
            sources.append(hca_specs)
        else:
            sources.append(csa_specs)
    return sources, hca_specs, csa_specs


def _hca_specs(hca_specs):
    import torch

    if hca_specs is not None:
        return {
            "hca_cmp_wkv": _stack_layer_specs(
                "hca_cmp_wkv",
                [hca_specs["cmp_wkv"]] * HCA_LAYER_COUNT,
            ),
            "hca_cmp_wgate": _stack_layer_specs(
                "hca_cmp_wgate",
                [hca_specs["cmp_wgate"]] * HCA_LAYER_COUNT,
            ),
            "hca_cmp_ape": _stack_layer_specs(
                "hca_cmp_ape",
                [hca_specs["cmp_ape"]] * HCA_LAYER_COUNT,
            ),
            "hca_cmp_norm_w": _stack_layer_specs(
                "hca_cmp_norm_w",
                [hca_specs["cmp_norm_w"]] * HCA_LAYER_COUNT,
            ),
            "hca_compress_state": _stack_layer_specs(
                "hca_compress_state",
                [hca_specs["compress_state"]] * HCA_LAYER_COUNT,
                is_output=True,
            ),
            "hca_compress_state_block_table": _clone_spec(
                hca_specs["compress_state_block_table"],
                "hca_compress_state_block_table",
            ),
            "hca_cmp_kv": _stack_layer_specs(
                "hca_cmp_kv",
                [hca_specs["cmp_kv"]] * HCA_LAYER_COUNT,
                is_output=True,
            ),
            "hca_cmp_block_table": _clone_spec(
                hca_specs["cmp_block_table"],
                "hca_cmp_block_table",
            ),
            "hca_cmp_slot_mapping": _clone_spec(
                hca_specs["cmp_slot_mapping"],
                "hca_cmp_slot_mapping",
            ),
            "hca_state_slot_mapping": _clone_spec(
                hca_specs["state_slot_mapping"],
                "hca_state_slot_mapping",
            ),
        }

    return {
        "hca_cmp_wkv": _placeholder_spec(
            "hca_cmp_wkv",
            [HCA_STORAGE_COUNT, HCA_MAIN_OUT_DIM, D],
            torch.bfloat16,
        ),
        "hca_cmp_wgate": _placeholder_spec(
            "hca_cmp_wgate",
            [HCA_STORAGE_COUNT, HCA_MAIN_OUT_DIM, D],
            torch.bfloat16,
        ),
        "hca_cmp_ape": _placeholder_spec(
            "hca_cmp_ape",
            [HCA_STORAGE_COUNT, HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM],
            torch.float32,
        ),
        "hca_cmp_norm_w": _placeholder_spec(
            "hca_cmp_norm_w",
            [HCA_STORAGE_COUNT, HEAD_DIM],
            torch.bfloat16,
        ),
        "hca_compress_state": _placeholder_spec(
            "hca_compress_state",
            [
                HCA_STORAGE_COUNT,
                HCA_MAIN_STATE_BLOCK_NUM,
                C128_COMPRESSOR_BLOCK_SIZE,
                HCA_STATE_DIM,
            ],
            torch.float32,
            is_output=True,
        ),
        "hca_compress_state_block_table": TensorSpec(
            "hca_compress_state_block_table",
            [B, HCA_STATE_MAX_BLOCKS],
            torch.int32,
            init_value=0,
        ),
        "hca_cmp_kv": _placeholder_spec(
            "hca_cmp_kv",
            [HCA_STORAGE_COUNT, HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            is_output=True,
        ),
        "hca_cmp_block_table": TensorSpec(
            "hca_cmp_block_table",
            [B, HCA_CMP_MAX_BLOCKS],
            torch.int32,
            init_value=0,
        ),
        "hca_cmp_slot_mapping": TensorSpec(
            "hca_cmp_slot_mapping",
            [T],
            torch.int64,
            init_value=-1,
        ),
        "hca_state_slot_mapping": TensorSpec(
            "hca_state_slot_mapping",
            [T],
            torch.int64,
            init_value=-1,
        ),
    }


def _csa_specs(csa_specs):
    import torch

    if csa_specs is not None:
        stack = lambda output_name, source_name, output=False: _stack_layer_specs(
            output_name,
            [csa_specs[source_name]] * CSA_LAYER_COUNT,
            is_output=output,
        )
        clone = lambda output_name, source_name: _clone_spec(
            csa_specs[source_name],
            output_name,
        )
        return {
            "csa_cmp_wkv": stack("csa_cmp_wkv", "cmp_wkv"),
            "csa_cmp_wgate": stack("csa_cmp_wgate", "cmp_wgate"),
            "csa_cmp_ape": stack("csa_cmp_ape", "cmp_ape"),
            "csa_cmp_norm_w": stack("csa_cmp_norm_w", "cmp_norm_w"),
            "csa_compress_state": stack("csa_compress_state", "compress_state", True),
            "csa_compress_state_block_table": clone(
                "csa_compress_state_block_table",
                "compress_state_block_table",
            ),
            "csa_idx_wq_b": stack("csa_idx_wq_b", "idx_wq_b"),
            "csa_idx_wq_b_scale": stack("csa_idx_wq_b_scale", "idx_wq_b_scale"),
            "csa_weights_proj": stack("csa_weights_proj", "weights_proj"),
            "csa_hadamard_idx": stack("csa_hadamard_idx", "hadamard_idx"),
            "csa_inner_wkv": stack("csa_inner_wkv", "inner_wkv"),
            "csa_inner_wgate": stack("csa_inner_wgate", "inner_wgate"),
            "csa_inner_ape": stack("csa_inner_ape", "inner_ape"),
            "csa_inner_norm_w": stack("csa_inner_norm_w", "inner_norm_w"),
            "csa_inner_compress_state": stack(
                "csa_inner_compress_state",
                "inner_compress_state",
                True,
            ),
            "csa_inner_compress_state_block_table": clone(
                "csa_inner_compress_state_block_table",
                "inner_compress_state_block_table",
            ),
            "csa_cmp_kv": stack("csa_cmp_kv", "cmp_kv", True),
            "csa_cmp_block_table": clone("csa_cmp_block_table", "cmp_block_table"),
            "csa_idx_kv_cache": stack("csa_idx_kv_cache", "idx_kv_cache", True),
            "csa_idx_kv_scale": stack("csa_idx_kv_scale", "idx_kv_scale", True),
            "csa_idx_block_table": clone("csa_idx_block_table", "idx_block_table"),
            "csa_cmp_slot_mapping": clone("csa_cmp_slot_mapping", "cmp_slot_mapping"),
            "csa_idx_slot_mapping": clone("csa_idx_slot_mapping", "idx_slot_mapping"),
            "csa_state_slot_mapping": clone("csa_state_slot_mapping", "state_slot_mapping"),
            "csa_inner_state_slot_mapping": clone(
                "csa_inner_state_slot_mapping",
                "inner_state_slot_mapping",
            ),
        }

    weight_shapes = {
        "csa_cmp_wkv": ([CSA_STORAGE_COUNT, CSA_MAIN_OUT_DIM, D], torch.bfloat16),
        "csa_cmp_wgate": ([CSA_STORAGE_COUNT, CSA_MAIN_OUT_DIM, D], torch.bfloat16),
        "csa_cmp_ape": (
            [CSA_STORAGE_COUNT, CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM],
            torch.float32,
        ),
        "csa_cmp_norm_w": ([CSA_STORAGE_COUNT, HEAD_DIM], torch.bfloat16),
        "csa_idx_wq_b": (
            [CSA_STORAGE_COUNT, Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM],
            torch.int8,
        ),
        "csa_idx_wq_b_scale": (
            [CSA_STORAGE_COUNT, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM],
            torch.float32,
        ),
        "csa_weights_proj": ([CSA_STORAGE_COUNT, D, CSA_IDX_N_HEADS], torch.bfloat16),
        "csa_hadamard_idx": (
            [CSA_STORAGE_COUNT, CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM],
            torch.bfloat16,
        ),
        "csa_inner_wkv": ([CSA_STORAGE_COUNT, CSA_INNER_OUT_DIM, D], torch.bfloat16),
        "csa_inner_wgate": ([CSA_STORAGE_COUNT, CSA_INNER_OUT_DIM, D], torch.bfloat16),
        "csa_inner_ape": (
            [CSA_STORAGE_COUNT, CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM],
            torch.float32,
        ),
        "csa_inner_norm_w": ([CSA_STORAGE_COUNT, CSA_IDX_HEAD_DIM], torch.bfloat16),
    }
    specs = {
        name: _placeholder_spec(name, shape, dtype)
        for name, (shape, dtype) in weight_shapes.items()
    }
    cache_shapes = {
        "csa_compress_state": (
            [
                CSA_STORAGE_COUNT,
                CSA_MAIN_STATE_BLOCK_NUM,
                CSA_MAIN_STATE_BLOCK_SIZE,
                CSA_MAIN_STATE_DIM,
            ],
            torch.float32,
        ),
        "csa_inner_compress_state": (
            [
                CSA_STORAGE_COUNT,
                CSA_INNER_STATE_BLOCK_NUM,
                CSA_INNER_STATE_BLOCK_SIZE,
                CSA_INNER_STATE_DIM,
            ],
            torch.float32,
        ),
        "csa_cmp_kv": (
            [CSA_STORAGE_COUNT, CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
        ),
        "csa_idx_kv_cache": (
            [CSA_STORAGE_COUNT, CSA_IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM],
            torch.int8,
        ),
        "csa_idx_kv_scale": (
            [CSA_STORAGE_COUNT, CSA_IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1],
            torch.float32,
        ),
    }
    specs.update(
        {
            name: _placeholder_spec(name, shape, dtype, is_output=True)
            for name, (shape, dtype) in cache_shapes.items()
        }
    )
    table_shapes = {
        "csa_compress_state_block_table": [B, CSA_MAIN_STATE_MAX_BLOCKS],
        "csa_inner_compress_state_block_table": [B, CSA_INNER_STATE_MAX_BLOCKS],
        "csa_cmp_block_table": [B, CSA_CMP_MAX_BLOCKS],
        "csa_idx_block_table": [B, CSA_IDX_CACHE_MAX_BLOCKS],
    }
    specs.update(
        {
            name: TensorSpec(name, shape, torch.int32, init_value=0)
            for name, shape in table_shapes.items()
        }
    )
    for name in (
        "csa_cmp_slot_mapping",
        "csa_idx_slot_mapping",
        "csa_state_slot_mapping",
        "csa_inner_state_slot_mapping",
    ):
        specs[name] = TensorSpec(name, [T], torch.int64, init_value=-1)
    return specs


def _apply_cache_policy(specs, cache_policy):
    """Validate the fixed-slot cache policy without changing tensor mappings."""

    if cache_policy not in CACHE_POLICIES:
        raise ValueError(f"unknown cache policy {cache_policy!r}")
    return specs


def build_tensor_specs(
    start_pos=DECODE_START_POS,
    workload="balanced",
    cache_policy="commit",
    layer_ids=LAYER_IDS,
):
    """Build the selected static-depth resident one-die ABI."""

    import torch

    if tuple(layer_ids) != LAYER_IDS:
        raise ValueError("forward specs must use the import-time manifest layer IDs")
    sources, hca_source, csa_source = _active_layer_sources(start_pos, workload)
    first = sources[0]
    hca_specs = _hca_specs(hca_source)
    csa_specs = _csa_specs(csa_source)

    specs = [_clone_spec(first["x_hc"])]
    for name in COMMON_ATTENTION_WEIGHT_NAMES[:10]:
        specs.append(
            _stack_layer_specs(
                name,
                [source[name] for source in sources],
            )
        )
    specs.extend(
        (
            _clone_spec(first["freqs_cos"], resident=0),
            _clone_spec(first["freqs_sin"], resident=0),
            _stack_layer_specs(
                "kv_cache",
                [source["kv_cache"] for source in sources],
                is_output=True,
            ),
        )
    )
    for name in COMMON_ATTENTION_WEIGHT_NAMES[10:]:
        specs.append(
            _stack_layer_specs(
                name,
                [source[name] for source in sources],
            )
        )

    hca_order = (
        "hca_cmp_wkv",
        "hca_cmp_wgate",
        "hca_cmp_ape",
        "hca_cmp_norm_w",
        "hca_compress_state",
        "hca_compress_state_block_table",
        "hca_cmp_kv",
        "hca_cmp_block_table",
        "hca_cmp_slot_mapping",
        "hca_state_slot_mapping",
    )
    specs.extend(hca_specs[name] for name in hca_order)

    csa_order = (
        "csa_cmp_wkv",
        "csa_cmp_wgate",
        "csa_cmp_ape",
        "csa_cmp_norm_w",
        "csa_compress_state",
        "csa_compress_state_block_table",
        "csa_idx_wq_b",
        "csa_idx_wq_b_scale",
        "csa_weights_proj",
        "csa_hadamard_idx",
        "csa_inner_wkv",
        "csa_inner_wgate",
        "csa_inner_ape",
        "csa_inner_norm_w",
        "csa_inner_compress_state",
        "csa_inner_compress_state_block_table",
        "csa_cmp_kv",
        "csa_cmp_block_table",
        "csa_idx_kv_cache",
        "csa_idx_kv_scale",
        "csa_idx_block_table",
        "csa_cmp_slot_mapping",
        "csa_idx_slot_mapping",
        "csa_state_slot_mapping",
        "csa_inner_state_slot_mapping",
    )
    specs.extend(csa_specs[name] for name in csa_order)

    metadata_names = (
        "ori_slot_mapping",
        "window_swa_indices",
        "window_swa_lens",
        "position_ids",
        "kv_seq_lens",
    )
    specs.extend(_clone_spec(first[name]) for name in metadata_names)

    for name in COMMON_FFN_WEIGHT_NAMES[:6]:
        specs.append(
            _stack_layer_specs(
                name,
                [source[name] for source in sources],
            )
        )

    hash_sources = [sources[LAYER_IDS.index(layer_id)]["tid2eid"] for layer_id in HASH_LAYER_IDS]
    specs.append(_stack_layer_specs("tid2eid", hash_sources))
    shared_input_names = (
        "input_ids",
        "recv_x",
        "recv_scale_dq",
        "recv_weights",
        "recv_expert_count",
    )
    specs.extend(_clone_spec(first[name]) for name in shared_input_names)

    for name in COMMON_FFN_WEIGHT_NAMES[6:]:
        specs.append(
            _stack_layer_specs(
                name,
                [source[name] for source in sources],
            )
        )
    specs.append(_clone_spec(first["route_to_recv"]))

    hca_positions = {layer_id: index for index, layer_id in enumerate(HCA_LAYER_IDS)}
    csa_positions = {layer_id: index for index, layer_id in enumerate(CSA_LAYER_IDS)}
    hash_positions = {layer_id: index for index, layer_id in enumerate(HASH_LAYER_IDS)}
    specs.extend(
        (
            TensorSpec(
                "layer_ids",
                [NUM_LAYERS],
                torch.int32,
                init_value=lambda: torch.tensor(LAYER_IDS, dtype=torch.int32),
            ),
            TensorSpec(
                "hca_stack_indices",
                [NUM_LAYERS],
                torch.int32,
                init_value=lambda: torch.tensor(
                    [hca_positions.get(layer_id, -1) for layer_id in LAYER_IDS],
                    dtype=torch.int32,
                ),
            ),
            TensorSpec(
                "csa_stack_indices",
                [NUM_LAYERS],
                torch.int32,
                init_value=lambda: torch.tensor(
                    [csa_positions.get(layer_id, -1) for layer_id in LAYER_IDS],
                    dtype=torch.int32,
                ),
            ),
            TensorSpec(
                "hash_stack_indices",
                [NUM_LAYERS],
                torch.int32,
                init_value=lambda: torch.tensor(
                    [hash_positions.get(layer_id, 0) for layer_id in LAYER_IDS],
                    dtype=torch.int32,
                ),
            ),
            TensorSpec(
                "layer_indices",
                [NUM_LAYERS, T, TOPK],
                torch.int32,
                is_output=True,
            ),
            TensorSpec(
                "layer_weights",
                [NUM_LAYERS, T, TOPK],
                torch.float32,
                is_output=True,
            ),
            TensorSpec(
                "x_out",
                [T, HC_MULT, D],
                torch.float32,
                is_output=True,
            ),
        )
    )

    for spec in specs:
        if spec.name in RESIDENT_WEIGHT_NAMES and spec.resident != 0:
            raise ValueError(f"weight {spec.name} is not resident on worker 0")
        if spec.name in RESIDENT_CACHE_NAMES:
            if spec.resident != 0 or not spec.is_output:
                raise ValueError(f"cache {spec.name} is not resident read-write state")
    return _apply_cache_policy(specs, cache_policy)


def _layer_golden_tensors(tensors, layer_index, kind_index, hash_index, x_hc, x_next):
    layer = {
        "x_hc": x_hc,
        "freqs_cos": tensors["freqs_cos"],
        "freqs_sin": tensors["freqs_sin"],
        "kv_cache": tensors["kv_cache"][layer_index],
        "ori_slot_mapping": tensors["ori_slot_mapping"],
        "window_swa_indices": tensors["window_swa_indices"],
        "window_swa_lens": tensors["window_swa_lens"],
        "position_ids": tensors["position_ids"],
        "kv_seq_lens": tensors["kv_seq_lens"],
        "tid2eid": tensors["tid2eid"][hash_index],
        "input_ids": tensors["input_ids"],
        "recv_x": tensors["recv_x"],
        "recv_scale_dq": tensors["recv_scale_dq"],
        "recv_weights": tensors["recv_weights"],
        "recv_expert_count": tensors["recv_expert_count"],
        "route_to_recv": tensors["route_to_recv"],
        "indices": tensors["layer_indices"][layer_index],
        "weights": tensors["layer_weights"][layer_index],
        "x_next": x_next,
        "layer_id": int(LAYER_IDS[layer_index]),
    }
    for name in (*COMMON_ATTENTION_WEIGHT_NAMES, *COMMON_FFN_WEIGHT_NAMES):
        layer[name] = tensors[name][layer_index]

    if LAYER_IDS[layer_index] in HCA_LAYER_IDS:
        layer.update(
            {
                "cmp_wkv": tensors["hca_cmp_wkv"][kind_index],
                "cmp_wgate": tensors["hca_cmp_wgate"][kind_index],
                "cmp_ape": tensors["hca_cmp_ape"][kind_index],
                "cmp_norm_w": tensors["hca_cmp_norm_w"][kind_index],
                "compress_state": tensors["hca_compress_state"][kind_index],
                "compress_state_block_table": tensors["hca_compress_state_block_table"],
                "cmp_kv": tensors["hca_cmp_kv"][kind_index],
                "cmp_block_table": tensors["hca_cmp_block_table"],
                "cmp_slot_mapping": tensors["hca_cmp_slot_mapping"],
                "state_slot_mapping": tensors["hca_state_slot_mapping"],
            }
        )
    else:
        layer.update(
            {
                "cmp_wkv": tensors["csa_cmp_wkv"][kind_index],
                "cmp_wgate": tensors["csa_cmp_wgate"][kind_index],
                "cmp_ape": tensors["csa_cmp_ape"][kind_index],
                "cmp_norm_w": tensors["csa_cmp_norm_w"][kind_index],
                "compress_state": tensors["csa_compress_state"][kind_index],
                "compress_state_block_table": tensors["csa_compress_state_block_table"],
                "idx_wq_b": tensors["csa_idx_wq_b"][kind_index],
                "idx_wq_b_scale": tensors["csa_idx_wq_b_scale"][kind_index],
                "weights_proj": tensors["csa_weights_proj"][kind_index],
                "hadamard_idx": tensors["csa_hadamard_idx"][kind_index],
                "inner_wkv": tensors["csa_inner_wkv"][kind_index],
                "inner_wgate": tensors["csa_inner_wgate"][kind_index],
                "inner_ape": tensors["csa_inner_ape"][kind_index],
                "inner_norm_w": tensors["csa_inner_norm_w"][kind_index],
                "inner_compress_state": tensors["csa_inner_compress_state"][kind_index],
                "inner_compress_state_block_table": tensors[
                    "csa_inner_compress_state_block_table"
                ],
                "cmp_kv": tensors["csa_cmp_kv"][kind_index],
                "cmp_block_table": tensors["csa_cmp_block_table"],
                "idx_kv_cache": tensors["csa_idx_kv_cache"][kind_index],
                "idx_kv_scale": tensors["csa_idx_kv_scale"][kind_index],
                "idx_block_table": tensors["csa_idx_block_table"],
                "cmp_slot_mapping": tensors["csa_cmp_slot_mapping"],
                "idx_slot_mapping": tensors["csa_idx_slot_mapping"],
                "state_slot_mapping": tensors["csa_state_slot_mapping"],
                "inner_state_slot_mapping": tensors["csa_inner_state_slot_mapping"],
            }
        )
    return layer


def golden_decode_fwd_compute(tensors):
    """Run bounded selected layers sequentially in Torch."""

    import torch

    hidden = tensors["x_hc"]
    hca_index = 0
    csa_index = 0
    hash_positions = {layer_id: index for index, layer_id in enumerate(HASH_LAYER_IDS)}
    for layer_index, layer_id in enumerate(LAYER_IDS):
        if layer_index == NUM_LAYERS - 1:
            hidden_next = tensors["x_out"]
        else:
            hidden_next = torch.zeros(T, HC_MULT, D, dtype=torch.float32)
        hash_index = hash_positions.get(layer_id, 0)
        if layer_id in HCA_LAYER_IDS:
            layer = _layer_golden_tensors(
                tensors,
                layer_index,
                hca_index,
                hash_index,
                hidden,
                hidden_next,
            )
            golden_decode_layer_compute_hca(layer)
            hca_index += 1
        else:
            layer = _layer_golden_tensors(
                tensors,
                layer_index,
                csa_index,
                hash_index,
                hidden,
                hidden_next,
            )
            golden_decode_layer_compute_csa(layer)
            csa_index += 1
        hidden = hidden_next


def _resident_spec_bytes(specs):
    from math import prod

    return sum(
        prod(spec.shape) * spec.dtype.itemsize
        for spec in specs
        if isinstance(spec, TensorSpec) and spec.is_resident
    )


def _retained_nonresident_temp_bytes(specs):
    """Return the runner's 1024-byte-aligned retained ABI staging size."""

    from math import prod

    alignment = RUNTIME_RETAINED_TEMP_ALIGNMENT_BYTES
    return sum(
        (prod(spec.shape) * spec.dtype.itemsize + alignment - 1) // alignment * alignment
        for spec in specs
        if isinstance(spec, TensorSpec) and not spec.is_resident
    )


def _preflight_specs(specs):
    """Validate the specs-aware known allocation against one target die."""

    known_allocation_bytes = (
        STATIC_MANIFEST.accounted_bytes
        + _known_runtime_resource_bytes()
        + _retained_nonresident_temp_bytes(specs)
    )
    if known_allocation_bytes >= 64 * GIB:
        raise ValueError(
            "the static main-model allocation plus known runtime resources and retained "
            "ABI staging exceeds one 64-GiB die"
        )


def _placeholder_spec_bytes(specs):
    from math import prod

    prefixes = []
    if HCA_LAYER_COUNT == 0:
        prefixes.append("hca_")
    if CSA_LAYER_COUNT == 0:
        prefixes.append("csa_")
    return sum(
        prod(spec.shape) * spec.dtype.itemsize
        for spec in specs
        if isinstance(spec, TensorSpec)
        and spec.is_resident
        and any(spec.name.startswith(prefix) for prefix in prefixes)
    )


def _print_runtime_manifest(specs):
    ring_heap_values = _effective_ring_heap_values()
    task_windows = _effective_ring_task_windows()
    dep_pools = _effective_ring_dep_pools()
    runtime_shared_memory_bytes = _runtime_shared_memory_bytes(task_windows)
    runtime_private_arena_bytes = _runtime_private_arena_bytes(task_windows, dep_pools)
    retained_nonresident_temp_bytes = _retained_nonresident_temp_bytes(specs)
    known_runtime_bytes = (
        sum(ring_heap_values)
        + runtime_shared_memory_bytes
        + runtime_private_arena_bytes
    )

    _print_manifest()
    print(f"active_hca_layers={HCA_LAYER_COUNT}")
    print(f"active_csa_layers={CSA_LAYER_COUNT}")
    print(f"hca_abi_storage={HCA_STORAGE_COUNT}")
    print(f"csa_abi_storage={CSA_STORAGE_COUNT}")
    print(f"resident_spec_bytes={_resident_spec_bytes(specs)}")
    print(f"inactive_placeholder_bytes={_placeholder_spec_bytes(specs)}")
    print(f"runtime_ring_heap_config={os.environ['PTO2_RING_HEAP']}")
    print(f"runtime_ring_heap_per_ring_bytes={ring_heap_values}")
    print(f"runtime_ring_heap_aggregate_bytes={sum(ring_heap_values)}")
    print(f"runtime_ring_task_window_config={os.environ['PTO2_RING_TASK_WINDOW']}")
    print(f"runtime_ring_task_window_per_ring_entries={task_windows}")
    print(f"runtime_ring_task_window_aggregate_entries={sum(task_windows)}")
    print(f"runtime_ring_dep_pool_config={os.environ['PTO2_RING_DEP_POOL']}")
    print(f"runtime_ring_dep_pool_per_ring_entries={dep_pools}")
    print(f"runtime_ring_dep_pool_aggregate_entries={sum(dep_pools)}")
    print(f"runtime_ring_shared_memory_bytes={runtime_shared_memory_bytes}")
    print(f"runtime_private_arena_bytes={runtime_private_arena_bytes}")
    print(f"runtime_private_arena_default_exact_bytes={PINNED_RUNTIME_DEFAULT_ARENA_EXACT_BYTES}")
    print("runtime_private_arena_accounting=pinned-runtime-derived-upper-bound")
    print(f"retained_nonresident_temp_bytes={retained_nonresident_temp_bytes}")
    print(f"runtime_known_allocation_bytes={known_runtime_bytes}")
    print(
        "static_accounted_plus_known_runtime_bytes="
        f"{STATIC_MANIFEST.accounted_bytes + known_runtime_bytes}"
    )
    print(
        "static_accounted_plus_known_runtime_and_temp_bytes="
        f"{STATIC_MANIFEST.accounted_bytes + known_runtime_bytes + retained_nonresident_temp_bytes}"
    )
    print("runtime_private_arena_in_known_allocation_bytes=True")
    print("runtime_unexposed_overhead_in_known_allocation_bytes=False")
    print("runtime_ring_resources_in_resident_spec_bytes=False")


def _unordered_route_pair_compare(index_name, weight_name, weight_compare):
    """Compare each top-k row as expert-ID and weight pairs."""

    def compare_route_pairs(
        actual,
        expected,
        *,
        actual_outputs,
        expected_outputs,
        inputs,
        rtol,
        atol,
    ):
        required_names = (index_name, weight_name)
        missing_actual = tuple(name for name in required_names if name not in actual_outputs)
        missing_expected = tuple(name for name in required_names if name not in expected_outputs)
        if missing_actual or missing_expected:
            return False, (
                f"    route-pair outputs missing: actual={missing_actual}, "
                f"expected={missing_expected}"
            )

        actual_indices = actual_outputs[index_name].cpu()
        expected_indices = expected_outputs[index_name].cpu()
        actual_weights = actual_outputs[weight_name].cpu()
        expected_weights = expected_outputs[weight_name].cpu()
        shapes = {
            tuple(actual_indices.shape),
            tuple(expected_indices.shape),
            tuple(actual_weights.shape),
            tuple(expected_weights.shape),
        }
        if len(shapes) != 1:
            return False, (
                "    route-pair shape mismatch: "
                f"actual_indices={tuple(actual_indices.shape)}, "
                f"expected_indices={tuple(expected_indices.shape)}, "
                f"actual_weights={tuple(actual_weights.shape)}, "
                f"expected_weights={tuple(expected_weights.shape)}"
            )

        actual_indices_sorted, actual_order = actual_indices.sort(dim=-1)
        expected_indices_sorted, expected_order = expected_indices.sort(dim=-1)
        if not actual_indices_sorted.equal(expected_indices_sorted):
            mismatch_count = int((actual_indices_sorted != expected_indices_sorted).sum().item())
            return False, (
                "    route expert-ID multisets differ after row-wise canonicalization: "
                f"mismatches={mismatch_count}/{actual_indices_sorted.numel()}"
            )

        actual_weights_sorted = actual_weights.gather(-1, actual_order)
        expected_weights_sorted = expected_weights.gather(-1, expected_order)
        passed, detail = weight_compare(
            actual_weights_sorted,
            expected_weights_sorted,
            actual_outputs=actual_outputs,
            expected_outputs=expected_outputs,
            inputs=inputs,
            rtol=rtol,
            atol=atol,
        )
        if not passed:
            return False, "    route weights differ after expert-ID canonicalization:\n" + detail
        return True, ""

    compare_route_pairs.__name__ = "unordered_route_pair_compare"
    return compare_route_pairs


def _state_slice_comparators(
    layer_ids,
    storage_count,
    strict_compare,
    accumulated_compare,
):
    """Select state comparators from each active layer's global schedule position."""

    if storage_count != max(1, len(layer_ids)):
        raise ValueError("state storage count does not match the active layer count")
    if not layer_ids:
        return (strict_compare,) * storage_count

    return tuple(
        strict_compare if LAYER_IDS.index(layer_id) == 0 else accumulated_compare
        for layer_id in layer_ids
    )


def _per_first_axis_slice(compares):
    """Apply the matching comparator to every first-axis state slice."""

    compares = tuple(compares)
    if not compares:
        raise ValueError("stacked state comparator requires at least one comparator")

    def compare_slices(actual, expected, **kwargs):
        if actual.shape != expected.shape:
            return False, f"    state shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}"
        if actual.ndim == 0:
            return False, "    stacked state comparator requires at least one axis"
        if actual.shape[0] != len(compares):
            return False, (
                f"    state comparator count mismatch: {len(compares)} comparators "
                f"for {actual.shape[0]} slices"
            )

        failures = []
        for slice_index, compare in enumerate(compares):
            passed, detail = compare(
                actual[slice_index],
                expected[slice_index],
                **kwargs,
            )
            if not passed:
                failures.append(
                    f"    slice[{slice_index}] ({compare.__name__}) failed:\n{detail}"
                )
        return not failures, "\n".join(failures)

    comparator_names = ",".join(compare.__name__ for compare in compares)
    compare_slices.__name__ = f"per_first_axis_slice({comparator_names})"
    return compare_slices


def _benchmark_result_error(stats, expected_ranks=1, expected_dispatches=1):
    """Return why a benchmark result is unusable, or ``None`` when complete."""

    import math

    if stats is None:
        return "runtime returned no benchmark statistics"
    if getattr(stats, "fallback_flattened", False):
        return "runtime flattened the round grid"
    if getattr(stats, "unstable_dispatch_slots", False):
        return "dispatch slots changed across measured rounds"

    rounds = getattr(stats, "rounds", None)
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds <= 0:
        return f"invalid measured-round count {rounds!r}"

    device_wall_us = getattr(stats, "device_wall_us", ())
    host_wall_us = getattr(stats, "host_wall_us", ())
    rounds_dispatches = getattr(stats, "rounds_dispatches", ())
    for metric_name, samples in (
        ("device wall", device_wall_us),
        ("host wall", host_wall_us),
        ("round grid", rounds_dispatches),
    ):
        if len(samples) != rounds:
            return f"{metric_name} has {len(samples)} samples for {rounds} rounds"

    expected_rank_ids = None
    invocation_count = 0
    for round_index, ranks in enumerate(rounds_dispatches):
        rank_ids = set(ranks)
        if len(rank_ids) != expected_ranks:
            return (
                f"round {round_index} has {len(rank_ids)} ranks; "
                f"expected {expected_ranks}"
            )
        if expected_rank_ids is None:
            expected_rank_ids = rank_ids
        elif rank_ids != expected_rank_ids:
            return f"round {round_index} changed the participating rank set"
        for rank_id, dispatches in ranks.items():
            if len(dispatches) != expected_dispatches:
                return (
                    f"round {round_index} rank {rank_id} has {len(dispatches)} dispatches; "
                    f"expected {expected_dispatches}"
                )
            invocation_count += len(dispatches)

    invocations = getattr(stats, "invocations", ())
    if len(invocations) != invocation_count:
        return (
            f"flat invocation list has {len(invocations)} entries; "
            f"round grid has {invocation_count}"
        )

    try:
        effective_us = stats.per_round("effective")
    except (AttributeError, TypeError, ValueError) as error:
        return f"effective timing is unavailable: {error}"
    if len(effective_us) != rounds:
        return f"effective timing has {len(effective_us)} samples for {rounds} rounds"

    for metric_name, samples in (
        ("device wall", device_wall_us),
        ("host wall", host_wall_us),
        ("effective", effective_us),
    ):
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, (int, float)) or not math.isfinite(sample) or sample <= 0:
                return f"{metric_name} sample {sample_index} is invalid: {sample!r}"
    return None


if __name__ == "__main__":
    from golden import ratio_allclose, ratio_reldiff, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, default="hca1")
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="a2a3",
        choices=("a2a3", "a2a3sim"),
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=DECODE_START_POS)
    parser.add_argument(
        "--workload",
        choices=tuple(ROUTED_WORKLOAD_COUNTS),
        default="balanced",
    )
    parser.add_argument(
        "--cache-policy",
        choices=CACHE_POLICIES,
        default="commit",
        help=(
            "commit is one-shot correctness/smoke; overwrite is a fixed-slot, "
            "state-mutating benchmark repeat"
        ),
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument(
        "--enable-l2-swimlane",
        type=int,
        nargs="?",
        const=4,
        default=0,
        choices=(0, 1, 2, 4),
    )
    parser.add_argument("--enable-scope-stats", action="store_true", default=False)
    parser.add_argument("--enable-dep-gen", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--print-manifest", action="store_true", default=False)
    args = parser.parse_args()

    if args.case != STATIC_CASE:
        parser.error("--case is an import-time static specialization and cannot change after import")
    benchmark_value = os.environ.get("PYPTO_BENCH", "").strip()
    benchmark_enabled = benchmark_value not in ("", "0", "false", "False")
    if benchmark_enabled and args.compile_only:
        parser.error("PYPTO_BENCH cannot be combined with --compile-only")
    if benchmark_enabled and args.cache_policy != "overwrite":
        parser.error("PYPTO_BENCH requires --cache-policy overwrite")
    if not benchmark_enabled and args.cache_policy == "overwrite":
        parser.error("--cache-policy overwrite requires PYPTO_BENCH")

    bounded_golden = STATIC_CASE in GOLDEN_CASES and args.cache_policy == "commit"
    if not bounded_golden and (args.golden_data is not None or args.save_data):
        parser.error("--golden-data/--save-data require a bounded commit-mode golden case")

    specs = build_tensor_specs(
        start_pos=args.start_pos,
        workload=args.workload,
        cache_policy=args.cache_policy,
        layer_ids=LAYER_IDS,
    )
    _preflight_specs(specs)
    if args.print_manifest:
        _print_runtime_manifest(specs)
        raise SystemExit(0)

    if bounded_golden:
        golden_fn = golden_decode_fwd_compute
        golden_data = args.golden_data
        print(
            f"[PHASE6] {STATIC_CASE}: bounded commit-mode golden for the main-model compute proxy; "
            "not EP128 end-to-end",
            flush=True,
        )
    else:
        golden_fn = None
        golden_data = None
        if args.cache_policy == "overwrite":
            print(
                f"[PHASE6] {STATIC_CASE}: fixed-slot state-mutating hot repeat with full cache "
                "writes; state is not reset between rounds, validation skipped, no golden, "
                "main-model compute proxy only",
                flush=True,
            )
        else:
            print(
                f"[PHASE6] {STATIC_CASE}: compile/smoke only; validation skipped, no golden, "
                "main-model compute proxy only and not EP128 end-to-end",
                flush=True,
            )

    strict_state_compare = ratio_allclose(atol=1e-3, rtol=1e-3)
    hca_accumulated_state_compare = ratio_reldiff(diff_thd=0.02, pct_thd=0.005)
    csa_accumulated_state_compare = ratio_reldiff(diff_thd=0.02, pct_thd=0.02)
    hca_state_compare = _per_first_axis_slice(
        _state_slice_comparators(
            HCA_LAYER_IDS,
            HCA_STORAGE_COUNT,
            strict_state_compare,
            hca_accumulated_state_compare,
        )
    )
    csa_state_compare = _per_first_axis_slice(
        _state_slice_comparators(
            CSA_LAYER_IDS,
            CSA_STORAGE_COUNT,
            strict_state_compare,
            csa_accumulated_state_compare,
        )
    )
    route_pair_compare = _unordered_route_pair_compare(
        "layer_indices",
        "layer_weights",
        ratio_allclose(atol=2.5e-4, rtol=5e-3),
    )

    compare_fn = {
        "x_out": ratio_reldiff(diff_thd=0.02, pct_thd=0.08),
        "layer_indices": route_pair_compare,
        "layer_weights": route_pair_compare,
        "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        "hca_cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        "hca_compress_state": hca_state_compare,
        "csa_cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        "csa_compress_state": csa_state_compare,
        "csa_inner_compress_state": csa_state_compare,
        "csa_idx_kv_cache": ratio_allclose(atol=1, rtol=0),
        "csa_idx_kv_scale": ratio_allclose(atol=1e-4, rtol=1e-3),
    }
    distributed_config = DistributedConfig(
        device_ids=[args.device],
        num_sub_workers=0,
    )
    result = run_jit(
        fn=l3_decode_fwd_compute,
        specs=specs,
        golden_fn=golden_fn,
        compile_only=args.compile_only,
        save_data=args.save_data,
        golden_data=golden_data,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=distributed_config,
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_l2_swimlane=args.enable_l2_swimlane,
            enable_scope_stats=args.enable_scope_stats,
            enable_dep_gen=args.enable_dep_gen,
        ),
        rtol=1e-2,
        atol=1e-2,
        compare_fn=compare_fn,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
    if benchmark_enabled:
        benchmark_error = _benchmark_result_error(result.bench)
        if benchmark_error is not None:
            print(f"[PHASE6] benchmark rejected: {benchmark_error}", flush=True)
            raise SystemExit(1)
