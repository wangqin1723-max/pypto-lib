# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Static HBM accounting for the DeepSeek-V4 Pro W8A8 main compute proxy."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Iterable, Literal

from config import (
    ACTIVE,
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    C128_COMPRESSOR_BLOCK_SIZE,
    CSA_COMPRESS_RATIO,
    CSA_INNER_STATE_PHYSICAL_BLOCKS,
    CSA_INNER_STATE_TABLE_MAX_BLOCKS,
    CSA_STATE_PHYSICAL_BLOCKS,
    CSA_STATE_TABLE_MAX_BLOCKS,
    DECODE_BATCH,
    HCA_COMPRESS_RATIO,
    HCA_STATE_PHYSICAL_BLOCKS,
    HCA_STATE_TABLE_MAX_BLOCKS,
    IDX_KV_BLOCK_NUM,
    KV_CMP_MAX_BLOCKS,
    KV_HCA_MAX_BLOCKS,
    KV_ORI_TABLE_MAX_BLOCKS,
    MAIN_COMPRESS_RATIOS,
    MOE_DEPLOYMENT_RECV_MAX,
    MOE_LOCAL_EXPERTS,
    ORI_KV_BLOCK_NUM,
)


DTypeName = Literal["bf16", "fp32", "int8", "int32"]
AllocationKind = Literal["shared_asset", "shared_metadata", "layer_weight", "layer_cache"]
AttentionKind = Literal["hca", "csa"]

_DTYPE_BYTES: tuple[tuple[DTypeName, int], ...] = (
    ("bf16", 2),
    ("fp32", 4),
    ("int8", 1),
    ("int32", 4),
)

MIB = 1024**2
GIB = 1024**3

HIDDEN_SIZE = ACTIVE.hidden_size
ATTENTION_HEADS = ACTIVE.num_attention_heads
HEAD_DIM = ACTIVE.head_dim
ROPE_HEAD_DIM = ACTIVE.qk_rope_head_dim
Q_LORA_RANK = ACTIVE.q_lora_rank
O_LORA_RANK = ACTIVE.o_lora_rank
O_GROUPS = ACTIVE.o_groups
O_GROUP_IN = ATTENTION_HEADS * HEAD_DIM // O_GROUPS
HC_MULT = ACTIVE.hc_mult
HC_DIM = ACTIVE.hc_dim
MIX_HC = ACTIVE.mix_hc
MOE_INTERMEDIATE_SIZE = ACTIVE.moe_intermediate_size
GLOBAL_EXPERTS = ACTIVE.n_routed_experts
ROUTES_PER_TOKEN = ACTIVE.num_experts_per_tok
VOCAB_SIZE = ACTIVE.vocab_size
MAX_SEQUENCE_LENGTH = ACTIVE.max_position_embeddings
INDEX_HEADS = ACTIVE.index_n_heads
INDEX_HEAD_DIM = ACTIVE.index_head_dim

HCA_COMPRESSED_BLOCKS = DECODE_BATCH * KV_HCA_MAX_BLOCKS
CSA_COMPRESSED_BLOCKS = DECODE_BATCH * KV_CMP_MAX_BLOCKS


def dtype_nbytes(dtype: DTypeName) -> int:
    """Return the storage width of one manifest dtype element."""

    for name, width in _DTYPE_BYTES:
        if name == dtype:
            return width
    raise ValueError(f"unsupported manifest dtype {dtype!r}")


@dataclass(frozen=True)
class ByteRange:
    """Half-open byte range inside the manifest's logical resident arena."""

    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    def overlaps(self, other: ByteRange) -> bool:
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True)
class TensorAllocation:
    """Shape-only allocation; constructing one never materializes a tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: DTypeName
    kind: AllocationKind
    byte_offset: int
    layer_id: int | None = None

    @property
    def nbytes(self) -> int:
        return prod(self.shape) * dtype_nbytes(self.dtype)

    @property
    def byte_range(self) -> ByteRange:
        return ByteRange(self.byte_offset, self.byte_offset + self.nbytes)


@dataclass(frozen=True)
class CachePool:
    """One physical cache tensor owned by exactly one main-model layer."""

    family: str
    layer_id: int
    attention_kind: AttentionKind
    physical_blocks: int
    logical_table_blocks: int
    block_table_name: str
    allocation: TensorAllocation

    def block_ids_are_local(self, block_ids: Iterable[int]) -> bool:
        """Check that block-table entries address only this layer-local pool."""

        return all(type(block_id) is int and 0 <= block_id < self.physical_blocks for block_id in block_ids)


@dataclass(frozen=True)
class HBMManifest:
    """Static resident allocation manifest for an explicit set of main layers."""

    layer_ids: tuple[int, ...]
    hca_layer_ids: tuple[int, ...]
    csa_layer_ids: tuple[int, ...]
    allocations: tuple[TensorAllocation, ...]
    cache_pools: tuple[CachePool, ...]

    def _bytes_for(self, *kinds: AllocationKind) -> int:
        return sum(allocation.nbytes for allocation in self.allocations if allocation.kind in kinds)

    @property
    def weight_bytes(self) -> int:
        return self._bytes_for("layer_weight")

    @property
    def cache_bytes(self) -> int:
        return self._bytes_for("layer_cache")

    @property
    def shared_asset_bytes(self) -> int:
        return self._bytes_for("shared_asset")

    @property
    def shared_metadata_bytes(self) -> int:
        return self._bytes_for("shared_metadata")

    @property
    def shared_bytes(self) -> int:
        return self.shared_asset_bytes + self.shared_metadata_bytes

    @property
    def layer_bytes(self) -> int:
        return self.weight_bytes + self.cache_bytes

    @property
    def model_bytes_before_metadata(self) -> int:
        """Documented layer total plus full-vocabulary shared model assets."""

        return self.layer_bytes + self.shared_asset_bytes

    @property
    def accounted_bytes(self) -> int:
        return sum(allocation.nbytes for allocation in self.allocations)

    @property
    def cache_ranges_are_independent(self) -> bool:
        ranges = [pool.allocation.byte_range for pool in self.cache_pools]
        return all(not left.overlaps(right) for index, left in enumerate(ranges) for right in ranges[index + 1 :])

    def cache_pool(self, layer_id: int, family: str) -> CachePool:
        matches = [pool for pool in self.cache_pools if pool.layer_id == layer_id and pool.family == family]
        if len(matches) != 1:
            raise KeyError(f"expected one {family!r} cache pool for layer {layer_id}, found {len(matches)}")
        return matches[0]


ShapeSpec = tuple[str, tuple[int, ...], DTypeName]
CacheSpec = tuple[str, tuple[int, ...], DTypeName, int, int, str]

_HC_WEIGHT_SPECS: tuple[ShapeSpec, ...] = (
    ("hc_attn_fn", (MIX_HC, HC_DIM), "fp32"),
    ("hc_attn_scale", (3,), "fp32"),
    ("hc_attn_base", (MIX_HC,), "fp32"),
)

_COMMON_ATTENTION_WEIGHT_SPECS: tuple[ShapeSpec, ...] = _HC_WEIGHT_SPECS + (
    ("attn_norm_w", (HIDDEN_SIZE,), "bf16"),
    ("wq_a", (HIDDEN_SIZE, Q_LORA_RANK), "bf16"),
    ("wq_b", (ATTENTION_HEADS, HEAD_DIM, Q_LORA_RANK), "int8"),
    ("wq_b_scale", (ATTENTION_HEADS * HEAD_DIM,), "fp32"),
    ("wkv", (HIDDEN_SIZE, HEAD_DIM), "bf16"),
    ("gamma_cq", (Q_LORA_RANK,), "bf16"),
    ("gamma_ckv", (HEAD_DIM,), "bf16"),
    ("attn_sink", (ATTENTION_HEADS,), "fp32"),
    ("wo_a", (O_GROUPS, O_LORA_RANK, O_GROUP_IN), "bf16"),
    ("wo_b", (HIDDEN_SIZE, O_GROUPS * O_LORA_RANK), "int8"),
    ("wo_b_scale", (HIDDEN_SIZE,), "fp32"),
)

_HCA_WEIGHT_SPECS: tuple[ShapeSpec, ...] = (
    ("hca_cmp_wkv", (HEAD_DIM, HIDDEN_SIZE), "bf16"),
    ("hca_cmp_wgate", (HEAD_DIM, HIDDEN_SIZE), "bf16"),
    ("hca_cmp_ape", (HCA_COMPRESS_RATIO, HEAD_DIM), "fp32"),
    ("hca_cmp_norm_w", (HEAD_DIM,), "bf16"),
)

_CSA_MAIN_OUT_DIM = 2 * HEAD_DIM
_CSA_INNER_OUT_DIM = 2 * INDEX_HEAD_DIM
_CSA_WEIGHT_SPECS: tuple[ShapeSpec, ...] = (
    ("csa_cmp_wkv", (_CSA_MAIN_OUT_DIM, HIDDEN_SIZE), "bf16"),
    ("csa_cmp_wgate", (_CSA_MAIN_OUT_DIM, HIDDEN_SIZE), "bf16"),
    ("csa_cmp_ape", (CSA_COMPRESS_RATIO, _CSA_MAIN_OUT_DIM), "fp32"),
    ("csa_cmp_norm_w", (HEAD_DIM,), "bf16"),
    ("csa_idx_wq_b", (Q_LORA_RANK, INDEX_HEADS * INDEX_HEAD_DIM), "int8"),
    ("csa_idx_wq_b_scale", (INDEX_HEADS * INDEX_HEAD_DIM,), "fp32"),
    ("csa_weights_proj", (HIDDEN_SIZE, INDEX_HEADS), "bf16"),
    ("csa_hadamard_idx", (INDEX_HEAD_DIM, INDEX_HEAD_DIM), "bf16"),
    ("csa_inner_wkv", (_CSA_INNER_OUT_DIM, HIDDEN_SIZE), "bf16"),
    ("csa_inner_wgate", (_CSA_INNER_OUT_DIM, HIDDEN_SIZE), "bf16"),
    ("csa_inner_ape", (CSA_COMPRESS_RATIO, _CSA_INNER_OUT_DIM), "fp32"),
    ("csa_inner_norm_w", (INDEX_HEAD_DIM,), "bf16"),
)

_MOE_WEIGHT_SPECS: tuple[ShapeSpec, ...] = (
    ("hc_ffn_fn", (MIX_HC, HC_DIM), "fp32"),
    ("hc_ffn_scale", (3,), "fp32"),
    ("hc_ffn_base", (MIX_HC,), "fp32"),
    ("norm_w", (HIDDEN_SIZE,), "bf16"),
    ("gate_w", (GLOBAL_EXPERTS, HIDDEN_SIZE), "fp32"),
    ("gate_bias", (GLOBAL_EXPERTS,), "fp32"),
    ("routed_w1", (MOE_LOCAL_EXPERTS, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE), "int8"),
    ("routed_w1_scale", (MOE_LOCAL_EXPERTS, MOE_INTERMEDIATE_SIZE), "fp32"),
    ("routed_w3", (MOE_LOCAL_EXPERTS, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE), "int8"),
    ("routed_w3_scale", (MOE_LOCAL_EXPERTS, MOE_INTERMEDIATE_SIZE), "fp32"),
    ("routed_w2", (MOE_LOCAL_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE), "int8"),
    ("routed_w2_scale", (MOE_LOCAL_EXPERTS, HIDDEN_SIZE), "fp32"),
    ("shared_w1", (MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE), "int8"),
    ("shared_w1_scale", (MOE_INTERMEDIATE_SIZE,), "fp32"),
    ("shared_w3", (MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE), "int8"),
    ("shared_w3_scale", (MOE_INTERMEDIATE_SIZE,), "fp32"),
    ("shared_w2", (HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE), "int8"),
    ("shared_w2_scale", (HIDDEN_SIZE,), "fp32"),
)

_HASH_WEIGHT_SPEC: ShapeSpec = ("tid2eid", (VOCAB_SIZE, ROUTES_PER_TOKEN), "int32")

_SHARED_ASSET_SPECS: tuple[ShapeSpec, ...] = (
    ("embed_weight", (VOCAB_SIZE, HIDDEN_SIZE), "bf16"),
    ("freqs_cos", (MAX_SEQUENCE_LENGTH, ROPE_HEAD_DIM), "bf16"),
    ("freqs_sin", (MAX_SEQUENCE_LENGTH, ROPE_HEAD_DIM), "bf16"),
    ("hc_head_fn", (HC_MULT, HC_DIM), "fp32"),
    ("hc_head_scale", (1,), "fp32"),
    ("hc_head_base", (HC_MULT,), "fp32"),
    ("final_norm_w", (HIDDEN_SIZE,), "bf16"),
    ("lm_head_weight", (VOCAB_SIZE, HIDDEN_SIZE), "bf16"),
)

_ORIGINAL_CACHE_SPEC: CacheSpec = (
    "original_kv",
    (ORI_KV_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM),
    "bf16",
    ORI_KV_BLOCK_NUM,
    KV_ORI_TABLE_MAX_BLOCKS,
    "block_table",
)

_HCA_CACHE_SPECS: tuple[CacheSpec, ...] = (
    _ORIGINAL_CACHE_SPEC,
    (
        "hca_compressed_kv",
        (HCA_COMPRESSED_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM),
        "bf16",
        HCA_COMPRESSED_BLOCKS,
        KV_HCA_MAX_BLOCKS,
        "hca_cmp_block_table",
    ),
    (
        "hca_compressor_state",
        (HCA_STATE_PHYSICAL_BLOCKS, C128_COMPRESSOR_BLOCK_SIZE, 2 * HEAD_DIM),
        "fp32",
        HCA_STATE_PHYSICAL_BLOCKS,
        HCA_STATE_TABLE_MAX_BLOCKS,
        "hca_state_block_table",
    ),
)

_CSA_CACHE_SPECS: tuple[CacheSpec, ...] = (
    _ORIGINAL_CACHE_SPEC,
    (
        "csa_compressed_kv",
        (CSA_COMPRESSED_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM),
        "bf16",
        CSA_COMPRESSED_BLOCKS,
        KV_CMP_MAX_BLOCKS,
        "csa_cmp_block_table",
    ),
    (
        "csa_index_kv",
        (IDX_KV_BLOCK_NUM, BLOCK_SIZE, 1, INDEX_HEAD_DIM),
        "int8",
        IDX_KV_BLOCK_NUM,
        KV_CMP_MAX_BLOCKS,
        "csa_idx_block_table",
    ),
    (
        "csa_index_scale",
        (IDX_KV_BLOCK_NUM, BLOCK_SIZE, 1, 1),
        "fp32",
        IDX_KV_BLOCK_NUM,
        KV_CMP_MAX_BLOCKS,
        "csa_idx_block_table",
    ),
    (
        "csa_compressor_state",
        (CSA_STATE_PHYSICAL_BLOCKS, C4A_COMPRESSOR_BLOCK_SIZE, 2 * _CSA_MAIN_OUT_DIM),
        "fp32",
        CSA_STATE_PHYSICAL_BLOCKS,
        CSA_STATE_TABLE_MAX_BLOCKS,
        "csa_state_block_table",
    ),
    (
        "csa_inner_compressor_state",
        (CSA_INNER_STATE_PHYSICAL_BLOCKS, C4A_COMPRESSOR_BLOCK_SIZE, 2 * _CSA_INNER_OUT_DIM),
        "fp32",
        CSA_INNER_STATE_PHYSICAL_BLOCKS,
        CSA_INNER_STATE_TABLE_MAX_BLOCKS,
        "csa_inner_state_block_table",
    ),
)

DEPTH_LADDER_LAYER_IDS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("hca1", (0,)),
    ("csa1", (2,)),
    ("depth2", tuple(range(2))),
    ("depth4", tuple(range(4))),
    ("depth16", tuple(range(16))),
    ("depth31", tuple(range(31))),
    ("depth61", tuple(range(61))),
)


def _attention_kind(layer_id: int) -> AttentionKind:
    ratio = MAIN_COMPRESS_RATIOS[layer_id]
    if ratio == HCA_COMPRESS_RATIO:
        return "hca"
    if ratio == CSA_COMPRESS_RATIO:
        return "csa"
    raise ValueError(f"main layer {layer_id} has unsupported compression ratio {ratio}")


def _shared_metadata_specs(has_hca: bool, has_csa: bool) -> tuple[ShapeSpec, ...]:
    specs: list[ShapeSpec] = [("block_table", (DECODE_BATCH, KV_ORI_TABLE_MAX_BLOCKS), "int32")]
    if has_hca:
        specs.extend(
            (
                ("hca_cmp_block_table", (DECODE_BATCH, KV_HCA_MAX_BLOCKS), "int32"),
                ("hca_state_block_table", (DECODE_BATCH, HCA_STATE_TABLE_MAX_BLOCKS), "int32"),
            )
        )
    if has_csa:
        specs.extend(
            (
                ("csa_cmp_block_table", (DECODE_BATCH, KV_CMP_MAX_BLOCKS), "int32"),
                ("csa_idx_block_table", (DECODE_BATCH, KV_CMP_MAX_BLOCKS), "int32"),
                ("csa_state_block_table", (DECODE_BATCH, CSA_STATE_TABLE_MAX_BLOCKS), "int32"),
                (
                    "csa_inner_state_block_table",
                    (DECODE_BATCH, CSA_INNER_STATE_TABLE_MAX_BLOCKS),
                    "int32",
                ),
            )
        )
    return tuple(specs)


def build_hbm_manifest(layer_ids: Iterable[int]) -> HBMManifest:
    """Build byte accounting for explicit Pro main-model layer IDs."""

    selected = tuple(layer_ids)
    if not selected:
        raise ValueError("at least one main-model layer is required")
    if any(type(layer_id) is not int for layer_id in selected):
        raise TypeError("layer IDs must be integers")
    if len(set(selected)) != len(selected):
        raise ValueError("layer IDs must be unique")
    for layer_id in selected:
        if not 0 <= layer_id < len(MAIN_COMPRESS_RATIOS):
            raise ValueError(f"layer ID {layer_id} is outside the 61-layer main model")

    kinds = tuple(_attention_kind(layer_id) for layer_id in selected)
    hca_layer_ids = tuple(layer_id for layer_id, kind in zip(selected, kinds) if kind == "hca")
    csa_layer_ids = tuple(layer_id for layer_id, kind in zip(selected, kinds) if kind == "csa")
    allocations: list[TensorAllocation] = []
    cache_pools: list[CachePool] = []
    offset = 0

    def append_allocation(
        name: str,
        shape: tuple[int, ...],
        dtype: DTypeName,
        kind: AllocationKind,
        layer_id: int | None = None,
    ) -> TensorAllocation:
        nonlocal offset
        allocation = TensorAllocation(name, shape, dtype, kind, offset, layer_id)
        allocations.append(allocation)
        offset += allocation.nbytes
        return allocation

    for name, shape, dtype in _SHARED_ASSET_SPECS:
        append_allocation(name, shape, dtype, "shared_asset")
    for name, shape, dtype in _shared_metadata_specs(bool(hca_layer_ids), bool(csa_layer_ids)):
        append_allocation(name, shape, dtype, "shared_metadata")

    common_weight_specs = _COMMON_ATTENTION_WEIGHT_SPECS + _MOE_WEIGHT_SPECS
    for layer_id, attention_kind in zip(selected, kinds):
        kind_specs = _HCA_WEIGHT_SPECS if attention_kind == "hca" else _CSA_WEIGHT_SPECS
        weight_specs = common_weight_specs + kind_specs
        if layer_id < ACTIVE.num_hash_layers:
            weight_specs += (_HASH_WEIGHT_SPEC,)
        for name, shape, dtype in weight_specs:
            append_allocation(f"layer{layer_id}.{name}", shape, dtype, "layer_weight", layer_id)

        cache_specs = _HCA_CACHE_SPECS if attention_kind == "hca" else _CSA_CACHE_SPECS
        for family, shape, dtype, physical_blocks, logical_blocks, table_name in cache_specs:
            allocation = append_allocation(
                f"layer{layer_id}.{family}",
                shape,
                dtype,
                "layer_cache",
                layer_id,
            )
            cache_pools.append(
                CachePool(
                    family,
                    layer_id,
                    attention_kind,
                    physical_blocks,
                    logical_blocks,
                    table_name,
                    allocation,
                )
            )

    manifest = HBMManifest(
        selected,
        hca_layer_ids,
        csa_layer_ids,
        tuple(allocations),
        tuple(cache_pools),
    )
    if not manifest.cache_ranges_are_independent:
        raise ValueError("layer cache byte ranges overlap")
    return manifest


def build_ladder_manifest(name: str) -> HBMManifest:
    """Build one named reduced-depth manifest from the Milestone 6 ladder."""

    for ladder_name, layer_ids in DEPTH_LADDER_LAYER_IDS:
        if ladder_name == name:
            return build_hbm_manifest(layer_ids)
    choices = ", ".join(ladder_name for ladder_name, _ in DEPTH_LADDER_LAYER_IDS)
    raise ValueError(f"unknown depth ladder entry {name!r}; expected one of {choices}")


if MOE_LOCAL_EXPERTS != 3 or MOE_DEPLOYMENT_RECV_MAX != 1024:
    raise ValueError("the static manifest requires the EP128 compute-only MoE shape")
