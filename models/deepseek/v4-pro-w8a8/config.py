# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Pro W8A8 128K bring-up configuration."""

from dataclasses import dataclass, replace
from typing import Literal, Optional, Tuple


@dataclass(frozen=True)
class DeepSeekV4Config:
    """Follows HuggingFace config.json"""

    name: str

    # ---- attention / hidden ----
    hidden_size: int
    num_attention_heads: int
    head_dim: int  # MLA value-head dim
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int  # grouped output projection
    sliding_window: int
    rms_norm_eps: float
    vocab_size: int

    # ---- MoE ----
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    scoring_func: Literal["softmax", "sigmoid", "sqrtsoftplus"]
    routed_scaling_factor: float
    swiglu_limit: float

    # ---- layers ----
    num_hidden_layers: int
    num_hash_layers: int  # first layers use hash routing
    num_nextn_predict_layers: int  # multi-token-prediction layers
    compress_ratios: Tuple[int, ...]  # per-layer KV compression ratio (0 / 4 / 128)

    # ---- lightning indexer ----
    index_n_heads: int
    index_head_dim: int
    index_topk: int  # compressed positions kept by the indexer

    # ---- hyper-connections (HC) ----
    hc_mult: int  # hc-stack width
    hc_sinkhorn_iters: int
    hc_eps: float

    # ---- context length / RoPE (YaRN; rope_scaling.* flattened) ----
    max_position_embeddings: int
    rope_theta: float
    compress_rope_theta: float
    rope_factor: float  # rope_scaling.factor
    beta_fast: int  # rope_scaling.beta_fast
    beta_slow: int  # rope_scaling.beta_slow
    original_max_position_embeddings: int  # rope_scaling.original_max_position_embeddings

    # ---- precision / quantization (quantization_config.* flattened; unused by decode kernels) ----
    dtype: Literal["bf16", "fp8"]  # quantization_config.quant_method
    scale_fmt: Optional[Literal["ue8m0"]]  # quantization_config.scale_fmt
    expert_dtype: Optional[Literal["fp4"]]  # MoE-expert weight dtype (None = same as `dtype`)
    scale_dtype: Literal["fp32", "fp8"]  # dequant-scale storage dtype

    # ---- deployment (not consumed by the decode kernels) ----
    max_batch_size: int  # max supported batch size (cache sizing)

    # ---- derived ----
    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.qk_rope_head_dim

    @property
    def softmax_scale(self) -> float:
        return self.head_dim**-0.5

    @property
    def index_nope_head_dim(self) -> int:
        return self.index_head_dim - self.qk_rope_head_dim

    @property
    def index_weights_scale(self) -> float:
        return self.index_head_dim**-0.5 * self.index_n_heads**-0.5

    @property
    def hc_dim(self) -> int:
        return self.hc_mult * self.hidden_size

    @property
    def mix_hc(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult


@dataclass(frozen=True)
class DecodeTarget:
    """Fixed deployment point for the staged decode bring-up."""

    model: DeepSeekV4Config
    batch: int
    sequence: int
    mtp_draft_depth: int
    start_position: int
    max_sequence_length: int
    deployment_ep: int
    communication_eps: Tuple[int, ...]
    routed_activation_dtype: Literal["int8"]
    routed_weight_dtype: Literal["int8"]

    @property
    def tokens(self) -> int:
        return self.batch * self.sequence


@dataclass(frozen=True)
class LayerRatioSchedule:
    """Main-model and MTP ownership of compression-ratio entries."""

    main: Tuple[int, ...]
    mtp: Tuple[int, ...]

    @property
    def main_swa_layers(self) -> int:
        return self.main.count(SWA_COMPRESS_RATIO)

    @property
    def main_csa_layers(self) -> int:
        return self.main.count(CSA_COMPRESS_RATIO)

    @property
    def main_hca_layers(self) -> int:
        return self.main.count(HCA_COMPRESS_RATIO)


@dataclass(frozen=True)
class ExpertParallelLayout:
    """Shape-only expert layout for one selected world size."""

    world_size: int
    global_experts: int
    local_experts: int
    recv_capacity: int


SWA_COMPRESS_RATIO = 0
CSA_COMPRESS_RATIO = 4
HCA_COMPRESS_RATIO = 128
SUPPORTED_COMPRESS_RATIOS = (
    SWA_COMPRESS_RATIO,
    CSA_COMPRESS_RATIO,
    HCA_COMPRESS_RATIO,
)


def split_layer_ratios(model: DeepSeekV4Config) -> LayerRatioSchedule:
    """Split the model-owned main and MTP compression-ratio entries."""

    expected_layers = model.num_hidden_layers + model.num_nextn_predict_layers
    if len(model.compress_ratios) != expected_layers:
        raise ValueError(
            f"{model.name} has {len(model.compress_ratios)} compression ratios "
            f"for {expected_layers} main-plus-MTP layers"
        )
    invalid_ratios = tuple(ratio for ratio in model.compress_ratios if ratio not in SUPPORTED_COMPRESS_RATIOS)
    if invalid_ratios:
        raise ValueError(f"{model.name} has unsupported compression ratios {invalid_ratios}")
    main_end = model.num_hidden_layers
    return LayerRatioSchedule(
        main=model.compress_ratios[:main_end],
        mtp=model.compress_ratios[main_end:],
    )


DEMO = DeepSeekV4Config(
    name="demo",
    hidden_size=4096,
    num_attention_heads=64,
    head_dim=512,
    qk_rope_head_dim=64,
    q_lora_rank=1024,
    o_lora_rank=1024,
    o_groups=8,
    sliding_window=128,
    rms_norm_eps=1e-6,
    vocab_size=129280,
    moe_intermediate_size=4096,
    n_routed_experts=16,
    n_shared_experts=1,
    num_experts_per_tok=2,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.0,
    swiglu_limit=0.0,
    num_hidden_layers=8,
    num_hash_layers=0,
    num_nextn_predict_layers=1,
    compress_ratios=(0, 0, 4, 128, 4, 128, 4, 0),
    index_n_heads=64,
    index_head_dim=128,
    index_topk=512,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    max_position_embeddings=4096,
    rope_theta=10000.0,
    compress_rope_theta=40000.0,
    rope_factor=40.0,
    beta_fast=32,
    beta_slow=1,
    original_max_position_embeddings=0,
    dtype="fp8",
    scale_fmt="ue8m0",
    expert_dtype=None,
    scale_dtype="fp8",
    max_batch_size=4,
)

FLASH = DeepSeekV4Config(
    name="flash",
    hidden_size=4096,
    num_attention_heads=64,
    head_dim=512,
    qk_rope_head_dim=64,
    q_lora_rank=1024,
    o_lora_rank=1024,
    o_groups=8,
    sliding_window=128,
    rms_norm_eps=1e-6,
    vocab_size=129280,
    moe_intermediate_size=2048,
    n_routed_experts=256,
    n_shared_experts=1,
    num_experts_per_tok=6,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.5,
    swiglu_limit=10.0,
    num_hidden_layers=43,
    num_hash_layers=3,
    num_nextn_predict_layers=1,
    compress_ratios=(
        0,
        0,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        0,
    ),
    index_n_heads=64,
    index_head_dim=128,
    index_topk=512,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    max_position_embeddings=16384,  # 8k prompt + 512 decode steps target; official 1M;
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    rope_factor=16.0,
    beta_fast=32,
    beta_slow=1,
    original_max_position_embeddings=65536,
    dtype="fp8",
    scale_fmt="ue8m0",
    expert_dtype="fp4",
    scale_dtype="fp8",
    max_batch_size=4,
)

PRO = DeepSeekV4Config(
    name="pro",
    hidden_size=7168,
    num_attention_heads=128,
    head_dim=512,
    qk_rope_head_dim=64,
    q_lora_rank=1536,
    o_lora_rank=1024,
    o_groups=16,
    sliding_window=128,
    rms_norm_eps=1e-6,
    vocab_size=129280,
    moe_intermediate_size=3072,
    n_routed_experts=384,
    n_shared_experts=1,
    num_experts_per_tok=6,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=2.5,
    swiglu_limit=10.0,
    num_hidden_layers=61,
    num_hash_layers=3,
    num_nextn_predict_layers=1,
    compress_ratios=(
        128,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        0,
    ),
    index_n_heads=64,
    index_head_dim=128,
    index_topk=1024,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    max_position_embeddings=1048576,
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    rope_factor=16.0,
    beta_fast=32,
    beta_slow=1,
    original_max_position_embeddings=65536,
    dtype="fp8",
    scale_fmt="ue8m0",
    expert_dtype=None,
    scale_dtype="fp8",
    max_batch_size=4,
)


KERNEL_MAX_SEQ_LEN = 131584

PRO_W8A8 = replace(
    PRO,
    name="pro-w8a8",
    max_position_embeddings=KERNEL_MAX_SEQ_LEN,
    max_batch_size=2,
    expert_dtype=None,
)

TARGET = DecodeTarget(
    model=PRO_W8A8,
    batch=2,
    sequence=4,
    mtp_draft_depth=3,
    start_position=131072,
    max_sequence_length=KERNEL_MAX_SEQ_LEN,
    deployment_ep=128,
    communication_eps=(2, 4, 8),
    routed_activation_dtype="int8",
    routed_weight_dtype="int8",
)

if TARGET.sequence != TARGET.mtp_draft_depth + 1:
    raise ValueError("decode sequence must contain one main token plus every MTP draft")
if TARGET.max_sequence_length != TARGET.model.max_position_embeddings:
    raise ValueError("target and kernel-preset sequence budgets must match")
if TARGET.model.n_routed_experts % TARGET.deployment_ep != 0:
    raise ValueError("global experts must divide evenly across deployment EP")

PRO_W8A8_LAYER_SCHEDULE = split_layer_ratios(PRO_W8A8)
EXPECTED_PRO_MAIN_RATIOS = (
    (
        HCA_COMPRESS_RATIO,
        HCA_COMPRESS_RATIO,
    )
    + (
        CSA_COMPRESS_RATIO,
        HCA_COMPRESS_RATIO,
    )
    * 29
    + (CSA_COMPRESS_RATIO,)
)

if PRO_W8A8_LAYER_SCHEDULE.main != EXPECTED_PRO_MAIN_RATIOS:
    raise ValueError("Pro main-layer compression order changed")
if PRO_W8A8_LAYER_SCHEDULE.mtp != (SWA_COMPRESS_RATIO,):
    raise ValueError("Pro trailing SWA entry must belong to the single MTP layer")
if (
    PRO_W8A8_LAYER_SCHEDULE.main_hca_layers,
    PRO_W8A8_LAYER_SCHEDULE.main_csa_layers,
    PRO_W8A8_LAYER_SCHEDULE.main_swa_layers,
) != (31, 30, 0):
    raise ValueError("Pro main-layer mix must be 31 HCA, 30 CSA, and zero SWA")

MAIN_COMPRESS_RATIOS = PRO_W8A8_LAYER_SCHEDULE.main
MTP_COMPRESS_RATIOS = PRO_W8A8_LAYER_SCHEDULE.mtp
MAIN_HCA_LAYERS = PRO_W8A8_LAYER_SCHEDULE.main_hca_layers
MAIN_CSA_LAYERS = PRO_W8A8_LAYER_SCHEDULE.main_csa_layers
MAIN_SWA_LAYERS = PRO_W8A8_LAYER_SCHEDULE.main_swa_layers

ACTIVE = PRO_W8A8
PRESETS = {p.name: p for p in (DEMO, FLASH, PRO, PRO_W8A8)}


# Deployment constants
DECODE_BATCH = TARGET.batch
DECODE_SEQ = TARGET.sequence
DECODE_TOKENS = TARGET.tokens
MTP_DRAFT_DEPTH = TARGET.mtp_draft_depth
DECODE_START_POS = TARGET.start_position
DECODE_MAX_SEQ_LEN = TARGET.max_sequence_length
PREFILL_BATCH = 1  # B: prefill batch for the current kernel programs
PREFILL_SEQ = 128  # S: prefill sequence for the current kernel programs
PREFILL_TOKENS = PREFILL_BATCH * PREFILL_SEQ
MOE_TOKENS = DECODE_TOKENS

# Implementation constants
BLOCK_SIZE = 128  # paged-KV page size / weight-quant block size
C4A_COMPRESSOR_BLOCK_SIZE = 4  # ratio-4 compressor state page size
C128_COMPRESSOR_BLOCK_SIZE = 8  # ratio-128 compressor state page size

# Static paged-cache pools shared by decode and prefill kernels. ``*_MAX_BLOCKS``
# is logical width per request; ``*_BLOCK_NUM`` is global physical capacity.
KV_ORI_TABLE_MAX_BLOCKS = (ACTIVE.max_position_embeddings + BLOCK_SIZE - 1) // BLOCK_SIZE
KV_ORI_MAX_BLOCKS = KV_ORI_TABLE_MAX_BLOCKS
KV_CMP_MAX_BLOCKS = (ACTIVE.max_position_embeddings + BLOCK_SIZE * CSA_COMPRESS_RATIO - 1) // (
    BLOCK_SIZE * CSA_COMPRESS_RATIO
)
KV_HCA_MAX_BLOCKS = (ACTIVE.max_position_embeddings + BLOCK_SIZE * HCA_COMPRESS_RATIO - 1) // (
    BLOCK_SIZE * HCA_COMPRESS_RATIO
)
IDX_CACHE_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
ORI_KV_BLOCK_NUM = 6
CMP_KV_BLOCK_NUM = DECODE_BATCH * KV_CMP_MAX_BLOCKS
IDX_KV_BLOCK_NUM = DECODE_BATCH * IDX_CACHE_MAX_BLOCKS
HCA_STATE_PHYSICAL_BLOCKS = 32
CSA_STATE_PHYSICAL_BLOCKS = 4
CSA_INNER_STATE_PHYSICAL_BLOCKS = 4
HCA_STATE_TABLE_MAX_BLOCKS = (
    ACTIVE.max_position_embeddings + C128_COMPRESSOR_BLOCK_SIZE - 1
) // C128_COMPRESSOR_BLOCK_SIZE
CSA_STATE_TABLE_MAX_BLOCKS = (
    ACTIVE.max_position_embeddings + C4A_COMPRESSOR_BLOCK_SIZE - 1
) // C4A_COMPRESSOR_BLOCK_SIZE
CSA_INNER_STATE_TABLE_MAX_BLOCKS = CSA_STATE_TABLE_MAX_BLOCKS
DECODE_ORI_BLOCK_NUM = ORI_KV_BLOCK_NUM
DECODE_CMP_BLOCK_NUM = CMP_KV_BLOCK_NUM
DECODE_IDX_BLOCK_NUM = IDX_KV_BLOCK_NUM
PREFILL_ORI_MAX_BLOCKS = KV_ORI_TABLE_MAX_BLOCKS
PREFILL_ORI_BLOCK_NUM = DECODE_ORI_BLOCK_NUM
PREFILL_CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
PREFILL_CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM  # shared global physical pool
PREFILL_IDX_MAX_BLOCKS = IDX_CACHE_MAX_BLOCKS
PREFILL_IDX_BLOCK_NUM = DECODE_IDX_BLOCK_NUM  # shared global physical pool

# Int8 quantization constants
INT8_SCALE_MAX = 127.0  # per-row INT8 quant: clamp scale so |q| <= 127
INT8_AMAX_EPS = 1e-4  # amax floor: avoids 127/0 on all-zero rows
FP32_NEG_INF = -3.4028234663852886e38  # most-negative finite fp32 (softmax masking)

# Expert deployment and measured communication are independent axes.
MOE_GLOBAL_EXPERTS = ACTIVE.n_routed_experts
DEPLOYMENT_EP = TARGET.deployment_ep
COMM_EP_CHOICES = TARGET.communication_eps
COMM_EP_DEFAULT = COMM_EP_CHOICES[0]


def expert_parallel_layout(world_size: int, tokens_per_rank: int) -> ExpertParallelLayout:
    """Build shape constants without creating distributed runtime state."""

    if world_size <= 0:
        raise ValueError("expert world size must be positive")
    if MOE_GLOBAL_EXPERTS % world_size != 0:
        raise ValueError(f"{MOE_GLOBAL_EXPERTS} global experts do not divide across EP{world_size}")
    return ExpertParallelLayout(
        world_size=world_size,
        global_experts=MOE_GLOBAL_EXPERTS,
        local_experts=MOE_GLOBAL_EXPERTS // world_size,
        recv_capacity=world_size * tokens_per_rank,
    )


DEPLOYMENT_LAYOUT = expert_parallel_layout(DEPLOYMENT_EP, DECODE_TOKENS)
COMMUNICATION_LAYOUTS = tuple(expert_parallel_layout(comm_ep, DECODE_TOKENS) for comm_ep in COMM_EP_CHOICES)
MOE_LOCAL_EXPERTS = DEPLOYMENT_LAYOUT.local_experts
MOE_DEPLOYMENT_RECV_MAX = DEPLOYMENT_LAYOUT.recv_capacity
MOE_GLOBAL_ROUTES = DEPLOYMENT_EP * DECODE_TOKENS * ACTIVE.num_experts_per_tok
if MOE_GLOBAL_ROUTES % MOE_GLOBAL_EXPERTS != 0:
    raise ValueError("balanced deployment route count must divide across global experts")
MOE_BALANCED_ROWS_PER_EXPERT = MOE_GLOBAL_ROUTES // MOE_GLOBAL_EXPERTS
MOE_BALANCED_ROWS_PER_SHARD = MOE_LOCAL_EXPERTS * MOE_BALANCED_ROWS_PER_EXPERT

# Shape aliases for standalone expert kernels; the distributed MoE entry selects
# its physical communication layout before importing shape-specialized kernels.
EP_WORLD_SIZE = DEPLOYMENT_EP
DECODE_RECV_MAX = MOE_DEPLOYMENT_RECV_MAX
PREFILL_RECV_MAX = DEPLOYMENT_EP * PREFILL_TOKENS
RECV_MAX = DECODE_RECV_MAX
