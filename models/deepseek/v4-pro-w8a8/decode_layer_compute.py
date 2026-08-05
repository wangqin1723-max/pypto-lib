# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One-die HCA/CSA layer compute with a synthetic local routed-row proxy.

This rung composes validated attention and FFN compute kernels.  Routed rows
are host fixtures, not an EP128 dispatch/combine implementation.
"""

import pypto.language as pl

from config import ACTIVE as M, BLOCK_SIZE, DECODE_START_POS
from decode_attention_csa import (
    CMP_BLOCK_NUM as CSA_CMP_BLOCK_NUM,
    CMP_MAX_BLOCKS as CSA_CMP_MAX_BLOCKS,
    COMPRESS_RATIO as CSA_COMPRESS_RATIO,
    IDX_CACHE_BLOCK_NUM as CSA_IDX_CACHE_BLOCK_NUM,
    IDX_CACHE_MAX_BLOCKS as CSA_IDX_CACHE_MAX_BLOCKS,
    IDX_HEAD_DIM as CSA_IDX_HEAD_DIM,
    IDX_N_HEADS as CSA_IDX_N_HEADS,
    INNER_OUT_DIM as CSA_INNER_OUT_DIM,
    INNER_STATE_BLOCK_NUM as CSA_INNER_STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE as CSA_INNER_STATE_BLOCK_SIZE,
    INNER_STATE_DIM as CSA_INNER_STATE_DIM,
    INNER_STATE_MAX_BLOCKS as CSA_INNER_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM as CSA_MAIN_OUT_DIM,
    MAIN_STATE_BLOCK_NUM as CSA_MAIN_STATE_BLOCK_NUM,
    MAIN_STATE_BLOCK_SIZE as CSA_MAIN_STATE_BLOCK_SIZE,
    MAIN_STATE_DIM as CSA_MAIN_STATE_DIM,
    MAIN_STATE_MAX_BLOCKS as CSA_MAIN_STATE_MAX_BLOCKS,
    ORI_BLOCK_NUM as CSA_ORI_BLOCK_NUM,
    attention_csa,
    build_tensor_specs as build_csa_attention_tensor_specs,
    golden_attention_csa,
)
from decode_attention_hca import (
    B,
    CMP_BLOCK_NUM_DYN as HCA_CMP_BLOCK_NUM_DYN,
    CMP_MAX_BLOCKS as HCA_CMP_MAX_BLOCKS,
    COMPRESS_RATIO as HCA_COMPRESS_RATIO,
    COMPRESS_STATE_BLOCK_NUM_DYN as HCA_STATE_BLOCK_NUM_DYN,
    COMPRESS_STATE_BLOCK_SIZE as HCA_STATE_BLOCK_SIZE,
    COMPRESS_STATE_DIM as HCA_STATE_DIM,
    COMPRESS_STATE_MAX_BLOCKS as HCA_STATE_MAX_BLOCKS,
    D,
    H,
    HC_DIM,
    HC_MULT,
    HEAD_DIM,
    MAX_SEQ_LEN,
    MAIN_OUT_DIM as HCA_MAIN_OUT_DIM,
    MIX_HC,
    O_GROUPS,
    O_GROUP_IN,
    O_LORA,
    ORI_BLOCK_NUM_DYN as HCA_ORI_BLOCK_NUM_DYN,
    Q_LORA,
    ROPE_HEAD_DIM,
    T,
    WIN,
    attention_hca,
    build_tensor_specs as build_hca_attention_tensor_specs,
    golden_attention_hca,
)
from expert_routed import (
    N_LOCAL_EXPERTS,
    RECV_MAX,
    ROUTED_WORKLOAD_COUNTS,
    build_tensor_specs as build_routed_tensor_specs,
    expert_routed,
    golden_expert_routed,
)
from expert_shared import (
    build_tensor_specs as build_shared_tensor_specs,
    expert_shared,
    golden_expert_shared,
)
from gate import (
    build_tensor_specs as build_gate_tensor_specs,
    gate,
    golden_gate_core,
)
from hc_post import golden_hc_post, hc_post
from hc_pre import (
    build_tensor_specs as build_hc_pre_tensor_specs,
    golden_hc_pre,
    hc_pre,
)


# model config
N_EXPERTS = M.n_routed_experts
TOPK = M.num_experts_per_tok
VOCAB = M.vocab_size
MOE_INTER = M.moe_intermediate_size

# representative layer entries
HCA_LAYER_ID = 0
CSA_LAYER_ID = 2


@pl.jit.inline
def local_proxy_combine(
    sh: pl.Tensor[[T, D], pl.BF16],
    recv_y: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
    route_to_recv: pl.Tensor[[T, TOPK], pl.INT32],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
):
    """Reduce each host-mapped synthetic routed row exactly once."""
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL_EXPERTS * RECV_MAX, D])
    for token in pl.spmd(T, name_hint="local_proxy_combine"):
        acc = pl.cast(sh[token : token + 1, :], target_type=pl.FP32)
        for route in pl.range(TOPK):
            recv_row_i32 = pl.read(route_to_recv, [token, route])
            recv_row = pl.cast(recv_row_i32, pl.INDEX)
            routed_row = recv_y_flat[recv_row : recv_row + 1, :]
            routed_row_fp32 = pl.cast(routed_row, target_type=pl.FP32)
            acc = pl.add(acc, routed_row_fp32)
        ffn_out[token : token + 1, :] = pl.cast(acc, target_type=pl.BF16, mode="rint")
    return ffn_out


@pl.jit.inline
def ffn_compute_proxy(
    x_attn: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    route_to_recv: pl.Tensor[[T, TOPK], pl.INT32],
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    x_next: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    layer_id: pl.Scalar[pl.INT32],
):
    """Compose FFN compute around synthetic one-die routed inputs."""
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_attn, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, x_mixed, post, comb)

    x_norm_i8 = pl.create_tensor([T, D], dtype=pl.INT8)
    x_norm_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    gate(
        x_mixed, norm_w, gate_w, gate_bias,
        layer_id, pl.const(T, pl.INT32),
        tid2eid, input_ids,
        x_norm_i8, x_norm_scale, indices, weights,
    )

    sh = pl.create_tensor([T, D], dtype=pl.BF16)
    expert_shared(
        x_norm_i8, x_norm_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        sh,
    )

    recv_y = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX, D], dtype=pl.BF16)
    expert_routed(
        recv_x, recv_scale_dq, recv_weights, recv_expert_count,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        recv_y,
    )

    ffn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    local_proxy_combine(sh, recv_y, route_to_recv, ffn_out)
    hc_post(ffn_out, x_attn, post, comb, x_next)
    return x_next


@pl.jit.inline
def decode_layer_compute_hca(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[H, HEAD_DIM, Q_LORA], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[
        [HCA_STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, HCA_STATE_DIM],
        pl.FP32,
    ],
    compress_state_block_table: pl.Tensor[[B, HCA_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Tensor[
        [HCA_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM],
        pl.BF16,
    ],
    cmp_kv: pl.Tensor[
        [HCA_CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM],
        pl.BF16,
    ],
    cmp_block_table: pl.Tensor[[B, HCA_CMP_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    window_swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    route_to_recv: pl.Tensor[[T, TOPK], pl.INT32],
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    x_next: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    layer_id: pl.Scalar[pl.INT32],
):
    """Run one reusable HCA layer body followed by the FFN proxy."""
    x_attn = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
    attention_hca(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale,
        wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        compress_state, compress_state_block_table,
        kv_cache, cmp_kv, cmp_block_table,
        ori_slot_mapping, window_swa_indices, window_swa_lens,
        cmp_slot_mapping, state_slot_mapping,
        position_ids, kv_seq_lens,
        attn_sink,
        wo_a, wo_b, wo_b_scale,
        x_attn,
    )
    ffn_compute_proxy(
        x_attn,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias, tid2eid, input_ids,
        recv_x, recv_scale_dq, recv_weights, recv_expert_count,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        route_to_recv,
        indices, weights, x_next,
        layer_id,
    )
    return x_next


@pl.jit(auto_scope=False)
def decode_layer_compute_hca_test(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[H, HEAD_DIM, Q_LORA], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[
            [HCA_STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, HCA_STATE_DIM],
            pl.FP32,
        ]
    ],
    compress_state_block_table: pl.Tensor[[B, HCA_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[
        pl.Tensor[
            [HCA_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    cmp_kv: pl.InOut[
        pl.Tensor[
            [HCA_CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    cmp_block_table: pl.Tensor[[B, HCA_CMP_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    window_swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    route_to_recv: pl.Tensor[[T, TOPK], pl.INT32],
    indices: pl.Out[pl.Tensor[[T, TOPK], pl.INT32]],
    weights: pl.Out[pl.Tensor[[T, TOPK], pl.FP32]],
    x_next: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
):
    decode_layer_compute_hca(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale,
        wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        compress_state, compress_state_block_table,
        kv_cache, cmp_kv, cmp_block_table,
        ori_slot_mapping, window_swa_indices, window_swa_lens,
        cmp_slot_mapping, state_slot_mapping,
        position_ids, kv_seq_lens,
        attn_sink,
        wo_a, wo_b, wo_b_scale,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias, tid2eid, input_ids,
        recv_x, recv_scale_dq, recv_weights, recv_expert_count,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        route_to_recv,
        indices, weights, x_next,
        pl.const(HCA_LAYER_ID, pl.INT32),
    )
    return indices, weights, x_next


@pl.jit.inline
def decode_layer_compute_csa(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[H, HEAD_DIM, Q_LORA], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[
        [CSA_MAIN_STATE_BLOCK_NUM, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM],
        pl.FP32,
    ],
    compress_state_block_table: pl.Tensor[[B, CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32],
    idx_wq_b: pl.Tensor[[Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, CSA_IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[CSA_IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.Tensor[
        [CSA_INNER_STATE_BLOCK_NUM, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM],
        pl.FP32,
    ],
    inner_compress_state_block_table: pl.Tensor[
        [B, CSA_INNER_STATE_MAX_BLOCKS],
        pl.INT32,
    ],
    kv_cache: pl.Tensor[
        [CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
        pl.BF16,
    ],
    cmp_kv: pl.Tensor[
        [CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
        pl.BF16,
    ],
    cmp_block_table: pl.Tensor[[B, CSA_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Tensor[
        [CSA_IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM],
        pl.INT8,
    ],
    idx_kv_scale: pl.Tensor[
        [CSA_IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1],
        pl.FP32,
    ],
    idx_block_table: pl.Tensor[[B, CSA_IDX_CACHE_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    window_swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    route_to_recv: pl.Tensor[[T, TOPK], pl.INT32],
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    x_next: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    layer_id: pl.Scalar[pl.INT32],
):
    """Run one reusable CSA layer body followed by the FFN proxy."""
    x_attn = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
    attention_csa(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale,
        wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        compress_state, compress_state_block_table,
        idx_wq_b, idx_wq_b_scale, weights_proj, hadamard_idx,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        inner_compress_state, inner_compress_state_block_table,
        kv_cache, cmp_kv, cmp_block_table,
        idx_kv_cache, idx_kv_scale, idx_block_table,
        ori_slot_mapping, window_swa_indices, window_swa_lens,
        cmp_slot_mapping, idx_slot_mapping,
        state_slot_mapping, inner_state_slot_mapping,
        position_ids, kv_seq_lens,
        attn_sink,
        wo_a, wo_b, wo_b_scale,
        x_attn,
    )
    ffn_compute_proxy(
        x_attn,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias, tid2eid, input_ids,
        recv_x, recv_scale_dq, recv_weights, recv_expert_count,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        route_to_recv,
        indices, weights, x_next,
        layer_id,
    )
    return x_next


@pl.jit(auto_scope=False)
def decode_layer_compute_csa_test(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[H, HEAD_DIM, Q_LORA], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[
            [CSA_MAIN_STATE_BLOCK_NUM, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM],
            pl.FP32,
        ]
    ],
    compress_state_block_table: pl.Tensor[[B, CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32],
    idx_wq_b: pl.Tensor[[Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, CSA_IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[CSA_IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.InOut[
        pl.Tensor[
            [CSA_INNER_STATE_BLOCK_NUM, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM],
            pl.FP32,
        ]
    ],
    inner_compress_state_block_table: pl.Tensor[
        [B, CSA_INNER_STATE_MAX_BLOCKS],
        pl.INT32,
    ],
    kv_cache: pl.InOut[
        pl.Tensor[
            [CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    cmp_kv: pl.InOut[
        pl.Tensor[
            [CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    cmp_block_table: pl.Tensor[[B, CSA_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.InOut[
        pl.Tensor[
            [CSA_IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM],
            pl.INT8,
        ]
    ],
    idx_kv_scale: pl.InOut[
        pl.Tensor[
            [CSA_IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1],
            pl.FP32,
        ]
    ],
    idx_block_table: pl.Tensor[[B, CSA_IDX_CACHE_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    window_swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    route_to_recv: pl.Tensor[[T, TOPK], pl.INT32],
    indices: pl.Out[pl.Tensor[[T, TOPK], pl.INT32]],
    weights: pl.Out[pl.Tensor[[T, TOPK], pl.FP32]],
    x_next: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
):
    decode_layer_compute_csa(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale,
        wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        compress_state, compress_state_block_table,
        idx_wq_b, idx_wq_b_scale, weights_proj, hadamard_idx,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        inner_compress_state, inner_compress_state_block_table,
        kv_cache, cmp_kv, cmp_block_table,
        idx_kv_cache, idx_kv_scale, idx_block_table,
        ori_slot_mapping, window_swa_indices, window_swa_lens,
        cmp_slot_mapping, idx_slot_mapping,
        state_slot_mapping, inner_state_slot_mapping,
        position_ids, kv_seq_lens,
        attn_sink,
        wo_a, wo_b, wo_b_scale,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias, tid2eid, input_ids,
        recv_x, recv_scale_dq, recv_weights, recv_expert_count,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        route_to_recv,
        indices, weights, x_next,
        pl.const(CSA_LAYER_ID, pl.INT32),
    )
    return indices, weights, x_next


HCA_ATTENTION_SPEC_NAMES = (
    "x_hc",
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
    "freqs_cos",
    "freqs_sin",
    "cmp_wkv",
    "cmp_wgate",
    "cmp_ape",
    "cmp_norm_w",
    "compress_state",
    "compress_state_block_table",
    "kv_cache",
    "cmp_kv",
    "cmp_block_table",
    "ori_slot_mapping",
    "window_swa_indices",
    "window_swa_lens",
    "cmp_slot_mapping",
    "state_slot_mapping",
    "position_ids",
    "kv_seq_lens",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
)

CSA_ATTENTION_SPEC_NAMES = (
    "x_hc",
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
    "freqs_cos",
    "freqs_sin",
    "cmp_wkv",
    "cmp_wgate",
    "cmp_ape",
    "cmp_norm_w",
    "compress_state",
    "compress_state_block_table",
    "idx_wq_b",
    "idx_wq_b_scale",
    "weights_proj",
    "hadamard_idx",
    "inner_wkv",
    "inner_wgate",
    "inner_ape",
    "inner_norm_w",
    "inner_compress_state",
    "inner_compress_state_block_table",
    "kv_cache",
    "cmp_kv",
    "cmp_block_table",
    "idx_kv_cache",
    "idx_kv_scale",
    "idx_block_table",
    "ori_slot_mapping",
    "window_swa_indices",
    "window_swa_lens",
    "cmp_slot_mapping",
    "idx_slot_mapping",
    "state_slot_mapping",
    "inner_state_slot_mapping",
    "position_ids",
    "kv_seq_lens",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
)

HCA_MUTABLE_NAMES = {"compress_state", "kv_cache", "cmp_kv"}
CSA_MUTABLE_NAMES = {
    "compress_state",
    "inner_compress_state",
    "kv_cache",
    "cmp_kv",
    "idx_kv_cache",
    "idx_kv_scale",
}


def build_route_to_recv(workload="balanced"):
    """Map all 48 active synthetic receive rows to the eight token routes."""
    import torch

    if workload not in ROUTED_WORKLOAD_COUNTS:
        raise ValueError(f"unknown routed workload {workload!r}")
    rows = []
    for expert, count in enumerate(ROUTED_WORKLOAD_COUNTS[workload]):
        for row in range(count):
            rows.append(expert * RECV_MAX + row)
    expected_rows = T * TOPK
    if len(rows) != expected_rows:
        raise ValueError(f"{workload!r} maps {len(rows)} routed rows, expected {expected_rows}")
    return torch.tensor(rows, dtype=torch.int32).reshape(T, TOPK)


def _clone_tensor_spec(spec, name=None, is_output=None):
    from golden import TensorSpec

    output = spec.is_output if is_output is None else is_output
    return TensorSpec(
        name or spec.name, list(spec.shape), spec.dtype,
        init_value=spec.init_value, is_output=output, resident=spec.resident,
    )


def _tensor_specs_by_name(specs):
    from golden import TensorSpec

    return {spec.name: spec for spec in specs if isinstance(spec, TensorSpec)}


def _build_ffn_tensor_specs(layer_id, workload):
    import torch
    from golden import TensorSpec

    hc_specs = _tensor_specs_by_name(build_hc_pre_tensor_specs(B, T // B))
    gate_specs = _tensor_specs_by_name(build_gate_tensor_specs(layer_id=layer_id, num_tokens=T, fixture="random"))
    routed_specs = _tensor_specs_by_name(build_routed_tensor_specs(workload=workload))
    shared_specs = _tensor_specs_by_name(build_shared_tensor_specs())

    return [
        _clone_tensor_spec(hc_specs["hc_fn"], "hc_ffn_fn", is_output=False),
        _clone_tensor_spec(hc_specs["hc_scale"], "hc_ffn_scale", is_output=False),
        _clone_tensor_spec(hc_specs["hc_base"], "hc_ffn_base", is_output=False),
        _clone_tensor_spec(gate_specs["norm_w"], is_output=False),
        _clone_tensor_spec(gate_specs["gate_w"], is_output=False),
        _clone_tensor_spec(gate_specs["gate_bias"], is_output=False),
        _clone_tensor_spec(gate_specs["tid2eid"], is_output=False),
        _clone_tensor_spec(gate_specs["input_ids"], is_output=False),
        _clone_tensor_spec(routed_specs["recv_x"], is_output=False),
        _clone_tensor_spec(routed_specs["recv_scale_dq"], is_output=False),
        _clone_tensor_spec(routed_specs["recv_weights"], is_output=False),
        _clone_tensor_spec(routed_specs["recv_expert_count"], is_output=False),
        _clone_tensor_spec(routed_specs["routed_w1"], is_output=False),
        _clone_tensor_spec(routed_specs["routed_w1_scale"], is_output=False),
        _clone_tensor_spec(routed_specs["routed_w3"], is_output=False),
        _clone_tensor_spec(routed_specs["routed_w3_scale"], is_output=False),
        _clone_tensor_spec(routed_specs["routed_w2"], is_output=False),
        _clone_tensor_spec(routed_specs["routed_w2_scale"], is_output=False),
        _clone_tensor_spec(shared_specs["shared_w1"], is_output=False),
        _clone_tensor_spec(shared_specs["shared_w1_scale"], is_output=False),
        _clone_tensor_spec(shared_specs["shared_w3"], is_output=False),
        _clone_tensor_spec(shared_specs["shared_w3_scale"], is_output=False),
        _clone_tensor_spec(shared_specs["shared_w2"], is_output=False),
        _clone_tensor_spec(shared_specs["shared_w2_scale"], is_output=False),
        TensorSpec("route_to_recv", [T, TOPK], torch.int32, init_value=lambda: build_route_to_recv(workload)),
        _clone_tensor_spec(gate_specs["indices"], is_output=True),
        _clone_tensor_spec(gate_specs["weights"], is_output=True),
        TensorSpec("x_next", [T, HC_MULT, D], torch.float32, is_output=True),
    ]


def build_hca_tensor_specs(start_pos=DECODE_START_POS, workload="balanced"):
    """Build the layer-0 HCA compute-proxy ABI."""
    attention_specs = _tensor_specs_by_name(build_hca_attention_tensor_specs(start_pos))
    specs = [
        _clone_tensor_spec(attention_specs[name], is_output=name in HCA_MUTABLE_NAMES)
        for name in HCA_ATTENTION_SPEC_NAMES
    ]
    specs.extend(_build_ffn_tensor_specs(HCA_LAYER_ID, workload))
    return specs


def build_csa_tensor_specs(start_pos=DECODE_START_POS, workload="balanced"):
    """Build the layer-2 CSA compute-proxy ABI."""
    attention_specs = _tensor_specs_by_name(build_csa_attention_tensor_specs(start_pos))
    specs = [
        _clone_tensor_spec(attention_specs[name], is_output=name in CSA_MUTABLE_NAMES)
        for name in CSA_ATTENTION_SPEC_NAMES
    ]
    specs.extend(_build_ffn_tensor_specs(CSA_LAYER_ID, workload))
    return specs


def _validate_route_to_recv(route_to_recv, recv_expert_count):
    expected = []
    for expert in range(N_LOCAL_EXPERTS):
        count = int(recv_expert_count[expert, 0].item())
        for row in range(count):
            expected.append(expert * RECV_MAX + row)
    actual = route_to_recv.reshape(-1).tolist()
    if sorted(actual) != sorted(expected):
        raise ValueError("route_to_recv must contain every valid synthetic receive row exactly once")


def _golden_ffn_compute_proxy(tensors, x_attn, layer_id):
    import torch

    x_mixed = torch.zeros(T, D, dtype=torch.bfloat16)
    post = torch.zeros(T, HC_MULT, dtype=torch.float32)
    comb = torch.zeros(T, HC_MULT * HC_MULT, dtype=torch.float32)
    golden_hc_pre(
        {
            "x": x_attn,
            "hc_fn": tensors["hc_ffn_fn"],
            "hc_scale": tensors["hc_ffn_scale"],
            "hc_base": tensors["hc_ffn_base"],
            "x_mixed": x_mixed,
            "post": post,
            "comb": comb,
        }
    )

    x_norm_i8 = torch.zeros(T, D, dtype=torch.int8)
    x_norm_scale = torch.zeros(T, 1, dtype=torch.float32)
    gate_tensors = dict(tensors)
    gate_tensors.update(
        {
            "x_mixed": x_mixed,
            "layer_id": layer_id,
            "num_tokens": T,
            "x_norm_i8": x_norm_i8,
            "x_norm_scale": x_norm_scale,
        }
    )
    golden_gate_core(gate_tensors)

    sh = torch.zeros(T, D, dtype=torch.bfloat16)
    golden_expert_shared(
        {
            "x_local_i8": x_norm_i8,
            "x_local_scale_dq": x_norm_scale,
            "shared_w1": tensors["shared_w1"],
            "shared_w1_scale": tensors["shared_w1_scale"],
            "shared_w3": tensors["shared_w3"],
            "shared_w3_scale": tensors["shared_w3_scale"],
            "shared_w2": tensors["shared_w2"],
            "shared_w2_scale": tensors["shared_w2_scale"],
            "sh": sh,
        }
    )

    recv_y = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.bfloat16)
    routed_tensors = dict(tensors)
    routed_tensors["recv_y"] = recv_y
    golden_expert_routed(routed_tensors)

    _validate_route_to_recv(tensors["route_to_recv"], tensors["recv_expert_count"])
    recv_y_flat = recv_y.reshape(N_LOCAL_EXPERTS * RECV_MAX, D)
    ffn_out = sh.float()
    for token in range(T):
        for route in range(TOPK):
            recv_row = int(tensors["route_to_recv"][token, route].item())
            ffn_out[token] = ffn_out[token] + recv_y_flat[recv_row].float()
    ffn_out = ffn_out.to(torch.bfloat16)

    golden_hc_post(
        {
            "x": ffn_out,
            "residual": x_attn,
            "post": post,
            "comb": comb,
            "y": tensors["x_next"],
        }
    )


def golden_decode_layer_compute_hca(tensors):
    """Sequential Torch reference for the reusable HCA layer body."""
    import torch

    x_attn = torch.zeros(T, HC_MULT, D, dtype=torch.float32)
    attention_tensors = dict(tensors)
    attention_tensors["x_out"] = x_attn
    golden_attention_hca(attention_tensors)
    layer_id = int(tensors.get("layer_id", HCA_LAYER_ID))
    _golden_ffn_compute_proxy(tensors, x_attn, layer_id)


def golden_decode_layer_compute_csa(tensors):
    """Sequential Torch reference for the reusable CSA layer body."""
    import torch

    x_attn = torch.zeros(T, HC_MULT, D, dtype=torch.float32)
    attention_tensors = dict(tensors)
    attention_tensors["x_out"] = x_attn
    golden_attention_csa(attention_tensors)
    layer_id = int(tensors.get("layer_id", CSA_LAYER_ID))
    _golden_ffn_compute_proxy(tensors, x_attn, layer_id)


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, ratio_reldiff, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3sim", choices=("a2a3sim", "a2a3"))
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--kind", choices=("hca", "csa"), default="hca")
    parser.add_argument(
        "--start-pos", type=int, default=DECODE_START_POS,
        help="Uniform target fixture start position for both decode requests.",
    )
    parser.add_argument("--workload", choices=tuple(ROUTED_WORKLOAD_COUNTS), default="balanced")
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=4, default=0, choices=(0, 1, 2, 4))
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    if args.kind == "hca":
        entry = decode_layer_compute_hca_test
        specs = build_hca_tensor_specs(args.start_pos, args.workload)
        golden_fn = golden_decode_layer_compute_hca
        compare_fn = {
            "x_next": ratio_reldiff(diff_thd=0.01, pct_thd=0.05),
            "weights": ratio_allclose(atol=2.5e-4, rtol=5e-3),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
        }
    else:
        entry = decode_layer_compute_csa_test
        specs = build_csa_tensor_specs(args.start_pos, args.workload)
        golden_fn = golden_decode_layer_compute_csa
        compare_fn = {
            "x_next": ratio_reldiff(diff_thd=0.02, pct_thd=0.08),
            "weights": ratio_allclose(atol=2.5e-4, rtol=5e-3),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3),
            "inner_compress_state": ratio_allclose(atol=1e-3, rtol=1e-3),
            "idx_kv_cache": ratio_allclose(atol=1, rtol=0),
            "idx_kv_scale": ratio_allclose(atol=1e-4, rtol=1e-3),
        }

    result = run_jit(
        fn=entry,
        specs=specs,
        golden_fn=golden_fn,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        golden_data=args.golden_data,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
            enable_dep_gen=True,
        ),
        rtol=1e-2,
        atol=1e-2,
        compare_fn=compare_fn,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
