# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=4
"""DeepSeek-V4 decode CSA heads, DSA-CP output exchange, projection, and reduce-scatter."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import BLOCK_SIZE, FLASH as M, INT8_AMAX_EPS, INT8_SCALE_MAX
from decode_attention_cp import (
    ATTENTION_WINDOW_ROWS,
    GROUP_T_PAD,
    LOCAL_O_GROUPS,
    LOCAL_T,
    LOCAL_T_PAD,
    O_GROUP_IN,
    O_WINDOW_ROWS,
    SP_SIZE,
    attention_token_head_all_to_all_step,
    o_projection_reduce_scatter_step,
)
from decode_o_projection_cp import LOCAL_O_WIDTH, decode_o_projection_cp
from decode_sparse_attn_csa import (
    B_DYN,
    CMP_BLOCK_NUM_DYN,
    CMP_MAX_BLOCKS,
    HEAD_DIM,
    H,
    INDEXER_SCORE_LEN,
    ORI_BLOCK_NUM_DYN,
    O_GROUPS,
    ROPE_DIM,
    S,
    SOFTMAX_SCALE,
    T_DYN,
    T_PAD,
    WIN,
    sparse_attn_csa_heads,
)


# model config
D = M.hidden_size
O_LORA = M.o_lora_rank
HEADS_PER_GROUP = H // O_GROUPS

# fixture
SUBCAPACITY_LOCAL_T = LOCAL_T - 8
FIXTURE_CACHE_ROWS = SP_SIZE * LOCAL_T
FIXTURE_CACHE_BLOCKS = (FIXTURE_CACHE_ROWS + BLOCK_SIZE - 1) // BLOCK_SIZE
FIXTURE_OUTPUT_SENTINEL = -7.0

if T_PAD != LOCAL_T_PAD:
    raise ValueError(f"CSA head rows {T_PAD} must match the DSA-CP local capacity {LOCAL_T_PAD}")
if LOCAL_T % 8 != 0 or SUBCAPACITY_LOCAL_T < 8 or SUBCAPACITY_LOCAL_T % 8 != 0:
    raise ValueError("CSA output fixtures require max and subcapacity token counts divisible by 8")
if LOCAL_T % S != 0 or SUBCAPACITY_LOCAL_T % S != 0:
    raise ValueError(f"CSA token counts must be divisible by decode sequence length {S}")


@pl.jit
def decode_csa_output_cp(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_MAX_BLOCKS], pl.INT32],
    idx_topk: pl.Tensor[[T_DYN, INDEXER_SCORE_LEN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN, 1], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    o_local: pl.InOut[pl.Tensor[[LOCAL_T_PAD, D], pl.BF16]],
    attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    attention_signal: pld.DistributedTensor[[SP_SIZE, 1], pl.INT32],
    o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.FP32],
    o_signal: pld.DistributedTensor[[SP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    sp_rank: pl.Scalar[pl.INT32],
):
    """Run the decode CSA attention-to-local-hidden DSA-CP path on one rank."""
    q.bind_dynamic(0, T_DYN)
    ori_kv.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    window_swa_indices.bind_dynamic(0, T_DYN)
    cmp_kv.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    cmp_block_table.bind_dynamic(0, B_DYN)
    idx_topk.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    local_t = pl.tensor.dim(q, 0)

    attention_grouped = pl.create_tensor([O_GROUPS * LOCAL_T_PAD, O_GROUP_IN], dtype=pl.BF16)
    attention_grouped, heads_tid = sparse_attn_csa_heads(
        q,
        ori_kv, window_swa_indices,
        cmp_kv, cmp_block_table, idx_topk,
        position_ids, attn_sink,
        freqs_cos, freqs_sin,
        attention_grouped,
    )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_heads_complete", deps=[heads_tid]):
        head_anchor = pl.read(attention_grouped, [0, 0])
        pl.write(attention_grouped, [0, 0], head_anchor)

    attention_local_flat = pl.create_tensor([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
    attention_local_flat, attention_signal = attention_token_head_all_to_all_step(
        attention_grouped, attention_local_flat,
        attention_window, attention_signal,
        group_base, sp_rank, local_t,
    )

    attention_local_groups = pl.reshape(attention_local_flat, [LOCAL_O_GROUPS, GROUP_T_PAD, O_GROUP_IN])
    o_partial = pl.create_tensor([GROUP_T_PAD, D], dtype=pl.FP32)
    o_partial, projection_tid = decode_o_projection_cp(
        attention_local_groups,
        wo_a, wo_b, wo_b_scale,
        local_t, o_partial,
    )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_o_projection_complete", deps=[projection_tid]):
        projection_anchor = pl.read(o_partial, [0, 0])
        pl.write(o_partial, [0, 0], projection_anchor)

    o_local, o_signal = o_projection_reduce_scatter_step(
        o_partial, o_local,
        o_window, o_signal,
        group_base, sp_rank, local_t,
    )
    return o_local, attention_signal, o_signal


@pl.jit.host
def l3_decode_csa_output_cp(
    q: pl.Tensor[[SP_SIZE, T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[SP_SIZE, ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[SP_SIZE, T_DYN, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[SP_SIZE, CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[SP_SIZE, B_DYN, CMP_MAX_BLOCKS], pl.INT32],
    idx_topk: pl.Tensor[[SP_SIZE, T_DYN, INDEXER_SCORE_LEN], pl.INT32],
    position_ids: pl.Tensor[[SP_SIZE, T_DYN, 1], pl.INT32],
    attn_sink: pl.Tensor[[SP_SIZE, H], pl.FP32],
    freqs_cos: pl.Tensor[[SP_SIZE, T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[SP_SIZE, T_DYN, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[SP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[SP_SIZE, D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[SP_SIZE, D], pl.FP32],
    o_local: pl.InOut[pl.Tensor[[SP_SIZE, LOCAL_T_PAD, D], pl.BF16]],
):
    """Launch one CSA output-half orchestration on each physical SP rank."""
    q.bind_dynamic(1, T_DYN)
    ori_kv.bind_dynamic(1, ORI_BLOCK_NUM_DYN)
    window_swa_indices.bind_dynamic(1, T_DYN)
    cmp_kv.bind_dynamic(1, CMP_BLOCK_NUM_DYN)
    cmp_block_table.bind_dynamic(1, B_DYN)
    idx_topk.bind_dynamic(1, T_DYN)
    position_ids.bind_dynamic(1, T_DYN)
    freqs_cos.bind_dynamic(1, T_DYN)
    freqs_sin.bind_dynamic(1, T_DYN)

    attention_window_buf = pld.alloc_window_buffer([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
    attention_signal_buf = pld.alloc_window_buffer([SP_SIZE, 1], dtype=pl.INT32)
    o_window_buf = pld.alloc_window_buffer([O_WINDOW_ROWS, D], dtype=pl.FP32)
    o_signal_buf = pld.alloc_window_buffer([SP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        attention_window = pld.window(attention_window_buf, [ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
        attention_signal = pld.window(attention_signal_buf, [SP_SIZE, 1], dtype=pl.INT32)
        o_window = pld.window(o_window_buf, [O_WINDOW_ROWS, D], dtype=pl.FP32)
        o_signal = pld.window(o_signal_buf, [SP_SIZE, 1], dtype=pl.INT32)
        decode_csa_output_cp(
            q[rank],
            ori_kv[rank], window_swa_indices[rank],
            cmp_kv[rank], cmp_block_table[rank], idx_topk[rank],
            position_ids[rank], attn_sink[rank],
            freqs_cos[rank], freqs_sin[rank],
            wo_a[rank], wo_b[rank], wo_b_scale[rank],
            o_local[rank],
            attention_window, attention_signal,
            o_window, o_signal,
            0, rank,
            device=rank,
        )


def build_tensor_specs(local_t):
    """Build a controlled four-rank CSA output-half fixture."""
    import torch

    from golden import TensorSpec

    if local_t < 8 or local_t > LOCAL_T or local_t % 8 != 0:
        raise ValueError(f"local_t must be a multiple of 8 in [8, {LOCAL_T}], got {local_t}")
    local_batch = local_t // S

    def init_q():
        return torch.zeros(SP_SIZE, local_t, H, HEAD_DIM, dtype=torch.bfloat16)

    def init_ori_kv():
        cache = torch.zeros(SP_SIZE, FIXTURE_CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
        token_values = torch.arange(FIXTURE_CACHE_BLOCKS * BLOCK_SIZE, dtype=torch.float32) / 128.0 + 1.0
        cache[:, :, :, 0, 0] = token_values.reshape(1, FIXTURE_CACHE_BLOCKS, BLOCK_SIZE).to(torch.bfloat16)
        return cache

    def init_window_swa_indices():
        indices = torch.full((SP_SIZE, local_t, WIN), -1, dtype=torch.int32)
        for rank in range(SP_SIZE):
            token_base = rank * LOCAL_T
            indices[rank, :, 0] = torch.arange(token_base, token_base + local_t, dtype=torch.int32)
        return indices

    def init_cmp_kv():
        return torch.zeros(SP_SIZE, FIXTURE_CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)

    def init_cmp_block_table():
        return torch.zeros(SP_SIZE, local_batch, CMP_MAX_BLOCKS, dtype=torch.int32)

    def init_idx_topk():
        return torch.full((SP_SIZE, local_t, INDEXER_SCORE_LEN), -1, dtype=torch.int32)

    def init_position_ids():
        return torch.zeros(SP_SIZE, local_t, 1, dtype=torch.int32)

    def init_attn_sink():
        return torch.zeros(SP_SIZE, H, dtype=torch.float32)

    def init_freqs_cos():
        return torch.ones(SP_SIZE, local_t, ROPE_DIM, dtype=torch.bfloat16)

    def init_freqs_sin():
        return torch.zeros(SP_SIZE, local_t, ROPE_DIM, dtype=torch.bfloat16)

    def init_wo_a():
        weight = torch.zeros(SP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN, dtype=torch.bfloat16)
        for rank in range(SP_SIZE):
            for local_group in range(LOCAL_O_GROUPS):
                global_group = rank * LOCAL_O_GROUPS + local_group
                for head in range(HEADS_PER_GROUP):
                    head_sign = 1.0 if head % 2 == 0 else -1.0
                    coefficient = head_sign * (global_group + 1) / 8.0
                    weight[rank, local_group, head, head * HEAD_DIM] = coefficient
        return weight

    def init_wo_b():
        weight = torch.zeros(SP_SIZE, D, LOCAL_O_WIDTH, dtype=torch.int8)
        for rank in range(SP_SIZE):
            output_channels = torch.arange(rank, D, SP_SIZE)
            features = (output_channels // SP_SIZE) % (LOCAL_O_GROUPS * HEADS_PER_GROUP)
            local_groups = features // HEADS_PER_GROUP
            heads = features % HEADS_PER_GROUP
            input_channels = local_groups * O_LORA + heads
            output_signs = torch.where((output_channels // 64) % 2 == 0, 1, -1).to(torch.int8)
            weight[rank, output_channels, input_channels] = output_signs
        return weight

    def init_wo_b_scale():
        channel_scale = torch.arange(D, dtype=torch.int32).remainder(4).to(torch.float32) * 0.25 + 0.5
        return channel_scale.reshape(1, D).expand(SP_SIZE, D).clone()

    def init_o_local():
        return torch.full((SP_SIZE, LOCAL_T_PAD, D), FIXTURE_OUTPUT_SENTINEL, dtype=torch.bfloat16)

    cache_shape = [SP_SIZE, FIXTURE_CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM]
    block_table_shape = [SP_SIZE, local_batch, CMP_MAX_BLOCKS]
    return [
        TensorSpec("q", [SP_SIZE, local_t, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", cache_shape, torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("window_swa_indices", [SP_SIZE, local_t, WIN], torch.int32, init_value=init_window_swa_indices),
        TensorSpec("cmp_kv", cache_shape, torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", block_table_shape, torch.int32, init_value=init_cmp_block_table),
        TensorSpec("idx_topk", [SP_SIZE, local_t, INDEXER_SCORE_LEN], torch.int32, init_value=init_idx_topk),
        TensorSpec("position_ids", [SP_SIZE, local_t, 1], torch.int32, init_value=init_position_ids),
        TensorSpec("attn_sink", [SP_SIZE, H], torch.float32, init_value=init_attn_sink),
        TensorSpec("freqs_cos", [SP_SIZE, local_t, ROPE_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [SP_SIZE, local_t, ROPE_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("wo_a", [SP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [SP_SIZE, D, LOCAL_O_WIDTH], torch.int8, init_value=init_wo_b),
        TensorSpec("wo_b_scale", [SP_SIZE, D], torch.float32, init_value=init_wo_b_scale),
        TensorSpec("o_local", [SP_SIZE, LOCAL_T_PAD, D], torch.bfloat16, init_value=init_o_local, is_output=True),
    ]


def golden_decode_csa_output_cp(tensors):
    """Compute the controlled CSA heads, output shards, and token-owner reduction."""
    import torch

    local_t = tensors["q"].shape[1]
    group_t = SP_SIZE * local_t
    q_first = tensors["q"][:, :, :, 0].float()
    window_indices = tensors["window_swa_indices"][:, :, 0].long()
    kv_rows = tensors["ori_kv"][:, :, :, 0, 0].reshape(SP_SIZE, -1).float()
    kv_first = torch.gather(kv_rows, 1, window_indices)
    scores = q_first * kv_first.unsqueeze(-1) * SOFTMAX_SCALE
    sink = tensors["attn_sink"].float().reshape(SP_SIZE, 1, H)
    head_first = kv_first.unsqueeze(-1) / (1.0 + torch.exp(sink - scores))
    heads = torch.zeros(SP_SIZE, local_t, H, HEAD_DIM, dtype=torch.bfloat16)
    heads[:, :, :, 0] = head_first.to(torch.bfloat16)
    grouped = heads.reshape(SP_SIZE, local_t, O_GROUPS, O_GROUP_IN).permute(0, 2, 1, 3)

    received = torch.zeros(SP_SIZE, LOCAL_O_GROUPS, group_t, O_GROUP_IN, dtype=torch.bfloat16)
    for destination_rank in range(SP_SIZE):
        for local_group in range(LOCAL_O_GROUPS):
            global_group = destination_rank * LOCAL_O_GROUPS + local_group
            received[destination_rank, local_group] = grouped[:, global_group].reshape(group_t, O_GROUP_IN)

    partials = []
    for rank in range(SP_SIZE):
        o_a = torch.zeros(LOCAL_O_GROUPS, group_t, O_LORA, dtype=torch.float32)
        for local_group in range(LOCAL_O_GROUPS):
            for head in range(HEADS_PER_GROUP):
                input_col = head * HEAD_DIM
                coefficient = tensors["wo_a"][rank, local_group, head, input_col].float()
                o_a[local_group, :, head] = received[rank, local_group, :, input_col].float() * coefficient
        row_amax = o_a.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
        scale_q = INT8_SCALE_MAX / row_amax
        o_a_i8 = torch.round(o_a * scale_q).to(torch.int32).to(torch.float16).to(torch.int8)
        scale_dq = 1.0 / scale_q
        wo_b = tensors["wo_b"][rank].reshape(D, LOCAL_O_GROUPS, O_LORA)
        o_partial = torch.zeros(group_t, D, dtype=torch.float32)
        for local_group in range(LOCAL_O_GROUPS):
            group_i32 = o_a_i8[local_group, :, :HEADS_PER_GROUP].to(torch.int32)
            weight_i32 = wo_b[:, local_group, :HEADS_PER_GROUP].to(torch.int32)
            group_partial = group_i32 @ weight_i32.T
            o_partial = o_partial + group_partial.float() * scale_dq[local_group]
        weight_scale = tensors["wo_b_scale"][rank].float().reshape(1, D)
        partials.append(o_partial * weight_scale)

    reduced = partials[0]
    for rank in range(1, SP_SIZE):
        reduced = reduced + partials[rank]
    tensors["o_local"].fill_(FIXTURE_OUTPUT_SENTINEL)
    for owner_rank in range(SP_SIZE):
        source_row = owner_rank * local_t
        tensors["o_local"][owner_rank, :local_t] = reduced[source_row : source_row + local_t].to(torch.bfloat16)


def build_o_local_compare(local_t):
    """Compare valid token rows and require the poisoned capacity tail to survive."""
    import torch

    from golden import ratio_allclose

    prefix_compare = ratio_allclose(
        atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0,
        valid_rows=local_t, valid_axis=1,
    )

    def compare(actual, expected, **kwargs):
        actual_tail = actual[:, local_t:]
        expected_tail = expected[:, local_t:]
        if not torch.equal(actual_tail, expected_tail):
            mismatch_count = int((actual_tail != expected_tail).sum().item())
            return False, f"    inactive token tail mismatch count={mismatch_count}"
        return prefix_compare(actual, expected, **kwargs)

    compare.__name__ = f"csa_output_prefix_and_tail(local_t={local_t})"
    return compare


if __name__ == "__main__":
    import argparse

    from golden import run_jit
    from pypto.ir.distributed_compiled_program import DistributedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=("a2a3", "a2a3sim", "a5", "a5sim"))
    parser.add_argument("-d", "--device", type=str, default=",".join(str(rank) for rank in range(SP_SIZE)))
    parser.add_argument("--case", choices=("all", "max", "subcapacity"), default="subcapacity")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) != SP_SIZE:
        parser.error(f"need exactly {SP_SIZE} devices, got {device_ids}")

    case_local_t = {"max": LOCAL_T, "subcapacity": SUBCAPACITY_LOCAL_T}
    selected_cases = tuple(case_local_t) if args.case == "all" else (args.case,)
    for case in selected_cases:
        local_t = case_local_t[case]
        result = run_jit(
            fn=l3_decode_csa_output_cp,
            specs=build_tensor_specs(local_t),
            golden_fn=golden_decode_csa_output_cp,
            compile_only=args.compile_only,
            compile_cfg=dict(
                dump_passes=args.dump_passes,
                distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
            ),
            runtime_cfg=dict(platform=args.platform),
            compare_fn={"o_local": build_o_local_compare(local_t)},
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
