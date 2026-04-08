# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-32B prefill Scope 2 — RoPE + KV cache update + causal attention.

Standalone test for the attention scope of the Qwen3-32B prefill layer,
with parameters aligned to qwen3_32b_prefill_tilelet.py.

For each batch element with seq_len_b tokens (processed per-token within
TOK_TILE=16 chunks):
  1. Apply RoPE to K projections (all KV heads) and store to cache.
  2. Copy V projections directly to cache.
  3. For each Q-head group:
     a. Gather Q heads and apply RoPE.
     b. Online flash-attention over the KV cache (up to ctx_len = pos + 1).
     c. Write normalised attention output.

a2a3 separation: every pl.incore() contains either vector-only or cube-only ops.
  - K/Q gather from GM into intermediate tensors is vector-only.
  - K/Q RoPE rotation + cache write is vector-only.
  - QK matmul and SV matmul are cube-only.
  - Softmax (scale, row_max, exp, row_sum) is vector-only.
  - Online rescale (m/l/o update) is vector-only.

A2/A3 textract restriction: sub-tile extraction (textract) from on-chip UB
tiles is not supported on A2/A3. K/Q projections are gathered per-head from
GM into intermediate GM tensors (k_group, q_group), then loaded as a whole
for RoPE — avoiding textract from large UB buffers.

Valid_shape handling aligned to decode_scope2 (fillpad approach):
  - K/V tiles loaded as full SEQ_TILE blocks without valid_shape.
  - valid_shape + fillpad applied only on QK scores before softmax.
  - row_sum uses BF16 round-trip precision (matching SV matmul weights).

Hardware TILELET / TILE sizing (at default HEAD_DIM=128):
  * K RoPE half-vectors [1, HEAD_DIM//2]      FP32 = [1,64]*4 = 256 B
  * Q attention group   [Q_HEAD_BATCH, HEAD_DIM] FP32 = [5,128]*4 = 2.5 KB
  * Attention K tile    [SEQ_TILE, HEAD_DIM]     BF16 = [64,128]*2 = 16 KB = MAX
  * Padded Q for cube   [Q_HEAD_PAD, HEAD_DIM]   BF16 = [16,128]*2 = 4 KB

Input projections are BF16; cos/sin tables are FP32; KV caches are BF16.
Output attention is BF16.
"""
from __future__ import annotations

import pypto.language as pl

BATCH = 16
MAX_SEQ = 96
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM  # 5120
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM  # 1024
Q_PER_KV = NUM_HEADS // NUM_KV_HEADS  # 5



ATTN_SCALE = 1.0 / (HEAD_DIM ** 0.5)

# Tiling constants (aligned to qwen3_32b_prefill_tilelet).
TOK_TILE = 16
Q_HEAD_BATCH = 5
Q_HEAD_PAD = 16       # padded Q rows for cube fractal alignment (M multiple of 16)
SEQ_TILE = 64


def build_prefill_attention_program(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
):
    hidden = num_heads * head_dim
    kv_hidden = num_kv_heads * head_dim
    q_per_kv = num_heads // num_kv_heads
    cache_rows = batch * num_kv_heads * max_seq
    half_dim = head_dim // 2
    q_groups = q_per_kv // Q_HEAD_BATCH
    total_q_groups = num_kv_heads * q_groups
    attn_scale = 1.0 / (head_dim ** 0.5)

    @pl.program
    class PrefillAttentionProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def prefill_attention(
            self,
            q_proj: pl.Tensor[[batch, max_seq, hidden], pl.BF16],
            k_proj: pl.Tensor[[batch, max_seq, kv_hidden], pl.BF16],
            v_proj: pl.Tensor[[batch, max_seq, kv_hidden], pl.BF16],
            seq_lens: pl.Tensor[[batch], pl.INT32],
            rope_cos: pl.Tensor[[max_seq, head_dim], pl.FP32],
            rope_sin: pl.Tensor[[max_seq, head_dim], pl.FP32],
            k_cache: pl.Tensor[[cache_rows, head_dim], pl.BF16],
            v_cache: pl.Tensor[[cache_rows, head_dim], pl.BF16],
            attn_out: pl.Out[pl.Tensor[[batch, max_seq, hidden], pl.BF16]],
        ) -> pl.Tensor[[batch, max_seq, hidden], pl.BF16]:
            for b in pl.parallel(0, batch, 1):
                seq_len_b = pl.tensor.read(seq_lens, [b])
                tok_blocks = (seq_len_b + TOK_TILE - 1) // TOK_TILE
                for p0_idx in pl.range(tok_blocks):
                    p0 = p0_idx * TOK_TILE
                    valid_tok = pl.min(TOK_TILE, seq_len_b - p0)

                    for ti in pl.range(valid_tok):
                        pos = p0 + ti
                        ctx_len = pos + 1
                        ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE

                        cos_row = pl.slice(
                            rope_cos, [1, head_dim], [pos, 0]
                        )
                        sin_row = pl.slice(
                            rope_sin, [1, head_dim], [pos, 0]
                        )
                        cos_lo = pl.slice(cos_row, [1, half_dim], [0, 0])
                        cos_hi = pl.slice(
                            cos_row, [1, half_dim], [0, half_dim]
                        )
                        sin_lo = pl.slice(sin_row, [1, half_dim], [0, 0])
                        sin_hi = pl.slice(
                            sin_row, [1, half_dim], [0, half_dim]
                        )

                        # A2/A3 textract restriction: sub-tile extraction from
                        # on-chip UB tiles is not supported. Use GM intermediate
                        # tensors for all cross-head data flow.
                        k_group = pl.create_tensor(
                            [num_kv_heads, head_dim], dtype=pl.FP32
                        )

                        # Vector-only: K gather from GM into k_group.
                        with pl.incore():
                            for ki in pl.range(num_kv_heads):
                                kv_col = ki * head_dim
                                k_group = pl.assemble(
                                    k_group,
                                    pl.cast(
                                        pl.reshape(
                                            pl.slice(
                                                k_proj,
                                                [1, 1, head_dim],
                                                [b, pos, kv_col],
                                            ),
                                            [1, head_dim],
                                        ),
                                        target_type=pl.FP32,
                                    ),
                                    [ki, 0],
                                )

                        # Vector-only: K RoPE → write rotated K to GM.
                        k_rot_gm = pl.create_tensor(
                            [num_kv_heads, head_dim], dtype=pl.BF16
                        )
                        with pl.incore():
                            k_lo = pl.slice(
                                k_group,
                                [num_kv_heads, half_dim],
                                [0, 0],
                            )
                            k_hi = pl.slice(
                                k_group,
                                [num_kv_heads, half_dim],
                                [0, half_dim],
                            )
                            k_rot = pl.concat(
                                pl.sub(
                                    pl.col_expand_mul(k_lo, cos_lo),
                                    pl.col_expand_mul(k_hi, sin_lo),
                                ),
                                pl.add(
                                    pl.col_expand_mul(k_hi, cos_hi),
                                    pl.col_expand_mul(k_lo, sin_hi),
                                ),
                            )
                            k_rot_gm = pl.cast(
                                k_rot, target_type=pl.BF16
                            )

                        # Vector-only: K/V cache write from GM.
                        with pl.incore():
                            for ki in pl.range(num_kv_heads):
                                cache_row = (
                                    b * num_kv_heads * max_seq
                                    + ki * max_seq
                                    + pos
                                )
                                k_cache = pl.assemble(
                                    k_cache,
                                    pl.slice(
                                        k_rot_gm,
                                        [1, head_dim],
                                        [ki, 0],
                                    ),
                                    [cache_row, 0],
                                )
                                v_cache = pl.assemble(
                                    v_cache,
                                    pl.reshape(
                                        pl.slice(
                                            v_proj,
                                            [1, 1, head_dim],
                                            [b, pos, ki * head_dim],
                                        ),
                                        [1, head_dim],
                                    ),
                                    [cache_row, 0],
                                )

                        attn_row = pl.create_tensor(
                            [1, hidden], dtype=pl.BF16
                        )

                        for gi in pl.parallel(0, total_q_groups, 1):
                            kvh = gi // q_groups
                            qg = gi - kvh * q_groups
                            q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH

                            q_padded = pl.create_tensor(
                                [Q_HEAD_PAD, head_dim], dtype=pl.BF16
                            )

                            # Vector-only: Q gather from GM into q_group.
                            q_group = pl.create_tensor(
                                [Q_HEAD_BATCH, head_dim], dtype=pl.FP32
                            )
                            with pl.incore():
                                for qi in pl.range(Q_HEAD_BATCH):
                                    q_col = (q_base + qi) * head_dim
                                    q_group = pl.assemble(
                                        q_group,
                                        pl.cast(
                                            pl.reshape(
                                                pl.slice(
                                                    q_proj,
                                                    [1, 1, head_dim],
                                                    [b, pos, q_col],
                                                ),
                                                [1, head_dim],
                                            ),
                                            target_type=pl.FP32,
                                        ),
                                        [qi, 0],
                                    )

                            # Vector-only: Q RoPE → write to GM.
                            q_rot_gm = pl.create_tensor(
                                [Q_HEAD_BATCH, head_dim], dtype=pl.BF16
                            )
                            with pl.incore():
                                q_lo = pl.slice(
                                    q_group,
                                    [Q_HEAD_BATCH, half_dim],
                                    [0, 0],
                                )
                                q_hi = pl.slice(
                                    q_group,
                                    [Q_HEAD_BATCH, half_dim],
                                    [0, half_dim],
                                )
                                q_rot = pl.concat(
                                    pl.sub(
                                        pl.col_expand_mul(
                                            q_lo, cos_lo
                                        ),
                                        pl.col_expand_mul(
                                            q_hi, sin_lo
                                        ),
                                    ),
                                    pl.add(
                                        pl.col_expand_mul(
                                            q_hi, cos_hi
                                        ),
                                        pl.col_expand_mul(
                                            q_lo, sin_hi
                                        ),
                                    ),
                                )
                                q_rot_gm = pl.cast(
                                    q_rot, target_type=pl.BF16
                                )

                            # Vector-only: assemble Q from GM into
                            # q_padded + init accumulators.
                            with pl.incore():
                                for qi in pl.range(Q_HEAD_BATCH):
                                    q_padded = pl.assemble(
                                        q_padded,
                                        pl.slice(
                                            q_rot_gm,
                                            [1, head_dim],
                                            [qi, 0],
                                        ),
                                        [qi, 0],
                                    )

                                oi = pl.full(
                                    [Q_HEAD_BATCH, head_dim],
                                    dtype=pl.FP32,
                                    value=0.0,
                                )
                                li_flat = pl.full(
                                    [1, Q_HEAD_BATCH],
                                    dtype=pl.FP32,
                                    value=0.0,
                                )
                                li = pl.reshape(
                                    li_flat, [Q_HEAD_BATCH, 1]
                                )
                                mi_flat = pl.full(
                                    [1, Q_HEAD_BATCH],
                                    dtype=pl.FP32,
                                    value=0.0,
                                )
                                mi = pl.reshape(
                                    mi_flat, [Q_HEAD_BATCH, 1]
                                )

                            for sb in pl.range(ctx_blocks):
                                s0 = sb * SEQ_TILE
                                valid_len = pl.min(
                                    SEQ_TILE, ctx_len - s0
                                )
                                cache_row0 = (
                                    b * num_kv_heads * max_seq
                                    + kvh * max_seq
                                    + s0
                                )

                                # Cube-only: QK matmul.
                                raw_scores_pad = pl.create_tensor(
                                    [Q_HEAD_PAD, SEQ_TILE],
                                    dtype=pl.FP32,
                                )
                                with pl.incore():
                                    k_tile = pl.slice(
                                        k_cache,
                                        [SEQ_TILE, head_dim],
                                        [cache_row0, 0],
                                    )
                                    raw_scores_pad = pl.matmul(
                                        q_padded,
                                        k_tile,
                                        b_trans=True,
                                        out_dtype=pl.FP32,
                                    )

                                # Vector-only: softmax with fillpad + BF16 round-trip.
                                exp_padded = pl.create_tensor(
                                    [Q_HEAD_PAD, SEQ_TILE],
                                    dtype=pl.BF16,
                                )
                                with pl.incore():
                                    scores_valid = pl.slice(
                                        raw_scores_pad,
                                        [Q_HEAD_BATCH, SEQ_TILE],
                                        [0, 0],
                                        valid_shape=[
                                            Q_HEAD_BATCH,
                                            valid_len,
                                        ],
                                    )
                                    scores_padded = pl.fillpad(
                                        scores_valid,
                                        pad_value=pl.PadValue.min,
                                    )
                                    scores = pl.mul(
                                        scores_padded, attn_scale
                                    )
                                    cur_mi = pl.row_max(scores)
                                    exp_scores = pl.exp(
                                        pl.row_expand_sub(
                                            scores, cur_mi
                                        )
                                    )
                                    exp_scores_bf16 = pl.cast(
                                        exp_scores,
                                        target_type=pl.BF16,
                                    )
                                    exp_scores_fp32 = pl.cast(
                                        exp_scores_bf16,
                                        target_type=pl.FP32,
                                    )
                                    cur_li = pl.row_sum(
                                        exp_scores_fp32
                                    )
                                    exp_padded = pl.assemble(
                                        exp_padded,
                                        exp_scores_bf16,
                                        [0, 0],
                                    )

                                # Cube-only: SV matmul.
                                oi_tmp_pad = pl.create_tensor(
                                    [Q_HEAD_PAD, head_dim],
                                    dtype=pl.FP32,
                                )
                                with pl.incore():
                                    v_tile = pl.slice(
                                        v_cache,
                                        [SEQ_TILE, head_dim],
                                        [cache_row0, 0],
                                    )
                                    oi_tmp_pad = pl.matmul(
                                        exp_padded,
                                        v_tile,
                                        out_dtype=pl.FP32,
                                    )

                                # Vector-only: online rescale.
                                with pl.incore():
                                    oi_tmp = pl.slice(
                                        oi_tmp_pad,
                                        [Q_HEAD_BATCH, head_dim],
                                        [0, 0],
                                    )
                                    if sb == 0:
                                        oi = oi_tmp
                                        li = cur_li
                                        mi = cur_mi
                                    else:
                                        mi_new = pl.maximum(
                                            mi, cur_mi
                                        )
                                        alpha = pl.exp(
                                            pl.sub(mi, mi_new)
                                        )
                                        beta = pl.exp(
                                            pl.sub(cur_mi, mi_new)
                                        )
                                        li = pl.add(
                                            pl.mul(alpha, li),
                                            pl.mul(beta, cur_li),
                                        )
                                        oi = pl.add(
                                            pl.row_expand_mul(
                                                oi, alpha
                                            ),
                                            pl.row_expand_mul(
                                                oi_tmp, beta
                                            ),
                                        )
                                        mi = mi_new

                            # Vector-only: ctx = oi/li, scatter into attn_row.
                            with pl.incore():
                                ctx = pl.row_expand_div(oi, li)
                                ctx_flat = pl.reshape(
                                    ctx,
                                    [1, Q_HEAD_BATCH * head_dim],
                                )
                                attn_row = pl.assemble(
                                    attn_row,
                                    pl.cast(
                                        ctx_flat,
                                        target_type=pl.BF16,
                                    ),
                                    [0, q_base * head_dim],
                                )

                        # Write attn_row into attn_out GM tensor.
                        attn_out = pl.assemble(
                            attn_out, attn_row, [b, pos, 0]
                        )

            return attn_out

    return PrefillAttentionProgram


def build_tensor_specs(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
):
    import torch
    from pypto.runtime import TensorSpec

    hidden = num_heads * head_dim
    kv_hidden = num_kv_heads * head_dim
    cache_rows = batch * num_kv_heads * max_seq

    def init_q_proj():
        return torch.rand(batch, max_seq, hidden) - 0.5

    def init_k_proj():
        return torch.rand(batch, max_seq, kv_hidden) - 0.5

    def init_v_proj():
        return torch.rand(batch, max_seq, kv_hidden) - 0.5

    def init_seq_lens():
        n_blocks = max_seq // TOK_TILE
        blocks = torch.randint(1, n_blocks + 1, (batch,), dtype=torch.int32)
        return blocks * TOK_TILE

    def init_rope_cos():
        return torch.rand(max_seq, head_dim) - 0.5

    def init_rope_sin():
        return torch.rand(max_seq, head_dim) - 0.5

    def init_k_cache():
        return torch.rand(cache_rows, head_dim) - 0.5

    def init_v_cache():
        return torch.rand(cache_rows, head_dim) - 0.5

    return [
        TensorSpec(
            "q_proj",
            [batch, max_seq, hidden],
            torch.bfloat16,
            init_value=init_q_proj,
        ),
        TensorSpec(
            "k_proj",
            [batch, max_seq, kv_hidden],
            torch.bfloat16,
            init_value=init_k_proj,
        ),
        TensorSpec(
            "v_proj",
            [batch, max_seq, kv_hidden],
            torch.bfloat16,
            init_value=init_v_proj,
        ),
        TensorSpec(
            "seq_lens", [batch], torch.int32, init_value=init_seq_lens
        ),
        TensorSpec(
            "rope_cos",
            [max_seq, head_dim],
            torch.float32,
            init_value=init_rope_cos,
        ),
        TensorSpec(
            "rope_sin",
            [max_seq, head_dim],
            torch.float32,
            init_value=init_rope_sin,
        ),
        TensorSpec(
            "k_cache",
            [cache_rows, head_dim],
            torch.bfloat16,
            init_value=init_k_cache,
        ),
        TensorSpec(
            "v_cache",
            [cache_rows, head_dim],
            torch.bfloat16,
            init_value=init_v_cache,
        ),
        TensorSpec(
            "attn_out",
            [batch, max_seq, hidden],
            torch.bfloat16,
            is_output=True,
        ),
    ]


def golden_prefill_attention(tensors, params):
    """PyTorch reference matching kernel BF16 precision path.

    Simulates the kernel's tiled online-softmax with BF16 matmuls:
      - Q/K projections cast from BF16 to FP32 for RoPE arithmetic.
      - Q cast to BF16 after RoPE (matching kernel QK matmul input).
      - QK/SV matmuls use BF16 inputs with FP32 accumulation.
      - BF16 round-trip on exp_scores before row_sum (matching kernel).
      - Full SEQ_TILE K/V loads with fillpad masking on scores.
    """
    import math

    import torch

    q_proj = tensors["q_proj"]     # [B, S, H], BF16
    k_proj = tensors["k_proj"]     # [B, S, KV_H], BF16
    v_proj = tensors["v_proj"]     # [B, S, KV_H], BF16
    seq_lens = tensors["seq_lens"]
    rope_cos = tensors["rope_cos"]
    rope_sin = tensors["rope_sin"]
    k_cache = tensors["k_cache"].clone()
    v_cache = tensors["v_cache"].clone()

    batch = q_proj.shape[0]
    hidden = q_proj.shape[2]
    kv_hidden = k_proj.shape[2]
    head_dim = rope_cos.shape[1]
    max_seq = rope_cos.shape[0]
    num_kv_heads = kv_hidden // head_dim
    num_heads = hidden // head_dim
    q_per_kv = num_heads // num_kv_heads
    q_groups = q_per_kv // Q_HEAD_BATCH
    half = head_dim // 2
    scale = 1.0 / math.sqrt(head_dim)

    attn_out = torch.zeros(batch, max_seq, hidden, dtype=torch.float32)

    for b in range(batch):
        seq_len_b = seq_lens[b].item()
        for p0 in range(0, seq_len_b, TOK_TILE):
            valid_tok = min(TOK_TILE, seq_len_b - p0)
            for ti in range(valid_tok):
                pos = p0 + ti
                ctx_len = pos + 1
                ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE

                cos_row = rope_cos[pos : pos + 1, :]
                sin_row = rope_sin[pos : pos + 1, :]
                cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
                sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

                # K RoPE: per KV head — cast BF16→FP32 for arithmetic.
                k_row = k_proj[b, pos, :].float().view(num_kv_heads, head_dim)
                k_lo, k_hi = k_row[:, :half], k_row[:, half:]
                k_rot = torch.cat(
                    [
                        k_lo * cos_lo - k_hi * sin_lo,
                        k_hi * cos_hi + k_lo * sin_hi,
                    ],
                    dim=-1,
                )

                # Update caches.
                for ki in range(num_kv_heads):
                    cr = (
                        b * num_kv_heads * max_seq
                        + ki * max_seq
                        + pos
                    )
                    k_cache[cr, :] = k_rot[ki].to(torch.bfloat16)
                    v_cache[cr, :] = v_proj[
                        b, pos, ki * head_dim : (ki + 1) * head_dim
                    ]

                # Q RoPE: per Q head — cast BF16→FP32 for arithmetic.
                q_row = q_proj[b, pos, :].float().view(num_heads, head_dim)
                q_lo, q_hi = q_row[:, :half], q_row[:, half:]
                q_rot = torch.cat(
                    [
                        q_lo * cos_lo - q_hi * sin_lo,
                        q_hi * cos_hi + q_lo * sin_hi,
                    ],
                    dim=-1,
                )

                # Grouped-query attention (tiled online softmax).
                for kvh in range(num_kv_heads):
                    for qg in range(q_groups):
                        q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                        q_grp = q_rot[
                            q_base : q_base + Q_HEAD_BATCH, :
                        ]
                        # Match kernel: Q cast to BF16 for QK matmul.
                        q_grp_bf16 = q_grp.to(torch.bfloat16)

                        oi = torch.zeros(
                            Q_HEAD_BATCH,
                            head_dim,
                            dtype=torch.float32,
                        )
                        li = torch.zeros(
                            Q_HEAD_BATCH, 1, dtype=torch.float32
                        )
                        mi = torch.zeros(
                            Q_HEAD_BATCH, 1, dtype=torch.float32
                        )

                        for sb in range(ctx_blocks):
                            s0 = sb * SEQ_TILE
                            valid_len = min(SEQ_TILE, ctx_len - s0)
                            cb = (
                                b * num_kv_heads * max_seq
                                + kvh * max_seq
                                + s0
                            )

                            # Full SEQ_TILE K/V tiles as BF16.
                            k_tile = k_cache[cb : cb + SEQ_TILE, :]
                            v_tile = v_cache[cb : cb + SEQ_TILE, :]

                            # QK matmul: BF16 * BF16 → FP32.
                            raw_scores = (
                                q_grp_bf16.float() @ k_tile.float().T
                            )

                            # Fillpad invalid positions.
                            if valid_len < SEQ_TILE:
                                raw_scores[:, valid_len:] = (
                                    torch.finfo(torch.float32).min
                                )
                            scores = raw_scores * scale

                            # Online softmax with BF16 round-trip.
                            cur_mi = scores.max(
                                dim=-1, keepdim=True
                            ).values
                            exp_scores = torch.exp(scores - cur_mi)
                            exp_scores_bf16 = exp_scores.to(
                                torch.bfloat16
                            )
                            cur_li = (
                                exp_scores_bf16.float().sum(
                                    dim=-1, keepdim=True
                                )
                            )

                            # SV matmul: BF16 * BF16 → FP32.
                            oi_tmp = (
                                exp_scores_bf16.float()
                                @ v_tile.float()
                            )

                            if sb == 0:
                                oi = oi_tmp
                                li = cur_li
                                mi = cur_mi
                            else:
                                mi_new = torch.maximum(mi, cur_mi)
                                alpha = torch.exp(mi - mi_new)
                                beta = torch.exp(cur_mi - mi_new)
                                li = alpha * li + beta * cur_li
                                oi = oi * alpha + oi_tmp * beta
                                mi = mi_new

                        ctx = oi / li
                        for qi in range(Q_HEAD_BATCH):
                            qh = q_base + qi
                            attn_out[
                                b,
                                pos,
                                qh * head_dim : (qh + 1) * head_dim,
                            ] = ctx[qi]

    tensors["attn_out"][:] = attn_out.to(torch.bfloat16)


def compile_and_run(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    platform: str = "a2a3",
    device_id: int = 0,
    dump_passes: bool = True,
    enable_profiling: bool = False,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    backend = (
        BackendType.Ascend950
        if platform.startswith("a5")
        else BackendType.Ascend910B
    )

    program = build_prefill_attention_program(
        batch=batch,
        max_seq=max_seq,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    tensor_specs = build_tensor_specs(
        batch=batch,
        max_seq=max_seq,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden_prefill_attention,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=2e-3,
            atol=2e-3,
            strategy=OptimizationStrategy.Default,
            dump_passes=dump_passes,
            backend_type=backend,
            enable_profiling=enable_profiling,
        ),
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--enable-profiling", action="store_true", default=False
    )
    args = parser.parse_args()

    result = compile_and_run(
        platform=args.platform,
        device_id=args.device,
        enable_profiling=args.enable_profiling,
    )
    if not result.passed:
        if result.error:
            print(f"Result: {result.error}")
        raise SystemExit(1)
