# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-32B prefill Scope 1 — input RMSNorm + Q/K/V projection.

Standalone test for the RMSNorm + projection scope of the Qwen3-32B prefill layer,
with model dimensions from Qwen3-32B. BATCH/MAX_SEQ reduced for device
memory debugging (full: BATCH=16, MAX_SEQ=4096).

For each batch element with seq_len_b tokens (processed in TOK_TILE=16 chunks):
  1. Compute RMSNorm of the input hidden states token tile.
  2. Project to Q (hidden_size), K (kv_hidden), V (kv_hidden).
     Projection results are cast from FP32 accumulation to BF16 output,
     aligned to the prefill_tilelet convention and reducing GM bandwidth.

TOK_TILE=16 is the minimum value that satisfies both hardware constraints:
  - Vector tilelet: [1, TOK_TILE] FP32 >= 32 B (1 block) -> TOK_TILE >= 8
  - Cube fractal:   matmul M-dim must be a multiple of 16  -> TOK_TILE >= 16
This matches decode's BATCH_TILE=16 exactly.

API alignment: this file mirrors qwen3_32b_decode_scope1.py as closely as
possible so that the only remaining difference is the 3D-tensor / variable-
length token loop.  Specifically:
  - RMSNorm uses pl.incore() (not pl.auto_incore()) with the same
    [1, TOK_TILE] -> reshape [TOK_TILE, 1] layout for partial_sq, and the
    same variance-based (not rsqrt-based) norm factor.
  - Q/K/V projections use pl.matmul + pl.matmul_acc (not pl.add + pl.matmul),
    with FP32 accumulation cast to BF16 output (matching tilelet convention).
  - Zero-init uses pl.full (not pl.create_tensor + pl.mul(x, 0.0)).

Hardware TILELET / TILE sizing (identical to decode_scope1 at BATCH_TILE=16):
  * RMSNorm accumulator [1, TOK_TILE]        FP32 = [1,16]*4  = 64 B (1xTILELET)
  * Normed tile chunk   [TOK_TILE, K_CHUNK]   BF16 = [16,128]*2 = 4 KB
  * Q/K/V accumulator   [TOK_TILE, OUT_CHUNK] FP32 = [16,64]*4 = 4 KB (2xTILELET)
  * Q/K/V output        [TOK_TILE, OUT_CHUNK] BF16 = [16,64]*2 = 2 KB (cast from acc)
  * Weight tiles         [K_CHUNK, OUT_CHUNK]  BF16 = [128,64]*2 = 16 KB = MAX
"""
from __future__ import annotations

import pypto.language as pl


# Reduced from full Qwen3-32B (BATCH=16, MAX_SEQ=4096) for device memory limits.
BATCH = 16
MAX_SEQ = 96
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM  # 5120
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM  # 1024
INTERMEDIATE = 25600


EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

# Tiling constants (aligned to qwen3_32b_prefill_tilelet).
K_CHUNK = 128
Q_OUT_CHUNK = 64
KV_OUT_CHUNK = 64
TOK_TILE = 16


def build_prefill_projection_program(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
):
    hidden = hidden_size
    kv_hidden = num_kv_heads * head_dim
    hidden_blocks = hidden // K_CHUNK
    q_out_blocks = hidden // Q_OUT_CHUNK
    kv_out_blocks = kv_hidden // KV_OUT_CHUNK
    hidden_inv = 1.0 / hidden

    @pl.program
    class PrefillProjectionProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def prefill_projection(
            self,
            hidden_states: pl.Tensor[[batch, max_seq, hidden], pl.BF16],
            seq_lens: pl.Tensor[[batch], pl.INT32],
            input_rms_weight: pl.Tensor[[1, hidden], pl.FP32],
            wq: pl.Tensor[[hidden, hidden], pl.BF16],
            wk: pl.Tensor[[hidden, kv_hidden], pl.BF16],
            wv: pl.Tensor[[hidden, kv_hidden], pl.BF16],
            q_proj: pl.Out[pl.Tensor[[batch, max_seq, hidden], pl.BF16]],
            k_proj: pl.Out[pl.Tensor[[batch, max_seq, kv_hidden], pl.BF16]],
            v_proj: pl.Out[pl.Tensor[[batch, max_seq, kv_hidden], pl.BF16]],
        ) -> tuple[
            pl.Tensor[[batch, max_seq, hidden], pl.BF16],
            pl.Tensor[[batch, max_seq, kv_hidden], pl.BF16],
            pl.Tensor[[batch, max_seq, kv_hidden], pl.BF16],
        ]:
            for b in pl.range(0, batch):
                seq_len_b = pl.tensor.read(seq_lens, [b])
                tok_blocks = (seq_len_b + TOK_TILE - 1) // TOK_TILE
                for p0_idx in pl.range(tok_blocks):
                    p0 = p0_idx * TOK_TILE
                    valid_tok = pl.min(TOK_TILE, seq_len_b - p0)

                    normed_tile = pl.create_tensor(
                        [TOK_TILE, hidden], dtype=pl.BF16
                    )

                    # Stage 1: RMSNorm (vector ops only in pl.incore).
                    # Mirrors decode_scope1: accumulate in [1, TOK_TILE],
                    # then reshape to [TOK_TILE, 1] for row_expand_mul.
                    with pl.incore():
                        partial_sq = pl.full(
                            [1, TOK_TILE], dtype=pl.FP32, value=0.0
                        )
                        for kb in pl.range(hidden_blocks):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.reshape(
                                pl.cast(
                                    pl.slice(
                                        hidden_states,
                                        [1, TOK_TILE, K_CHUNK],
                                        [b, p0, k0],
                                        valid_shape=[1, valid_tok, K_CHUNK],
                                    ),
                                    target_type=pl.FP32,
                                ),
                                [TOK_TILE, K_CHUNK],
                            )
                            partial_sq = pl.add(
                                partial_sq,
                                pl.reshape(
                                    pl.row_sum(pl.mul(x_chunk, x_chunk)),
                                    [1, TOK_TILE],
                                ),
                            )

                        # Compute variance in [1, TOK_TILE], then reshape to
                        # [TOK_TILE, 1] for row_expand_mul broadcasting.
                        variance = pl.reshape(
                            pl.add(
                                pl.mul(partial_sq, hidden_inv), EPS
                            ),
                            [TOK_TILE, 1],
                        )

                        for kb in pl.range(hidden_blocks):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.reshape(
                                pl.cast(
                                    pl.slice(
                                        hidden_states,
                                        [1, TOK_TILE, K_CHUNK],
                                        [b, p0, k0],
                                        valid_shape=[1, valid_tok, K_CHUNK],
                                    ),
                                    target_type=pl.FP32,
                                ),
                                [TOK_TILE, K_CHUNK],
                            )
                            gamma = pl.slice(
                                input_rms_weight, [1, K_CHUNK], [0, k0]
                            )
                            normed = pl.col_expand_mul(
                                pl.row_expand_mul(x_chunk, variance), gamma
                            )
                            normed_tile = pl.assemble(
                                normed_tile,
                                pl.cast(normed, target_type=pl.BF16),
                                [0, k0],
                            )

                    # Stage 2: Q projection (matmul + matmul_acc, cube only).
                    for ob in pl.range(q_out_blocks):
                        q0 = ob * Q_OUT_CHUNK
                        with pl.incore():
                            tile_a = pl.slice(
                                normed_tile,
                                [TOK_TILE, K_CHUNK],
                                [0, 0],
                            )
                            tile_w = pl.slice(
                                wq, [K_CHUNK, Q_OUT_CHUNK], [0, q0]
                            )
                            q_acc = pl.matmul(
                                tile_a, tile_w, out_dtype=pl.FP32
                            )
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                tile_a_i = pl.slice(
                                    normed_tile,
                                    [TOK_TILE, K_CHUNK],
                                    [0, k0],
                                )
                                tile_w_i = pl.slice(
                                    wq,
                                    [K_CHUNK, Q_OUT_CHUNK],
                                    [k0, q0],
                                )
                                q_acc = pl.matmul_acc(
                                    q_acc, tile_a_i, tile_w_i
                                )
                            q_proj = pl.assemble(
                                q_proj,
                                pl.cast(q_acc, target_type=pl.BF16),
                                [b, p0, q0],
                            )

                    # Stage 3: K projection (matmul + matmul_acc, cube only).
                    for ob in pl.range(kv_out_blocks):
                        kv0 = ob * KV_OUT_CHUNK
                        with pl.incore():
                            tile_a = pl.slice(
                                normed_tile,
                                [TOK_TILE, K_CHUNK],
                                [0, 0],
                            )
                            tile_wk = pl.slice(
                                wk,
                                [K_CHUNK, KV_OUT_CHUNK],
                                [0, kv0],
                            )
                            k_acc = pl.matmul(
                                tile_a, tile_wk, out_dtype=pl.FP32
                            )
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                tile_a_i = pl.slice(
                                    normed_tile,
                                    [TOK_TILE, K_CHUNK],
                                    [0, k0],
                                )
                                tile_wk_i = pl.slice(
                                    wk,
                                    [K_CHUNK, KV_OUT_CHUNK],
                                    [k0, kv0],
                                )
                                k_acc = pl.matmul_acc(
                                    k_acc, tile_a_i, tile_wk_i
                                )
                            k_proj = pl.assemble(
                                k_proj,
                                pl.cast(k_acc, target_type=pl.BF16),
                                [b, p0, kv0],
                            )

                    # Stage 4: V projection (matmul + matmul_acc, cube only).
                    for ob in pl.range(kv_out_blocks):
                        kv0 = ob * KV_OUT_CHUNK
                        with pl.incore():
                            tile_a = pl.slice(
                                normed_tile,
                                [TOK_TILE, K_CHUNK],
                                [0, 0],
                            )
                            tile_wv = pl.slice(
                                wv,
                                [K_CHUNK, KV_OUT_CHUNK],
                                [0, kv0],
                            )
                            v_acc = pl.matmul(
                                tile_a, tile_wv, out_dtype=pl.FP32
                            )
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                tile_a_i = pl.slice(
                                    normed_tile,
                                    [TOK_TILE, K_CHUNK],
                                    [0, k0],
                                )
                                tile_wv_i = pl.slice(
                                    wv,
                                    [K_CHUNK, KV_OUT_CHUNK],
                                    [k0, kv0],
                                )
                                v_acc = pl.matmul_acc(
                                    v_acc, tile_a_i, tile_wv_i
                                )
                            v_proj = pl.assemble(
                                v_proj,
                                pl.cast(v_acc, target_type=pl.BF16),
                                [b, p0, kv0],
                            )

            return q_proj, k_proj, v_proj

    return PrefillProjectionProgram


def build_tensor_specs(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
):
    import torch
    from pypto.runtime import TensorSpec

    kv_hidden = num_kv_heads * head_dim

    def init_hidden_states():
        return torch.rand(batch, max_seq, hidden_size) - 0.5

    def init_seq_lens():
        n_blocks = max_seq // TOK_TILE
        blocks = torch.randint(1, n_blocks + 1, (batch,), dtype=torch.int32)
        return blocks * TOK_TILE

    def init_rms_weight():
        return torch.rand(1, hidden_size) - 0.5

    def init_wq():
        return torch.rand(hidden_size, hidden_size) - 0.5

    def init_wk():
        return torch.rand(hidden_size, kv_hidden) - 0.5

    def init_wv():
        return torch.rand(hidden_size, kv_hidden) - 0.5

    return [
        TensorSpec(
            "hidden_states",
            [batch, max_seq, hidden_size],
            torch.bfloat16,
            init_value=init_hidden_states,
        ),
        TensorSpec(
            "seq_lens", [batch], torch.int32, init_value=init_seq_lens
        ),
        TensorSpec(
            "input_rms_weight",
            [1, hidden_size],
            torch.float32,
            init_value=init_rms_weight,
        ),
        TensorSpec(
            "wq",
            [hidden_size, hidden_size],
            torch.bfloat16,
            init_value=init_wq,
        ),
        TensorSpec(
            "wk",
            [hidden_size, kv_hidden],
            torch.bfloat16,
            init_value=init_wk,
        ),
        TensorSpec(
            "wv",
            [hidden_size, kv_hidden],
            torch.bfloat16,
            init_value=init_wv,
        ),
        TensorSpec(
            "q_proj",
            [batch, max_seq, hidden_size],
            torch.bfloat16,
            is_output=True,
        ),
        TensorSpec(
            "k_proj",
            [batch, max_seq, kv_hidden],
            torch.bfloat16,
            is_output=True,
        ),
        TensorSpec(
            "v_proj",
            [batch, max_seq, kv_hidden],
            torch.bfloat16,
            is_output=True,
        ),
    ]


def golden_prefill_projection(tensors, params):
    """PyTorch reference matching kernel precision path.

    RMSNorm uses variance (not rsqrt) to match the kernel's
    row_expand_mul(x_chunk, variance) path.
    Projections: BF16 matmul with FP32 accumulation, cast to BF16 output
    (aligned to prefill_tilelet).
    """
    import torch

    hidden_states = tensors["hidden_states"]
    seq_lens = tensors["seq_lens"]
    input_rms_weight = tensors["input_rms_weight"]
    wq = tensors["wq"]
    wk = tensors["wk"]
    wv = tensors["wv"]

    batch = hidden_states.shape[0]
    hidden_size = hidden_states.shape[2]
    kv_hidden = wk.shape[1]

    q_proj = tensors["q_proj"]
    k_proj = tensors["k_proj"]
    v_proj = tensors["v_proj"]

    for b in range(batch):
        seq_len_b = seq_lens[b].item()
        for p0 in range(0, seq_len_b, TOK_TILE):
            valid_tok = min(TOK_TILE, seq_len_b - p0)
            x_tile = hidden_states[b, p0 : p0 + valid_tok, :].float()

            # RMSNorm: chunked squared sum, variance-based (matches kernel).
            sq_sum = torch.zeros(valid_tok, 1, dtype=torch.float32)
            for k0 in range(0, hidden_size, K_CHUNK):
                x_chunk = x_tile[:, k0 : k0 + K_CHUNK]
                sq_sum = sq_sum + (x_chunk ** 2).sum(dim=-1, keepdim=True)
            variance = sq_sum / hidden_size + EPS
            normed = (
                x_tile * variance * input_rms_weight.float()
            ).bfloat16()

            # Q/K/V projection: BF16 matmul, FP32 acc, cast to BF16 output.
            q_proj[b, p0 : p0 + valid_tok, :] = (
                normed.float() @ wq.float()
            ).bfloat16()
            k_proj[b, p0 : p0 + valid_tok, :] = (
                normed.float() @ wk.float()
            ).bfloat16()
            v_proj[b, p0 : p0 + valid_tok, :] = (
                normed.float() @ wv.float()
            ).bfloat16()


def compile_and_run(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    platform: str = "a5",
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

    program = build_prefill_projection_program(
        batch=batch,
        max_seq=max_seq,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    tensor_specs = build_tensor_specs(
        batch=batch,
        max_seq=max_seq,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden_prefill_projection,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=1e-3,
            atol=1e-3,
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
        default="a5",
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
