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
with parameters aligned to qwen3_32b_prefill_tilelet.py.

For each batch element with seq_len_b tokens (processed in TOK_TILE=4 chunks):
  1. Compute RMSNorm of the input hidden states token tile.
  2. Project to Q (hidden_size), K (kv_hidden), V (kv_hidden).

a2a3 separation: every pl.incore() contains either vector-only or cube-only ops.
  - RMSNorm (sq_sum, rsqrt, gamma multiply) is vector-only.
  - Q/K/V projections use chained matmul + matmul_acc (cube-only).
  - Output tensors are FP32 to avoid pl.cast (vector op) inside cube incore.

Hardware TILELET / TILE sizing:
  * RMSNorm accumulator [TOK_TILE, 1]        FP32 = [4,1]*4 = 16 B
  * Normed tile chunk   [TOK_TILE, K_CHUNK]   BF16 = [4,128]*2 = 1 KB
  * Q/K/V accumulator   [TOK_TILE, OUT_CHUNK] FP32 = [4,64]*4 = 1 KB
  * Weight tiles         [K_CHUNK, OUT_CHUNK]  BF16 = [128,64]*2 = 16 KB = MAX
"""
from __future__ import annotations

import pypto.language as pl

BATCH = 16
MAX_SEQ = 4096
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM  # 8192
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM
INTERMEDIATE = 25600

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

# Tiling constants (aligned to qwen3_32b_prefill_tilelet).
K_CHUNK = 128
Q_OUT_CHUNK = 64
KV_OUT_CHUNK = 64
TOK_TILE = 4


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
            q_proj: pl.Out[pl.Tensor[[batch, max_seq, hidden], pl.FP32]],
            k_proj: pl.Out[pl.Tensor[[batch, max_seq, kv_hidden], pl.FP32]],
            v_proj: pl.Out[pl.Tensor[[batch, max_seq, kv_hidden], pl.FP32]],
        ) -> tuple[
            pl.Tensor[[batch, max_seq, hidden], pl.FP32],
            pl.Tensor[[batch, max_seq, kv_hidden], pl.FP32],
            pl.Tensor[[batch, max_seq, kv_hidden], pl.FP32],
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

                    # Vector-only incore: RMSNorm — two-pass column-chunked.
                    with pl.auto_incore():
                        sq_sum = pl.create_tensor(
                            [TOK_TILE, 1], dtype=pl.FP32
                        )
                        sq_sum = pl.mul(sq_sum, 0.0)
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
                            sq_sum = pl.add(
                                sq_sum,
                                pl.row_sum(pl.mul(x_chunk, x_chunk)),
                            )

                        inv_rms = pl.rsqrt(
                            pl.add(pl.mul(sq_sum, hidden_inv), EPS)
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
                                pl.row_expand_mul(x_chunk, inv_rms), gamma
                            )
                            normed_tile = pl.assemble(
                                normed_tile,
                                pl.cast(normed, target_type=pl.BF16),
                                [0, k0],
                            )

                    # Cube-only incore: Q projection (chained matmul_acc).
                    # Output is FP32 — no pl.cast inside cube incore.
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
                                q_proj, q_acc, [b, p0, q0]
                            )

                    # Cube-only incore: K projection (chained matmul_acc).
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
                                k_proj, k_acc, [b, p0, kv0]
                            )

                    # Cube-only incore: V projection (chained matmul_acc).
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
                                v_proj, v_acc, [b, p0, kv0]
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
            torch.float32,
            is_output=True,
        ),
        TensorSpec(
            "k_proj",
            [batch, max_seq, kv_hidden],
            torch.float32,
            is_output=True,
        ),
        TensorSpec(
            "v_proj",
            [batch, max_seq, kv_hidden],
            torch.float32,
            is_output=True,
        ),
    ]


def golden_prefill_projection(tensors, params):
    """PyTorch reference matching kernel precision path.

    RMSNorm in FP32, then cast normed to BF16.
    Projections: BF16 matmul with FP32 accumulation, final FP32 output.
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

            # RMSNorm: chunked squared sum.
            sq_sum = torch.zeros(valid_tok, 1, dtype=torch.float32)
            for k0 in range(0, hidden_size, K_CHUNK):
                x_chunk = x_tile[:, k0 : k0 + K_CHUNK]
                sq_sum = sq_sum + (x_chunk ** 2).sum(dim=-1, keepdim=True)
            inv_rms = torch.rsqrt(sq_sum / hidden_size + EPS)
            normed = (
                x_tile * inv_rms * input_rms_weight.float()
            ).bfloat16()

            # Q/K/V projection: BF16 matmul, FP32 output.
            q_proj[b, p0 : p0 + valid_tok, :] = (
                normed.float() @ wq.float()
            ).float()
            k_proj[b, p0 : p0 + valid_tok, :] = (
                normed.float() @ wk.float()
            ).float()
            v_proj[b, p0 : p0 + valid_tok, :] = (
                normed.float() @ wv.float()
            ).float()


def compile_and_run(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
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
