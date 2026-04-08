# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-32B prefill Scope 3 — output projection + residual + post RMSNorm + MLP.

Standalone test for the output + MLP scope of the Qwen3-32B prefill layer,
with parameters aligned to qwen3_32b_prefill_tilelet.py.

For each batch element with seq_len_b tokens (processed in TOK_TILE=16 chunks):
  1. Output projection: attn_out x wo, tiled by Q_OUT_CHUNK, + first residual.
  2. Post-attention RMSNorm over resid1.
  3. MLP: gate/up projections -> SiLU activation -> down projection.
  4. Final residual addition -> BF16 output.

a2a3 separation: uses pl.auto_incore(split=pl.SplitMode.UP_DOWN) so the
compiler automatically splits cube (AIC) and vector (AIV) operations.
  - Output projection uses chained matmul + matmul_acc (cube).
  - Residual addition, RMSNorm, SiLU are vector ops.
  - MLP gate/up projections use chained matmul + matmul_acc (cube).
  - Down projection uses matmul_acc to accumulate (cube).

Hardware TILELET / TILE sizing:
  * RMSNorm accumulator  [TOK_TILE, 1]          FP32 = [16,1]*4 = 64 B
  * Output/MLP accum     [TOK_TILE, OUT_CHUNK]   FP32 = [16,64]*4 = 4 KB
  * Weight tiles          [K_CHUNK, OUT_CHUNK]    BF16 = [128,64]*2 = 16 KB = MAX
  * Down weight tiles     [MLP_OUT_CHUNK, K_CHUNK] BF16 = [64,128]*2 = 16 KB = MAX
"""
from __future__ import annotations

import pypto.language as pl

BATCH = 16
MAX_SEQ = 96
NUM_HEADS = 40
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM
INTERMEDIATE = 25600

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

# Tiling constants (aligned to qwen3_32b_prefill_tilelet).
K_CHUNK = 128
Q_OUT_CHUNK = 64
MLP_OUT_CHUNK = 64
TOK_TILE = 16


def build_prefill_scope3_program(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
):
    hidden = hidden_size
    inter = intermediate_size
    hidden_blocks = hidden // K_CHUNK
    q_out_blocks = hidden // Q_OUT_CHUNK
    mlp_out_blocks = inter // MLP_OUT_CHUNK
    hidden_inv = 1.0 / hidden

    @pl.program
    class PrefillScope3Program:
        @pl.function(type=pl.FunctionType.Opaque)
        def prefill_scope3(
            self,
            attn_out: pl.Tensor[[batch, max_seq, hidden], pl.BF16],
            seq_lens: pl.Tensor[[batch], pl.INT32],
            hidden_states: pl.Tensor[[batch, max_seq, hidden], pl.BF16],
            wo: pl.Tensor[[hidden, hidden], pl.BF16],
            post_rms_weight: pl.Tensor[[1, hidden], pl.FP32],
            w_gate: pl.Tensor[[hidden, inter], pl.BF16],
            w_up: pl.Tensor[[hidden, inter], pl.BF16],
            w_down: pl.Tensor[[inter, hidden], pl.BF16],
            out: pl.Out[pl.Tensor[[batch, max_seq, hidden], pl.BF16]],
        ) -> pl.Tensor[[batch, max_seq, hidden], pl.BF16]:
            with pl.auto_incore(split=pl.SplitMode.UP_DOWN):
                for b in pl.parallel(0, batch, 1):
                    seq_len_b = pl.tensor.read(seq_lens, [b])
                    tok_blocks = (seq_len_b + TOK_TILE - 1) // TOK_TILE
                    for p0_idx in pl.range(tok_blocks):
                        p0 = p0_idx * TOK_TILE
                        valid_tok = pl.min(TOK_TILE, seq_len_b - p0)

                        resid1_tile = pl.create_tensor(
                            [TOK_TILE, hidden], dtype=pl.FP32
                        )

                        # --- Output projection + first residual ---
                        for ob in pl.range(q_out_blocks):
                            o0 = ob * Q_OUT_CHUNK

                            # Chained matmul_acc over hidden blocks.
                            tile_a0 = pl.reshape(
                                pl.slice(
                                    attn_out,
                                    [1, TOK_TILE, K_CHUNK],
                                    [b, p0, 0],
                                    valid_shape=[1, valid_tok, K_CHUNK],
                                ),
                                [TOK_TILE, K_CHUNK],
                            )
                            tile_w0 = pl.slice(
                                wo, [K_CHUNK, Q_OUT_CHUNK], [0, o0]
                            )
                            o_acc = pl.matmul(
                                tile_a0, tile_w0, out_dtype=pl.FP32
                            )
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                tile_a_i = pl.reshape(
                                    pl.slice(
                                        attn_out,
                                        [1, TOK_TILE, K_CHUNK],
                                        [b, p0, k0],
                                        valid_shape=[
                                            1,
                                            valid_tok,
                                            K_CHUNK,
                                        ],
                                    ),
                                    [TOK_TILE, K_CHUNK],
                                )
                                tile_w_i = pl.slice(
                                    wo,
                                    [K_CHUNK, Q_OUT_CHUNK],
                                    [k0, o0],
                                )
                                o_acc = pl.matmul_acc(
                                    o_acc, tile_a_i, tile_w_i
                                )

                            # Add residual from hidden_states.
                            resid_chunk = pl.reshape(
                                pl.cast(
                                    pl.slice(
                                        hidden_states,
                                        [1, TOK_TILE, Q_OUT_CHUNK],
                                        [b, p0, o0],
                                        valid_shape=[
                                            1,
                                            valid_tok,
                                            Q_OUT_CHUNK,
                                        ],
                                    ),
                                    target_type=pl.FP32,
                                ),
                                [TOK_TILE, Q_OUT_CHUNK],
                            )
                            resid1_chunk = pl.add(o_acc, resid_chunk)
                            resid1_tile = pl.assemble(
                                resid1_tile, resid1_chunk, [0, o0]
                            )

                        # --- Post-attention RMSNorm ---
                        sq_sum = pl.full(
                            [TOK_TILE, 1], dtype=pl.FP32, value=0.0
                        )
                        for kb in pl.range(hidden_blocks):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.slice(
                                resid1_tile,
                                [TOK_TILE, K_CHUNK],
                                [0, k0],
                            )
                            sq_sum = pl.add(
                                sq_sum,
                                pl.row_sum(
                                    pl.mul(x_chunk, x_chunk)
                                ),
                            )
                        inv_rms = pl.rsqrt(
                            pl.add(pl.mul(sq_sum, hidden_inv), EPS)
                        )

                        # Normalize and apply gamma.
                        post_norm_tile = pl.create_tensor(
                            [TOK_TILE, hidden], dtype=pl.BF16
                        )
                        for kb in pl.range(hidden_blocks):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.slice(
                                resid1_tile,
                                [TOK_TILE, K_CHUNK],
                                [0, k0],
                            )
                            gamma = pl.slice(
                                post_rms_weight,
                                [1, K_CHUNK],
                                [0, k0],
                            )
                            normed = pl.col_expand_mul(
                                pl.row_expand_mul(x_chunk, inv_rms),
                                gamma,
                            )
                            post_norm_tile = pl.assemble(
                                post_norm_tile,
                                pl.cast(
                                    normed, target_type=pl.BF16
                                ),
                                [0, k0],
                            )

                        # --- Zero-init down_proj accumulator ---
                        down_proj_tile = pl.create_tensor(
                            [TOK_TILE, hidden], dtype=pl.FP32
                        )
                        for zi in pl.range(hidden_blocks):
                            z0 = zi * K_CHUNK
                            zero = pl.full(
                                [TOK_TILE, K_CHUNK],
                                dtype=pl.FP32,
                                value=0.0,
                            )
                            down_proj_tile = pl.assemble(
                                down_proj_tile, zero, [0, z0]
                            )

                        # --- MLP: gate/up projections + SiLU + down projection ---
                        for ob in pl.range(mlp_out_blocks):
                            o0 = ob * MLP_OUT_CHUNK

                            # Gate matmul (chained acc).
                            pc0 = pl.slice(
                                post_norm_tile,
                                [TOK_TILE, K_CHUNK],
                                [0, 0],
                            )
                            wg0 = pl.slice(
                                w_gate,
                                [K_CHUNK, MLP_OUT_CHUNK],
                                [0, o0],
                            )
                            gate_acc = pl.matmul(
                                pc0, wg0, out_dtype=pl.FP32
                            )
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                pci = pl.slice(
                                    post_norm_tile,
                                    [TOK_TILE, K_CHUNK],
                                    [0, k0],
                                )
                                wgi = pl.slice(
                                    w_gate,
                                    [K_CHUNK, MLP_OUT_CHUNK],
                                    [k0, o0],
                                )
                                gate_acc = pl.matmul_acc(
                                    gate_acc, pci, wgi
                                )

                            # Up matmul (chained acc).
                            pc0 = pl.slice(
                                post_norm_tile,
                                [TOK_TILE, K_CHUNK],
                                [0, 0],
                            )
                            wu0 = pl.slice(
                                w_up,
                                [K_CHUNK, MLP_OUT_CHUNK],
                                [0, o0],
                            )
                            up_acc = pl.matmul(
                                pc0, wu0, out_dtype=pl.FP32
                            )
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                pci = pl.slice(
                                    post_norm_tile,
                                    [TOK_TILE, K_CHUNK],
                                    [0, k0],
                                )
                                wui = pl.slice(
                                    w_up,
                                    [K_CHUNK, MLP_OUT_CHUNK],
                                    [k0, o0],
                                )
                                up_acc = pl.matmul_acc(
                                    up_acc, pci, wui
                                )

                            # SiLU(gate) * up -> BF16.
                            sigmoid = pl.recip(
                                pl.add(
                                    pl.exp(pl.neg(gate_acc)), 1.0
                                )
                            )
                            mlp_chunk = pl.cast(
                                pl.mul(
                                    pl.mul(gate_acc, sigmoid), up_acc
                                ),
                                target_type=pl.BF16,
                            )

                            # Down projection — accumulate into down_proj_tile.
                            for dob in pl.range(hidden_blocks):
                                d0 = dob * K_CHUNK
                                wd = pl.slice(
                                    w_down,
                                    [MLP_OUT_CHUNK, K_CHUNK],
                                    [o0, d0],
                                )
                                down_prev = pl.slice(
                                    down_proj_tile,
                                    [TOK_TILE, K_CHUNK],
                                    [0, d0],
                                )
                                down_next = pl.matmul_acc(
                                    down_prev, mlp_chunk, wd
                                )
                                down_proj_tile = pl.assemble(
                                    down_proj_tile,
                                    down_next,
                                    [0, d0],
                                )

                        # --- Final residual: down_proj + resid1 -> BF16 -> out ---
                        for ob in pl.range(hidden_blocks):
                            o0 = ob * K_CHUNK
                            down_acc = pl.add(
                                pl.slice(
                                    down_proj_tile,
                                    [TOK_TILE, K_CHUNK],
                                    [0, o0],
                                ),
                                pl.slice(
                                    resid1_tile,
                                    [TOK_TILE, K_CHUNK],
                                    [0, o0],
                                ),
                            )
                            out = pl.assemble(
                                out,
                                pl.cast(
                                    down_acc, target_type=pl.BF16
                                ),
                                [b, p0, o0],
                            )

                return out

    return PrefillScope3Program


def golden_prefill_scope3(tensors, params):
    """Reference computation for Scope 3 (prefill).

    Steps:
      1. Output projection: attn_out (BF16) x wo, FP32 accumulation + residual
      2. Post-attention RMSNorm
      3. SwiGLU MLP: gate/up projections -> silu(gate) * up -> down projection
      4. Final residual addition -> BF16 output
    """
    import torch

    attn_out_t = tensors["attn_out"]       # [B, S, H], BF16
    seq_lens = tensors["seq_lens"]          # [B], INT32
    hidden_states = tensors["hidden_states"]  # [B, S, H], BF16
    wo = tensors["wo"]                       # [H, H], BF16
    post_rms_weight = tensors["post_rms_weight"]  # [1, H], FP32
    w_gate = tensors["w_gate"]               # [H, I], BF16
    w_up = tensors["w_up"]                   # [H, I], BF16
    w_down = tensors["w_down"]               # [I, H], BF16

    batch = attn_out_t.shape[0]
    hidden_size = attn_out_t.shape[2]
    eps = EPS

    out_t = tensors["out"]

    for b in range(batch):
        seq_len_b = seq_lens[b].item()
        for p0 in range(0, seq_len_b, TOK_TILE):
            valid_tok = min(TOK_TILE, seq_len_b - p0)

            attn_tile = attn_out_t[b, p0 : p0 + valid_tok, :]

            # 1. Output projection (BF16 inputs, FP32 accumulation) + residual.
            o_proj = torch.matmul(attn_tile.float(), wo.float())
            resid1 = o_proj + hidden_states[
                b, p0 : p0 + valid_tok, :
            ].float()

            # 2. Post-attention RMSNorm.
            variance = resid1.pow(2).mean(dim=-1, keepdim=True)
            inv_rms = torch.rsqrt(variance + eps)
            normed_bf16 = (
                resid1 * inv_rms * post_rms_weight.float()
            ).bfloat16()

            # 3. SwiGLU MLP: gate/up projections, silu activation, down projection.
            gate = torch.matmul(
                normed_bf16.float(), w_gate.float()
            )
            up = torch.matmul(
                normed_bf16.float(), w_up.float()
            )
            mlp_bf16 = (
                gate * torch.sigmoid(gate) * up
            ).bfloat16()
            down = torch.matmul(
                mlp_bf16.float(), w_down.float()
            )

            # 4. Final residual + cast to BF16.
            out_t[b, p0 : p0 + valid_tok, :] = (
                down + resid1
            ).bfloat16()


def build_tensor_specs(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
):
    import torch
    from pypto.runtime import TensorSpec

    def xavier_bf16(shape):
        fan_in = shape[-1]
        return (
            torch.randn(shape, dtype=torch.float32) / (fan_in**0.5)
        ).to(torch.bfloat16)

    def xavier_fp32(shape):
        fan_in = shape[-1]
        return torch.randn(shape, dtype=torch.float32) / (fan_in**0.5)

    def init_seq_lens():
        n_blocks = max_seq // TOK_TILE
        blocks = torch.randint(
            1, n_blocks + 1, (batch,), dtype=torch.int32
        )
        return blocks * TOK_TILE

    return [
        TensorSpec(
            "attn_out",
            [batch, max_seq, hidden_size],
            torch.bfloat16,
            init_value=xavier_bf16([batch, max_seq, hidden_size]),
        ),
        TensorSpec(
            "seq_lens",
            [batch],
            torch.int32,
            init_value=init_seq_lens,
        ),
        TensorSpec(
            "hidden_states",
            [batch, max_seq, hidden_size],
            torch.bfloat16,
            init_value=xavier_bf16([batch, max_seq, hidden_size]),
        ),
        TensorSpec(
            "wo",
            [hidden_size, hidden_size],
            torch.bfloat16,
            init_value=xavier_bf16([hidden_size, hidden_size]),
        ),
        TensorSpec(
            "post_rms_weight",
            [1, hidden_size],
            torch.float32,
            init_value=xavier_fp32([1, hidden_size]),
        ),
        TensorSpec(
            "w_gate",
            [hidden_size, intermediate_size],
            torch.bfloat16,
            init_value=xavier_bf16([hidden_size, intermediate_size]),
        ),
        TensorSpec(
            "w_up",
            [hidden_size, intermediate_size],
            torch.bfloat16,
            init_value=xavier_bf16([hidden_size, intermediate_size]),
        ),
        TensorSpec(
            "w_down",
            [intermediate_size, hidden_size],
            torch.bfloat16,
            init_value=xavier_bf16([intermediate_size, hidden_size]),
        ),
        TensorSpec(
            "out",
            [batch, max_seq, hidden_size],
            torch.bfloat16,
            is_output=True,
        ),
    ]


def compile_and_run(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
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

    program = build_prefill_scope3_program(
        batch=batch,
        max_seq=max_seq,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    tensor_specs = build_tensor_specs(
        batch=batch,
        max_seq=max_seq,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden_prefill_scope3,
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
