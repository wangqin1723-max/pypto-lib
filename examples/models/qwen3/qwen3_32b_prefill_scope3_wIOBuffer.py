# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-32B prefill Scope 3 — with I/O buffer caching.

Identical kernel program to qwen3_32b_prefill_scope3.py, but caches
input tensors and golden output to disk so that re-runs skip the
expensive golden computation.

Usage:
  # First run: generates random inputs, computes golden, saves cache
  python qwen3_32b_prefill_scope3_wIOBuffer.py -p a2a3 -d 5

  # Subsequent runs: loads cached inputs and golden output (< 1s)
  python qwen3_32b_prefill_scope3_wIOBuffer.py -p a2a3 -d 5

  # Force regeneration (delete cache first)
  python qwen3_32b_prefill_scope3_wIOBuffer.py -p a2a3 -d 5 --clear-cache

Cache location: build_output/prefill_scope3_io_cache/ (override with --cache-dir).
"""
from __future__ import annotations

import pypto.language as pl

BATCH = 16
MAX_SEQ = 256
HIDDEN = 1024      
INTERMEDIATE = 2048 

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

# Tiling constants.
K_CHUNK = 128
Q_OUT_CHUNK = 64
MLP_OUT_CHUNK = 64
TOK_TILE = 16

# Default cache directory (relative to working directory).
DEFAULT_CACHE_DIR = "build_output/prefill_scope3_io_cache"


# ---------------------------------------------------------------------------
# Program definition (unchanged from qwen3_32b_prefill_scope3.py)
# ---------------------------------------------------------------------------
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
            for b in pl.parallel(0, batch, 1):
                seq_len_b = pl.tensor.read(seq_lens, [b])
                tok_blocks = (seq_len_b + TOK_TILE - 1) // TOK_TILE
                for p0_idx in pl.range(tok_blocks):
                    p0 = p0_idx * TOK_TILE
                    valid_tok = pl.min(TOK_TILE, seq_len_b - p0)

                    # GM intermediate tensors.
                    resid1_tile = pl.create_tensor([TOK_TILE, hidden], dtype=pl.FP32)
                    attn_tile = pl.create_tensor([TOK_TILE, hidden], dtype=pl.BF16)

                    # --- Stage 1a: Copy attn_out 3D -> attn_tile 2D ---
                    with pl.incore():
                        for kb in pl.range(hidden_blocks):
                            k0 = kb * K_CHUNK
                            a_chunk_fp32 = pl.reshape(
                                pl.cast(
                                    pl.slice(attn_out, [1, TOK_TILE, K_CHUNK], [b, p0, k0],
                                             valid_shape=[1, valid_tok, K_CHUNK]),
                                    target_type=pl.FP32,
                                ),
                                [TOK_TILE, K_CHUNK],
                            )
                            a_chunk_bf16 = pl.cast(a_chunk_fp32, target_type=pl.BF16)
                            attn_tile = pl.assemble(attn_tile, a_chunk_bf16, [0, k0])

                    # Stage 1b: Initialize resid1_tile accumulator.
                    with pl.auto_incore():
                        for ob in pl.parallel(0, q_out_blocks, chunk=8):
                            o0 = ob * Q_OUT_CHUNK
                            zero_resid1 = pl.full([TOK_TILE, Q_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                            resid1_tile = pl.assemble(resid1_tile, zero_resid1, [0, o0])

                    # --- Stage 2: Output projection + first residual ---
                    for ob in pl.range(q_out_blocks):
                        o0 = ob * Q_OUT_CHUNK

                        # Cube: chained matmul, assemble to GM.
                        with pl.incore():
                            tile_a = pl.slice(attn_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            tile_w = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [0, o0])
                            o_acc = pl.matmul(tile_a, tile_w, out_dtype=pl.FP32)
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                tile_a_i = pl.slice(attn_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                tile_w_i = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [k0, o0])
                                o_acc = pl.matmul_acc(o_acc, tile_a_i, tile_w_i)

                            resid1_tile = pl.assemble(resid1_tile, o_acc, [0, o0])

                        # Vector: read back matmul result, add residual.
                        with pl.incore():
                            resid_chunk = pl.reshape(
                                pl.cast(
                                    pl.slice(hidden_states, [1, TOK_TILE, Q_OUT_CHUNK], [b, p0, o0],
                                             valid_shape=[1, valid_tok, Q_OUT_CHUNK]),
                                    target_type=pl.FP32,
                                ),
                                [TOK_TILE, Q_OUT_CHUNK],
                            )
                            mm_out = pl.slice(resid1_tile, [TOK_TILE, Q_OUT_CHUNK], [0, o0])
                            resid_sum = pl.add(mm_out, resid_chunk)
                            resid1_tile = pl.assemble(resid1_tile, resid_sum, [0, o0])

                    # --- Stage 3/4/5: Post-attention RMSNorm + normalize + init down_proj ---
                    post_norm_tile = pl.create_tensor([TOK_TILE, hidden], dtype=pl.BF16)
                    down_fp32_tile = pl.create_tensor([TOK_TILE, hidden], dtype=pl.FP32)
                    with pl.auto_incore():
                        sq_sum = pl.full([1, TOK_TILE], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(hidden_blocks):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, k0])
                            sq_sum = pl.add(
                                sq_sum,
                                pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, TOK_TILE]),
                            )
                        inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, hidden_inv), EPS)))

                        # Normalize, apply gamma, and zero-init down_proj accumulator.
                        for kb in pl.range(hidden_blocks):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, k0])
                            gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [0, k0])
                            normed = pl.col_expand_mul(
                                pl.row_expand_mul(x_chunk, pl.reshape(inv_rms, [TOK_TILE, 1])),
                                gamma,
                            )
                            normed_bf16 = pl.cast(normed, target_type=pl.BF16)
                            post_norm_tile = pl.assemble(
                                post_norm_tile, normed_bf16, [0, k0])
                            down_zero_chunk = pl.full([TOK_TILE, K_CHUNK], dtype=pl.FP32, value=0.0)
                            down_fp32_tile = pl.assemble(down_fp32_tile, down_zero_chunk, [0, k0])

                    # --- Stage 6: MLP gate/up + SiLU + down projection ---
                    for ob in pl.range(mlp_out_blocks):
                        o0 = ob * MLP_OUT_CHUNK

                        # Stage 6a: Gate matmul chain.
                        with pl.incore():
                            pc0 = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            wg0 = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
                            gate_acc = pl.matmul(pc0, wg0, out_dtype=pl.FP32)
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                pci = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                wgi = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                                gate_acc = pl.matmul_acc(gate_acc, pci, wgi)

                        # Stage 6b: Up matmul chain.
                        with pl.incore():
                            pc0 = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            wu0 = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
                            up_acc = pl.matmul(pc0, wu0, out_dtype=pl.FP32)
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                pci = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                wui = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                                up_acc = pl.matmul_acc(up_acc, pci, wui)

                        # Stage 6c: SiLU activation and cast to BF16.
                        with pl.auto_incore():
                            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                            mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                            mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)

                        # Stage 7: Down projection, cube matmul + vector accumulate per chunk.
                        for dob in pl.range(hidden_blocks):
                            d0 = dob * K_CHUNK

                            with pl.incore():
                                w_down_chunk = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [o0, d0])
                                down_next = pl.matmul(mlp_chunk_bf16, w_down_chunk, out_dtype=pl.FP32)

                            with pl.incore():
                                down_prev = pl.slice(down_fp32_tile, [TOK_TILE, K_CHUNK], [0, d0])
                                accum = pl.add(down_prev, down_next)
                                down_fp32_tile = pl.assemble(down_fp32_tile, accum, [0, d0])

                    # --- Stage 8: Final residual add, write BF16 output ---
                    for ob in pl.range(hidden_blocks):
                        o0 = ob * K_CHUNK
                        with pl.incore():
                            final_sum = pl.add(
                                pl.slice(down_fp32_tile, [TOK_TILE, K_CHUNK], [0, o0]),
                                pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, o0]),
                            )
                            final_bf16 = pl.cast(final_sum, target_type=pl.BF16)
                            out = pl.assemble(out, final_bf16, [b, p0, o0])

            return out

    return PrefillScope3Program


# ---------------------------------------------------------------------------
# Tensor specs with I/O buffer caching
# ---------------------------------------------------------------------------
def build_tensor_specs(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    cache_dir: str = None,
):
    import torch
    from pypto.runtime import TensorSpec

    def init_seq_lens():
        import os
        if cache_dir is not None:
            p = os.path.join(cache_dir, "seq_lens.pt")
            if os.path.exists(p):
                return torch.load(p)
            n_blocks = max_seq // TOK_TILE
            t = torch.randint(1, n_blocks + 1, (batch,), dtype=torch.int32) * TOK_TILE
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(t, p)
            return t
        n_blocks = max_seq // TOK_TILE
        return torch.randint(1, n_blocks + 1, (batch,), dtype=torch.int32) * TOK_TILE

    def init_attn_out():
        import os
        if cache_dir is not None:
            p = os.path.join(cache_dir, "attn_out.pt")
            if os.path.exists(p):
                return torch.load(p)
            t = torch.rand(batch, max_seq, hidden_size) - 0.5
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(t, p)
            return t
        return torch.rand(batch, max_seq, hidden_size) - 0.5

    def init_hidden_states():
        import os
        if cache_dir is not None:
            p = os.path.join(cache_dir, "hidden_states.pt")
            if os.path.exists(p):
                return torch.load(p)
            t = torch.rand(batch, max_seq, hidden_size) - 0.5
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(t, p)
            return t
        return torch.rand(batch, max_seq, hidden_size) - 0.5

    def init_wo():
        import os
        if cache_dir is not None:
            p = os.path.join(cache_dir, "wo.pt")
            if os.path.exists(p):
                return torch.load(p)
            t = (torch.rand(hidden_size, hidden_size) - 0.5) / hidden_size ** 0.5
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(t, p)
            return t
        return (torch.rand(hidden_size, hidden_size) - 0.5) / hidden_size ** 0.5

    def init_post_rms_weight():
        import os
        if cache_dir is not None:
            p = os.path.join(cache_dir, "post_rms_weight.pt")
            if os.path.exists(p):
                return torch.load(p)
            t = torch.ones(1, hidden_size)
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(t, p)
            return t
        return torch.ones(1, hidden_size)

    def init_w_gate():
        import os
        if cache_dir is not None:
            p = os.path.join(cache_dir, "w_gate.pt")
            if os.path.exists(p):
                return torch.load(p)
            t = (torch.rand(hidden_size, intermediate_size) - 0.5) / hidden_size ** 0.5
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(t, p)
            return t
        return (torch.rand(hidden_size, intermediate_size) - 0.5) / hidden_size ** 0.5

    def init_w_up():
        import os
        if cache_dir is not None:
            p = os.path.join(cache_dir, "w_up.pt")
            if os.path.exists(p):
                return torch.load(p)
            t = (torch.rand(hidden_size, intermediate_size) - 0.5) / hidden_size ** 0.5
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(t, p)
            return t
        return (torch.rand(hidden_size, intermediate_size) - 0.5) / hidden_size ** 0.5

    def init_w_down():
        import os
        if cache_dir is not None:
            p = os.path.join(cache_dir, "w_down.pt")
            if os.path.exists(p):
                return torch.load(p)
            t = (torch.rand(intermediate_size, hidden_size) - 0.5) / intermediate_size ** 0.5
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(t, p)
            return t
        return (torch.rand(intermediate_size, hidden_size) - 0.5) / intermediate_size ** 0.5

    return [
        TensorSpec("attn_out", [batch, max_seq, hidden_size], torch.bfloat16,
                   init_value=init_attn_out),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=init_seq_lens),
        TensorSpec("hidden_states", [batch, max_seq, hidden_size], torch.bfloat16,
                   init_value=init_hidden_states),
        TensorSpec("wo", [hidden_size, hidden_size], torch.bfloat16,
                   init_value=init_wo),
        TensorSpec("post_rms_weight", [1, hidden_size], torch.float32,
                   init_value=init_post_rms_weight),
        TensorSpec("w_gate", [hidden_size, intermediate_size], torch.bfloat16,
                   init_value=init_w_gate),
        TensorSpec("w_up", [hidden_size, intermediate_size], torch.bfloat16,
                   init_value=init_w_up),
        TensorSpec("w_down", [intermediate_size, hidden_size], torch.bfloat16,
                   init_value=init_w_down),
        TensorSpec("out", [batch, max_seq, hidden_size], torch.bfloat16, is_output=True),
    ]


# ---------------------------------------------------------------------------
# Compile and run with cached golden
# ---------------------------------------------------------------------------
def compile_and_run(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    platform: str = "a2a3",
    device_id: int = 0,
    dump_passes: bool = True,
    enable_profiling: bool = False,
    cache_dir: str = None,
):
    import torch
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    backend = BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B

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
        cache_dir=cache_dir,
    )

    # Golden function defined here to capture cache_dir as a closure variable.
    def golden_prefill_scope3(tensors, params):
        import os
        golden_path = os.path.join(cache_dir, "golden_out.pt") if cache_dir is not None else None
        if golden_path is not None and os.path.exists(golden_path):
            tensors["out"][:] = torch.load(golden_path)
            return

        attn_out_t = tensors["attn_out"]
        seq_lens = tensors["seq_lens"]
        hidden_states = tensors["hidden_states"]
        wo = tensors["wo"]
        post_rms_weight = tensors["post_rms_weight"]
        w_gate = tensors["w_gate"]
        w_up = tensors["w_up"]
        w_down = tensors["w_down"]

        batch_sz = attn_out_t.shape[0]
        hidden_sz = attn_out_t.shape[2]
        eps = EPS

        out_t = tensors["out"]

        for b in range(batch_sz):
            seq_len_b = seq_lens[b].item()
            for p0 in range(0, seq_len_b, TOK_TILE):
                valid_tok = min(TOK_TILE, seq_len_b - p0)

                attn_tile = attn_out_t[b, p0 : p0 + valid_tok, :]

                # 1. Output projection (BF16 inputs, FP32 accumulation) + residual.
                o_proj = torch.matmul(attn_tile.float(), wo.float())
                resid1 = o_proj + hidden_states[b, p0 : p0 + valid_tok, :].float()

                # 2. Post-attention RMSNorm.
                variance = resid1.pow(2).mean(dim=-1, keepdim=True)
                inv_rms = torch.rsqrt(variance + eps)
                normed_bf16 = (resid1 * inv_rms * post_rms_weight.float()).bfloat16()

                # 3. SwiGLU MLP: gate/up projections, silu activation, down projection.
                gate = torch.matmul(normed_bf16.float(), w_gate.float())
                up = torch.matmul(normed_bf16.float(), w_up.float())
                mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
                down = torch.matmul(mlp_bf16.float(), w_down.float())

                # 4. Final residual + cast to BF16.
                out_t[b, p0 : p0 + valid_tok, :] = (down + resid1).bfloat16()

        if golden_path is not None:
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(tensors["out"].clone(), golden_path)

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden_prefill_scope3,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=3e-3,
            atol=3e-3,
            strategy=OptimizationStrategy.Default,
            dump_passes=dump_passes,
            backend_type=backend,
            enable_profiling=enable_profiling,
        ),
    )
    return result


if __name__ == "__main__":
    import argparse
    import os
    import shutil

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR,
                        help="directory for cached I/O tensors (default: %(default)s)")
    parser.add_argument("--clear-cache", action="store_true", default=False,
                        help="delete cached tensors before running")
    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir)
    if args.clear_cache and os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cleared cache: {cache_dir}")

    if os.path.isdir(cache_dir):
        print(f"Using cached I/O from: {cache_dir}")
    else:
        print(f"No cache found, will generate and save to: {cache_dir}")

    result = compile_and_run(
        platform=args.platform,
        device_id=args.device,
        enable_profiling=args.enable_profiling,
        cache_dir=cache_dir,
    )
    if not result.passed:
        if result.error:
            print(f"Result: {result.error}")
        raise SystemExit(1)
