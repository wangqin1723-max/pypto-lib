# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-32B prefill Scope 2 — with I/O buffer caching (via io_cache plugin).

Identical kernel program to qwen3_32b_prefill_scope2.py.  All program,
tensor-spec, and golden definitions are imported from the original file;
this wrapper only adds disk-backed caching of input tensors and golden output.

Usage:
  python qwen3_32b_prefill_scope2_wIOBuffer.py -p a2a3 -d 5
  python qwen3_32b_prefill_scope2_wIOBuffer.py -p a2a3 -d 5 --clear-cache
"""
from qwen3_32b_prefill_scope2 import (
    build_prefill_scope2_program,
    build_tensor_specs,
    golden_prefill_scope2,
    BATCH, MAX_SEQ, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
)
from io_cache import (
    make_cache_dir, wrap_tensor_specs, wrap_golden,
    add_cache_args, setup_cache_dir,
)

DEFAULT_CACHE_DIR = make_cache_dir(
    "prefill_scope2",
    BATCH=BATCH, MAX_SEQ=MAX_SEQ, NUM_HEADS=NUM_HEADS,
    NUM_KV_HEADS=NUM_KV_HEADS, HEAD_DIM=HEAD_DIM,
)


def compile_and_run(
    batch=BATCH, max_seq=MAX_SEQ,
    num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
    platform="a2a3", device_id=0, dump_passes=True,
    enable_profiling=False, cache_dir=None,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    backend = BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B

    program = build_prefill_scope2_program(
        batch=batch, max_seq=max_seq, num_heads=num_heads,
        num_kv_heads=num_kv_heads, head_dim=head_dim,
    )
    tensor_specs = wrap_tensor_specs(
        build_tensor_specs(
            batch=batch, max_seq=max_seq, num_heads=num_heads,
            num_kv_heads=num_kv_heads, head_dim=head_dim,
        ),
        cache_dir,
    )
    golden = wrap_golden(golden_prefill_scope2, "attn_out", cache_dir, tensor_specs)

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=2e-3,
            atol=2e-3,
            strategy=OptimizationStrategy.Default,
            dump_passes=dump_passes,
            backend_type=backend,
            runtime_profiling=enable_profiling,
        ),
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    add_cache_args(parser, DEFAULT_CACHE_DIR)
    args = parser.parse_args()

    cache_dir = setup_cache_dir(args)

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
