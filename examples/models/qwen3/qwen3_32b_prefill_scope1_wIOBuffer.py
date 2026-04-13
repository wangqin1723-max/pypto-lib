# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-32B prefill Scope 1 — with I/O buffer caching (via io_cache plugin).

Identical kernel program to qwen3_32b_prefill_scope1.py. All program,
tensor-spec, and golden definitions are imported from the original file;
this wrapper only adds disk-backed caching of input tensors and golden output.

Usage:
  python qwen3_32b_prefill_scope1_wIOBuffer.py -p a5 -d 0
  python qwen3_32b_prefill_scope1_wIOBuffer.py -p a5 -d 0 --clear-cache
"""
from qwen3_32b_prefill_scope1 import (
    build_prefill_projection_program,
    build_tensor_specs,
    golden_prefill_projection,
    BATCH,
    MAX_SEQ,
    HIDDEN,
    NUM_KV_HEADS,
    HEAD_DIM,
)
from io_cache import (
    make_cache_dir,
    wrap_tensor_specs,
    add_cache_args,
    setup_cache_dir,
)

DEFAULT_CACHE_DIR = make_cache_dir(
    "prefill_scope1",
    BATCH=BATCH,
    MAX_SEQ=MAX_SEQ,
    HIDDEN=HIDDEN,
    NUM_KV_HEADS=NUM_KV_HEADS,
    HEAD_DIM=HEAD_DIM,
)


def wrap_golden_multi(golden_fn, output_names, cache_dir, tensor_specs=None):
    """Cache and reload multiple golden outputs for multi-output kernels."""
    if cache_dir is None:
        return golden_fn

    import os

    golden_paths = {
        name: os.path.join(cache_dir, f"golden_{name}.pt")
        for name in output_names
    }

    if not all(os.path.exists(path) for path in golden_paths.values()):
        import time

        import torch

        if tensor_specs is None:
            raise ValueError(
                "tensor_specs is required for first-run golden pre-computation. "
                "Pass the (wrapped) tensor_specs from wrap_tensor_specs()."
            )

        tensors = {}
        for spec in tensor_specs:
            if spec.is_output:
                tensors[spec.name] = torch.zeros(spec.shape, dtype=spec.dtype)
            else:
                t = spec.init_value()
                tensors[spec.name] = t.to(spec.dtype) if hasattr(t, "to") else t

        print("Pre-computing golden for caching...")
        t0 = time.time()
        golden_fn(tensors, {})
        elapsed = time.time() - t0
        print(f"Golden pre-computation took {elapsed:.3f}s")
        os.makedirs(cache_dir, exist_ok=True)
        for name, path in golden_paths.items():
            torch.save(tensors[name].clone(), path)

    # Closure variables must be simple scalar types (str, int, float, bool,
    # None) so that golden_writer._extract_closure_constants can serialize
    # them into the generated golden.py.  A dict like golden_paths is silently
    # skipped, causing a NameError at runtime.  Use two strings instead.
    output_names_str = ",".join(output_names)

    def _load_golden(tensors, params):
        import os
        import torch

        for name in output_names_str.split(","):
            path = os.path.join(cache_dir, f"golden_{name}.pt")
            tensors[name][:] = torch.load(path)

    return _load_golden


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
    cache_dir=None,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    backend = BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B

    program = build_prefill_projection_program(
        batch=batch,
        max_seq=max_seq,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    tensor_specs = wrap_tensor_specs(
        build_tensor_specs(
            batch=batch,
            max_seq=max_seq,
            hidden_size=hidden_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        ),
        cache_dir,
    )
    golden = wrap_golden_multi(
        golden_prefill_projection,
        ("q_proj", "k_proj", "v_proj"),
        cache_dir,
        tensor_specs,
    )

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=1e-3,
            atol=1e-3,
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
    parser.add_argument("-p", "--platform", type=str, default="a5",
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
