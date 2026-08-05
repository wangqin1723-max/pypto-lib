# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Pro W8A8 single-die MoE compute proxy: gate, shared expert, and routed experts."""


import pypto.language as pl

from config import ACTIVE as M, MOE_DEPLOYMENT_RECV_MAX, MOE_LOCAL_EXPERTS, MOE_TOKENS
from expert_routed import (
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
    GATE_FIXTURES,
    build_tensor_specs as build_gate_tensor_specs,
    gate,
    gate_tile_prefix_compare,
    golden_gate_core,
)


# model config
T = MOE_TOKENS
D = M.hidden_size
N_EXPERTS = M.n_routed_experts
TOPK = M.num_experts_per_tok
VOCAB = M.vocab_size
MOE_INTER = M.moe_intermediate_size

# deployment shape
N_LOCAL_EXPERTS = MOE_LOCAL_EXPERTS
RECV_MAX = MOE_DEPLOYMENT_RECV_MAX


@pl.jit
def moe_compute(
    x_mixed: pl.Tensor[[T, D], pl.BF16],
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
    x_norm_i8: pl.Out[pl.Tensor[[T, D], pl.INT8]],
    x_norm_scale: pl.Out[pl.Tensor[[T, 1], pl.FP32]],
    indices: pl.Out[pl.Tensor[[T, TOPK], pl.INT32]],
    weights: pl.Out[pl.Tensor[[T, TOPK], pl.FP32]],
    sh: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    recv_y: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16]],
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    gate(
        x_mixed,
        norm_w, gate_w, gate_bias,
        layer_id, num_tokens,
        tid2eid, input_ids,
        x_norm_i8, x_norm_scale, indices, weights,
    )

    expert_shared(
        x_norm_i8, x_norm_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        sh,
    )

    expert_routed(
        recv_x, recv_scale_dq, recv_weights, recv_expert_count,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        recv_y,
    )
    return x_norm_i8, x_norm_scale, indices, weights, sh, recv_y


def golden_moe_compute(tensors):
    gate_tensors = dict(tensors)
    golden_gate_core(gate_tensors)

    shared_tensors = {
        "x_local_i8": gate_tensors["x_norm_i8"],
        "x_local_scale_dq": gate_tensors["x_norm_scale"],
        "shared_w1": tensors["shared_w1"],
        "shared_w1_scale": tensors["shared_w1_scale"],
        "shared_w3": tensors["shared_w3"],
        "shared_w3_scale": tensors["shared_w3_scale"],
        "shared_w2": tensors["shared_w2"],
        "shared_w2_scale": tensors["shared_w2_scale"],
        "sh": tensors["sh"],
    }
    golden_expert_shared(shared_tensors)
    golden_expert_routed(tensors)


def build_tensor_specs(layer_id=10, num_tokens=T, fixture="random", workload="balanced"):
    import torch
    from golden import TensorSpec

    gate_specs = {
        spec.name: spec
        for spec in build_gate_tensor_specs(layer_id=layer_id, num_tokens=num_tokens, fixture=fixture)
    }
    # Compute-proxy gate input: stable unit-magnitude rows with the selected routing fixture.
    gate_specs["x_mixed"] = TensorSpec(
        "x_mixed",
        [T, D],
        torch.bfloat16,
        init_value=lambda: torch.ones(T, D),
    )
    shared_specs = {spec.name: spec for spec in build_shared_tensor_specs()}
    routed_specs = {spec.name: spec for spec in build_routed_tensor_specs(workload=workload)}

    return [
        gate_specs["x_mixed"],
        gate_specs["norm_w"],
        gate_specs["gate_w"],
        gate_specs["gate_bias"],
        gate_specs["tid2eid"],
        gate_specs["input_ids"],
        routed_specs["recv_x"],
        routed_specs["recv_scale_dq"],
        routed_specs["recv_weights"],
        routed_specs["recv_expert_count"],
        routed_specs["routed_w1"],
        routed_specs["routed_w1_scale"],
        routed_specs["routed_w3"],
        routed_specs["routed_w3_scale"],
        routed_specs["routed_w2"],
        routed_specs["routed_w2_scale"],
        shared_specs["shared_w1"],
        shared_specs["shared_w1_scale"],
        shared_specs["shared_w3"],
        shared_specs["shared_w3_scale"],
        shared_specs["shared_w2"],
        shared_specs["shared_w2_scale"],
        gate_specs["x_norm_i8"],
        gate_specs["x_norm_scale"],
        gate_specs["indices"],
        gate_specs["weights"],
        shared_specs["sh"],
        routed_specs["recv_y"],
        gate_specs["layer_id"],
        gate_specs["num_tokens"],
    ]


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, ratio_reldiff, run_jit, topk_pair_compare

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=("a2a3", "a2a3sim", "a5", "a5sim"))
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--layer-id", type=int, default=10)
    parser.add_argument("--num-tokens", type=int, default=T)
    parser.add_argument(
        "--fixture",
        choices=GATE_FIXTURES,
        default="random",
        help="gate routing fixture; the compute proxy uses stable unit-magnitude input rows",
    )
    parser.add_argument("--workload", choices=tuple(ROUTED_WORKLOAD_COUNTS), default="balanced")
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=4, default=0, choices=(0, 1, 2, 4))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    args = parser.parse_args()
    if args.fixture == "tail-expert" and args.layer_id < M.num_hash_layers:
        parser.error("--fixture tail-expert requires a score-routing layer")

    compare_fn = {
        "x_norm_i8": gate_tile_prefix_compare(
            args.num_tokens,
            ratio_allclose(atol=1, rtol=0, max_error_ratio=0.001),
        ),
        "sh": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
        "recv_y": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
    }
    if args.fixture != "tail-expert":
        compare_fn["indices"] = topk_pair_compare("weights")

    result = run_jit(
        fn=moe_compute,
        specs=build_tensor_specs(
            layer_id=args.layer_id,
            num_tokens=args.num_tokens,
            fixture=args.fixture,
            workload=args.workload,
        ),
        golden_fn=golden_moe_compute,
        compile_only=args.compile_only,
        save_data=args.save_data,
        golden_data=args.golden_data,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn=compare_fn,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
