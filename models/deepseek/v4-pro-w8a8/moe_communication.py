# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host layout contract for future DeepSeek-V4 Pro W8A8 MoE communication measurements.

Deployment EP128 describes the model shard used by the single-die compute
proxy.  Physical COMM_EP2/4/8 describes a separate dispatch/combine
measurement.  This module defines that communication layout without importing
PyPTO's distributed runtime.  Hardware execution intentionally fails until a
dedicated multi-device kernel is implemented and validated on this server.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from operator import index as integer_index
from typing import NoReturn, Sequence

from config import (
    ACTIVE as M,
    COMM_EP_CHOICES,
    DECODE_TOKENS,
    DEPLOYMENT_EP,
    MOE_GLOBAL_EXPERTS,
)


INT8_BYTES = 1
BF16_BYTES = 2
FP32_BYTES = 4
INT32_BYTES = 4
AUX_PAD = 8
ROUTE_PAD = 8
MEASUREMENT_STATUS = "unvalidated-contract-only"


class DistributedCommunicationUnavailable(RuntimeError):
    """Raised when unvalidated distributed execution is requested."""


@dataclass(frozen=True)
class MoeCommunicationContract:
    """Static dispatch/combine layout for one physical communication world."""

    deployment_ep: int
    communication_ep: int
    global_experts: int
    local_experts: int
    tokens_per_rank: int
    topk: int
    hidden_size: int
    recv_capacity_per_expert: int

    @property
    def routes_per_rank(self) -> int:
        return self.tokens_per_rank * self.topk

    @property
    def deployment_local_experts(self) -> int:
        return self.global_experts // self.deployment_ep

    @property
    def deployment_recv_capacity_per_expert(self) -> int:
        return self.deployment_ep * self.tokens_per_rank

    @property
    def dispatch_logical_row_bytes(self) -> int:
        # INT8 activation + FP32 dequant scale + FP32 route weight + INT32 return route.
        return self.hidden_size * INT8_BYTES + 2 * FP32_BYTES + INT32_BYTES

    @property
    def dispatch_window_row_bytes(self) -> int:
        # Existing distributed kernels pad auxiliary and route windows to 32 B each.
        return self.hidden_size * INT8_BYTES + AUX_PAD * FP32_BYTES + ROUTE_PAD * INT32_BYTES

    @property
    def combine_row_bytes(self) -> int:
        return self.hidden_size * BF16_BYTES

    @property
    def count_metadata_bytes_per_rank(self) -> int:
        # Each source publishes one INT32 count for every global expert.
        return self.global_experts * INT32_BYTES

    def report_row(self) -> dict[str, int | str | bool]:
        """Return fields suitable for an explicitly unmeasured report row."""

        return {
            "label": f"COMM_EP{self.communication_ep} dispatch/combine contract",
            "measurement_status": MEASUREMENT_STATUS,
            "hardware_execution_available": False,
            "deployment_ep": self.deployment_ep,
            "communication_ep": self.communication_ep,
            "global_experts": self.global_experts,
            "deployment_local_experts": self.deployment_local_experts,
            "deployment_recv_capacity_per_expert": self.deployment_recv_capacity_per_expert,
            "communication_local_experts": self.local_experts,
            "tokens_per_rank": self.tokens_per_rank,
            "topk": self.topk,
            "routes_per_rank": self.routes_per_rank,
            "communication_recv_capacity_per_expert": self.recv_capacity_per_expert,
            "dispatch_logical_row_bytes": self.dispatch_logical_row_bytes,
            "dispatch_window_row_bytes": self.dispatch_window_row_bytes,
            "combine_row_bytes": self.combine_row_bytes,
            "count_metadata_bytes_per_rank": self.count_metadata_bytes_per_rank,
        }


@dataclass(frozen=True)
class DispatchRoute:
    """One route copy and the key needed to return its expert result."""

    source_rank: int
    source_token: int
    topk_slot: int
    return_route: int
    global_expert: int
    destination_rank: int
    local_expert: int


@dataclass(frozen=True)
class AllToAllPlan:
    """Validated logical all-to-all plan; it does not submit device work."""

    contract: MoeCommunicationContract
    routes: tuple[DispatchRoute, ...]
    send_counts: tuple[tuple[int, ...], ...]
    recv_expert_counts: tuple[tuple[int, ...], ...]

    @property
    def remote_routes(self) -> int:
        return sum(route.source_rank != route.destination_rank for route in self.routes)

    @property
    def remote_dispatch_window_bytes(self) -> int:
        return self.remote_routes * self.contract.dispatch_window_row_bytes

    @property
    def remote_combine_bytes(self) -> int:
        return self.remote_routes * self.contract.combine_row_bytes


def communication_contract(comm_ep: int) -> MoeCommunicationContract:
    """Build the fixed-384-expert contract for a supported physical COMM_EP."""

    if comm_ep not in COMM_EP_CHOICES:
        choices = ", ".join(str(value) for value in COMM_EP_CHOICES)
        raise ValueError(f"COMM_EP must be one of ({choices}); got {comm_ep}")
    if MOE_GLOBAL_EXPERTS % comm_ep != 0:
        raise ValueError(f"{MOE_GLOBAL_EXPERTS} global experts do not divide across COMM_EP{comm_ep}")
    return MoeCommunicationContract(
        deployment_ep=DEPLOYMENT_EP,
        communication_ep=comm_ep,
        global_experts=MOE_GLOBAL_EXPERTS,
        local_experts=MOE_GLOBAL_EXPERTS // comm_ep,
        tokens_per_rank=DECODE_TOKENS,
        topk=M.num_experts_per_tok,
        hidden_size=M.hidden_size,
        recv_capacity_per_expert=comm_ep * DECODE_TOKENS,
    )


def expert_destination(global_expert: int, comm_ep: int) -> tuple[int, int]:
    """Map a global expert to a contiguous physical rank shard."""

    contract = communication_contract(comm_ep)
    if global_expert < 0 or global_expert >= contract.global_experts:
        raise ValueError(
            f"global expert must be in [0, {contract.global_experts}); got {global_expert}"
        )
    return divmod(global_expert, contract.local_experts)


def build_all_to_all_plan(
    routes_by_rank: Sequence[Sequence[Sequence[int]]],
    comm_ep: int,
) -> AllToAllPlan:
    """Validate ``[COMM_EP, tokens, topk]`` routes and lower their data movement."""

    contract = communication_contract(comm_ep)
    if len(routes_by_rank) != contract.communication_ep:
        raise ValueError(
            f"route matrix needs {contract.communication_ep} source ranks; got {len(routes_by_rank)}"
        )

    send_counts = [
        [0 for _ in range(contract.communication_ep)]
        for _ in range(contract.communication_ep)
    ]
    recv_expert_counts = [
        [0 for _ in range(contract.local_experts)]
        for _ in range(contract.communication_ep)
    ]
    lowered_routes: list[DispatchRoute] = []

    for source_rank, token_routes in enumerate(routes_by_rank):
        if len(token_routes) != contract.tokens_per_rank:
            raise ValueError(
                f"source rank {source_rank} needs {contract.tokens_per_rank} token rows; "
                f"got {len(token_routes)}"
            )
        for source_token, selected_experts in enumerate(token_routes):
            if len(selected_experts) != contract.topk:
                raise ValueError(
                    f"rank {source_rank} token {source_token} needs topk={contract.topk}; "
                    f"got {len(selected_experts)}"
                )
            if any(isinstance(expert, bool) for expert in selected_experts):
                raise ValueError(
                    f"rank {source_rank} token {source_token} expert routes must be integers"
                )
            try:
                expert_ids = tuple(integer_index(expert) for expert in selected_experts)
            except TypeError as error:
                raise ValueError(
                    f"rank {source_rank} token {source_token} expert routes must be integers"
                ) from error
            if len(set(expert_ids)) != contract.topk:
                raise ValueError(
                    f"rank {source_rank} token {source_token} contains duplicate expert routes"
                )
            for topk_slot, global_expert in enumerate(expert_ids):
                destination_rank, local_expert = expert_destination(global_expert, comm_ep)
                send_counts[source_rank][destination_rank] += 1
                recv_expert_counts[destination_rank][local_expert] += 1
                lowered_routes.append(
                    DispatchRoute(
                        source_rank=source_rank,
                        source_token=source_token,
                        topk_slot=topk_slot,
                        return_route=source_token * contract.topk + topk_slot,
                        global_expert=global_expert,
                        destination_rank=destination_rank,
                        local_expert=local_expert,
                    )
                )

    for destination_rank, expert_counts in enumerate(recv_expert_counts):
        for local_expert, count in enumerate(expert_counts):
            if count > contract.recv_capacity_per_expert:
                raise ValueError(
                    f"COMM_EP{comm_ep} rank {destination_rank} local expert {local_expert} "
                    f"receives {count} rows, above capacity {contract.recv_capacity_per_expert}"
                )

    return AllToAllPlan(
        contract=contract,
        routes=tuple(lowered_routes),
        send_counts=tuple(tuple(row) for row in send_counts),
        recv_expert_counts=tuple(tuple(row) for row in recv_expert_counts),
    )


def execute_distributed_communication(
    *,
    comm_ep: int,
    device_ids: Sequence[int],
) -> NoReturn:
    """Validate launch shape, then fail because hardware execution is unvalidated."""

    contract = communication_contract(comm_ep)
    normalized_devices = tuple(int(device) for device in device_ids)
    if len(normalized_devices) != contract.communication_ep:
        raise ValueError(
            f"COMM_EP{comm_ep} needs exactly {contract.communication_ep} devices; "
            f"got {normalized_devices}"
        )
    if len(set(normalized_devices)) != len(normalized_devices):
        raise ValueError(f"COMM_EP{comm_ep} device ids must be unique; got {normalized_devices}")
    if any(device < 0 for device in normalized_devices):
        raise ValueError(f"device ids must be non-negative; got {normalized_devices}")
    raise DistributedCommunicationUnavailable(
        f"COMM_EP{comm_ep} dispatch/combine is contract-only on this server. "
        "No dedicated distributed kernel has been implemented or validated; "
        "do not report this path as measured EP128 or measured communication."
    )


def _device_ids(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comm-ep", type=int, choices=COMM_EP_CHOICES, default=COMM_EP_CHOICES[0])
    parser.add_argument(
        "-d",
        "--device",
        type=_device_ids,
        default=(),
        help="comma-separated device ids; used only by the fail-fast execution adapter",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="request distributed execution; currently fails because the path is unvalidated",
    )
    args = parser.parse_args()

    if args.execute:
        try:
            execute_distributed_communication(comm_ep=args.comm_ep, device_ids=args.device)
        except (DistributedCommunicationUnavailable, ValueError) as error:
            parser.error(str(error))

    contract = communication_contract(args.comm_ep)
    print(json.dumps(contract.report_row(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
