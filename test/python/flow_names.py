# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""Test that a program names its routes the same in every build.

An unnamed Flow or PacketFlow is named by the configuration it is registered
with, in registration order, however many others the process built before it.
"""

import numpy as np
from aie.dialects._aie_enum_gen import AIETileType
from aie.iron import Flow, PacketFlow, Program, Runtime
from aie.iron.device import NPU2Col1, Tile

ty = np.ndarray[(256,), np.dtype[np.int32]]


def build(names=(None, None)):
    shim = Tile(0, 0)
    mem = Tile(tile_type=AIETileType.MemTile)
    # Built as arguments to a call, the flows take fallback names rather
    # than the list's.
    flows = list(
        (
            Flow(shim, mem, name=names[0]),
            Flow(mem, shim, name=names[1]),
            PacketFlow(3, shim, mem),
        )
    )

    def sequence(a, c):
        flows[0].fill(a)
        flows[2].fill(a)
        flows[1].drain(c, wait=True)

    rt = Runtime(sequence, [ty, ty])
    for flow in flows:
        rt.add_flow(flow)
    return str(Program(NPU2Col1(), rt).resolve_program())


print("\nTEST: names_routes_per_program")
first = build()
Flow(Tile(0, 0), Tile(0, 1))
second = build()
assert first == second
print(second)

# CHECK-LABEL: names_routes_per_program
# CHECK-DAG: aie.route_endpoint @flow0_src
# CHECK-DAG: aie.route_endpoint @flow1_dst
# CHECK-DAG: aie.shim_dma_allocation @packetflow0_src
# CHECK-DAG: aiex.dma_configure_task_for @flow0_src
# CHECK-DAG: aiex.dma_configure_task_for @packetflow0_src
# CHECK-DAG: aiex.dma_configure_task_for @flow1_dst


print("\nTEST: unnamed_routes_skip_given_names")
print(build(names=(None, "flow0")))

# CHECK-LABEL: unnamed_routes_skip_given_names
# CHECK-DAG: aiex.dma_configure_task_for @flow1_src
# CHECK-DAG: aiex.dma_configure_task_for @flow0_dst


print("\nTEST: unregistered_route_has_no_name")
loose = Flow(Tile(0, 0), Tile(0, 1))
try:
    loose.endpoint(loose.src).symbol
except ValueError as e:
    print(e)
assert loose.name is None

# CHECK-LABEL: unregistered_route_has_no_name
# CHECK: This Flow has no name yet: an unnamed route is named by the configuration it is registered with (rt.add_flow) when that resolves.
