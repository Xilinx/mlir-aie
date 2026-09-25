# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""Test PacketFlow.fill / PacketFlow.drain.

A fill stamps its route's packet ID into the shim descriptor, which is what
steers the packets once they leave the shim. Packet routes are how several
routes share one shim channel, so those routes share that channel's allocation
rather than each defining it.
"""

import numpy as np
from aie.iron import PacketDest, PacketFlow, Program, Runtime
from aie.iron.device import NPU2Col1, Tile

ty = np.ndarray[(256,), np.dtype[np.int32]]


def build(body, register=True, extra_dsts=(), fill_extra_dsts=()):
    shim, mem = Tile(0, 0), Tile(0, 1)
    flows = [
        PacketFlow(0, shim, mem, keep_pkt_header=True, extra_dsts=fill_extra_dsts),
        PacketFlow(1, shim, mem, keep_pkt_header=True),
        PacketFlow(2, mem, shim, src_channel=2, extra_dsts=extra_dsts),
    ]

    def sequence(a, c):
        body(*flows, a, c)

    rt = Runtime(sequence, [ty, ty])
    if register:
        for f in flows:
            rt.add_flow(f)
    return Program(NPU2Col1(), rt).resolve_program()


def _move(in0, in1, out, a, c):
    in0.fill(a)
    in1.fill(a, tap=a[128:])
    out.drain(c, wait=True)


print("\nTEST: stamps_packet_ids_on_a_shared_channel")
module = build(_move)
print(module)
assert str(module).count("aie.shim_dma_allocation ") == 2

# CHECK-LABEL: stamps_packet_ids_on_a_shared_channel
# CHECK: aiex.dma_configure_task_for @shim_0_0_mm2s_0
# CHECK:   aie.dma_bd({{.*}} len = 256 {{.*}}) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
# CHECK: aiex.dma_configure_task_for @shim_0_0_mm2s_0
# CHECK:   aie.dma_bd({{.*}} offset = 128 len = 128 {{.*}}) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
# CHECK: aiex.dma_configure_task_for @shim_0_0_s2mm_0
# CHECK-NOT: packet_info
# CHECK: } {issue_token = true}
# CHECK-DAG: aie.shim_dma_allocation @shim_0_0_mm2s_0(%{{.*}}, MM2S, 0)
# CHECK-DAG: aie.shim_dma_allocation @shim_0_0_s2mm_0(%{{.*}}, S2MM, 0)


print("\nTEST: fills_a_packet_broadcast")
module = build(
    lambda i0, i1, out, a, c: i0.fill(a),
    fill_extra_dsts=[PacketDest(Tile(0, 2)), PacketDest(Tile(0, 3))],
)
assert module.operation.verify()
print(module)
assert str(module).count("aie.shim_dma_allocation ") == 1

# CHECK-LABEL: fills_a_packet_broadcast
# CHECK: aiex.dma_configure_task_for @shim_0_0_mm2s_0
# CHECK: aie.dma_bd({{.*}} len = 256 {{.*}}) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
# CHECK: aie.packet_flow(0)
# CHECK: aie.packet_source
# CHECK: aie.packet_dest
# CHECK: aie.packet_dest
# CHECK: aie.packet_dest
# CHECK: aie.shim_dma_allocation @shim_0_0_mm2s_0(%{{.*}}, MM2S, 0)


print("\nTEST: rejects_what_would_not_be_emitted")
cases = {
    "unregistered": dict(body=_move, register=False),
    "fill from a mem tile": dict(body=lambda i0, i1, out, a, c: out.fill(a)),
    "drain into a mem tile": dict(body=lambda i0, i1, out, a, c: i0.drain(c)),
    "drain of a fan-out": dict(
        body=lambda i0, i1, out, a, c: out.drain(c),
        extra_dsts=[PacketDest(Tile(0, 2))],
    ),
    "fill with an extra shim": dict(
        body=lambda i0, i1, out, a, c: i0.fill(a),
        fill_extra_dsts=[PacketDest(Tile(0, 2)), PacketDest(Tile(0, 0), channel=1)],
    ),
}
for name, kwargs in cases.items():
    try:
        build(**kwargs)
    except ValueError as e:
        print(f"{name}: {e}")
    else:
        raise AssertionError(f"Expected {name} to fail")

# CHECK-LABEL: rejects_what_would_not_be_emitted
# CHECK: unregistered: PacketFlow must be registered with rt.add_flow(flow)
# CHECK: fill from a mem tile: fill() sends data into the array
# CHECK: drain into a mem tile: drain() reads results back out of the array
# CHECK: drain of a fan-out: drain() reads results back out of the array
# CHECK: fill with an extra shim: PacketFlow.fill()/drain() require exactly one shim endpoint
