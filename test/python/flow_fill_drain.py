# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""Test Flow.fill / Flow.drain.

They name the shim channel after the Flow and emit the matching
aie.shim_dma_allocation, so a route driven from the runtime sequence needs no
symbol invented by the caller. The slice handed to them is what reaches the
buffer descriptor.

A Flow never registered would otherwise emit a transfer naming an allocation
nothing declares -- which verifies clean and fails much later, so it is rejected
where it happens.
"""

import numpy as np
from aie.dialects._aie_enum_gen import AIETileType
from aie.iron import Flow, Program, Runtime, Worker
from aie.iron.device import NPU2, NPU2Col1, Tile

devmem_ty = np.ndarray[(16, 16, 512), np.dtype[np.int8]]
back_ty = np.ndarray[(8, 8, 512), np.dtype[np.int8]]


def build(register=True, body=None, symbols=(None, None), worker_args=False):
    # No tile_type on either tile: the Device infers both from coordinates, and
    # fill/drain have to follow that rather than the unset hint.
    shim, core = Tile(0, 0), Tile(0, 2)
    into = Flow(shim, core, src_channel=0, dst_channel=1, shim_symbol=symbols[0])
    out = Flow(core, shim, src_channel=0, dst_channel=1, shim_symbol=symbols[1])

    def sequence(a, c):
        (body or _move)(into, out, a, c)

    rt = Runtime(sequence, [devmem_ty, back_ty])
    if register:
        rt.add_flow(into)
        rt.add_flow(out)
    workers = []
    if worker_args:
        worker_tile = Tile(0, 3, tile_type=AIETileType.CoreTile)
        workers.append(
            Worker(lambda *_: None, [into, out], tile=worker_tile, while_true=False)
        )
    return Program(NPU2Col1(), rt, workers=workers).resolve_program()


def _move(into, out, a, c):
    into.fill(a, tap=a[0::2, 1::2, ...])
    out.drain(c, tap=c[...], wait=True)


print("\nTEST: names_its_own_shim_channels")
print(build())

# Flows and allocations resolve after the sequence; symbols are position-independent.
# CHECK-LABEL: names_its_own_shim_channels
# The slice reaches the descriptor as the offset and steps it describes.
# CHECK: aiex.dma_configure_task_for @flow0_src
# CHECK:   aie.dma_bd({{.*}} offset = 512 len = 32768 sizes = [1, 8, 8, 512] strides = [0, 16384, 1024, 1])
# CHECK: aiex.dma_configure_task_for @flow1_dst
# CHECK:   aie.dma_bd({{.*}} offset = 0 len = 32768 sizes = [1, 8, 8, 512] strides = [0, 4096, 512, 1])
# CHECK-DAG: aie.shim_dma_allocation @flow0_src(%{{.*}}, MM2S, 0)
# CHECK-DAG: aie.shim_dma_allocation @flow1_dst(%{{.*}}, S2MM, 1)


print("\nTEST: rejects_what_would_not_be_emitted")
try:
    build(register=False)
except ValueError as e:
    print(f"unregistered: {e}")
else:
    raise AssertionError("Expected an unregistered Flow to fail")

try:
    build(body=lambda into, out, a, c: out.fill(a))
except ValueError as e:
    print(f"wrong direction: {e}")
else:
    raise AssertionError("Expected fill() on a non-shim source to fail")

try:
    build(body=lambda into, out, a, c: into.drain(c))
except ValueError as e:
    print(f"wrong direction: {e}")
else:
    raise AssertionError("Expected drain() on a non-shim destination to fail")

# CHECK-LABEL: rejects_what_would_not_be_emitted
# CHECK: unregistered: Flow must be registered with rt.add_flow(flow)
# CHECK: wrong direction: fill() sends data into the array
# CHECK: wrong direction: drain() reads results back out of the array


print("\nTEST: fills_and_drains_shim_to_shim")


def shim_to_shim(verbs, channels=(1, 0), symbol=None):
    # Two shim tiles, as a gather's feed and drain: fill() is MM2S at the
    # source, drain() S2MM at the destination, each with its own allocation.
    feed, drain = Tile(0, 0), Tile(1, 0)
    flow = Flow(
        feed,
        drain,
        src_channel=channels[0],
        dst_channel=channels[1],
        shim_symbol=symbol,
        name="loop",
    )

    def sequence(a, c):
        for verb, buffer in zip(verbs, (a, c)):
            getattr(flow, verb)(buffer, tap=buffer[...], wait=verb == "drain")

    rt = Runtime(sequence, [back_ty, back_ty])
    rt.add_flow(flow)
    module = Program(NPU2(), rt).resolve_program()
    assert module.operation.verify()
    return module


print(shim_to_shim(("fill", "drain")))

# CHECK-LABEL: fills_and_drains_shim_to_shim
# CHECK: aiex.dma_configure_task_for @loop_src
# CHECK: aiex.dma_configure_task_for @loop_dst
# CHECK: aie.flow(%[[FEED:[a-z_0-9]+]], DMA : 1, %[[DRAIN:[a-z_0-9]+]], DMA : 0)
# CHECK: aie.shim_dma_allocation @loop_src(%[[FEED]], MM2S, 1)
# CHECK: aie.shim_dma_allocation @loop_dst(%[[DRAIN]], S2MM, 0)

print("\nTEST: drains_shim_to_shim_with_a_named_source")
print(shim_to_shim(("drain",), symbol="explicit_shim"))

# shim_symbol names the source, which gets its allocation as before; the
# drain takes the destination's own symbol.
# CHECK-LABEL: drains_shim_to_shim_with_a_named_source
# CHECK: aiex.dma_configure_task_for @loop_dst
# CHECK: aie.shim_dma_allocation @explicit_shim(%{{.*}}, MM2S, 1)
# CHECK: aie.shim_dma_allocation @loop_dst(%{{.*}}, S2MM, 0)

print("\nTEST: routes_shim_to_shim")
print(shim_to_shim(("fill", "drain"), channels=(None, None)))

# Unassigned channels lower to route endpoints, each shim end a transfer
# uses carrying the fifoName that allocates its channel.
# CHECK-LABEL: routes_shim_to_shim
# CHECK: aie.route_endpoint @loop_src(%{{.*}}) DMA {fifoName = "loop_src"}
# CHECK: aie.route_endpoint @loop_dst(%{{.*}}) DMA {fifoName = "loop_dst"}
# CHECK: aie.route from @loop_src to [@loop_dst]


print("\nTEST: one_allocation_per_flow")


def repeat_transfers(into, out, a, c):
    _move(into, out, a, c)
    _move(into, out, a, c)


for symbols in ((None, None), ("input", "output")):
    module = build(body=repeat_transfers, symbols=symbols, worker_args=True)
    text = str(module)
    assert text.count("aie.shim_dma_allocation ") == 2
    assert text.count("aiex.dma_configure_task_for ") == 4
print("repeated transfers reuse automatic and explicit allocations")

# CHECK-LABEL: one_allocation_per_flow
# CHECK: repeated transfers reuse automatic and explicit allocations
