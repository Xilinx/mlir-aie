# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""Test Flow.fill / Flow.drain.

They name the shim channel themselves and emit the matching
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
from aie.iron.device import NPU2Col1, Tile

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
# CHECK: aiex.dma_configure_task_for @shim_0_0_mm2s_0
# CHECK:   aie.dma_bd({{.*}} offset = 512 len = 32768 sizes = [8, 8, 512] strides = [16384, 1024, 1])
# CHECK: aiex.dma_configure_task_for @shim_0_0_s2mm_1
# CHECK:   aie.dma_bd({{.*}} offset = 0 len = 32768 sizes = [8, 8, 512] strides = [4096, 512, 1])
# CHECK-DAG: aie.shim_dma_allocation @shim_0_0_mm2s_0(%{{.*}}, MM2S, 0)
# CHECK-DAG: aie.shim_dma_allocation @shim_0_0_s2mm_1(%{{.*}}, S2MM, 1)


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


print("\nTEST: rejects_shim_to_shim_transfers")
for symbol in (None, "explicit_shim"):
    for verb in ("fill", "drain"):
        shim = Tile(0, 0)
        flow = Flow(shim, shim, src_channel=0, dst_channel=1, shim_symbol=symbol)

        def sequence(a):
            getattr(flow, verb)(a, tap=a[...])

        rt = Runtime(sequence, [back_ty])
        rt.add_flow(flow)
        try:
            Program(NPU2Col1(), rt).resolve_program()
        except ValueError as e:
            assert "require exactly one shim endpoint" in str(e)
        else:
            raise AssertionError(f"Expected shim-to-shim {verb}() to fail")
print("rejected fill and drain with automatic and explicit symbols")

# CHECK-LABEL: rejects_shim_to_shim_transfers
# CHECK: rejected fill and drain with automatic and explicit symbols


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
