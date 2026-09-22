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
from aie.iron import Flow, Program, Runtime
from aie.iron.device import NPU2Col1, Tile

devmem_ty = np.ndarray[(16, 16, 512), np.dtype[np.int8]]
back_ty = np.ndarray[(8, 8, 512), np.dtype[np.int8]]


def build(register=True, body=None):
    # No tile_type on either tile: the Device infers both from coordinates, and
    # fill/drain have to follow that rather than the unset hint.
    shim, core = Tile(0, 0), Tile(0, 2)
    into = Flow(shim, core, src_channel=0, dst_channel=1)
    out = Flow(core, shim, src_channel=0, dst_channel=1)

    def sequence(a, c):
        (body or _move)(into, out, a, c)

    rt = Runtime(sequence, [devmem_ty, back_ty])
    if register:
        rt.add_flow(into)
        rt.add_flow(out)
    return Program(NPU2Col1(), rt).resolve_program()


def _move(into, out, a, c):
    into.fill(a, tap=a[0::2, 1::2, ...])
    out.drain(c, tap=c[...], wait=True)


print("\nTEST: names_its_own_shim_channels")
print(build())

# Named after the channel each route ends at, and declared for both. The lazy
# declarations follow the runtime sequence; symbols are position-independent.
# CHECK-LABEL: names_its_own_shim_channels
# The slice reaches the descriptor as the offset and steps it describes.
# CHECK: aiex.dma_configure_task_for @shim_0_0_mm2s_0
# CHECK:   aie.dma_bd({{.*}} offset = 512 len = 32768 sizes = [1, 8, 8, 512] strides = [0, 16384, 1024, 1])
# CHECK: aiex.dma_configure_task_for @shim_0_0_s2mm_1
# CHECK:   aie.dma_bd({{.*}} offset = 0 len = 32768 sizes = [1, 8, 8, 512] strides = [0, 4096, 512, 1])
# CHECK-DAG: aie.shim_dma_allocation @shim_0_0_mm2s_0(%{{.*}}, MM2S, 0)
# CHECK-DAG: aie.shim_dma_allocation @shim_0_0_s2mm_1(%{{.*}}, S2MM, 1)


print("\nTEST: rejects_what_would_not_be_emitted")
try:
    build(register=False)
except ValueError as e:
    print(f"unregistered: {e}")

try:
    build(body=lambda into, out, a, c: out.fill(a))
except ValueError as e:
    print(f"wrong direction: {e}")

# CHECK-LABEL: rejects_what_would_not_be_emitted
# CHECK: unregistered: Flow must be registered with rt.add_flow(flow)
# CHECK: wrong direction: fill() sends data into the array
