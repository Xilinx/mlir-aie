# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""
Test IRON patterns with Workers and ObjectFifos.
"""

import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from util import construct_and_print_module


# CHECK-LABEL: TEST: passthrough_dma_forward
# CHECK: aie.device(npu2) {
# CHECK:   %[[SHIM_IN:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK:   %[[MEM:.*]] = aie.logical_tile<MemTile>
# CHECK:   %[[SHIM_OUT:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK:   aie.objectfifo @in(%[[SHIM_IN]], {%[[MEM]]}, {{.*}})
# CHECK:   aie.objectfifo @in_fwd(%[[MEM]], {%[[SHIM_OUT]]}, {{.*}})
# CHECK:   aie.objectfifo.link [@in] -> [@in_fwd]
@construct_and_print_module
def passthrough_dma_forward(module):
    """Test ObjectFifo.forward() creates link with LogicalTileOps."""
    N = 4096
    line_size = 1024
    vector_ty = np.ndarray[(N,), np.dtype[np.int32]]
    line_ty = np.ndarray[(line_size,), np.dtype[np.int32]]

    # Pattern: forward() creates intermediate endpoint
    of_in = ObjectFifo(line_ty, name="in")
    of_out = of_in.cons().forward()

    rt = Runtime()
    with rt.sequence(vector_ty, vector_ty, vector_ty) as (a_in, _, c_out):
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), c_out, wait=True)

    module = Program(NPU2(), rt).resolve_program()
    return module


# CHECK-LABEL: TEST: worker_multiple_objectfifos
# CHECK: aie.device(npu2) {
# CHECK:   %[[WORKER:.*]] = aie.logical_tile<CoreTile>(?, ?)
# CHECK:   %[[SHIM1:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK:   %[[SHIM2:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK:   %[[SHIM3:.*]] = aie.logical_tile<ShimNOCTile>
# CHECK:   aie.objectfifo @in1(%[[SHIM1]], {%[[WORKER]]}, {{.*}})
# CHECK:   aie.objectfifo @in2(%[[SHIM2]], {%[[WORKER]]}, {{.*}})
# CHECK:   aie.objectfifo @out(%[[WORKER]], {%[[SHIM3]]}, {{.*}})
# CHECK:   aie.core(%[[WORKER]])
# CHECK:   aie.runtime_sequence
@construct_and_print_module
def worker_multiple_objectfifos(module):
    """Test Worker with multiple ObjectFifos"""
    N = 256
    n = 16
    tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]
    tile_ty = np.ndarray[(n,), np.dtype[np.int32]]

    of_in1 = ObjectFifo(tile_ty, name="in1")
    of_in2 = ObjectFifo(tile_ty, name="in2")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_body(of_in1, of_in2, of_out):
        for _ in range_(4):
            elem_in1 = of_in1.acquire(1)
            elem_in2 = of_in2.acquire(1)
            elem_out = of_out.acquire(1)
            for i in range_(n):
                elem_out[i] = elem_in1[i] * elem_in2[i]
            of_in1.release(1)
            of_in2.release(1)
            of_out.release(1)

    worker = Worker(core_body, [of_in1.cons(), of_in2.cons(), of_out.prod()])

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (A, B, C):
        rt.start(worker)
        rt.fill(of_in1.prod(), A)
        rt.fill(of_in2.prod(), B)
        rt.drain(of_out.cons(), C, wait=True)

    module = Program(NPU2(), rt).resolve_program()
    return module
