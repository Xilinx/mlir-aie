# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s
# RUN: %python %s | aie-opt --aie-place-tiles --aie-objectFifo-stateful-transform --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu | FileCheck %s --check-prefix=NPU

# Runtime scalars of several widths and signedness as tap sizes and offsets:
# each reaches the BD at its own width, and the lowering widens it to i64
# for its field and bounds guards. Unsigned scalars are zero-extended with
# emitc.cast first so taplib's signed checks see their true value.

import numpy as np

from aie.helpers.taplib import TensorAccessPattern
from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.device import NPU1Col1

N = 4096
vec_ty = np.ndarray[(N,), np.dtype[np.int32]]
tile_ty = np.ndarray[(16,), np.dtype[np.int32]]

of_in = ObjectFifo(tile_ty, name="of_in")
of_out = of_in.cons().forward(name="of_out")


def seq(a, b, n, m, u8, u32, in_prod, out_cons):
    in_prod.fill(a, tap=TensorAccessPattern((N,), 0, [1, 1, n, 16], [0, 0, 16, 1]))
    in_prod.fill(a, tap=TensorAccessPattern((N,), m, [1, 1, 4, 16], [0, 0, 16, 1]))
    in_prod.fill(a, tap=TensorAccessPattern((N,), u32, [1, 1, u8, 16], [0, 0, 32, 1]))
    out_cons.drain(b, wait=True)


rt = Runtime(
    seq,
    [
        vec_ty,
        vec_ty,
        np.int32,
        np.int64,
        np.uint8,
        np.uint32,
        of_in.prod(),
        of_out.cons(),
    ],
)
print(Program(NPU1Col1(), rt).resolve_program())

# CHECK-LABEL: aie.runtime_sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: i32, %arg3: i64, %arg4: ui8, %arg5: ui32)
# CHECK:         cf.assert %{{.*}}, "All sizes must be >= 1, but got [1, 1, <runtime>, 16]"
# CHECK:         aiex.dma_configure_task_for @of_in {
# CHECK-NEXT:      aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 sizes = [1, 1, %arg2 : i32, 16] strides = [0, 0, 16, 1])
# CHECK:         aiex.dma_configure_task_for @of_in {
# CHECK-NEXT:      aie.dma_bd(%arg0 : memref<4096xi32> offset = %arg3 : i64 len = 64

# CHECK:         %[[U32:.*]] = emitc.cast %arg5 : ui32 to i64
# CHECK-NEXT:    %[[U8:.*]] = emitc.cast %arg4 : ui8 to i32
# CHECK:         aiex.dma_configure_task_for @of_in {
# CHECK-NEXT:      aie.dma_bd(%arg0 : memref<4096xi32> offset = %[[U32]] : i64 sizes = [1, 1, %[[U8]] : i32, 16] strides = [0, 0, 32, 1])

# A contiguous runtime walk is encoded linear, so only its transfer length is
# guarded; a strided one is encoded ND and its size gets the 10-bit field
# guard.
# NPU-LABEL: aie.runtime_sequence
# NPU-NOT:   aiex.dma_configure_task_for
# NPU:       arith.extui %arg2 : i32 to i64
# NPU:       cf.assert %{{.*}}, "a runtime DMA transfer exceeds the 4294967295-granule BD buffer_length"
# NPU:       cf.assert %{{.*}}, "a runtime DMA access runs past the end of its 4096-element host buffer"
# NPU:       aiex.npu.blockwrite_values
# NPU:       arith.muli %arg3, %{{.*}} : i64
# NPU:       aiex.npu.address_patch

# NPU:       %[[U8:.*]] = emitc.cast %arg4 : ui8 to i32
# NPU:       arith.extui %[[U8]] : i32 to i64
# NPU:       cf.assert %{{.*}}, "a runtime DMA d1 size must be in [1:1023]"
# NPU:       aiex.npu.blockwrite_values
# NPU-NOT:   aiex.dma_configure_task_for
