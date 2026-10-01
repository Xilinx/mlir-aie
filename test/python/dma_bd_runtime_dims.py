# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s
# RUN: %python %s | aie-opt --aie-place-tiles --aie-objectFifo-stateful-transform --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu | FileCheck %s --check-prefix=NPU

# Runtime scalars of several widths and signedness as tap sizes and offsets:
# each reaches the BD as a signless i64 size or i32 offset. A narrower signed
# value is sign-extended, an unsigned one zero-extended, and a wider one
# guarded with aiex.npu.require before it is truncated.

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

# A signed i32 size is sign-extended; the i64 transfer length it feeds is
# guarded and truncated to dma_bd's i32 operand.
# CHECK-LABEL: aie.runtime_sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: i32, %arg3: i64, %arg4: ui8, %arg5: ui32)
# CHECK:       aiex.dma_configure_task_for @of_in {
# CHECK-NEXT:    %[[N:.*]] = arith.extsi %arg2 : i32 to i64
# CHECK:         aiex.npu.require(%{{.*}}) {message = "a runtime DMA transfer length does not fit in 32 bits"}
# CHECK-NEXT:    %[[LEN:.*]] = arith.trunci %{{.*}} : i64 to i32
# CHECK-NEXT:    aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = %[[LEN]] sizes = [1, 1, %[[N]], 16] strides = [0, 0, 16, 1])

# An i64 offset is guarded and truncated to i32.
# CHECK:       aiex.dma_configure_task_for @of_in {
# CHECK:         aiex.npu.require(%{{.*}}) {message = "a runtime DMA offset does not fit in 32 bits"}
# CHECK-NEXT:    %[[OFF:.*]] = arith.trunci %arg3 : i64 to i32
# CHECK-NEXT:    aie.dma_bd(%arg0 : memref<4096xi32> offset = %[[OFF]] len = 64

# Unsigned scalars are zero-extended with emitc.cast before taplib's bounds
# checks see them, so a large ui32 cannot pass as a negative i32.
# CHECK:       %[[U32:.*]] = emitc.cast %arg5 : ui32 to i64
# CHECK-NEXT:  %[[U8:.*]] = emitc.cast %arg4 : ui8 to i32
# CHECK:       aiex.dma_configure_task_for @of_in {
# CHECK-NEXT:    %[[U8W:.*]] = arith.extsi %[[U8]] : i32 to i64
# CHECK:         arith.trunci %[[U32]] : i64 to i32
# CHECK:         aie.dma_bd(%arg0 : memref<4096xi32> offset = %{{.*}} len = %{{.*}} sizes = [1, 1, %[[U8W]], 16] strides = [0, 0, 32, 1])

# The lowering hoists every cast and guard out of the BD blocks and encodes
# each BD into words.
# NPU-LABEL: aie.runtime_sequence
# NPU-NOT:   aiex.dma_configure_task_for
# NPU:       arith.extsi %arg2 : i32 to i64
# NPU:       aiex.npu.require(%{{.*}}) {message = "a runtime DMA size or stride does not fit in 31 bits"}
# NPU:       aiex.npu.blockwrite_values
# NPU:       arith.trunci %arg3 : i64 to i32
# NPU:       aiex.npu.blockwrite_values

# A runtime size in ND mode gets its 10-bit field guard.
# NPU:       emitc.cast %arg5 : ui32 to i64
# NPU:       aiex.npu.assert_bd_field(%{{.*}}) {max = 1023 : i32} : i32
# NPU:       aiex.npu.blockwrite_values
# NPU-NOT:   aiex.dma_configure_task_for
