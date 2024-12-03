#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.

# REQUIRES: ryzen_ai, valid_xchess_license
#
# RUN: %python %s | FileCheck %s
#CHECK: %core_0_2 = aie.core(%tile_0_2) {
#CHECK:       %c0 = arith.constant 0 : index
#CHECK:       %c5 = arith.constant 5 : index
#CHECK:       %c1 = arith.constant 1 : index
#CHECK:       scf.for %arg0 = %c0 to %c5 step %c1 {
#CHECK:         %0 = aie.objectfifo.acquire @in(Consume, 1) : !aie.objectfifosubview<memref<10xi32>>
#CHECK:         %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
#CHECK:         %c0_0 = arith.constant 0 : index
#CHECK:         %c5_1 = arith.constant 5 : index
#CHECK:         %c1_2 = arith.constant 1 : index
#CHECK:         scf.for %arg1 = %c0_0 to %c5_1 step %c1_2 {
#CHECK:           %2 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<10xi32>>
#CHECK:           %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
#CHECK:           func.call @passthrough_10_i32(%1, %3) : (memref<10xi32>, memref<10xi32>) -> ()
#CHECK:           aie.objectfifo.release @out(Produce, 1)
#CHECK:         }
#CHECK:         aie.objectfifo.release @in(Consume, 1)
#CHECK:       }
#CHECK:       aie.end
#CHECK:     } {link_with = "kernel.o"}

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.extras.context import mlir_mod_ctx

N = 50
O = 250
n_rows = 5
dev = AIEDevice.npu1_1col
col = 0


def nested_loops():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            tensor_ty = np.ndarray[(N // n_rows,), np.dtype[np.int32]]

            # Tile declarations
            ShimTile = tile(col, 0)
            ComputeTile = tile(col, 2)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, ComputeTile, 2, tensor_ty)
            of_out = object_fifo("out", ComputeTile, ShimTile, 2, tensor_ty)

            # AIE Core Function declarations
            passthrough_10_i32 = external_func(
                "passthrough_10_i32", inputs=[tensor_ty, tensor_ty]
            )

            # Set up compute tiles
            @core(ComputeTile, "kernel.o")
            def core_body():
                for _ in range_(5):
                    elemIn = of_in.acquire(ObjectFifoPort.Consume, 1)
                    for _ in range_(5):
                        elemOut = of_out.acquire(ObjectFifoPort.Produce, 1)
                        passthrough_10_i32(elemIn, elemOut)
                        of_out.release(ObjectFifoPort.Produce, 1)
                    of_in.release(ObjectFifoPort.Consume, 1)

            # To/from AIE-array data movement
            @runtime_sequence(tensor_ty, tensor_ty)
            def sequence(A, C):
                npu_dma_memcpy_nd(
                    metadata=of_in, bd_id=1, mem=A, sizes=[1, 1, 1, N], issue_token=True
                )
                npu_dma_memcpy_nd(metadata=of_out, bd_id=0, mem=C, sizes=[1, 1, 1, O])
                dma_wait(of_in, of_out)

    print(ctx.module)


nested_loops()
