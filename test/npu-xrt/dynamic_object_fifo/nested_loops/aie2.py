#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.

# REQUIRES: ryzen_ai, valid_xchess_license
#
# RUN: %python %S/aie2.py > ./aie2.mlir
# RUN: aie-opt --aie-objectFifo-stateful-transform=dynamic-objFifos ./aie2.mlir | FileCheck %s
# CHECK: %tile_0_0 = aie.tile(0, 0)
# CHECK: %tile_0_2 = aie.tile(0, 2)
# CHECK: %out_cons_prod_lock = aie.lock(%tile_0_0, 2) {init = 1 : i32, sym_name = "out_cons_prod_lock"}
# CHECK: %out_cons_cons_lock = aie.lock(%tile_0_0, 3) {init = 0 : i32, sym_name = "out_cons_cons_lock"}
# CHECK: %out_buff_0 = aie.buffer(%tile_0_2) {sym_name = "out_buff_0"} : memref<10xi32> 
# CHECK: %out_buff_1 = aie.buffer(%tile_0_2) {sym_name = "out_buff_1"} : memref<10xi32> 
# CHECK: %out_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "out_prod_lock"}
# CHECK: %out_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "out_cons_lock"}
# CHECK: %in_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "in_cons_buff_0"} : memref<10xi32> 
# CHECK: %in_cons_buff_1 = aie.buffer(%tile_0_2) {sym_name = "in_cons_buff_1"} : memref<10xi32> 
# CHECK: %in_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "in_cons_prod_lock"}
# CHECK: %in_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "in_cons_cons_lock"}
# CHECK: %in_prod_lock = aie.lock(%tile_0_0, 0) {init = 1 : i32, sym_name = "in_prod_lock"}
# CHECK: %in_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "in_cons_lock"}
# CHECK: aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
# CHECK: aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)
# CHECK: func.func private @passthrough_10_i32(memref<10xi32>, memref<10xi32>)
# CHECK: %buffer_0_2 = aie.buffer(%tile_0_2) : memref<2xindex> 
# CHECK: %core_0_2 = aie.core(%tile_0_2) {
# CHECK:   %c0 = arith.constant 0 : index
# CHECK:   %c0_0 = arith.constant 0 : index
# CHECK:   %c2 = arith.constant 2 : index
# CHECK:   memref.store %c0, %buffer_0_2[%c0_0] : memref<2xindex>
# CHECK:   %c1 = arith.constant 1 : index
# CHECK:   %c2_1 = arith.constant 2 : index
# CHECK:   memref.store %c0, %buffer_0_2[%c1] : memref<2xindex>
# CHECK:   %c0_2 = arith.constant 0 : index
# CHECK:   %c5 = arith.constant 5 : index
# CHECK:   %c1_3 = arith.constant 1 : index
# CHECK:   scf.for %arg0 = %c0_2 to %c5 step %c1_3 {
# CHECK:     aie.use_lock(%in_cons_cons_lock, AcquireGreaterEqual, 1)
# CHECK:     %0 = memref.load %buffer_0_2[%c1] : memref<2xindex>
# CHECK:     %1 = scf.index_switch %0 -> memref<10xi32> 
# CHECK:     case 0 {
# CHECK:       scf.yield %in_cons_buff_0 : memref<10xi32>
# CHECK:     }
# CHECK:     case 1 {
# CHECK:       scf.yield %in_cons_buff_1 : memref<10xi32>
# CHECK:     }
# CHECK:     default {
# CHECK:       scf.yield %in_cons_buff_0 : memref<10xi32>
# CHECK:     }
# CHECK:     %c0_4 = arith.constant 0 : index
# CHECK:     %c5_5 = arith.constant 5 : index
# CHECK:     %c1_6 = arith.constant 1 : index
# CHECK:     scf.for %arg1 = %c0_4 to %c5_5 step %c1_6 {
# CHECK:       aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)
# CHECK:       %5 = memref.load %buffer_0_2[%c0_0] : memref<2xindex>
# CHECK:       %6 = scf.index_switch %5 -> memref<10xi32> 
# CHECK:       case 0 {
# CHECK:         scf.yield %out_buff_0 : memref<10xi32>
# CHECK:       }
# CHECK:       case 1 {
# CHECK:         scf.yield %out_buff_1 : memref<10xi32>
# CHECK:       }
# CHECK:       default {
# CHECK:         scf.yield %out_buff_0 : memref<10xi32>
# CHECK:       }
# CHECK:       func.call @passthrough_10_i32(%1, %6) : (memref<10xi32>, memref<10xi32>) -> ()
# CHECK:       aie.use_lock(%out_cons_lock, Release, 1)
# CHECK:       %7 = memref.load %buffer_0_2[%c0_0] : memref<2xindex>
# CHECK:       %c1_8 = arith.constant 1 : index
# CHECK:       %8 = arith.addi %7, %c1_8 : index
# CHECK:       %9 = arith.remsi %8, %c2 : index
# CHECK:       memref.store %9, %buffer_0_2[%c0_0] : memref<2xindex>
# CHECK:     }
# CHECK:     aie.use_lock(%in_cons_prod_lock, Release, 1)
# CHECK:     %2 = memref.load %buffer_0_2[%c1] : memref<2xindex>
# CHECK:     %c1_7 = arith.constant 1 : index
# CHECK:     %3 = arith.addi %2, %c1_7 : index
# CHECK:     %4 = arith.remsi %3, %c2_1 : index
# CHECK:     memref.store %4, %buffer_0_2[%c1] : memref<2xindex>
# CHECK:   }
# CHECK:   aie.end
# CHECK: } {link_with = "kernel.o"}

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
