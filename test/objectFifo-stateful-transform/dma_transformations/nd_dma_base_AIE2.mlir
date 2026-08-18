//===- nd_dma_base_AIE2.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: May 9th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_2"} : memref<256xi32>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_3"} : memref<256xi32>
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_0]]) {init = 4 : i32, sym_name = "of0_prod_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of0_cons_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of1_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of1_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_0]]) {init = 2 : i32, sym_name = "of1_prod_lock_0"}
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_12:.*]] = aie.buffer(%[[VAL_11]]) {sym_name = "of0_cons_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_11]]) {sym_name = "of0_cons_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_11]]) {sym_name = "of0_cons_buff_2"} : memref<256xi32>
// CHECK:           %[[VAL_15:.*]] = aie.buffer(%[[VAL_11]]) {sym_name = "of0_cons_buff_3"} : memref<256xi32>
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_11]]) {init = 4 : i32, sym_name = "of0_cons_prod_lock_0"}
// CHECK:           %[[VAL_17:.*]] = aie.lock(%[[VAL_11]]) {init = 0 : i32, sym_name = "of0_cons_cons_lock_0"}
// CHECK:           %[[VAL_18:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_19:.*]] = aie.buffer(%[[VAL_18]]) {sym_name = "of1_cons_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_20:.*]] = aie.buffer(%[[VAL_18]]) {sym_name = "of1_cons_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_21:.*]] = aie.lock(%[[VAL_18]]) {init = 2 : i32, sym_name = "of1_cons_prod_lock_0"}
// CHECK:           %[[VAL_22:.*]] = aie.lock(%[[VAL_18]]) {init = 0 : i32, sym_name = "of1_cons_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_11]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 1, %[[VAL_18]], DMA : 0)
// CHECK:           %[[VAL_23:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_24:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_25:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_24]])
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<256xi32> offset = 0 len = 256 sizes = [16, 16, 1] strides = [1, 16, 1])
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_24]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_24]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<256xi32> offset = 0 len = 256 sizes = [16, 16, 1] strides = [1, 16, 1])
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_24]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_24]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<256xi32> offset = 0 len = 256 sizes = [16, 16, 1] strides = [1, 16, 1])
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_24]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_24]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<256xi32> offset = 0 len = 256 sizes = [16, 16, 1] strides = [1, 16, 1])
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_24]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             %[[VAL_26:.*]] = aie.dma_start(MM2S, 1, ^bb6, ^bb8)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_24]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<256xi32> offset = 0 len = 256 sizes = [128] strides = [2])
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_24]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_24]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<256xi32> offset = 0 len = 256 sizes = [128] strides = [2])
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_24]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb8:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = aie.mem(%[[VAL_11]]) {
// CHECK:             %[[VAL_28:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_29:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, %[[VAL_28]])
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<256xi32> offset = 0 len = 256 sizes = [1] strides = [1])
// CHECK:             aie.use_lock(%[[VAL_17]], Release, %[[VAL_28]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, %[[VAL_28]])
// CHECK:             aie.dma_bd(%[[VAL_13]] : memref<256xi32> offset = 0 len = 256 sizes = [1] strides = [1])
// CHECK:             aie.use_lock(%[[VAL_17]], Release, %[[VAL_28]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, %[[VAL_28]])
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<256xi32> offset = 0 len = 256 sizes = [1] strides = [1])
// CHECK:             aie.use_lock(%[[VAL_17]], Release, %[[VAL_28]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, %[[VAL_28]])
// CHECK:             aie.dma_bd(%[[VAL_15]] : memref<256xi32> offset = 0 len = 256 sizes = [1] strides = [1])
// CHECK:             aie.use_lock(%[[VAL_17]], Release, %[[VAL_28]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_30:.*]] = aie.mem(%[[VAL_18]]) {
// CHECK:             %[[VAL_31:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_32:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, %[[VAL_31]])
// CHECK:             aie.dma_bd(%[[VAL_19]] : memref<256xi32> offset = 0 len = 256)
// CHECK:             aie.use_lock(%[[VAL_22]], Release, %[[VAL_31]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, %[[VAL_31]])
// CHECK:             aie.dma_bd(%[[VAL_20]] : memref<256xi32> offset = 0 len = 256)
// CHECK:             aie.use_lock(%[[VAL_22]], Release, %[[VAL_31]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @ndDMAObjFifoAIE2 {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile33 = aie.tile(3, 3)

    // Even if an objectFifo could be implemented in shared memory, as with
    // this case between two adjacent tiles, we need to use DMAs if a data
    // layout transformation with dimensionsToStream and dimensionsFromStream was specified.
    aie.objectfifo @of0 (%tile12 dimensionsToStream [<size = 16, stride = 1>, <size = 16, stride = 16>, <size = 1, stride = 1>], // transpose
                         {%tile13 dimensionsFromStream [<size = 1, stride = 1>]},
                         4 : i32) : !aie.objectfifo<memref<256xi32>>

    aie.objectfifo @of1 (%tile12 dimensionsToStream [<size = 128, stride = 2>], {%tile33},
                         2 : i32) : !aie.objectfifo<memref<256xi32>>
 }
}
