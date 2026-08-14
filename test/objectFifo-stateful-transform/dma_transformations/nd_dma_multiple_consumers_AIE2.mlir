//===- nd_dma_multiple_consumers_AIE2.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: May 9th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_2:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_3:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_4:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "of3_cons_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "of3_cons_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_4]]) {init = 2 : i32, sym_name = "of3_cons_prod_lock_0"}
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_4]]) {init = 0 : i32, sym_name = "of3_cons_cons_lock_0"}
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "of3_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "of3_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_3]]) {init = 2 : i32, sym_name = "of3_prod_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_3]]) {init = 0 : i32, sym_name = "of3_cons_lock_0"}
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of1_cons_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of1_cons_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_2]]) {init = 2 : i32, sym_name = "of1_cons_prod_lock_0"}
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_2]]) {init = 0 : i32, sym_name = "of1_cons_cons_lock_0"}
// CHECK:           %[[VAL_17:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of1_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_18:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of1_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_19:.*]] = aie.lock(%[[VAL_0]]) {init = 2 : i32, sym_name = "of1_prod_lock_0"}
// CHECK:           %[[VAL_20:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK:           %[[VAL_21:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of0_1_cons_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_22:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of0_1_cons_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_23:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of0_1_cons_buff_2"} : memref<256xi32>
// CHECK:           %[[VAL_24:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of0_1_cons_buff_3"} : memref<256xi32>
// CHECK:           %[[VAL_25:.*]] = aie.lock(%[[VAL_2]]) {init = 4 : i32, sym_name = "of0_1_cons_prod_lock_0"}
// CHECK:           %[[VAL_26:.*]] = aie.lock(%[[VAL_2]]) {init = 0 : i32, sym_name = "of0_1_cons_cons_lock_0"}
// CHECK:           %[[VAL_27:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of0_0_cons_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_28:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of0_0_cons_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_29:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of0_0_cons_buff_2"} : memref<256xi32>
// CHECK:           %[[VAL_30:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of0_0_cons_buff_3"} : memref<256xi32>
// CHECK:           %[[VAL_31:.*]] = aie.lock(%[[VAL_1]]) {init = 4 : i32, sym_name = "of0_0_cons_prod_lock_0"}
// CHECK:           %[[VAL_32:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "of0_0_cons_cons_lock_0"}
// CHECK:           %[[VAL_33:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_34:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_35:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_2"} : memref<256xi32>
// CHECK:           %[[VAL_36:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_3"} : memref<256xi32>
// CHECK:           %[[VAL_37:.*]] = aie.lock(%[[VAL_0]]) {init = 4 : i32, sym_name = "of0_prod_lock_0"}
// CHECK:           %[[VAL_38:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of0_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 1, %[[VAL_2]], DMA : 1)
// CHECK:           aie.flow(%[[VAL_3]], DMA : 0, %[[VAL_4]], DMA : 0)
// CHECK:           %[[VAL_39:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_40:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_41:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_38]], AcquireGreaterEqual, %[[VAL_40]])
// CHECK:             aie.dma_bd(%[[VAL_33]] : memref<256xi32> offset = 0 len = 256 sizes = [16, 16, 1] strides = [1, 16, 1])
// CHECK:             aie.use_lock(%[[VAL_37]], Release, %[[VAL_40]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_38]], AcquireGreaterEqual, %[[VAL_40]])
// CHECK:             aie.dma_bd(%[[VAL_34]] : memref<256xi32> offset = 0 len = 256 sizes = [16, 16, 1] strides = [1, 16, 1])
// CHECK:             aie.use_lock(%[[VAL_37]], Release, %[[VAL_40]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_38]], AcquireGreaterEqual, %[[VAL_40]])
// CHECK:             aie.dma_bd(%[[VAL_35]] : memref<256xi32> offset = 0 len = 256 sizes = [16, 16, 1] strides = [1, 16, 1])
// CHECK:             aie.use_lock(%[[VAL_37]], Release, %[[VAL_40]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_38]], AcquireGreaterEqual, %[[VAL_40]])
// CHECK:             aie.dma_bd(%[[VAL_36]] : memref<256xi32> offset = 0 len = 256 sizes = [16, 16, 1] strides = [1, 16, 1])
// CHECK:             aie.use_lock(%[[VAL_37]], Release, %[[VAL_40]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             %[[VAL_42:.*]] = aie.dma_start(MM2S, 1, ^bb6, ^bb8)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[VAL_20]], AcquireGreaterEqual, %[[VAL_40]])
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<256xi32> offset = 0 len = 256 sizes = [128] strides = [2])
// CHECK:             aie.use_lock(%[[VAL_19]], Release, %[[VAL_40]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[VAL_20]], AcquireGreaterEqual, %[[VAL_40]])
// CHECK:             aie.dma_bd(%[[VAL_18]] : memref<256xi32> offset = 0 len = 256 sizes = [128] strides = [2])
// CHECK:             aie.use_lock(%[[VAL_19]], Release, %[[VAL_40]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb8:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_43:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_44:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_45:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_31]], AcquireGreaterEqual, %[[VAL_44]])
// CHECK:             aie.dma_bd(%[[VAL_27]] : memref<256xi32> offset = 0 len = 256 sizes = [1] strides = [1])
// CHECK:             aie.use_lock(%[[VAL_32]], Release, %[[VAL_44]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_31]], AcquireGreaterEqual, %[[VAL_44]])
// CHECK:             aie.dma_bd(%[[VAL_28]] : memref<256xi32> offset = 0 len = 256 sizes = [1] strides = [1])
// CHECK:             aie.use_lock(%[[VAL_32]], Release, %[[VAL_44]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_31]], AcquireGreaterEqual, %[[VAL_44]])
// CHECK:             aie.dma_bd(%[[VAL_29]] : memref<256xi32> offset = 0 len = 256 sizes = [1] strides = [1])
// CHECK:             aie.use_lock(%[[VAL_32]], Release, %[[VAL_44]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_31]], AcquireGreaterEqual, %[[VAL_44]])
// CHECK:             aie.dma_bd(%[[VAL_30]] : memref<256xi32> offset = 0 len = 256 sizes = [1] strides = [1])
// CHECK:             aie.use_lock(%[[VAL_32]], Release, %[[VAL_44]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_46:.*]] = aie.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_47:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_48:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_25]], AcquireGreaterEqual, %[[VAL_47]])
// CHECK:             aie.dma_bd(%[[VAL_21]] : memref<256xi32> offset = 0 len = 256 sizes = [3] strides = [4])
// CHECK:             aie.use_lock(%[[VAL_26]], Release, %[[VAL_47]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_25]], AcquireGreaterEqual, %[[VAL_47]])
// CHECK:             aie.dma_bd(%[[VAL_22]] : memref<256xi32> offset = 0 len = 256 sizes = [3] strides = [4])
// CHECK:             aie.use_lock(%[[VAL_26]], Release, %[[VAL_47]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_25]], AcquireGreaterEqual, %[[VAL_47]])
// CHECK:             aie.dma_bd(%[[VAL_23]] : memref<256xi32> offset = 0 len = 256 sizes = [3] strides = [4])
// CHECK:             aie.use_lock(%[[VAL_26]], Release, %[[VAL_47]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_25]], AcquireGreaterEqual, %[[VAL_47]])
// CHECK:             aie.dma_bd(%[[VAL_24]] : memref<256xi32> offset = 0 len = 256 sizes = [3] strides = [4])
// CHECK:             aie.use_lock(%[[VAL_26]], Release, %[[VAL_47]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             %[[VAL_49:.*]] = aie.dma_start(S2MM, 1, ^bb6, ^bb8)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_47]])
// CHECK:             aie.dma_bd(%[[VAL_13]] : memref<256xi32> offset = 0 len = 256)
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_47]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_47]])
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<256xi32> offset = 0 len = 256)
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_47]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb8:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_50:.*]] = aie.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_51:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_52:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_51]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<256xi32> offset = 0 len = 256)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_51]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_51]])
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<256xi32> offset = 0 len = 256)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_51]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_53:.*]] = aie.mem(%[[VAL_4]]) {
// CHECK:             %[[VAL_54:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_55:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_54]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<256xi32> offset = 0 len = 256 sizes = [9] strides = [9])
// CHECK:             aie.use_lock(%[[VAL_8]], Release, %[[VAL_54]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_54]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<256xi32> offset = 0 len = 256 sizes = [9] strides = [9])
// CHECK:             aie.use_lock(%[[VAL_8]], Release, %[[VAL_54]])
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
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @of0 (%tile12 dimensionsToStream [<size = 16, stride = 1>, <size = 16, stride = 16>, <size = 1, stride = 1>], // transpose
                         {%tile13 dimensionsFromStream [<size = 1, stride = 1>],
                          %tile33 dimensionsFromStream [<size = 3, stride = 4>]},
                         4 : i32) : !aie.objectfifo<memref<256xi32>>

    aie.objectfifo @of1 (%tile12 dimensionsToStream [<size = 128, stride = 2>], {%tile33},
                         2 : i32) : !aie.objectfifo<memref<256xi32>>

    aie.objectfifo @of3 (%tile22, {%tile23 dimensionsFromStream [<size = 9, stride = 9>]},
                         2 : i32) : !aie.objectfifo<memref<256xi32>>
 }
}
