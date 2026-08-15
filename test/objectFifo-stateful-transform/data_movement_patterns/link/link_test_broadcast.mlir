//===- link_test_broadcast.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: July 31st 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "mem_in_1_cons_buff_0"} : memref<3000xi32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "mem_in_1_cons_buff_1"} : memref<3000xi32>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "mem_in_1_cons_buff_2"} : memref<3000xi32>
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "mem_in_1_cons_buff_3"} : memref<3000xi32>
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "mem_in_1_cons_buff_4"} : memref<3000xi32>
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "mem_in_1_cons_buff_5"} : memref<3000xi32>
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "mem_in_1_cons_buff_6"} : memref<3000xi32>
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_1]]) {init = 7 : i32, sym_name = "mem_in_1_cons_prod_lock_0"}
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "mem_in_1_cons_cons_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_12:.*]] = aie.buffer(%[[VAL_11]]) {sym_name = "mem_in_0_cons_buff_0"} : memref<3000xi32>
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_11]]) {sym_name = "mem_in_0_cons_buff_1"} : memref<3000xi32>
// CHECK:           %[[VAL_14:.*]] = aie.lock(%[[VAL_11]]) {init = 2 : i32, sym_name = "mem_in_0_cons_prod_lock_0"}
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_11]]) {init = 0 : i32, sym_name = "mem_in_0_cons_cons_lock_0"}
// CHECK:           %[[VAL_16:.*]] = aie.tile(0, 3)
// CHECK:           %[[VAL_17:.*]] = aie.buffer(%[[VAL_16]]) {sym_name = "mem_out_cons_buff_0"} : memref<3000xi32>
// CHECK:           %[[VAL_18:.*]] = aie.buffer(%[[VAL_16]]) {sym_name = "mem_out_cons_buff_1"} : memref<3000xi32>
// CHECK:           %[[VAL_19:.*]] = aie.buffer(%[[VAL_16]]) {sym_name = "mem_out_cons_buff_2"} : memref<3000xi32>
// CHECK:           %[[VAL_20:.*]] = aie.buffer(%[[VAL_16]]) {sym_name = "mem_out_cons_buff_3"} : memref<3000xi32>
// CHECK:           %[[VAL_21:.*]] = aie.lock(%[[VAL_16]]) {init = 4 : i32, sym_name = "mem_out_cons_prod_lock_0"}
// CHECK:           %[[VAL_22:.*]] = aie.lock(%[[VAL_16]]) {init = 0 : i32, sym_name = "mem_out_cons_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_11]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_16]], DMA : 0)
// CHECK:           %[[VAL_23:.*]] = aie.core(%[[VAL_11]]) {
// CHECK:             %[[VAL_24:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_25:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_26:.*]] = arith.constant 11 : i32
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_24]])
// CHECK:             memref.store %[[VAL_26]], %[[VAL_12]]{{\[}}%[[VAL_25]]] : memref<3000xi32>
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = aie.core(%[[VAL_16]]) {
// CHECK:             %[[VAL_28:.*]] = arith.constant 3 : i32
// CHECK:             %[[VAL_29:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_30:.*]] = arith.constant 11 : i32
// CHECK:             aie.use_lock(%[[VAL_22]], AcquireGreaterEqual, %[[VAL_28]])
// CHECK:             memref.store %[[VAL_30]], %[[VAL_17]]{{\[}}%[[VAL_29]]] : memref<3000xi32>
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @mem_in_shim_alloc(%[[VAL_0]], MM2S, 0)
// CHECK:           %[[VAL_31:.*]] = aie.mem(%[[VAL_11]]) {
// CHECK:             %[[VAL_32:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_33:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_32]])
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<3000xi32> offset = 0 len = 3000)
// CHECK:             aie.use_lock(%[[VAL_15]], Release, %[[VAL_32]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_32]])
// CHECK:             aie.dma_bd(%[[VAL_13]] : memref<3000xi32> offset = 0 len = 3000)
// CHECK:             aie.use_lock(%[[VAL_15]], Release, %[[VAL_32]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_34:.*]] = aie.memtile_dma(%[[VAL_1]]) {
// CHECK:             %[[VAL_35:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_36:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb8)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<3000xi32> offset = 0 len = 3000)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<3000xi32> offset = 0 len = 3000)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<3000xi32> offset = 0 len = 3000)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<3000xi32> offset = 0 len = 3000)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<3000xi32> offset = 0 len = 3000)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<3000xi32> offset = 0 len = 3000)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<3000xi32> offset = 0 len = 3000)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb8:
// CHECK:             %[[VAL_37:.*]] = aie.dma_start(MM2S, 0, ^bb9, ^bb16)
// CHECK:           ^bb9:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<3000xi32> offset = 0 len = 3000)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb10
// CHECK:           ^bb10:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<3000xi32> offset = 0 len = 3000)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb11
// CHECK:           ^bb11:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<3000xi32> offset = 0 len = 3000)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb12
// CHECK:           ^bb12:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<3000xi32> offset = 0 len = 3000)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb13
// CHECK:           ^bb13:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<3000xi32> offset = 0 len = 3000)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb14
// CHECK:           ^bb14:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<3000xi32> offset = 0 len = 3000)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb15
// CHECK:           ^bb15:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<3000xi32> offset = 0 len = 3000)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb9
// CHECK:           ^bb16:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_38:.*]] = aie.mem(%[[VAL_16]]) {
// CHECK:             %[[VAL_39:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_40:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, %[[VAL_39]])
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<3000xi32> offset = 0 len = 3000)
// CHECK:             aie.use_lock(%[[VAL_22]], Release, %[[VAL_39]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, %[[VAL_39]])
// CHECK:             aie.dma_bd(%[[VAL_18]] : memref<3000xi32> offset = 0 len = 3000)
// CHECK:             aie.use_lock(%[[VAL_22]], Release, %[[VAL_39]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, %[[VAL_39]])
// CHECK:             aie.dma_bd(%[[VAL_19]] : memref<3000xi32> offset = 0 len = 3000)
// CHECK:             aie.use_lock(%[[VAL_22]], Release, %[[VAL_39]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, %[[VAL_39]])
// CHECK:             aie.dma_bd(%[[VAL_20]] : memref<3000xi32> offset = 0 len = 3000)
// CHECK:             aie.use_lock(%[[VAL_22]], Release, %[[VAL_39]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @link_test_broadcast {
    aie.device(xcve2302) {
        %tile00 = aie.tile(0, 0)
        %tile01 = aie.tile(0, 1)
        %tile02 = aie.tile(0, 2)
        %tile03 = aie.tile(0, 3)

        aie.objectfifo @mem_in (%tile00, {%tile02, %tile01}, [2,2,7]) : !aie.objectfifo<memref<3000xi32>>
        aie.objectfifo @mem_out (%tile01, {%tile03}, 7 : i32) : !aie.objectfifo<memref<3000xi32>>
        aie.objectfifo.link [@mem_in] -> [@mem_out] ([] [])

        %core02 = aie.core(%tile02) {
            %v11 = arith.constant 11 : i32
            %c0 = arith.constant 0 : index

            %subview_obj = aie.objectfifo.acquire @mem_in(Consume) : memref<3000xi32>
            memref.store %v11, %subview_obj[%c0] : memref<3000xi32>
            aie.end
        }

        %core03 = aie.core(%tile03) {
            %v11 = arith.constant 11 : i32
            %c0 = arith.constant 0 : index

            %subview_obj, %subview_obj1, %subview_obj2 = aie.objectfifo.acquire @mem_out(Consume) : memref<3000xi32>, memref<3000xi32>, memref<3000xi32>
            memref.store %v11, %subview_obj[%c0] : memref<3000xi32>
            aie.end
        }
    }
}
