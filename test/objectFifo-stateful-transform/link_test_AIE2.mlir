//===- link_test_AIE2.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: July 31st 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           memref.global "public" @mem_out_cons : memref<3000xi32>
// CHECK:           memref.global "public" @mem_out : memref<3000xi32>
// CHECK:           memref.global "public" @mem_in_0_cons : memref<3000xi32>
// CHECK:           memref.global "public" @mem_in_1_cons : memref<3000xi32>
// CHECK:           memref.global "public" @mem_in : memref<3000xi32>
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_3:.*]] = aie.tile(0, 3)
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "mem_out_cons_buff_0"} : memref<3000xi32>
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "mem_out_cons_buff_1"} : memref<3000xi32>
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "mem_out_cons_buff_2"} : memref<3000xi32>
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "mem_out_cons_buff_3"} : memref<3000xi32>
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_3]], 0) {init = 4 : i32, sym_name = "mem_out_cons_prod_lock"}
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_3]], 1) {init = 0 : i32, sym_name = "mem_out_cons_cons_lock"}
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "mem_in_0_cons_buff_0"} : memref<3000xi32>
// CHECK:           %[[VAL_11:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "mem_in_0_cons_buff_1"} : memref<3000xi32>
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_2]], 0) {init = 2 : i32, sym_name = "mem_in_0_cons_prod_lock"}
// CHECK:           %[[VAL_13:.*]] = aie.lock(%[[VAL_2]], 1) {init = 0 : i32, sym_name = "mem_in_0_cons_cons_lock"}
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "mem_in_1_cons_buff_0"} : memref<3000xi32>
// CHECK:           %[[VAL_15:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "mem_in_1_cons_buff_1"} : memref<3000xi32>
// CHECK:           %[[VAL_16:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "mem_in_1_cons_buff_2"} : memref<3000xi32>
// CHECK:           %[[VAL_17:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "mem_in_1_cons_buff_3"} : memref<3000xi32>
// CHECK:           %[[VAL_18:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "mem_in_1_cons_buff_4"} : memref<3000xi32>
// CHECK:           %[[VAL_19:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "mem_in_1_cons_buff_5"} : memref<3000xi32>
// CHECK:           %[[VAL_20:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "mem_in_1_cons_buff_6"} : memref<3000xi32>
// CHECK:           %[[VAL_21:.*]] = aie.lock(%[[VAL_1]], 0) {init = 7 : i32, sym_name = "mem_in_1_cons_prod_lock"}
// CHECK:           %[[VAL_22:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "mem_in_1_cons_cons_lock"}
// CHECK:           %[[VAL_23:.*]] = aie.lock(%[[VAL_0]], 0) {init = 0 : i32, sym_name = "mem_in_prod_lock"}
// CHECK:           %[[VAL_24:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "mem_in_cons_lock"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK:           %[[VAL_25:.*]] = aie.core(%[[VAL_2]]) {
// CHECK:             %[[VAL_26:.*]] = arith.constant 11 : i32
// CHECK:             %[[VAL_27:.*]] = arith.constant 0 : index
// CHECK:             aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, 1)
// CHECK:             memref.store %[[VAL_26]], %[[VAL_10]]{{\[}}%[[VAL_27]]] : memref<3000xi32>
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @mem_in(MM2S, 0, 0)
// CHECK:           %[[VAL_28:.*]] = aie.core(%[[VAL_3]]) {
// CHECK:             %[[VAL_29:.*]] = arith.constant 11 : i32
// CHECK:             %[[VAL_30:.*]] = arith.constant 0 : index
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, 3)
// CHECK:             memref.store %[[VAL_29]], %[[VAL_4]]{{\[}}%[[VAL_30]]] : memref<3000xi32>
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = aie.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_32:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[VAL_13]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_11]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[VAL_13]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = aie.memtile_dma(%[[VAL_1]]) {
// CHECK:             %[[VAL_34:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb8)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb7
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[VAL_22]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_15]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[VAL_22]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:  // pred: ^bb2
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_16]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[VAL_22]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:  // pred: ^bb3
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[VAL_22]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:  // pred: ^bb4
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_18]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[VAL_22]], Release, 1)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:  // pred: ^bb5
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_19]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[VAL_22]], Release, 1)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:  // pred: ^bb6
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_20]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[VAL_22]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb8:  // pred: ^bb0
// CHECK:             %[[VAL_35:.*]] = aie.dma_start(MM2S, 0, ^bb9, ^bb16)
// CHECK:           ^bb9:  // 2 preds: ^bb8, ^bb15
// CHECK:             aie.use_lock(%[[VAL_22]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[VAL_21]], Release, 1)
// CHECK:             aie.next_bd ^bb10
// CHECK:           ^bb10:  // pred: ^bb9
// CHECK:             aie.use_lock(%[[VAL_22]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_15]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[VAL_21]], Release, 1)
// CHECK:             aie.next_bd ^bb11
// CHECK:           ^bb11:  // pred: ^bb10
// CHECK:             aie.use_lock(%[[VAL_22]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_16]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[VAL_21]], Release, 1)
// CHECK:             aie.next_bd ^bb12
// CHECK:           ^bb12:  // pred: ^bb11
// CHECK:             aie.use_lock(%[[VAL_22]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[VAL_21]], Release, 1)
// CHECK:             aie.next_bd ^bb13
// CHECK:           ^bb13:  // pred: ^bb12
// CHECK:             aie.use_lock(%[[VAL_22]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_18]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[VAL_21]], Release, 1)
// CHECK:             aie.next_bd ^bb14
// CHECK:           ^bb14:  // pred: ^bb13
// CHECK:             aie.use_lock(%[[VAL_22]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_19]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[VAL_21]], Release, 1)
// CHECK:             aie.next_bd ^bb15
// CHECK:           ^bb15:  // pred: ^bb14
// CHECK:             aie.use_lock(%[[VAL_22]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_20]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[VAL_21]], Release, 1)
// CHECK:             aie.next_bd ^bb9
// CHECK:           ^bb16:  // pred: ^bb8
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_36:.*]] = aie.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_37:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb4
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:  // pred: ^bb2
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:  // pred: ^bb3
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<3000xi32>, 0, 3000)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @link_AIE2 {
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

            %subview = aie.objectfifo.acquire @mem_in (Consume, 1) : !aie.objectfifosubview<memref<3000xi32>>
            %subview_obj = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<3000xi32>> -> memref<3000xi32>
            memref.store %v11, %subview_obj[%c0] : memref<3000xi32>
            aie.end
        }

        %core03 = aie.core(%tile03) {
            %v11 = arith.constant 11 : i32
            %c0 = arith.constant 0 : index

            %subview = aie.objectfifo.acquire @mem_out (Consume, 3) : !aie.objectfifosubview<memref<3000xi32>>
            %subview_obj = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<3000xi32>> -> memref<3000xi32>
            memref.store %v11, %subview_obj[%c0] : memref<3000xi32>
            aie.end
        }
    }
}
