//===- broadcast_test.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Date: September 5th 2022
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           memref.global "public" @broadcast_of_0_cons : memref<16xi32>
// CHECK:           memref.global "public" @broadcast_of_1_cons : memref<16xi32>
// CHECK:           memref.global "public" @broadcast_of_2_cons : memref<16xi32>
// CHECK:           memref.global "public" @broadcast_of_3_cons : memref<16xi32>
// CHECK:           memref.global "public" @broadcast_of : memref<16xi32>
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_2:.*]] = aie.tile(1, 4)
// CHECK:           %[[VAL_3:.*]] = aie.tile(3, 2)
// CHECK:           %[[VAL_4:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "broadcast_of_0_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "broadcast_of_0_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_0]], 0) {init = 0 : i32, sym_name = "broadcast_of_0_cons_lock_0"}
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "broadcast_of_0_cons_lock_1"}
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "broadcast_of_1_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "broadcast_of_1_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_11:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "broadcast_of_1_cons_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_2]], 0) {init = 0 : i32, sym_name = "broadcast_of_1_cons_lock_0"}
// CHECK:           %[[VAL_13:.*]] = aie.lock(%[[VAL_2]], 1) {init = 0 : i32, sym_name = "broadcast_of_1_cons_lock_1"}
// CHECK:           %[[VAL_14:.*]] = aie.lock(%[[VAL_2]], 2) {init = 0 : i32, sym_name = "broadcast_of_1_cons_lock_2"}
// CHECK:           %[[VAL_15:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "broadcast_of_2_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_16:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "broadcast_of_2_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_17:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "broadcast_of_2_cons_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_18:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "broadcast_of_2_cons_buff_3"} : memref<16xi32>
// CHECK:           %[[VAL_19:.*]] = aie.lock(%[[VAL_3]], 0) {init = 0 : i32, sym_name = "broadcast_of_2_cons_lock_0"}
// CHECK:           %[[VAL_20:.*]] = aie.lock(%[[VAL_3]], 1) {init = 0 : i32, sym_name = "broadcast_of_2_cons_lock_1"}
// CHECK:           %[[VAL_21:.*]] = aie.lock(%[[VAL_3]], 2) {init = 0 : i32, sym_name = "broadcast_of_2_cons_lock_2"}
// CHECK:           %[[VAL_22:.*]] = aie.lock(%[[VAL_3]], 3) {init = 0 : i32, sym_name = "broadcast_of_2_cons_lock_3"}
// CHECK:           %[[VAL_23:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "broadcast_of_3_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_24:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "broadcast_of_3_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_25:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "broadcast_of_3_cons_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_26:.*]] = aie.lock(%[[VAL_4]], 0) {init = 0 : i32, sym_name = "broadcast_of_3_cons_lock_0"}
// CHECK:           %[[VAL_27:.*]] = aie.lock(%[[VAL_4]], 1) {init = 0 : i32, sym_name = "broadcast_of_3_cons_lock_1"}
// CHECK:           %[[VAL_28:.*]] = aie.lock(%[[VAL_4]], 2) {init = 0 : i32, sym_name = "broadcast_of_3_cons_lock_2"}
// CHECK:           %[[VAL_29:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "broadcast_of_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_30:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "broadcast_of_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_31:.*]] = aie.lock(%[[VAL_1]], 0) {init = 0 : i32, sym_name = "broadcast_of_lock_0"}
// CHECK:           %[[VAL_32:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "broadcast_of_lock_1"}
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_4]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:           func.func @some_work(%[[VAL_33:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_34:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:             %[[VAL_35:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_36:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_37:.*]] = arith.constant 12 : index
// CHECK:             %[[VAL_38:.*]] = arith.constant 2 : index
// CHECK:             scf.for %[[VAL_39:.*]] = %[[VAL_35]] to %[[VAL_37]] step %[[VAL_38]] {
// CHECK:               aie.use_lock(%[[VAL_31]], Acquire, 0)
// CHECK:               func.call @some_work(%[[VAL_29]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_31]], Release, 1)
// CHECK:               aie.use_lock(%[[VAL_32]], Acquire, 0)
// CHECK:               func.call @some_work(%[[VAL_30]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_32]], Release, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_40:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_41:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_42:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_43:.*]] = arith.constant 12 : index
// CHECK:             %[[VAL_44:.*]] = arith.constant 2 : index
// CHECK:             scf.for %[[VAL_45:.*]] = %[[VAL_41]] to %[[VAL_43]] step %[[VAL_44]] {
// CHECK:               aie.use_lock(%[[VAL_7]], Acquire, 1)
// CHECK:               func.call @some_work(%[[VAL_5]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_7]], Release, 0)
// CHECK:               aie.use_lock(%[[VAL_8]], Acquire, 1)
// CHECK:               func.call @some_work(%[[VAL_6]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_8]], Release, 0)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_46:.*]] = aie.core(%[[VAL_2]]) {
// CHECK:             %[[VAL_47:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_48:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_49:.*]] = arith.constant 12 : index
// CHECK:             %[[VAL_50:.*]] = arith.constant 3 : index
// CHECK:             scf.for %[[VAL_51:.*]] = %[[VAL_47]] to %[[VAL_49]] step %[[VAL_50]] {
// CHECK:               aie.use_lock(%[[VAL_12]], Acquire, 1)
// CHECK:               aie.use_lock(%[[VAL_13]], Acquire, 1)
// CHECK:               func.call @some_work(%[[VAL_9]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[VAL_10]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_12]], Release, 0)
// CHECK:               aie.use_lock(%[[VAL_13]], Release, 0)
// CHECK:               aie.use_lock(%[[VAL_14]], Acquire, 1)
// CHECK:               aie.use_lock(%[[VAL_12]], Acquire, 1)
// CHECK:               func.call @some_work(%[[VAL_11]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[VAL_9]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_14]], Release, 0)
// CHECK:               aie.use_lock(%[[VAL_12]], Release, 0)
// CHECK:               aie.use_lock(%[[VAL_13]], Acquire, 1)
// CHECK:               aie.use_lock(%[[VAL_14]], Acquire, 1)
// CHECK:               func.call @some_work(%[[VAL_10]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[VAL_11]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_13]], Release, 0)
// CHECK:               aie.use_lock(%[[VAL_14]], Release, 0)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_52:.*]] = aie.core(%[[VAL_3]]) {
// CHECK:             %[[VAL_53:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_54:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_55:.*]] = arith.constant 12 : index
// CHECK:             %[[VAL_56:.*]] = arith.constant 4 : index
// CHECK:             scf.for %[[VAL_57:.*]] = %[[VAL_53]] to %[[VAL_55]] step %[[VAL_56]] {
// CHECK:               aie.use_lock(%[[VAL_19]], Acquire, 1)
// CHECK:               aie.use_lock(%[[VAL_20]], Acquire, 1)
// CHECK:               aie.use_lock(%[[VAL_21]], Acquire, 1)
// CHECK:               func.call @some_work(%[[VAL_15]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[VAL_16]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[VAL_17]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_19]], Release, 0)
// CHECK:               aie.use_lock(%[[VAL_22]], Acquire, 1)
// CHECK:               func.call @some_work(%[[VAL_16]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[VAL_17]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[VAL_18]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_20]], Release, 0)
// CHECK:               aie.use_lock(%[[VAL_19]], Acquire, 1)
// CHECK:               func.call @some_work(%[[VAL_17]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[VAL_18]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[VAL_15]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_21]], Release, 0)
// CHECK:               aie.use_lock(%[[VAL_20]], Acquire, 1)
// CHECK:               func.call @some_work(%[[VAL_18]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[VAL_15]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[VAL_16]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_22]], Release, 0)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_58:.*]] = aie.core(%[[VAL_4]]) {
// CHECK:             %[[VAL_59:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_60:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_61:.*]] = arith.constant 12 : index
// CHECK:             %[[VAL_62:.*]] = arith.constant 3 : index
// CHECK:             scf.for %[[VAL_63:.*]] = %[[VAL_59]] to %[[VAL_61]] step %[[VAL_62]] {
// CHECK:               aie.use_lock(%[[VAL_26]], Acquire, 1)
// CHECK:               aie.use_lock(%[[VAL_27]], Acquire, 1)
// CHECK:               func.call @some_work(%[[VAL_23]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[VAL_24]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_26]], Release, 0)
// CHECK:               aie.use_lock(%[[VAL_28]], Acquire, 1)
// CHECK:               func.call @some_work(%[[VAL_24]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[VAL_25]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_27]], Release, 0)
// CHECK:               aie.use_lock(%[[VAL_26]], Acquire, 1)
// CHECK:               func.call @some_work(%[[VAL_25]]) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%[[VAL_23]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_28]], Release, 0)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_64:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_65:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(%[[VAL_31]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_29]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_31]], Release, 0)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_32]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_30]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_32]], Release, 0)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_66:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_67:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(%[[VAL_7]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_8]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_68:.*]] = aie.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_69:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK:             aie.use_lock(%[[VAL_12]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_12]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_13]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_13]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:  // pred: ^bb2
// CHECK:             aie.use_lock(%[[VAL_14]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_11]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_14]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_70:.*]] = aie.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_71:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb4
// CHECK:             aie.use_lock(%[[VAL_19]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_15]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_19]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_20]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_16]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_20]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:  // pred: ^bb2
// CHECK:             aie.use_lock(%[[VAL_21]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_21]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:  // pred: ^bb3
// CHECK:             aie.use_lock(%[[VAL_22]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_18]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_22]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_72:.*]] = aie.mem(%[[VAL_4]]) {
// CHECK:             %[[VAL_73:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK:             aie.use_lock(%[[VAL_26]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_23]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_26]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_27]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_24]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_27]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:  // pred: ^bb2
// CHECK:             aie.use_lock(%[[VAL_28]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_25]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_28]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @broadcast {
 aie.device(xcvc1902) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile14 = aie.tile(1, 4)
    %tile32 = aie.tile(3, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @broadcast_of (%tile13, {%tile12, %tile14, %tile32, %tile33}, [2, 2, 3, 4, 3]) : !aie.objectfifo<memref<16xi32>>

    func.func @some_work(%lineOut : memref<16xi32>) -> () {
        return
    }

    %core13 = aie.core(%tile13) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        scf.for %indexInHeight = %c0 to %height step %c1 {
            %subview = aie.objectfifo.acquire @broadcast_of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
            %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Produce, 1)
        }
        
        aie.end
    }

    %core12 = aie.core(%tile12) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        scf.for %indexInHeight = %c0 to %height step %c1 {
            %subview = aie.objectfifo.acquire @broadcast_of (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
            %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Consume, 1)
        }
        
        aie.end
    }

    %core14 = aie.core(%tile14) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        scf.for %indexInHeight = %c0 to %height step %c1 {
            %subview = aie.objectfifo.acquire @broadcast_of (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem1 = aie.objectfifo.subview.access %subview[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            func.call @some_work(%elem1) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Consume, 2)
        }
        
        aie.end
    }

    %core32 = aie.core(%tile32) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        scf.for %indexInHeight = %c0 to %height step %c1 {
            %subview = aie.objectfifo.acquire @broadcast_of (Consume, 3) : !aie.objectfifosubview<memref<16xi32>>
            %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem1 = aie.objectfifo.subview.access %subview[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem2 = aie.objectfifo.subview.access %subview[2] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            func.call @some_work(%elem1) : (memref<16xi32>) -> ()
            func.call @some_work(%elem2) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Consume, 1)
        }
        
        aie.end
    }

    %core33 = aie.core(%tile33) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        scf.for %indexInHeight = %c0 to %height step %c1 {
            %subview = aie.objectfifo.acquire @broadcast_of (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem1 = aie.objectfifo.subview.access %subview[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            func.call @some_work(%elem1) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Consume, 1)
        }
        
        aie.end
    }
 }
}
