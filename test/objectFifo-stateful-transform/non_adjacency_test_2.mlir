//===- non_adjacency_test_2.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
// Date: May 24th 2022
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:           memref.global "public" @objfifo_cons : memref<16xi32>
// CHECK:           memref.global "public" @objfifo : memref<16xi32>
// CHECK:           %[[VAL_0:.*]] = AIE.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(3, 3)
// CHECK:           %[[VAL_2:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "objfifo_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "objfifo_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "objfifo_cons_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "objfifo_cons_buff_3"} : memref<16xi32>
// CHECK:           %[[VAL_6:.*]] = AIE.lock(%[[VAL_1]], 0) {init = 0 : i32, sym_name = "objfifo_cons_lock_0"}
// CHECK:           %[[VAL_7:.*]] = AIE.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "objfifo_cons_lock_1"}
// CHECK:           %[[VAL_8:.*]] = AIE.lock(%[[VAL_1]], 2) {init = 0 : i32, sym_name = "objfifo_cons_lock_2"}
// CHECK:           %[[VAL_9:.*]] = AIE.lock(%[[VAL_1]], 3) {init = 0 : i32, sym_name = "objfifo_cons_lock_3"}
// CHECK:           %[[VAL_10:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "objfifo_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_11:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "objfifo_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_12:.*]] = AIE.lock(%[[VAL_0]], 0) {init = 0 : i32, sym_name = "objfifo_lock_0"}
// CHECK:           %[[VAL_13:.*]] = AIE.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "objfifo_lock_1"}
// CHECK:           AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           func.func @some_work(%[[VAL_14:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_16:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_17:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_18:.*]] = arith.constant 12 : index
// CHECK:             %[[VAL_19:.*]] = arith.constant 2 : index
// CHECK:             scf.for %[[VAL_20:.*]] = %[[VAL_16]] to %[[VAL_18]] step %[[VAL_19]] {
// CHECK:               AIE.useLock(%[[VAL_12]], Acquire, 0)
// CHECK:               func.call @some_work(%[[VAL_10]]) : (memref<16xi32>) -> ()
// CHECK:               AIE.useLock(%[[VAL_12]], Release, 1)
// CHECK:               AIE.useLock(%[[VAL_13]], Acquire, 0)
// CHECK:               func.call @some_work(%[[VAL_11]]) : (memref<16xi32>) -> ()
// CHECK:               AIE.useLock(%[[VAL_13]], Release, 1)
// CHECK:             }
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = AIE.core(%[[VAL_1]]) {
// CHECK:             %[[VAL_22:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_23:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_24:.*]] = arith.constant 12 : index
// CHECK:             %[[VAL_25:.*]] = arith.constant 4 : index
// CHECK:             scf.for %[[VAL_26:.*]] = %[[VAL_22]] to %[[VAL_24]] step %[[VAL_25]] {
// CHECK:               AIE.useLock(%[[VAL_6]], Acquire, 1)
// CHECK:               AIE.useLock(%[[VAL_7]], Acquire, 1)
// CHECK:               AIE.useLock(%[[VAL_8]], Acquire, 1)
// CHECK:               func.call @some_work(%[[VAL_2]]) : (memref<16xi32>) -> ()
// CHECK:               AIE.useLock(%[[VAL_6]], Release, 0)
// CHECK:               AIE.useLock(%[[VAL_9]], Acquire, 1)
// CHECK:               func.call @some_work(%[[VAL_3]]) : (memref<16xi32>) -> ()
// CHECK:               AIE.useLock(%[[VAL_7]], Release, 0)
// CHECK:               AIE.useLock(%[[VAL_6]], Acquire, 1)
// CHECK:               func.call @some_work(%[[VAL_4]]) : (memref<16xi32>) -> ()
// CHECK:               AIE.useLock(%[[VAL_8]], Release, 0)
// CHECK:               AIE.useLock(%[[VAL_7]], Acquire, 1)
// CHECK:               func.call @some_work(%[[VAL_5]]) : (memref<16xi32>) -> ()
// CHECK:               AIE.useLock(%[[VAL_9]], Release, 0)
// CHECK:             }
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = AIE.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_28:.*]] = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             AIE.useLock(%[[VAL_12]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_10]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_12]], Release, 0)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_13]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_11]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_13]], Release, 0)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = AIE.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_30:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb4
// CHECK:             AIE.useLock(%[[VAL_6]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_2]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_6]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_7]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_3]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_7]], Release, 1)
// CHECK:             AIE.nextBd ^bb3
// CHECK:           ^bb3:  // pred: ^bb2
// CHECK:             AIE.useLock(%[[VAL_8]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_4]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_8]], Release, 1)
// CHECK:             AIE.nextBd ^bb4
// CHECK:           ^bb4:  // pred: ^bb3
// CHECK:             AIE.useLock(%[[VAL_9]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_5]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_9]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb5:  // pred: ^bb0
// CHECK:             AIE.end
// CHECK:           }
// CHECK:         }

module @non_adjacency {
    AIE.device(xcvc1902) {
        %tile12 = AIE.tile(1, 2)
        %tile33 = AIE.tile(3, 3)

        AIE.objectfifo @objfifo (%tile12, {%tile33}, 2 : i32) : !AIE.objectfifo<memref<16xi32>>

        func.func @some_work(%lineOut : memref<16xi32>) -> () {
            return
        }

        %core12 = AIE.core(%tile12) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %height = arith.constant 12 : index

            scf.for %indexInHeight = %c0 to %height step %c1 {
                %subview = AIE.objectfifo.acquire @objfifo (Produce, 1) : !AIE.objectfifosubview<memref<16xi32>>
                %elem0 = AIE.objectfifo.subview.access %subview[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @some_work(%elem0) : (memref<16xi32>) -> ()
                AIE.objectfifo.release @objfifo (Produce, 1)
            }

            AIE.end
        }

        %core33 = AIE.core(%tile33) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %height = arith.constant 12 : index

            scf.for %indexInHeight = %c0 to %height step %c1 {
                %subview = AIE.objectfifo.acquire @objfifo (Consume, 3) : !AIE.objectfifosubview<memref<16xi32>>
                %elem0 = AIE.objectfifo.subview.access %subview[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %elem1 = AIE.objectfifo.subview.access %subview[1] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %elem2 = AIE.objectfifo.subview.access %subview[2] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @some_work(%elem0) : (memref<16xi32>) -> ()
                AIE.objectfifo.release @objfifo (Consume, 1)
            }

            AIE.end
        }
    }
}
