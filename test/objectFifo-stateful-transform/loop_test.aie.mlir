//===- loop_test.aie.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
// Date: February 9th 2022
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:           memref.global "public" @loop_of : memref<16xi32>
// CHECK:           %[[VAL_0:.*]] = AIE.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(1, 3)
// CHECK:           %[[VAL_2:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "loop_of_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "loop_of_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "loop_of_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "loop_of_buff_3"} : memref<16xi32>
// CHECK:           %[[VAL_6:.*]] = AIE.lock(%[[VAL_0]], 0) {init = 0 : i32, sym_name = "loop_of_lock_0"}
// CHECK:           %[[VAL_7:.*]] = AIE.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "loop_of_lock_1"}
// CHECK:           %[[VAL_8:.*]] = AIE.lock(%[[VAL_0]], 2) {init = 0 : i32, sym_name = "loop_of_lock_2"}
// CHECK:           %[[VAL_9:.*]] = AIE.lock(%[[VAL_0]], 3) {init = 0 : i32, sym_name = "loop_of_lock_3"}
// CHECK:           func.func @some_work(%[[VAL_10:.*]]: memref<16xi32>, %[[VAL_11:.*]]: index) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_13:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_14:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_15:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_16:.*]] = arith.constant 4 : index
// CHECK:             %[[VAL_17:.*]] = arith.constant 21 : index
// CHECK:             AIE.useLock(%[[VAL_6]], Acquire, 0)
// CHECK:             func.call @some_work(%[[VAL_2]], %[[VAL_13]]) : (memref<16xi32>, index) -> ()
// CHECK:             AIE.useLock(%[[VAL_6]], Release, 1)
// CHECK:             %[[VAL_18:.*]] = arith.constant 16 : index
// CHECK:             %[[VAL_19:.*]] = arith.constant 8 : index
// CHECK:             scf.for %[[VAL_20:.*]] = %[[VAL_14]] to %[[VAL_18]] step %[[VAL_19]] {
// CHECK:               AIE.useLock(%[[VAL_7]], Acquire, 0)
// CHECK:               func.call @some_work(%[[VAL_3]], %[[VAL_20]]) : (memref<16xi32>, index) -> ()
// CHECK:               AIE.useLock(%[[VAL_7]], Release, 1)
// CHECK:               AIE.useLock(%[[VAL_8]], Acquire, 0)
// CHECK:               %[[VAL_21:.*]] = arith.constant 2 : index
// CHECK:               %[[VAL_22:.*]] = arith.addi %[[VAL_20]], %[[VAL_21]] : index
// CHECK:               func.call @some_work(%[[VAL_4]], %[[VAL_22]]) : (memref<16xi32>, index) -> ()
// CHECK:               AIE.useLock(%[[VAL_8]], Release, 1)
// CHECK:               AIE.useLock(%[[VAL_9]], Acquire, 0)
// CHECK:               %[[VAL_23:.*]] = arith.constant 4 : index
// CHECK:               %[[VAL_24:.*]] = arith.addi %[[VAL_20]], %[[VAL_23]] : index
// CHECK:               func.call @some_work(%[[VAL_5]], %[[VAL_24]]) : (memref<16xi32>, index) -> ()
// CHECK:               AIE.useLock(%[[VAL_9]], Release, 1)
// CHECK:               AIE.useLock(%[[VAL_6]], Acquire, 0)
// CHECK:               %[[VAL_25:.*]] = arith.constant 6 : index
// CHECK:               %[[VAL_26:.*]] = arith.addi %[[VAL_20]], %[[VAL_25]] : index
// CHECK:               func.call @some_work(%[[VAL_2]], %[[VAL_26]]) : (memref<16xi32>, index) -> ()
// CHECK:               AIE.useLock(%[[VAL_6]], Release, 1)
// CHECK:             }
// CHECK:             AIE.useLock(%[[VAL_7]], Acquire, 0)
// CHECK:             %[[VAL_27:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_28:.*]] = arith.addi %[[VAL_18]], %[[VAL_27]] : index
// CHECK:             func.call @some_work(%[[VAL_3]], %[[VAL_28]]) : (memref<16xi32>, index) -> ()
// CHECK:             AIE.useLock(%[[VAL_7]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_8]], Acquire, 0)
// CHECK:             %[[VAL_29:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_30:.*]] = arith.addi %[[VAL_18]], %[[VAL_29]] : index
// CHECK:             func.call @some_work(%[[VAL_4]], %[[VAL_30]]) : (memref<16xi32>, index) -> ()
// CHECK:             AIE.useLock(%[[VAL_8]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_9]], Acquire, 0)
// CHECK:             %[[VAL_31:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_32:.*]] = arith.addi %[[VAL_14]], %[[VAL_31]] : index
// CHECK:             func.call @some_work(%[[VAL_5]], %[[VAL_32]]) : (memref<16xi32>, index) -> ()
// CHECK:             AIE.useLock(%[[VAL_9]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_6]], Acquire, 0)
// CHECK:             %[[VAL_33:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_34:.*]] = arith.addi %[[VAL_14]], %[[VAL_33]] : index
// CHECK:             func.call @some_work(%[[VAL_2]], %[[VAL_34]]) : (memref<16xi32>, index) -> ()
// CHECK:             AIE.useLock(%[[VAL_6]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_7]], Acquire, 0)
// CHECK:             %[[VAL_35:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_36:.*]] = arith.addi %[[VAL_14]], %[[VAL_35]] : index
// CHECK:             func.call @some_work(%[[VAL_3]], %[[VAL_36]]) : (memref<16xi32>, index) -> ()
// CHECK:             AIE.useLock(%[[VAL_7]], Release, 1)
// CHECK:             AIE.end
// CHECK:           }
// CHECK:         }

module @loop  {
    AIE.device(xcvc1902) {
        %tile12 = AIE.tile(1, 2)
        %tile13 = AIE.tile(1, 3)

        AIE.objectfifo @loop_of (%tile12, {%tile13}, 4 : i32) : !AIE.objectfifo<memref<16xi32>>

        func.func @some_work(%line_in:memref<16xi32>, %index:index) -> () {
            return
        }

        %core12 = AIE.core(%tile12) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %c4 = arith.constant 4 : index
            %c21 = arith.constant 21 : index

            %subviewTop0 = AIE.objectfifo.acquire @loop_of (Produce, 1) : !AIE.objectfifosubview<memref<16xi32>>
            %elemTop0 = AIE.objectfifo.subview.access %subviewTop0[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elemTop0, %c0) : (memref<16xi32>,index) -> ()
            AIE.objectfifo.release @loop_of (Produce, 1)

            scf.for %indexInHeight = %c1 to %c21 step %c2 {
                %subview = AIE.objectfifo.acquire @loop_of (Produce, 1) : !AIE.objectfifosubview<memref<16xi32>>
                %elem0 = AIE.objectfifo.subview.access %subview[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @some_work(%elem0,%indexInHeight) : (memref<16xi32>,index) -> ()
                AIE.objectfifo.release @loop_of (Produce, 1)
            }

            scf.for %indexInHeight = %c1 to %c4 step %c1 {
                %subview = AIE.objectfifo.acquire @loop_of (Produce, 1) : !AIE.objectfifosubview<memref<16xi32>>
                %elem0 = AIE.objectfifo.subview.access %subview[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @some_work(%elem0,%indexInHeight) : (memref<16xi32>,index) -> ()
                AIE.objectfifo.release @loop_of (Produce, 1)
            }

            AIE.end
        }
    }
}
