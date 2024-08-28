//===- loop_test_unroll_remainder.mlir -------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           memref.global "public" @of_2 : memref<16xi32>
// CHECK:           memref.global "public" @of_1 : memref<16xi32>
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of_2_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of_2_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of_2_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of_2_buff_3"} : memref<16xi32>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_0]], 0) {init = 0 : i32, sym_name = "of_2_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "of_2_lock_1"}
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_0]], 2) {init = 0 : i32, sym_name = "of_2_lock_2"}
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_0]], 3) {init = 0 : i32, sym_name = "of_2_lock_3"}
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_1_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_11:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_1_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_1]], 0) {init = 0 : i32, sym_name = "of_1_lock_0"}
// CHECK:           %[[VAL_13:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "of_1_lock_1"}
// CHECK:           func.func @some_work(%[[VAL_14:.*]]: memref<16xi32>, %[[VAL_15:.*]]: memref<16xi32>, %[[VAL_16:.*]]: index) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_18:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_19:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_20:.*]] = arith.constant 10 : index
// CHECK:             %[[VAL_21:.*]] = arith.constant 8 : index
// CHECK:             %[[VAL_22:.*]] = arith.constant 4 : index
// CHECK:             scf.for %[[VAL_23:.*]] = %[[VAL_18]] to %[[VAL_21]] step %[[VAL_22]] {
// CHECK:               aie.use_lock(%[[VAL_12]], Acquire, 1)
// CHECK:               aie.use_lock(%[[VAL_6]], Acquire, 0)
// CHECK:               func.call @some_work(%[[VAL_10]], %[[VAL_2]], %[[VAL_23]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_12]], Release, 0)
// CHECK:               aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:               %[[VAL_24:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_25:.*]] = arith.muli %[[VAL_19]], %[[VAL_24]] : index
// CHECK:               %[[VAL_26:.*]] = arith.addi %[[VAL_23]], %[[VAL_25]] : index
// CHECK:               aie.use_lock(%[[VAL_13]], Acquire, 1)
// CHECK:               aie.use_lock(%[[VAL_7]], Acquire, 0)
// CHECK:               func.call @some_work(%[[VAL_11]], %[[VAL_3]], %[[VAL_26]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_13]], Release, 0)
// CHECK:               aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:               %[[VAL_27:.*]] = arith.constant 2 : index
// CHECK:               %[[VAL_28:.*]] = arith.muli %[[VAL_19]], %[[VAL_27]] : index
// CHECK:               %[[VAL_29:.*]] = arith.addi %[[VAL_23]], %[[VAL_28]] : index
// CHECK:               aie.use_lock(%[[VAL_12]], Acquire, 1)
// CHECK:               aie.use_lock(%[[VAL_8]], Acquire, 0)
// CHECK:               func.call @some_work(%[[VAL_10]], %[[VAL_4]], %[[VAL_29]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_12]], Release, 0)
// CHECK:               aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:               %[[VAL_30:.*]] = arith.constant 3 : index
// CHECK:               %[[VAL_31:.*]] = arith.muli %[[VAL_19]], %[[VAL_30]] : index
// CHECK:               %[[VAL_32:.*]] = arith.addi %[[VAL_23]], %[[VAL_31]] : index
// CHECK:               aie.use_lock(%[[VAL_13]], Acquire, 1)
// CHECK:               aie.use_lock(%[[VAL_9]], Acquire, 0)
// CHECK:               func.call @some_work(%[[VAL_11]], %[[VAL_5]], %[[VAL_32]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_13]], Release, 0)
// CHECK:               aie.use_lock(%[[VAL_9]], Release, 1)
// CHECK:             }
// CHECK:             %[[VAL_33:.*]] = arith.constant 2 : index
// CHECK:             aie.use_lock(%[[VAL_12]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_6]], Acquire, 0)
// CHECK:             func.call @some_work(%[[VAL_10]], %[[VAL_2]], %[[VAL_21]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:             aie.use_lock(%[[VAL_12]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:             %[[VAL_34:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_35:.*]] = arith.muli %[[VAL_19]], %[[VAL_34]] : index
// CHECK:             %[[VAL_36:.*]] = arith.addi %[[VAL_21]], %[[VAL_35]] : index
// CHECK:             aie.use_lock(%[[VAL_13]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_7]], Acquire, 0)
// CHECK:             func.call @some_work(%[[VAL_11]], %[[VAL_3]], %[[VAL_36]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:             aie.use_lock(%[[VAL_13]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }}
// CHECK:        }

module {
  aie.device(xcvc1902) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    aie.objectfifo @of_1 (%tile13, {%tile12}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_2 (%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    func.func @some_work(%line_inA:memref<16xi32>, %line_inB:memref<16xi32>, %index:index) -> () {
      return
    }
    %core12 = aie.core(%tile12) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index

      scf.for %indexInHeight = %c0 to %c10 step %c1 {
        %subviewIn = aie.objectfifo.acquire @of_1 (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
        %subviewOut = aie.objectfifo.acquire @of_2 (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @some_work(%elemIn, %elemOut, %indexInHeight) : (memref<16xi32>, memref<16xi32>, index) -> ()
        aie.objectfifo.release @of_1 (Consume, 1)
        aie.objectfifo.release @of_2 (Produce, 1)
      }
      aie.end
    }
  }
}