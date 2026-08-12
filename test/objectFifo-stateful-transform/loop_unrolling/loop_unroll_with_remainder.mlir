//===- loop_unroll_with_remainder.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL: module {
// CHECK:         aie.device(xcvc1902) {
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
// CHECK:           func.func @some_work(%[[A:.*]]: memref<16xi32>, %[[B:.*]]: memref<16xi32>, %[[I:.*]]: index) {
// CHECK:             return
// CHECK:           }
// unroll factor = lcm(of_1 depth 2, of_2 depth 4) = 4: rolled loop over 4 sub-
// iterations (trip 8) then 2 straight-line remainder iterations (indices 8, 9).
// Binary-lock polarity: of_1 acquire=1/release=0, of_2 acquire=0/release=1.
// CHECK:           %[[CORE:.*]] = aie.core(%[[VAL_0]]) {
// CHECK-DAG:             %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:             %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:             %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:             %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:             %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:             %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:             %[[C9:.*]] = arith.constant 9 : index
// CHECK-DAG:             %[[L0:.*]] = arith.constant 0 : i32
// CHECK-DAG:             %[[L1:.*]] = arith.constant 1 : i32
// CHECK:             scf.for %[[IV:.*]] = %[[C0]] to %[[C8]] step %[[C4]] {
// CHECK:               aie.use_lock(%[[VAL_12]], Acquire, %[[L1]])
// CHECK:               aie.use_lock(%[[VAL_6]], Acquire, %[[L0]])
// CHECK:               func.call @some_work(%[[VAL_10]], %[[VAL_2]], %[[IV]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_12]], Release, %[[L0]])
// CHECK:               aie.use_lock(%[[VAL_6]], Release, %[[L1]])
// CHECK:               %[[J1:.*]] = arith.addi %[[IV]], %[[C1]] : index
// CHECK:               aie.use_lock(%[[VAL_13]], Acquire, %[[L1]])
// CHECK:               aie.use_lock(%[[VAL_7]], Acquire, %[[L0]])
// CHECK:               func.call @some_work(%[[VAL_11]], %[[VAL_3]], %[[J1]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_13]], Release, %[[L0]])
// CHECK:               aie.use_lock(%[[VAL_7]], Release, %[[L1]])
// CHECK:               %[[J2:.*]] = arith.addi %[[IV]], %[[C2]] : index
// CHECK:               aie.use_lock(%[[VAL_12]], Acquire, %[[L1]])
// CHECK:               aie.use_lock(%[[VAL_8]], Acquire, %[[L0]])
// CHECK:               func.call @some_work(%[[VAL_10]], %[[VAL_4]], %[[J2]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_12]], Release, %[[L0]])
// CHECK:               aie.use_lock(%[[VAL_8]], Release, %[[L1]])
// CHECK:               %[[J3:.*]] = arith.addi %[[IV]], %[[C3]] : index
// CHECK:               aie.use_lock(%[[VAL_13]], Acquire, %[[L1]])
// CHECK:               aie.use_lock(%[[VAL_9]], Acquire, %[[L0]])
// CHECK:               func.call @some_work(%[[VAL_11]], %[[VAL_5]], %[[J3]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_13]], Release, %[[L0]])
// CHECK:               aie.use_lock(%[[VAL_9]], Release, %[[L1]])
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_12]], Acquire, %[[L1]])
// CHECK:             aie.use_lock(%[[VAL_6]], Acquire, %[[L0]])
// CHECK:             func.call @some_work(%[[VAL_10]], %[[VAL_2]], %[[C8]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %[[L0]])
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[L1]])
// CHECK:             aie.use_lock(%[[VAL_13]], Acquire, %[[L1]])
// CHECK:             aie.use_lock(%[[VAL_7]], Acquire, %[[L0]])
// CHECK:             func.call @some_work(%[[VAL_11]], %[[VAL_3]], %[[C9]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:             aie.use_lock(%[[VAL_13]], Release, %[[L0]])
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[L1]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }
// CHECK:       }

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
