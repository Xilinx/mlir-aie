//===- nested_loop_unroll_with_remainder.mlir ------------------*- MLIR -*-===//
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
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_0]], 0) {init = 0 : i32, sym_name = "of_2_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "of_2_lock_1"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_0]], 2) {init = 0 : i32, sym_name = "of_2_lock_2"}
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_1_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_1_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_1_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_1]], 0) {init = 0 : i32, sym_name = "of_1_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "of_1_lock_1"}
// CHECK:           %[[VAL_13:.*]] = aie.lock(%[[VAL_1]], 2) {init = 0 : i32, sym_name = "of_1_lock_2"}
// CHECK:           func.func @some_work(%{{.*}}: memref<16xi32>, %{{.*}}: memref<16xi32>, %{{.*}}: index, %{{.*}}: index) {
// CHECK:             return
// CHECK:           }
// of_1 (consume, depth 3) nested inside of_2 (produce, depth 3). Outer unrolled
// x3 over of_2 buff0/1/2; inner unrolled x3 (+2 remainder) over of_1, whose
// buffer rotates continuously mod 3 across the whole nest. Binary-lock polarity:
// of_1 acquire=1/release=0, of_2 acquire=0/release=1.
// CHECK:           %[[CORE:.*]] = aie.core(%[[VAL_0]]) {
// CHECK-DAG:             %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:             %[[C9:.*]] = arith.constant 9 : index
// CHECK-DAG:             %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:             %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:             %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:             %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:             %[[C7:.*]] = arith.constant 7 : index
// CHECK-DAG:             %[[L0:.*]] = arith.constant 0 : i32
// CHECK-DAG:             %[[L1:.*]] = arith.constant 1 : i32
// CHECK:             scf.for %[[IV0:.*]] = %[[C0]] to %[[C9]] step %[[C3]] {
// CHECK:               aie.use_lock(%[[VAL_5]], Acquire, %[[L0]])
// CHECK:               scf.for %[[IA:.*]] = %[[C0]] to %[[C6]] step %[[C3]] {
// CHECK:                 aie.use_lock(%[[VAL_11]], Acquire, %[[L1]])
// CHECK:                 func.call @some_work(%[[VAL_8]], %[[VAL_2]], %[[IV0]], %[[IA]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                 aie.use_lock(%[[VAL_11]], Release, %[[L0]])
// CHECK:                 %[[A1:.*]] = arith.addi %[[IA]], %[[C1]] : index
// CHECK:                 aie.use_lock(%[[VAL_12]], Acquire, %[[L1]])
// CHECK:                 func.call @some_work(%[[VAL_9]], %[[VAL_2]], %[[IV0]], %[[A1]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                 aie.use_lock(%[[VAL_12]], Release, %[[L0]])
// CHECK:                 %[[A2:.*]] = arith.addi %[[IA]], %[[C2]] : index
// CHECK:                 aie.use_lock(%[[VAL_13]], Acquire, %[[L1]])
// CHECK:                 func.call @some_work(%[[VAL_10]], %[[VAL_2]], %[[IV0]], %[[A2]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                 aie.use_lock(%[[VAL_13]], Release, %[[L0]])
// CHECK:               }
// CHECK:               aie.use_lock(%[[VAL_11]], Acquire, %[[L1]])
// CHECK:               func.call @some_work(%[[VAL_8]], %[[VAL_2]], %[[IV0]], %[[C6]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_11]], Release, %[[L0]])
// CHECK:               aie.use_lock(%[[VAL_12]], Acquire, %[[L1]])
// CHECK:               func.call @some_work(%[[VAL_9]], %[[VAL_2]], %[[IV0]], %[[C7]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_12]], Release, %[[L0]])
// CHECK:               aie.use_lock(%[[VAL_5]], Release, %[[L1]])
// CHECK:               %[[O1:.*]] = arith.addi %[[IV0]], %[[C1]] : index
// CHECK:               aie.use_lock(%[[VAL_6]], Acquire, %[[L0]])
// CHECK:               scf.for %[[IB:.*]] = %[[C0]] to %[[C6]] step %[[C3]] {
// CHECK:                 aie.use_lock(%[[VAL_13]], Acquire, %[[L1]])
// CHECK:                 func.call @some_work(%[[VAL_10]], %[[VAL_3]], %[[O1]], %[[IB]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                 aie.use_lock(%[[VAL_13]], Release, %[[L0]])
// CHECK:                 %[[B1:.*]] = arith.addi %[[IB]], %[[C1]] : index
// CHECK:                 aie.use_lock(%[[VAL_11]], Acquire, %[[L1]])
// CHECK:                 func.call @some_work(%[[VAL_8]], %[[VAL_3]], %[[O1]], %[[B1]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                 aie.use_lock(%[[VAL_11]], Release, %[[L0]])
// CHECK:                 %[[B2:.*]] = arith.addi %[[IB]], %[[C2]] : index
// CHECK:                 aie.use_lock(%[[VAL_12]], Acquire, %[[L1]])
// CHECK:                 func.call @some_work(%[[VAL_9]], %[[VAL_3]], %[[O1]], %[[B2]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                 aie.use_lock(%[[VAL_12]], Release, %[[L0]])
// CHECK:               }
// CHECK:               aie.use_lock(%[[VAL_13]], Acquire, %[[L1]])
// CHECK:               func.call @some_work(%[[VAL_10]], %[[VAL_3]], %[[O1]], %[[C6]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_13]], Release, %[[L0]])
// CHECK:               aie.use_lock(%[[VAL_11]], Acquire, %[[L1]])
// CHECK:               func.call @some_work(%[[VAL_8]], %[[VAL_3]], %[[O1]], %[[C7]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_11]], Release, %[[L0]])
// CHECK:               aie.use_lock(%[[VAL_6]], Release, %[[L1]])
// CHECK:               %[[O2:.*]] = arith.addi %[[IV0]], %[[C2]] : index
// CHECK:               aie.use_lock(%[[VAL_7]], Acquire, %[[L0]])
// CHECK:               scf.for %[[IC:.*]] = %[[C0]] to %[[C6]] step %[[C3]] {
// CHECK:                 aie.use_lock(%[[VAL_12]], Acquire, %[[L1]])
// CHECK:                 func.call @some_work(%[[VAL_9]], %[[VAL_4]], %[[O2]], %[[IC]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                 aie.use_lock(%[[VAL_12]], Release, %[[L0]])
// CHECK:                 %[[K1:.*]] = arith.addi %[[IC]], %[[C1]] : index
// CHECK:                 aie.use_lock(%[[VAL_13]], Acquire, %[[L1]])
// CHECK:                 func.call @some_work(%[[VAL_10]], %[[VAL_4]], %[[O2]], %[[K1]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                 aie.use_lock(%[[VAL_13]], Release, %[[L0]])
// CHECK:                 %[[K2:.*]] = arith.addi %[[IC]], %[[C2]] : index
// CHECK:                 aie.use_lock(%[[VAL_11]], Acquire, %[[L1]])
// CHECK:                 func.call @some_work(%[[VAL_8]], %[[VAL_4]], %[[O2]], %[[K2]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                 aie.use_lock(%[[VAL_11]], Release, %[[L0]])
// CHECK:               }
// CHECK:               aie.use_lock(%[[VAL_12]], Acquire, %[[L1]])
// CHECK:               func.call @some_work(%[[VAL_9]], %[[VAL_4]], %[[O2]], %[[C6]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_12]], Release, %[[L0]])
// CHECK:               aie.use_lock(%[[VAL_13]], Acquire, %[[L1]])
// CHECK:               func.call @some_work(%[[VAL_10]], %[[VAL_4]], %[[O2]], %[[C7]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_13]], Release, %[[L0]])
// CHECK:               aie.use_lock(%[[VAL_7]], Release, %[[L1]])
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }

module {
  aie.device(xcvc1902) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    aie.objectfifo @of_1 (%tile13, {%tile12}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_2 (%tile12, {%tile13}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    func.func @some_work(%line_inA:memref<16xi32>, %line_inB:memref<16xi32>, %index:index, %index1:index) -> () {
      return
    }
    %core12 = aie.core(%tile12) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c9 = arith.constant 9 : index

      scf.for %indexInHeight = %c0 to %c9 step %c1 {
        %subviewOut = aie.objectfifo.acquire @of_2 (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        scf.for %indexInHeight1 = %c0 to %c8 step %c1 {
            %subviewIn = aie.objectfifo.acquire @of_1 (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
            %elemIn = aie.objectfifo.subview.access %subviewIn[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elemOut = aie.objectfifo.subview.access %subviewOut[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elemIn, %elemOut, %indexInHeight, %indexInHeight1) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
            aie.objectfifo.release @of_1 (Consume, 1)
        }
        aie.objectfifo.release @of_2 (Produce, 1)
      }

      aie.end
    }
  }
}
