//===- acquire_before_loop.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: February 9th 2022
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL: module {
// CHECK:         aie.device(xcvc1902) {
// CHECK-DAG:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK-DAG:           aie.buffer @[[VAL_2:loop_of_buff_0]](%[[VAL_0]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_3:loop_of_buff_1]](%[[VAL_0]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_4:loop_of_buff_2]](%[[VAL_0]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_5:loop_of_buff_3]](%[[VAL_0]]) : memref<16xi32>
// CHECK-DAG:           aie.lock @[[VAL_6:loop_of_lock_0]](%[[VAL_0]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @[[VAL_7:loop_of_lock_1]](%[[VAL_0]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @[[VAL_8:loop_of_lock_2]](%[[VAL_0]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @[[VAL_9:loop_of_lock_3]](%[[VAL_0]]) {init = 0 : i32}
// CHECK:           func.func @some_work(%[[VAL_10:.*]]: memref<16xi32>, %[[VAL_11:.*]]: index) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = aie.core(%[[VAL_0]]) {
// CHECK-DAG:             %[[IDX9:.*]] = arith.constant 9 : index
// CHECK-DAG:             %[[IDX4:.*]] = arith.constant 4 : index
// CHECK-DAG:             %[[IDX1:.*]] = arith.constant 1 : index
// CHECK-DAG:             %[[IDX0:.*]] = arith.constant 0 : index
// CHECK-DAG:             %[[C0I:.*]] = arith.constant 0 : i32
// CHECK-DAG:             %[[C1I:.*]] = arith.constant 1 : i32
// CHECK-DAG:             %[[IDX2:.*]] = arith.constant 2 : index
// CHECK-DAG:             %[[IDX3:.*]] = arith.constant 3 : index
// CHECK:             aie.use_lock(@[[VAL_6]], Acquire, %[[C0I]])
// CHECK:             func.call @some_work(%{{[^ ,)]+}}, %[[IDX0]]) : (memref<16xi32>, index) -> ()
// CHECK:             aie.use_lock(@[[VAL_6]], Release, %[[C1I]])
// CHECK:             scf.for %[[IV:.*]] = %[[IDX1]] to %[[IDX9]] step %[[IDX4]] {
// CHECK:               aie.use_lock(@[[VAL_7]], Acquire, %[[C0I]])
// CHECK:               func.call @some_work(%{{[^ ,)]+}}, %[[IV]]) : (memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(@[[VAL_7]], Release, %[[C1I]])
// CHECK:               %[[A0:.*]] = arith.addi %[[IV]], %[[IDX1]] : index
// CHECK:               aie.use_lock(@[[VAL_8]], Acquire, %[[C0I]])
// CHECK:               func.call @some_work(%{{[^ ,)]+}}, %[[A0]]) : (memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(@[[VAL_8]], Release, %[[C1I]])
// CHECK:               %[[A1:.*]] = arith.addi %[[IV]], %[[IDX2]] : index
// CHECK:               aie.use_lock(@[[VAL_9]], Acquire, %[[C0I]])
// CHECK:               func.call @some_work(%{{[^ ,)]+}}, %[[A1]]) : (memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(@[[VAL_9]], Release, %[[C1I]])
// CHECK:               %[[A2:.*]] = arith.addi %[[IV]], %[[IDX3]] : index
// CHECK:               aie.use_lock(@[[VAL_6]], Acquire, %[[C0I]])
// CHECK:               func.call @some_work(%{{[^ ,)]+}}, %[[A2]]) : (memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(@[[VAL_6]], Release, %[[C1I]])
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }
// CHECK:       }

module {
  aie.device(xcvc1902) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    aie.objectfifo @loop_of (%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    func.func @some_work(%line_in:memref<16xi32>, %index:index) -> () {
      return
    }
    %core12 = aie.core(%tile12) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c9 = arith.constant 9 : index
      %elemTop0 = aie.objectfifo.acquire @loop_of (Produce, 1) : memref<16xi32>
      func.call @some_work(%elemTop0, %c0) : (memref<16xi32>,index) -> ()
      aie.objectfifo.release @loop_of (Produce, 1)
      scf.for %indexInHeight = %c1 to %c9 step %c1 {
        %elem0 = aie.objectfifo.acquire @loop_of (Produce, 1) : memref<16xi32>
        func.call @some_work(%elem0,%indexInHeight) : (memref<16xi32>,index) -> ()
        aie.objectfifo.release @loop_of (Produce, 1)
      }
      aie.end
    }
  }
}
