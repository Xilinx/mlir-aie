//===- loop_test_common.mlir ------------------------------------*- MLIR -*-===//
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

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK-DAG:       %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK-DAG:       %[[BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "loop_of_buff_0"} : memref<16xi32>
// CHECK-DAG:       %[[BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "loop_of_buff_1"} : memref<16xi32>
// CHECK-DAG:       %[[BUFF_2:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "loop_of_buff_2"} : memref<16xi32>
// CHECK-DAG:       %[[BUFF_3:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "loop_of_buff_3"} : memref<16xi32>
// CHECK-DAG:       %[[LOCK_0:.*]] = aie.lock(%[[TILE_1_2]], 0) {init = 0 : i32, sym_name = "loop_of_lock_0"}
// CHECK-DAG:       %[[LOCK_1:.*]] = aie.lock(%[[TILE_1_2]], 1) {init = 0 : i32, sym_name = "loop_of_lock_1"}
// CHECK-DAG:       %[[LOCK_2:.*]] = aie.lock(%[[TILE_1_2]], 2) {init = 0 : i32, sym_name = "loop_of_lock_2"}
// CHECK-DAG:       %[[LOCK_3:.*]] = aie.lock(%[[TILE_1_2]], 3) {init = 0 : i32, sym_name = "loop_of_lock_3"}
// CHECK:           func.func @some_work(%[[VAL_10:.*]]: memref<16xi32>, %[[VAL_11:.*]]: index) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[CORE_1_2:.*]] = aie.core(%[[TILE_1_2]]) {
// CHECK:             %[[VAL_13:.*]] = arith.constant 0 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C2:.*]] = arith.constant 2 : index
// CHECK:             %[[C4:.*]] = arith.constant 4 : index
// CHECK:             %[[C21:.*]] = arith.constant 21 : index
// CHECK:             aie.use_lock(%[[LOCK_0]], Acquire, 0)
// CHECK:             func.call @some_work(%[[BUFF_0]], %[[VAL_13]]) : (memref<16xi32>, index) -> ()
// CHECK:             aie.use_lock(%[[LOCK_0]], Release, 1)
// CHECK:             %[[C17:.*]] = arith.constant 17 : index
// CHECK:             %[[VAL_19:.*]] = arith.constant 8 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C1]] to %[[C17]] step %[[VAL_19]] {
// CHECK-NEXT:          aie.use_lock(%[[LOCK_1]], Acquire, 0)
// CHECK-NEXT:          func.call @some_work(%[[BUFF_1]], %[[ARG0]]) : (memref<16xi32>, index) -> ()
// CHECK-NEXT:          aie.use_lock(%[[LOCK_1]], Release, 1)
// CHECK-DAG:           %[[C1_1:.*]] = arith.constant 1 : index
// CHECK-DAG:           %[[MUL_0:.*]] = arith.muli %[[C2]], %[[C1_1]] : index
// CHECK-DAG:           %[[ADD_0:.*]] = arith.addi %[[ARG0]], %[[MUL_0]] : index
// CHECK-DAG:           aie.use_lock(%[[LOCK_2]], Acquire, 0)
// CHECK:               func.call @some_work(%[[BUFF_2]], %[[ADD_0]]) : (memref<16xi32>, index) -> ()
// CHECK-NEXT:          aie.use_lock(%[[LOCK_2]], Release, 1)
// CHECK-DAG:           %[[C2_1:.*]] = arith.constant 2 : index
// CHECK-DAG:           %[[MUL_1:.*]] = arith.muli %[[C2]], %[[C2_1]] : index
// CHECK-DAG:           %[[ADD_1:.*]] = arith.addi %[[ARG0]], %[[MUL_1]] : index
// CHECK-DAG:           aie.use_lock(%[[LOCK_3]], Acquire, 0)
// CHECK:               func.call @some_work(%[[BUFF_3]], %[[ADD_1]]) : (memref<16xi32>, index) -> ()
// CHECK-NEXT:          aie.use_lock(%[[LOCK_3]], Release, 1)
// CHECK-DAG:           %[[C3_1:.*]] = arith.constant 3 : index
// CHECK-DAG:           %[[MUL_2:.*]] = arith.muli %[[C2]], %[[C3_1]] : index
// CHECK-DAG:           %[[ADD_2:.*]] = arith.addi %[[ARG0]], %[[MUL_2]] : index
// CHECK-DAG:           aie.use_lock(%[[LOCK_0]], Acquire, 0)
// CHECK:               func.call @some_work(%[[BUFF_0]], %[[ADD_2]]) : (memref<16xi32>, index) -> ()
// CHECK-NEXT:          aie.use_lock(%[[LOCK_0]], Release, 1)
// CHECK-NEXT:        }
// CHECK:             scf.for %[[ARG0:.+]] = %[[C17]] to %[[C21]] step %c2 {
// CHECK-DAG:           aie.use_lock(%[[LOCK_1]], Acquire, 0)
// CHECK:               func.call @some_work(%[[BUFF_1]], %[[ARG0]]) : (memref<16xi32>, index) -> ()
// CHECK-NEXT:          aie.use_lock(%[[LOCK_1]], Release, 1)
// CHECK-NEXT:        }
// CHECK:             %[[C3:.*]] = arith.constant 3 : index
// CHECK:             %[[C2_0:.*]] = arith.constant 2 : index
// CHECK:             aie.use_lock(%[[LOCK_2]], Acquire, 0)
// CHECK:             func.call @some_work(%[[BUFF_2]], %[[C1]]) : (memref<16xi32>, index) -> ()
// CHECK:             aie.use_lock(%[[LOCK_2]], Release, 1)
// CHECK:             %[[C1_1:.*]] = arith.constant 1 : index
// CHECK:             %[[MUL_0:.*]] = arith.muli %c1, %[[C1_1]] : index
// CHECK:             %[[ADD_0:.*]] = arith.addi %c1, %[[MUL_0]] : index
// CHECK:             aie.use_lock(%[[LOCK_3]], Acquire, 0)
// CHECK:             func.call @some_work(%[[BUFF_3]], %[[ADD_0]]) : (memref<16xi32>, index) -> ()
// CHECK:             aie.use_lock(%[[LOCK_3]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_0]], Acquire, 0)
// CHECK:             func.call @some_work(%[[BUFF_0]], %[[C3]]) : (memref<16xi32>, index) -> ()
// CHECK:             aie.use_lock(%[[LOCK_0]], Release, 1)

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
      %c21 = arith.constant 21 : index
      %subviewTop0 = aie.objectfifo.acquire @loop_of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elemTop0 = aie.objectfifo.subview.access %subviewTop0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      func.call @some_work(%elemTop0, %c0) : (memref<16xi32>,index) -> ()
      aie.objectfifo.release @loop_of (Produce, 1)
      scf.for %indexInHeight = %c1 to %c21 step %c2 {
        %subview = aie.objectfifo.acquire @loop_of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @some_work(%elem0,%indexInHeight) : (memref<16xi32>,index) -> ()
        aie.objectfifo.release @loop_of (Produce, 1)
      }
      scf.for %indexInHeight = %c1 to %c4 step %c1 {
        %subview = aie.objectfifo.acquire @loop_of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @some_work(%elem0,%indexInHeight) : (memref<16xi32>,index) -> ()
        aie.objectfifo.release @loop_of (Produce, 1)
      }
      aie.end
    }
  }
}