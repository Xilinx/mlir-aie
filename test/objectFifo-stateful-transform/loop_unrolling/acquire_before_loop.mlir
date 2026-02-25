//===- acquire_before_loop.mlir ---------------------------------*- MLIR -*-===//
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

// CHECK:  module {
// CHECK:    aie.device(xcvc1902) {
// CHECK:      %{{.*}}tile_1_2 = aie.tile(1, 2)
// CHECK:      %{{.*}}tile_1_3 = aie.tile(1, 3)
// CHECK:      %[[VAL_0:.*]] = aie.buffer(%{{.*}}tile_1_2) {sym_name = "loop_of_buff_0"} : memref<16xi32> 
// CHECK:      %[[VAL_1:.*]] = aie.buffer(%{{.*}}tile_1_2) {sym_name = "loop_of_buff_1"} : memref<16xi32> 
// CHECK:      %[[VAL_2:.*]] = aie.buffer(%{{.*}}tile_1_2) {sym_name = "loop_of_buff_2"} : memref<16xi32> 
// CHECK:      %[[VAL_3:.*]] = aie.buffer(%{{.*}}tile_1_2) {sym_name = "loop_of_buff_3"} : memref<16xi32> 
// CHECK:      %[[VAL_4:.*]] = aie.lock(%{{.*}}tile_1_2, 0) {init = 0 : i32, sym_name = "loop_of_lock_0"}
// CHECK:      %[[VAL_5:.*]] = aie.lock(%{{.*}}tile_1_2, 1) {init = 0 : i32, sym_name = "loop_of_lock_1"}
// CHECK:      %[[VAL_6:.*]] = aie.lock(%{{.*}}tile_1_2, 2) {init = 0 : i32, sym_name = "loop_of_lock_2"}
// CHECK:      %[[VAL_7:.*]] = aie.lock(%{{.*}}tile_1_2, 3) {init = 0 : i32, sym_name = "loop_of_lock_3"}
// CHECK:      func.func @some_work(%arg0: memref<16xi32>, %arg1: index) {
// CHECK:        return
// CHECK:      }
// CHECK:      %core_1_2 = aie.core(%{{.*}}tile_1_2) {
// CHECK:        %c0 = arith.constant 0 : index
// CHECK:        %c1 = arith.constant 1 : index
// CHECK:        %c2 = arith.constant 2 : index
// CHECK:        %c4 = arith.constant 4 : index
// CHECK:        %c9 = arith.constant 9 : index
// CHECK:        aie.use_lock(%[[VAL_4]], Acquire, 0)
// CHECK:        func.call @some_work(%[[VAL_0]], %c0) : (memref<16xi32>, index) -> ()
// CHECK:        aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:        %c4_0 = arith.constant 4 : index
// CHECK:        scf.for %arg0 = %c1 to %c9 step %c4_0 {
// CHECK:          aie.use_lock(%[[VAL_5]], Acquire, 0)
// CHECK:          func.call @some_work(%[[VAL_1]], %arg0) : (memref<16xi32>, index) -> ()
// CHECK:          aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:          %c1_1 = arith.constant 1 : index
// CHECK:          %0 = arith.muli %c1, %c1_1 : index
// CHECK:          %1 = arith.addi %arg0, %0 : index
// CHECK:          aie.use_lock(%[[VAL_6]], Acquire, 0)
// CHECK:          func.call @some_work(%[[VAL_2]], %1) : (memref<16xi32>, index) -> ()
// CHECK:          aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:          %c2_2 = arith.constant 2 : index
// CHECK:          %2 = arith.muli %c1, %c2_2 : index
// CHECK:          %3 = arith.addi %arg0, %2 : index
// CHECK:          aie.use_lock(%[[VAL_7]], Acquire, 0)
// CHECK:          func.call @some_work(%[[VAL_3]], %3) : (memref<16xi32>, index) -> ()
// CHECK:          aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:          %c3 = arith.constant 3 : index
// CHECK:          %4 = arith.muli %c1, %c3 : index
// CHECK:          %5 = arith.addi %arg0, %4 : index
// CHECK:          aie.use_lock(%[[VAL_4]], Acquire, 0)
// CHECK:          func.call @some_work(%[[VAL_0]], %5) : (memref<16xi32>, index) -> ()
// CHECK:          aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:        }
// CHECK:        aie.end
// CHECK:      }
// CHECK:    }
// CHECK:  }

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
      %subviewTop0 = aie.objectfifo.acquire @loop_of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elemTop0 = aie.objectfifo.subview.access %subviewTop0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      func.call @some_work(%elemTop0, %c0) : (memref<16xi32>,index) -> ()
      aie.objectfifo.release @loop_of (Produce, 1)
      scf.for %indexInHeight = %c1 to %c9 step %c1 {
        %subview = aie.objectfifo.acquire @loop_of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @some_work(%elem0,%indexInHeight) : (memref<16xi32>,index) -> ()
        aie.objectfifo.release @loop_of (Produce, 1)
      }
      aie.end
    }
  }
}
