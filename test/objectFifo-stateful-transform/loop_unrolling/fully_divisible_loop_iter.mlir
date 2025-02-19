//===- fully_divisible_loop_iter.mlir ---------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK:  module {
// CHECK:    aie.device(xcvc1902) {
// CHECK:      memref.global "public" @loop_of : memref<16xi32>
// CHECK:      %{{.*}}tile_1_2 = aie.tile(1, 2)
// CHECK:      %{{.*}}tile_1_3 = aie.tile(1, 3)
// CHECK:      %loop_of_buff_0 = aie.buffer(%{{.*}}tile_1_2) {sym_name = "loop_of_buff_0"} : memref<16xi32> 
// CHECK:      %loop_of_buff_1 = aie.buffer(%{{.*}}tile_1_2) {sym_name = "loop_of_buff_1"} : memref<16xi32> 
// CHECK:      %loop_of_lock_0 = aie.lock(%{{.*}}tile_1_2, 0) {init = 0 : i32, sym_name = "loop_of_lock_0"}
// CHECK:      %loop_of_lock_1 = aie.lock(%{{.*}}tile_1_2, 1) {init = 0 : i32, sym_name = "loop_of_lock_1"}
// CHECK:      func.func @some_work(%arg0: memref<16xi32>, %arg1: index) {
// CHECK:        return
// CHECK:      }
// CHECK:      %core_1_2 = aie.core(%{{.*}}tile_1_2) {
// CHECK:        %c0 = arith.constant 0 : index
// CHECK:        %c1 = arith.constant 1 : index
// CHECK:        %c4 = arith.constant 4 : index
// CHECK:        %c2 = arith.constant 2 : index
// CHECK:        scf.for %arg0 = %c0 to %c4 step %c2 {
// CHECK:          aie.use_lock(%loop_of_lock_0, Acquire, 0)
// CHECK:          func.call @some_work(%loop_of_buff_0, %arg0) : (memref<16xi32>, index) -> ()
// CHECK:          aie.use_lock(%loop_of_lock_0, Release, 1)
// CHECK:          %c1_0 = arith.constant 1 : index
// CHECK:          %0 = arith.muli %c1, %c1_0 : index
// CHECK:          %1 = arith.addi %arg0, %0 : index
// CHECK:          aie.use_lock(%loop_of_lock_1, Acquire, 0)
// CHECK:          func.call @some_work(%loop_of_buff_1, %1) : (memref<16xi32>, index) -> ()
// CHECK:          aie.use_lock(%loop_of_lock_1, Release, 1)
// CHECK:        }
// CHECK:        aie.end
// CHECK:      }
// CHECK:    }
// CHECK:  }

module {
  aie.device(xcvc1902) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    aie.objectfifo @loop_of (%tile12, {%tile13}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    func.func @some_work(%line_in:memref<16xi32>, %index:index) -> () {
      return
    }
    %core12 = aie.core(%tile12) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %indexInHeight = %c0 to %c4 step %c1 {
        %subview = aie.objectfifo.acquire @loop_of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @some_work(%elem0,%indexInHeight) : (memref<16xi32>,index) -> ()
        aie.objectfifo.release @loop_of (Produce, 1)
      }
      aie.end
    }
  }
}