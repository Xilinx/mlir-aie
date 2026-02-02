//===- unroll_factor_multiple_objectfifos.mlir ------------------*- MLIR -*-===//
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
// CHECK:      %{{.*}}tile_1_2 = aie.tile(1, 2)
// CHECK:      %{{.*}}tile_1_3 = aie.tile(1, 3)
// CHECK:      %[[VAL_0:.*]] = aie.buffer(%{{.*}}tile_1_2) {sym_name = "of_2_buff_0"} : memref<16xi32> 
// CHECK:      %[[VAL_1:.*]] = aie.buffer(%{{.*}}tile_1_2) {sym_name = "of_2_buff_1"} : memref<16xi32> 
// CHECK:      %[[VAL_2:.*]] = aie.buffer(%{{.*}}tile_1_2) {sym_name = "of_2_buff_2"} : memref<16xi32> 
// CHECK:      %[[VAL_3:.*]] = aie.lock(%{{.*}}tile_1_2, 0) {init = 0 : i32, sym_name = "of_2_lock_0"}
// CHECK:      %[[VAL_4:.*]] = aie.lock(%{{.*}}tile_1_2, 1) {init = 0 : i32, sym_name = "of_2_lock_1"}
// CHECK:      %[[VAL_5:.*]] = aie.lock(%{{.*}}tile_1_2, 2) {init = 0 : i32, sym_name = "of_2_lock_2"}
// CHECK:      %[[VAL_6:.*]] = aie.buffer(%{{.*}}tile_1_3) {sym_name = "of_1_buff_0"} : memref<16xi32> 
// CHECK:      %[[VAL_7:.*]] = aie.buffer(%{{.*}}tile_1_3) {sym_name = "of_1_buff_1"} : memref<16xi32> 
// CHECK:      %[[VAL_8:.*]] = aie.lock(%{{.*}}tile_1_3, 0) {init = 0 : i32, sym_name = "of_1_lock_0"}
// CHECK:      %[[VAL_9:.*]] = aie.lock(%{{.*}}tile_1_3, 1) {init = 0 : i32, sym_name = "of_1_lock_1"}
// CHECK:      func.func @some_work(%arg0: memref<16xi32>, %arg1: memref<16xi32>, %arg2: index) {
// CHECK:        return
// CHECK:      }
// CHECK:      %core_1_2 = aie.core(%{{.*}}tile_1_2) {
// CHECK:        %c0 = arith.constant 0 : index
// CHECK:        %c1 = arith.constant 1 : index
// CHECK:        %c12 = arith.constant 12 : index
// CHECK:        %c6 = arith.constant 6 : index
// CHECK:        scf.for %arg0 = %c0 to %c12 step %c6 {
// CHECK:          aie.use_lock(%[[VAL_8]], Acquire, 1)
// CHECK:          aie.use_lock(%[[VAL_3]], Acquire, 0)
// CHECK:          func.call @some_work(%[[VAL_6]], %[[VAL_0]], %arg0) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:          aie.use_lock(%[[VAL_8]], Release, 0)
// CHECK:          aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:          %c1_0 = arith.constant 1 : index
// CHECK:          %0 = arith.muli %c1, %c1_0 : index
// CHECK:          %1 = arith.addi %arg0, %0 : index
// CHECK:          aie.use_lock(%[[VAL_9]], Acquire, 1)
// CHECK:          aie.use_lock(%[[VAL_4]], Acquire, 0)
// CHECK:          func.call @some_work(%[[VAL_7]], %[[VAL_1]], %1) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:          aie.use_lock(%[[VAL_9]], Release, 0)
// CHECK:          aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:          %c2 = arith.constant 2 : index
// CHECK:          %2 = arith.muli %c1, %c2 : index
// CHECK:          %3 = arith.addi %arg0, %2 : index
// CHECK:          aie.use_lock(%[[VAL_8]], Acquire, 1)
// CHECK:          aie.use_lock(%[[VAL_5]], Acquire, 0)
// CHECK:          func.call @some_work(%[[VAL_6]], %[[VAL_2]], %3) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:          aie.use_lock(%[[VAL_8]], Release, 0)
// CHECK:          aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:          %c3 = arith.constant 3 : index
// CHECK:          %4 = arith.muli %c1, %c3 : index
// CHECK:          %5 = arith.addi %arg0, %4 : index
// CHECK:          aie.use_lock(%[[VAL_9]], Acquire, 1)
// CHECK:          aie.use_lock(%[[VAL_3]], Acquire, 0)
// CHECK:          func.call @some_work(%[[VAL_7]], %[[VAL_0]], %5) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:          aie.use_lock(%[[VAL_9]], Release, 0)
// CHECK:          aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:          %c4 = arith.constant 4 : index
// CHECK:          %6 = arith.muli %c1, %c4 : index
// CHECK:          %7 = arith.addi %arg0, %6 : index
// CHECK:          aie.use_lock(%[[VAL_8]], Acquire, 1)
// CHECK:          aie.use_lock(%[[VAL_4]], Acquire, 0)
// CHECK:          func.call @some_work(%[[VAL_6]], %[[VAL_1]], %7) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:          aie.use_lock(%[[VAL_8]], Release, 0)
// CHECK:          aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:          %c5 = arith.constant 5 : index
// CHECK:          %8 = arith.muli %c1, %c5 : index
// CHECK:          %9 = arith.addi %arg0, %8 : index
// CHECK:          aie.use_lock(%[[VAL_9]], Acquire, 1)
// CHECK:          aie.use_lock(%[[VAL_5]], Acquire, 0)
// CHECK:          func.call @some_work(%[[VAL_7]], %[[VAL_2]], %9) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:          aie.use_lock(%[[VAL_9]], Release, 0)
// CHECK:          aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:        }
// CHECK:        aie.end
// CHECK:      }
// CHECK:    }
// CHECK:  }

module {
  aie.device(xcvc1902) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    aie.objectfifo @of_1 (%tile13, {%tile12}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_2 (%tile12, {%tile13}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    func.func @some_work(%line_inA:memref<16xi32>, %line_inB:memref<16xi32>, %index:index) -> () {
      return
    }
    %core12 = aie.core(%tile12) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c12 = arith.constant 12 : index

      scf.for %indexInHeight = %c0 to %c12 step %c1 {
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
