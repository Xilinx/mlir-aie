//===- loop_test_unroll_remainder_nested.mlir ------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   module {
// CHECK:          aie.device(xcvc1902) {
// CHECK:              memref.global "public" @of_2 : memref<16xi32>
// CHECK:              memref.global "public" @of_1 : memref<16xi32>
// CHECK:              %tile_1_2 = aie.tile(1, 2)
// CHECK:              %tile_1_3 = aie.tile(1, 3)
// CHECK:              %of_2_buff_0 = aie.buffer(%tile_1_2) {sym_name = "of_2_buff_0"} : memref<16xi32> 
// CHECK:              %of_2_buff_1 = aie.buffer(%tile_1_2) {sym_name = "of_2_buff_1"} : memref<16xi32> 
// CHECK:              %of_2_buff_2 = aie.buffer(%tile_1_2) {sym_name = "of_2_buff_2"} : memref<16xi32> 
// CHECK:              %of_2_lock_0 = aie.lock(%tile_1_2, 0) {init = 0 : i32, sym_name = "of_2_lock_0"}
// CHECK:              %of_2_lock_1 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "of_2_lock_1"}
// CHECK:              %of_2_lock_2 = aie.lock(%tile_1_2, 2) {init = 0 : i32, sym_name = "of_2_lock_2"}
// CHECK:              %of_1_buff_0 = aie.buffer(%tile_1_3) {sym_name = "of_1_buff_0"} : memref<16xi32> 
// CHECK:              %of_1_buff_1 = aie.buffer(%tile_1_3) {sym_name = "of_1_buff_1"} : memref<16xi32> 
// CHECK:              %of_1_buff_2 = aie.buffer(%tile_1_3) {sym_name = "of_1_buff_2"} : memref<16xi32> 
// CHECK:              %of_1_lock_0 = aie.lock(%tile_1_3, 0) {init = 0 : i32, sym_name = "of_1_lock_0"}
// CHECK:              %of_1_lock_1 = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "of_1_lock_1"}
// CHECK:              %of_1_lock_2 = aie.lock(%tile_1_3, 2) {init = 0 : i32, sym_name = "of_1_lock_2"}
// CHECK:              func.func @some_work(%arg0: memref<16xi32>, %arg1: memref<16xi32>, %arg2: index, %arg3: index) {
// CHECK:                return
// CHECK:              }
// CHECK:              %core_1_2 = aie.core(%tile_1_2) {
// CHECK:                %c0 = arith.constant 0 : index
// CHECK:                %c1 = arith.constant 1 : index
// CHECK:                %c8 = arith.constant 8 : index
// CHECK:                %c9 = arith.constant 9 : index
// CHECK:                %c3 = arith.constant 3 : index
// CHECK:                scf.for %arg0 = %c0 to %c9 step %c3 {
// CHECK:                  aie.use_lock(%of_2_lock_0, Acquire, 0)
// CHECK:                  %c6 = arith.constant 6 : index
// CHECK:                  %c3_0 = arith.constant 3 : index
// CHECK:                  scf.for %arg1 = %c0 to %c6 step %c3_0 {
// CHECK:                    aie.use_lock(%of_1_lock_0, Acquire, 1)
// CHECK:                    func.call @some_work(%of_1_buff_0, %of_2_buff_0, %arg0, %arg1) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                    aie.use_lock(%of_1_lock_0, Release, 0)
// CHECK:                    %c1_12 = arith.constant 1 : index
// CHECK:                    %10 = arith.muli %c1, %c1_12 : index
// CHECK:                    %11 = arith.addi %arg1, %10 : index
// CHECK:                    aie.use_lock(%of_1_lock_1, Acquire, 1)
// CHECK:                    func.call @some_work(%of_1_buff_1, %of_2_buff_0, %arg0, %11) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                    aie.use_lock(%of_1_lock_1, Release, 0)
// CHECK:                    %c2_13 = arith.constant 2 : index
// CHECK:                    %12 = arith.muli %c1, %c2_13 : index
// CHECK:                    %13 = arith.addi %arg1, %12 : index
// CHECK:                    aie.use_lock(%of_1_lock_2, Acquire, 1)
// CHECK:                    func.call @some_work(%of_1_buff_2, %of_2_buff_0, %arg0, %13) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                    aie.use_lock(%of_1_lock_2, Release, 0)
// CHECK:                  }
// CHECK:                  %c2 = arith.constant 2 : index
// CHECK:                  aie.use_lock(%of_1_lock_0, Acquire, 1)
// CHECK:                  func.call @some_work(%of_1_buff_0, %of_2_buff_0, %arg0, %c6) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                  aie.use_lock(%of_1_lock_0, Release, 0)
// CHECK:                  %c1_1 = arith.constant 1 : index
// CHECK:                  %0 = arith.muli %c1, %c1_1 : index
// CHECK:                  %1 = arith.addi %c6, %0 : index
// CHECK:                  aie.use_lock(%of_1_lock_1, Acquire, 1)
// CHECK:                  func.call @some_work(%of_1_buff_1, %of_2_buff_0, %arg0, %1) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                  aie.use_lock(%of_1_lock_1, Release, 0)
// CHECK:                  aie.use_lock(%of_2_lock_0, Release, 1)
// CHECK:                  %c1_2 = arith.constant 1 : index
// CHECK:                  %2 = arith.muli %c1, %c1_2 : index
// CHECK:                  %3 = arith.addi %arg0, %2 : index
// CHECK:                  aie.use_lock(%of_2_lock_1, Acquire, 0)
// CHECK:                  %c6_3 = arith.constant 6 : index
// CHECK:                  %c3_4 = arith.constant 3 : index
// CHECK:                  scf.for %arg1 = %c0 to %c6_3 step %c3_4 {
// CHECK:                    aie.use_lock(%of_1_lock_2, Acquire, 1)
// CHECK:                    func.call @some_work(%of_1_buff_2, %of_2_buff_1, %3, %arg1) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                    aie.use_lock(%of_1_lock_2, Release, 0)
// CHECK:                    %c1_12 = arith.constant 1 : index
// CHECK:                    %10 = arith.muli %c1, %c1_12 : index
// CHECK:                    %11 = arith.addi %arg1, %10 : index
// CHECK:                    aie.use_lock(%of_1_lock_0, Acquire, 1)
// CHECK:                    func.call @some_work(%of_1_buff_0, %of_2_buff_1, %3, %11) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                    aie.use_lock(%of_1_lock_0, Release, 0)
// CHECK:                    %c2_13 = arith.constant 2 : index
// CHECK:                    %12 = arith.muli %c1, %c2_13 : index
// CHECK:                    %13 = arith.addi %arg1, %12 : index
// CHECK:                    aie.use_lock(%of_1_lock_1, Acquire, 1)
// CHECK:                    func.call @some_work(%of_1_buff_1, %of_2_buff_1, %3, %13) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                    aie.use_lock(%of_1_lock_1, Release, 0)
// CHECK:                  }
// CHECK:                  %c2_5 = arith.constant 2 : index
// CHECK:                  aie.use_lock(%of_1_lock_2, Acquire, 1)
// CHECK:                  func.call @some_work(%of_1_buff_2, %of_2_buff_1, %3, %c6_3) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                  aie.use_lock(%of_1_lock_2, Release, 0)
// CHECK:                  %c1_6 = arith.constant 1 : index
// CHECK:                  %4 = arith.muli %c1, %c1_6 : index
// CHECK:                  %5 = arith.addi %c6_3, %4 : index
// CHECK:                  aie.use_lock(%of_1_lock_0, Acquire, 1)
// CHECK:                  func.call @some_work(%of_1_buff_0, %of_2_buff_1, %3, %5) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                  aie.use_lock(%of_1_lock_0, Release, 0)
// CHECK:                  aie.use_lock(%of_2_lock_1, Release, 1)
// CHECK:                  %c2_7 = arith.constant 2 : index
// CHECK:                  %6 = arith.muli %c1, %c2_7 : index
// CHECK:                  %7 = arith.addi %arg0, %6 : index
// CHECK:                  aie.use_lock(%of_2_lock_2, Acquire, 0)
// CHECK:                  %c6_8 = arith.constant 6 : index
// CHECK:                  %c3_9 = arith.constant 3 : index
// CHECK:                  scf.for %arg1 = %c0 to %c6_8 step %c3_9 {
// CHECK:                    aie.use_lock(%of_1_lock_1, Acquire, 1)
// CHECK:                    func.call @some_work(%of_1_buff_1, %of_2_buff_2, %7, %arg1) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                    aie.use_lock(%of_1_lock_1, Release, 0)
// CHECK:                    %c1_12 = arith.constant 1 : index
// CHECK:                    %10 = arith.muli %c1, %c1_12 : index
// CHECK:                    %11 = arith.addi %arg1, %10 : index
// CHECK:                    aie.use_lock(%of_1_lock_2, Acquire, 1)
// CHECK:                    func.call @some_work(%of_1_buff_2, %of_2_buff_2, %7, %11) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                    aie.use_lock(%of_1_lock_2, Release, 0)
// CHECK:                    %c2_13 = arith.constant 2 : index
// CHECK:                    %12 = arith.muli %c1, %c2_13 : index
// CHECK:                    %13 = arith.addi %arg1, %12 : index
// CHECK:                    aie.use_lock(%of_1_lock_0, Acquire, 1)
// CHECK:                    func.call @some_work(%of_1_buff_0, %of_2_buff_2, %7, %13) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                    aie.use_lock(%of_1_lock_0, Release, 0)
// CHECK:                  }
// CHECK:                  %c2_10 = arith.constant 2 : index
// CHECK:                  aie.use_lock(%of_1_lock_1, Acquire, 1)
// CHECK:                  func.call @some_work(%of_1_buff_1, %of_2_buff_2, %7, %c6_8) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                  aie.use_lock(%of_1_lock_1, Release, 0)
// CHECK:                  %c1_11 = arith.constant 1 : index
// CHECK:                  %8 = arith.muli %c1, %c1_11 : index
// CHECK:                  %9 = arith.addi %c6_8, %8 : index
// CHECK:                  aie.use_lock(%of_1_lock_2, Acquire, 1)
// CHECK:                  func.call @some_work(%of_1_buff_2, %of_2_buff_2, %7, %9) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                  aie.use_lock(%of_1_lock_2, Release, 0)
// CHECK:                  aie.use_lock(%of_2_lock_2, Release, 1)
// CHECK:                }
// CHECK:                aie.end
// CHECK:              }
// CHECK:            }
// CHECK:          }

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
