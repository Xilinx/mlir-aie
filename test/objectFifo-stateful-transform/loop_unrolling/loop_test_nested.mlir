//===- loop_test_nested.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           memref.global "public" @loop_of : memref<16xi32>
// CHECK-DAG:       %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK-DAG:       %[[BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "loop_of_buff_0"} : memref<16xi32>
// CHECK-DAG:       %[[BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "loop_of_buff_1"} : memref<16xi32>
// CHECK-DAG:       %[[LOCK_0:.*]] = aie.lock(%[[TILE_1_2]], 0) {init = 0 : i32, sym_name = "loop_of_lock_0"}
// CHECK-DAG:       %[[LOCK_1:.*]] = aie.lock(%[[TILE_1_2]], 1) {init = 0 : i32, sym_name = "loop_of_lock_1"}
// CHECK:           func.func @some_work(%{{.+}}: memref<4x4xi32>, %{{.+}}: index) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[CORE_1_2:.*]] = aie.core(%[[TILE_1_2]]) {
// CHECK-DAG:         %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:         %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:         %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:         %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:         %[[C21:.*]] = arith.constant 21 : index
// CHECK-DAG:         %[[C4294967295:.*]] = arith.constant 4294967295 : index
// CHECK-DAG:         %[[C4294967294:.*]] = arith.constant 4294967294 : index
// CHECK-DAG:         %[[C2_0:.*]] = arith.constant 2 : index
// CHECK:             scf.for %[[ARG0:.+]] = %[[C0]] to %[[C4294967294]] step %[[C2_0]] {
// CHECK:               aie.use_lock(%[[LOCK_0]], Acquire, 0)
// CHECK-NEXT:          %[[REINTERPRET_0:.+]] = memref.reinterpret_cast %[[BUFF_0]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK-NEXT:          func.call @some_work(%[[REINTERPRET_0]], %[[C0]]) : (memref<4x4xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[LOCK_0]], Release, 1)
// CHECK-DAG:           %[[C2_4:.*]] = arith.constant 2 : index
// CHECK:               scf.for %[[ARG1:.+]] = %[[C1]] to %[[C21]] step %[[C2_4]] {
// CHECK-NEXT:            aie.use_lock(%[[LOCK_1]], Acquire, 0)
// CHECK-NEXT:            %[[REINTERPRET_1:.+]] = memref.reinterpret_cast %[[BUFF_1]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK-NEXT:            func.call @some_work(%[[REINTERPRET_1]], %[[ARG1]]) : (memref<4x4xi32>, index) -> ()
// CHECK-NEXT:            aie.use_lock(%[[LOCK_1]], Release, 1)
// CHECK-DAG:             %[[C1_1:.*]] = arith.constant 1 : index
// CHECK-DAG:             %[[MUL_0:.*]] = arith.muli %[[C1]], %[[C1_1]] : index
// CHECK-DAG:             %[[ADD_0:.*]] = arith.addi %[[ARG1]], %[[MUL_0]] : index
// CHECK-DAG:             aie.use_lock(%[[LOCK_0]], Acquire, 0)
// CHECK:                 %[[REINTERPRET_2:.+]] = memref.reinterpret_cast %[[BUFF_0]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK-NEXT:            func.call @some_work(%[[REINTERPRET_2]], %[[ADD_0]]) : (memref<4x4xi32>, index) -> ()
// CHECK-NEXT:            aie.use_lock(%[[LOCK_0]], Release, 1)
// CHECK-NEXT:          }
// CHECK:               aie.use_lock(%[[LOCK_1]], Acquire, 0)
// CHECK-NEXT:          %[[REINTERPRET_3:.+]] = memref.reinterpret_cast %[[BUFF_1]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK-NEXT:          func.call @some_work(%[[REINTERPRET_3]], %[[C0]]) : (memref<4x4xi32>, index) -> ()
// CHECK-NEXT:          aie.use_lock(%[[LOCK_1]], Release, 1)
// CHECK:               aie.use_lock(%[[LOCK_0]], Acquire, 0)
// CHECK-NEXT:          %[[REINTERPRET_4:.+]] = memref.reinterpret_cast %[[BUFF_0]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK-NEXT:          func.call @some_work(%[[REINTERPRET_4]], %[[C0]]) : (memref<4x4xi32>, index) -> ()
// CHECK-NEXT:          aie.use_lock(%[[LOCK_0]], Release, 1)
// CHECK:               %[[C2_3:.*]] = arith.constant 2 : index
// CHECK:               scf.for %[[ARG1:.+]] = %[[C1]] to %[[C21]] step %[[C2_3]] {
// CHECK-NEXT:            aie.use_lock(%[[LOCK_1]], Acquire, 0)
// CHECK-NEXT:            %[[REINTERPRET_5:.+]] = memref.reinterpret_cast %[[BUFF_1]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK-NEXT:            func.call @some_work(%[[REINTERPRET_5]], %[[ARG1]]) : (memref<4x4xi32>, index) -> ()
// CHECK-NEXT:            aie.use_lock(%[[LOCK_1]], Release, 1)
// CHECK-DAG:             %[[C1_1:.*]] = arith.constant 1 : index
// CHECK-DAG:             %[[MUL_1:.*]] = arith.muli %[[C1]], %[[C1_1]] : index
// CHECK-DAG:             %[[ADD_1:.*]] = arith.addi %[[ARG1]], %[[MUL_1]] : index
// CHECK-DAG:             aie.use_lock(%[[LOCK_0]], Acquire, 0)
// CHECK:                 %[[REINTERPRET_6:.+]] = memref.reinterpret_cast %[[BUFF_0]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK-NEXT:            func.call @some_work(%[[REINTERPRET_6]], %[[ADD_1]]) : (memref<4x4xi32>, index) -> ()
// CHECK-NEXT:            aie.use_lock(%[[LOCK_0]], Release, 1)
// CHECK-NEXT:          }
// CHECK:               aie.use_lock(%[[LOCK_1]], Acquire, 0)
// CHECK-NEXT:          %[[REINTERPRET_7:.+]] = memref.reinterpret_cast %[[BUFF_1]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK-NEXT:          func.call @some_work(%[[REINTERPRET_7]], %[[C0]]) : (memref<4x4xi32>, index) -> ()
// CHECK-NEXT:          aie.use_lock(%[[LOCK_1]], Release, 1)
// CHECK-NEXT:        }
// CHECK:             aie.use_lock(%[[LOCK_0]], Acquire, 0)
// CHECK-NEXT:        %[[REINTERPRET_8:.+]] = memref.reinterpret_cast %[[BUFF_0]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK-NEXT:        func.call @some_work(%[[REINTERPRET_8]], %[[C0]]) : (memref<4x4xi32>, index) -> ()
// CHECK:             aie.use_lock(%[[LOCK_0]], Release, 1)
// CHECK:             %[[C2_4:.*]] = arith.constant 2 : index
// CHECK:             scf.for %[[ARG0:.+]] = %[[C1]] to %[[C21]] step %[[C2_4]] {
// CHECK-NEXT:          aie.use_lock(%[[LOCK_1]], Acquire, 0)
// CHECK-NEXT:          %[[REINTERPRET_9:.+]] = memref.reinterpret_cast %[[BUFF_1]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK-NEXT:          func.call @some_work(%[[REINTERPRET_9]], %[[ARG0]]) : (memref<4x4xi32>, index) -> ()
// CHECK-NEXT:          aie.use_lock(%[[LOCK_1]], Release, 1)
// CHECK-DAG:           %[[C1_4:.*]] = arith.constant 1 : index
// CHECK-DAG:           %[[MUL_2:.*]] = arith.muli %[[C1]], %[[C1_4]] : index
// CHECK-DAG:           %[[ADD_2:.*]] = arith.addi %[[ARG0]], %[[MUL_2]] : index
// CHECK-DAG:           aie.use_lock(%[[LOCK_0]], Acquire, 0)
// CHECK:               %[[REINTERPRET_10:.+]] = memref.reinterpret_cast %[[BUFF_0]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK-NEXT:          func.call @some_work(%[[REINTERPRET_10]], %[[ADD_1]]) : (memref<4x4xi32>, index) -> ()
// CHECK-NEXT:          aie.use_lock(%[[LOCK_0]], Release, 1)
// CHECK-NEXT:        }
// CHECK:             aie.use_lock(%[[LOCK_1]], Acquire, 0)
// CHECK-NEXT:        %[[REINTERPRET_11:.+]] = memref.reinterpret_cast %[[BUFF_1]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK-NEXT:        func.call @some_work(%[[REINTERPRET_11]], %[[C0]]) : (memref<4x4xi32>, index) -> ()
// CHECK-NEXT:        aie.use_lock(%[[LOCK_1]], Release, 1)
module {
  aie.device(xcvc1902) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    aie.objectfifo @loop_of (%tile12, {%tile13}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    func.func @some_work(%line_in: memref<4x4xi32>, %index: index) -> () {
      return
    }
    %core12 = aie.core(%tile12) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c21 = arith.constant 21 : index
      %cmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg0 = %c0 to %cmax step %c1 {
        %subviewTop0 = aie.objectfifo.acquire @loop_of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %elemTop0 = aie.objectfifo.subview.access %subviewTop0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %reinterpret_cast_0 = memref.reinterpret_cast %elemTop0 to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
        func.call @some_work(%reinterpret_cast_0, %c0) : (memref<4x4xi32>, index) -> ()
        aie.objectfifo.release @loop_of (Produce, 1)
        scf.for %indexInHeight = %c1 to %c21 step %c1 {
          %subview = aie.objectfifo.acquire @loop_of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
          %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
          %reinterpret_cast_1 = memref.reinterpret_cast %elem0 to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
          func.call @some_work(%reinterpret_cast_1, %indexInHeight) : (memref<4x4xi32>, index) -> ()
          aie.objectfifo.release @loop_of (Produce, 1)
        }
        %subviewTop1 = aie.objectfifo.acquire @loop_of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %elemTop1 = aie.objectfifo.subview.access %subviewTop1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %reinterpret_cast_2 = memref.reinterpret_cast %elemTop1 to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
        func.call @some_work(%reinterpret_cast_2, %c0) : (memref<4x4xi32>, index) -> ()
        aie.objectfifo.release @loop_of (Produce, 1)
      }
      aie.end
    }
  }
}
