//===- nested_loop_unroll_inner_then_outer.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "loop_of_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "loop_of_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "loop_of_lock_0"}
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "loop_of_lock_1"}
// CHECK:           func.func @some_work(%[[A:.*]]: memref<4x4xi32>, %[[I:.*]]: index) {
// CHECK:             return
// CHECK:           }
// depth-2 fifo (buff0/1, lock0/1); binary-lock polarity acquire=0/release=1.
// Inner loop unrolled x2 then outer unrolled x2 (+ outer remainder), rotating
// buff0/buff1 continuously across the whole nest.
// CHECK:           %[[CORE:.*]] = aie.core(%[[VAL_0]]) {
// CHECK-DAG:             %[[CMAX:.*]] = arith.constant 4294967294 : index
// CHECK-DAG:             %[[C21:.*]] = arith.constant 21 : index
// CHECK-DAG:             %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:             %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:             %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:             %[[L0:.*]] = arith.constant 0 : i32
// CHECK-DAG:             %[[L1:.*]] = arith.constant 1 : i32
// CHECK:             scf.for %[[IV0:.*]] = %[[C0]] to %[[CMAX]] step %[[C2]] {
// CHECK:               aie.use_lock(%[[VAL_3]], Acquire, %[[L0]])
// CHECK:               %[[RCA:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK:               func.call @some_work(%[[RCA]], %[[C0]]) : (memref<4x4xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_3]], Release, %[[L1]])
// CHECK:               scf.for %[[IV1:.*]] = %[[C1]] to %[[C21]] step %[[C2]] {
// CHECK:                 aie.use_lock(%[[VAL_4]], Acquire, %[[L0]])
// CHECK:                 %[[RCB:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK:                 func.call @some_work(%[[RCB]], %[[IV1]]) : (memref<4x4xi32>, index) -> ()
// CHECK:                 aie.use_lock(%[[VAL_4]], Release, %[[L1]])
// CHECK:                 %[[J:.*]] = arith.addi %[[IV1]], %[[C1]] : index
// CHECK:                 aie.use_lock(%[[VAL_3]], Acquire, %[[L0]])
// CHECK:                 %[[RCC:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK:                 func.call @some_work(%[[RCC]], %[[J]]) : (memref<4x4xi32>, index) -> ()
// CHECK:                 aie.use_lock(%[[VAL_3]], Release, %[[L1]])
// CHECK:               }
// CHECK:               aie.use_lock(%[[VAL_4]], Acquire, %[[L0]])
// CHECK:               %[[RCD:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK:               func.call @some_work(%[[RCD]], %[[C0]]) : (memref<4x4xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_4]], Release, %[[L1]])
// CHECK:               aie.use_lock(%[[VAL_3]], Acquire, %[[L0]])
// CHECK:               %[[RCE:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK:               func.call @some_work(%[[RCE]], %[[C0]]) : (memref<4x4xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_3]], Release, %[[L1]])
// CHECK:               scf.for %[[IV1B:.*]] = %[[C1]] to %[[C21]] step %[[C2]] {
// CHECK:                 aie.use_lock(%[[VAL_4]], Acquire, %[[L0]])
// CHECK:                 %[[RCF:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK:                 func.call @some_work(%[[RCF]], %[[IV1B]]) : (memref<4x4xi32>, index) -> ()
// CHECK:                 aie.use_lock(%[[VAL_4]], Release, %[[L1]])
// CHECK:                 %[[JB:.*]] = arith.addi %[[IV1B]], %[[C1]] : index
// CHECK:                 aie.use_lock(%[[VAL_3]], Acquire, %[[L0]])
// CHECK:                 %[[RCG:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK:                 func.call @some_work(%[[RCG]], %[[JB]]) : (memref<4x4xi32>, index) -> ()
// CHECK:                 aie.use_lock(%[[VAL_3]], Release, %[[L1]])
// CHECK:               }
// CHECK:               aie.use_lock(%[[VAL_4]], Acquire, %[[L0]])
// CHECK:               %[[RCH:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK:               func.call @some_work(%[[RCH]], %[[C0]]) : (memref<4x4xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_4]], Release, %[[L1]])
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_3]], Acquire, %[[L0]])
// CHECK:             %[[RCI:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK:             func.call @some_work(%[[RCI]], %[[C0]]) : (memref<4x4xi32>, index) -> ()
// CHECK:             aie.use_lock(%[[VAL_3]], Release, %[[L1]])
// CHECK:             scf.for %[[IV0R:.*]] = %[[C1]] to %[[C21]] step %[[C2]] {
// CHECK:               aie.use_lock(%[[VAL_4]], Acquire, %[[L0]])
// CHECK:               %[[RCJ:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK:               func.call @some_work(%[[RCJ]], %[[IV0R]]) : (memref<4x4xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_4]], Release, %[[L1]])
// CHECK:               %[[JR:.*]] = arith.addi %[[IV0R]], %[[C1]] : index
// CHECK:               aie.use_lock(%[[VAL_3]], Acquire, %[[L0]])
// CHECK:               %[[RCK:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK:               func.call @some_work(%[[RCK]], %[[JR]]) : (memref<4x4xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_3]], Release, %[[L1]])
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_4]], Acquire, %[[L0]])
// CHECK:             %[[RCL:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
// CHECK:             func.call @some_work(%[[RCL]], %[[C0]]) : (memref<4x4xi32>, index) -> ()
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[L1]])
// CHECK:             aie.end
// CHECK:           }

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
        %elemTop0 = aie.objectfifo.acquire @loop_of(Produce) : memref<16xi32>
        %reinterpret_cast_0 = memref.reinterpret_cast %elemTop0 to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
        func.call @some_work(%reinterpret_cast_0, %c0) : (memref<4x4xi32>, index) -> ()
        aie.objectfifo.release @loop_of(Produce) [1]
        scf.for %indexInHeight = %c1 to %c21 step %c1 {
          %elem0 = aie.objectfifo.acquire @loop_of(Produce) : memref<16xi32>
          %reinterpret_cast_1 = memref.reinterpret_cast %elem0 to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
          func.call @some_work(%reinterpret_cast_1, %indexInHeight) : (memref<4x4xi32>, index) -> ()
          aie.objectfifo.release @loop_of(Produce) [1]
        }
        %elemTop1 = aie.objectfifo.acquire @loop_of(Produce) : memref<16xi32>
        %reinterpret_cast_2 = memref.reinterpret_cast %elemTop1 to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<16xi32> to memref<4x4xi32>
        func.call @some_work(%reinterpret_cast_2, %c0) : (memref<4x4xi32>, index) -> ()
        aie.objectfifo.release @loop_of(Produce) [1]
      }
      aie.end
    }
  }
}
