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
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "loop_of_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "loop_of_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "loop_of_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "loop_of_buff_3"} : memref<16xi32>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "loop_of_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "loop_of_lock_1"}
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "loop_of_lock_2"}
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "loop_of_lock_3"}
// CHECK:           func.func @some_work(%[[VAL_10:.*]]: memref<16xi32>, %[[VAL_11:.*]]: index) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[IDX9:.*]] = arith.constant 9 : index
// CHECK:             %[[IDX4:.*]] = arith.constant 4 : index
// CHECK:             %[[IDX1:.*]] = arith.constant 1 : index
// CHECK:             %[[IDX0:.*]] = arith.constant 0 : index
// CHECK:             %[[C0I:.*]] = arith.constant 0 : i32
// CHECK:             %[[C1I:.*]] = arith.constant 1 : i32
// CHECK:             %[[IDX2:.*]] = arith.constant 2 : index
// CHECK:             %[[IDX3:.*]] = arith.constant 3 : index
// CHECK:             aie.use_lock(%[[VAL_6]], Acquire, %[[C0I]])
// CHECK:             func.call @some_work(%[[VAL_2]], %[[IDX0]]) : (memref<16xi32>, index) -> ()
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[C1I]])
// CHECK:             scf.for %[[IV:.*]] = %[[IDX1]] to %[[IDX9]] step %[[IDX4]] {
// CHECK:               aie.use_lock(%[[VAL_7]], Acquire, %[[C0I]])
// CHECK:               func.call @some_work(%[[VAL_3]], %[[IV]]) : (memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_7]], Release, %[[C1I]])
// CHECK:               %[[A0:.*]] = arith.addi %[[IV]], %[[IDX1]] : index
// CHECK:               aie.use_lock(%[[VAL_8]], Acquire, %[[C0I]])
// CHECK:               func.call @some_work(%[[VAL_4]], %[[A0]]) : (memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_8]], Release, %[[C1I]])
// CHECK:               %[[A1:.*]] = arith.addi %[[IV]], %[[IDX2]] : index
// CHECK:               aie.use_lock(%[[VAL_9]], Acquire, %[[C0I]])
// CHECK:               func.call @some_work(%[[VAL_5]], %[[A1]]) : (memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_9]], Release, %[[C1I]])
// CHECK:               %[[A2:.*]] = arith.addi %[[IV]], %[[IDX3]] : index
// CHECK:               aie.use_lock(%[[VAL_6]], Acquire, %[[C0I]])
// CHECK:               func.call @some_work(%[[VAL_2]], %[[A2]]) : (memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_6]], Release, %[[C1I]])
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
      %elemTop0 = aie.objectfifo.acquire @loop_of(Produce) : memref<16xi32>
      func.call @some_work(%elemTop0, %c0) : (memref<16xi32>,index) -> ()
      aie.objectfifo.release @loop_of(Produce) [1]
      scf.for %indexInHeight = %c1 to %c9 step %c1 {
        %elem0 = aie.objectfifo.acquire @loop_of(Produce) : memref<16xi32>
        func.call @some_work(%elem0,%indexInHeight) : (memref<16xi32>,index) -> ()
        aie.objectfifo.release @loop_of(Produce) [1]
      }
      aie.end
    }
  }
}
