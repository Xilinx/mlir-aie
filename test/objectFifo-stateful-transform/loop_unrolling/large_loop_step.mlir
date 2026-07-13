//===- large_loop_step.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=false" %s | FileCheck %s

// CHECK-LABEL: module {
// CHECK:         aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "loop_of_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "loop_of_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "loop_of_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "loop_of_buff_3"} : memref<16xi32>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_0]], 0) {init = 0 : i32, sym_name = "loop_of_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "loop_of_lock_1"}
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_0]], 2) {init = 0 : i32, sym_name = "loop_of_lock_2"}
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_0]], 3) {init = 0 : i32, sym_name = "loop_of_lock_3"}
// CHECK:           func.func @some_work(%[[VAL_10:.*]]: memref<16xi32>, %[[VAL_11:.*]]: index) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_13:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_14:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_15:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_16:.*]] = arith.constant 4 : index
// CHECK:             %[[VAL_17:.*]] = arith.constant 21 : index
// CHECK:             %[[VAL_18:.*]] = arith.constant 17 : index
// CHECK:             %[[VAL_19:.*]] = arith.constant 8 : index
// CHECK:             scf.for %[[VAL_20:.*]] = %[[VAL_14]] to %[[VAL_18]] step %[[VAL_19]] {
// CHECK:               %[[VAL_21:.*]] = arith.constant 0 : i32
// CHECK:               aie.use_lock(%[[VAL_6]], Acquire, %[[VAL_21]])
// CHECK:               func.call @some_work(%[[VAL_2]], %[[VAL_20]]) : (memref<16xi32>, index) -> ()
// CHECK:               %[[VAL_22:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_6]], Release, %[[VAL_22]])
// CHECK:               %[[VAL_23:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_24:.*]] = arith.muli %[[VAL_15]], %[[VAL_23]] : index
// CHECK:               %[[VAL_25:.*]] = arith.addi %[[VAL_20]], %[[VAL_24]] : index
// CHECK:               %[[VAL_26:.*]] = arith.constant 0 : i32
// CHECK:               aie.use_lock(%[[VAL_7]], Acquire, %[[VAL_26]])
// CHECK:               func.call @some_work(%[[VAL_3]], %[[VAL_25]]) : (memref<16xi32>, index) -> ()
// CHECK:               %[[VAL_27:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_7]], Release, %[[VAL_27]])
// CHECK:               %[[VAL_28:.*]] = arith.constant 2 : index
// CHECK:               %[[VAL_29:.*]] = arith.muli %[[VAL_15]], %[[VAL_28]] : index
// CHECK:               %[[VAL_30:.*]] = arith.addi %[[VAL_20]], %[[VAL_29]] : index
// CHECK:               %[[VAL_31:.*]] = arith.constant 0 : i32
// CHECK:               aie.use_lock(%[[VAL_8]], Acquire, %[[VAL_31]])
// CHECK:               func.call @some_work(%[[VAL_4]], %[[VAL_30]]) : (memref<16xi32>, index) -> ()
// CHECK:               %[[VAL_32:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_8]], Release, %[[VAL_32]])
// CHECK:               %[[VAL_33:.*]] = arith.constant 3 : index
// CHECK:               %[[VAL_34:.*]] = arith.muli %[[VAL_15]], %[[VAL_33]] : index
// CHECK:               %[[VAL_35:.*]] = arith.addi %[[VAL_20]], %[[VAL_34]] : index
// CHECK:               %[[VAL_36:.*]] = arith.constant 0 : i32
// CHECK:               aie.use_lock(%[[VAL_9]], Acquire, %[[VAL_36]])
// CHECK:               func.call @some_work(%[[VAL_5]], %[[VAL_35]]) : (memref<16xi32>, index) -> ()
// CHECK:               %[[VAL_37:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_9]], Release, %[[VAL_37]])
// CHECK:             }
// CHECK:             scf.for %[[VAL_38:.*]] = %[[VAL_18]] to %[[VAL_17]] step %[[VAL_15]] {
// CHECK:               %[[VAL_39:.*]] = arith.constant 0 : i32
// CHECK:               aie.use_lock(%[[VAL_6]], Acquire, %[[VAL_39]])
// CHECK:               func.call @some_work(%[[VAL_2]], %[[VAL_38]]) : (memref<16xi32>, index) -> ()
// CHECK:               %[[VAL_40:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_6]], Release, %[[VAL_40]])
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
      %c21 = arith.constant 21 : index
      scf.for %indexInHeight = %c1 to %c21 step %c2 {
        %subview = aie.objectfifo.acquire @loop_of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @some_work(%elem0,%indexInHeight) : (memref<16xi32>,index) -> ()
        aie.objectfifo.release @loop_of (Produce, 1)
      }
      aie.end
    }
  }
}
