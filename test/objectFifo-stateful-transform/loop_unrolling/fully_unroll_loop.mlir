//===- fully_unroll_loop.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=false" %s | FileCheck %s

// CHECK-LABEL: module {
// CHECK:         aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of_2_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of_2_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of_2_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_0]], 0) {init = 0 : i32, sym_name = "of_2_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "of_2_lock_1"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_0]], 2) {init = 0 : i32, sym_name = "of_2_lock_2"}
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_1_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_1_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_1]], 0) {init = 0 : i32, sym_name = "of_1_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "of_1_lock_1"}
// CHECK:           func.func @some_work(%[[VAL_12:.*]]: memref<16xi32>, %[[VAL_13:.*]]: memref<16xi32>, %[[VAL_14:.*]]: index) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_16:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_17:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_18:.*]] = arith.constant 4 : index
// CHECK:             %[[VAL_19:.*]] = arith.constant 4 : index
// CHECK:             %[[VAL_20:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], Acquire, %[[VAL_20]])
// CHECK:             %[[VAL_21:.*]] = arith.constant 0 : i32
// CHECK:             aie.use_lock(%[[VAL_5]], Acquire, %[[VAL_21]])
// CHECK:             func.call @some_work(%[[VAL_8]], %[[VAL_2]], %[[VAL_16]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:             %[[VAL_22:.*]] = arith.constant 0 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_22]])
// CHECK:             %[[VAL_23:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_23]])
// CHECK:             %[[VAL_24:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_25:.*]] = arith.muli %[[VAL_17]], %[[VAL_24]] : index
// CHECK:             %[[VAL_26:.*]] = arith.addi %[[VAL_16]], %[[VAL_25]] : index
// CHECK:             %[[VAL_27:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Acquire, %[[VAL_27]])
// CHECK:             %[[VAL_28:.*]] = arith.constant 0 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], Acquire, %[[VAL_28]])
// CHECK:             func.call @some_work(%[[VAL_9]], %[[VAL_3]], %[[VAL_26]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:             %[[VAL_29:.*]] = arith.constant 0 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_29]])
// CHECK:             %[[VAL_30:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_30]])
// CHECK:             %[[VAL_31:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_32:.*]] = arith.muli %[[VAL_17]], %[[VAL_31]] : index
// CHECK:             %[[VAL_33:.*]] = arith.addi %[[VAL_16]], %[[VAL_32]] : index
// CHECK:             %[[VAL_34:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], Acquire, %[[VAL_34]])
// CHECK:             %[[VAL_35:.*]] = arith.constant 0 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Acquire, %[[VAL_35]])
// CHECK:             func.call @some_work(%[[VAL_8]], %[[VAL_4]], %[[VAL_33]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:             %[[VAL_36:.*]] = arith.constant 0 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_36]])
// CHECK:             %[[VAL_37:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_37]])
// CHECK:             %[[VAL_38:.*]] = arith.constant 3 : index
// CHECK:             %[[VAL_39:.*]] = arith.muli %[[VAL_17]], %[[VAL_38]] : index
// CHECK:             %[[VAL_40:.*]] = arith.addi %[[VAL_16]], %[[VAL_39]] : index
// CHECK:             %[[VAL_41:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Acquire, %[[VAL_41]])
// CHECK:             %[[VAL_42:.*]] = arith.constant 0 : i32
// CHECK:             aie.use_lock(%[[VAL_5]], Acquire, %[[VAL_42]])
// CHECK:             func.call @some_work(%[[VAL_9]], %[[VAL_2]], %[[VAL_40]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:             %[[VAL_43:.*]] = arith.constant 0 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_43]])
// CHECK:             %[[VAL_44:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_44]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }
// CHECK:       }

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
      %c4 = arith.constant 4 : index

      scf.for %indexInHeight = %c0 to %c4 step %c1 {
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
