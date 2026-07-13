//===- unroll_factor_multiple_objectfifos.mlir ------------------*- MLIR -*-===//
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
// CHECK:             %[[VAL_18:.*]] = arith.constant 12 : index
// CHECK:             %[[VAL_19:.*]] = arith.constant 6 : index
// CHECK:             scf.for %[[VAL_20:.*]] = %[[VAL_16]] to %[[VAL_18]] step %[[VAL_19]] {
// CHECK:               %[[VAL_21:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_10]], Acquire, %[[VAL_21]])
// CHECK:               %[[VAL_22:.*]] = arith.constant 0 : i32
// CHECK:               aie.use_lock(%[[VAL_5]], Acquire, %[[VAL_22]])
// CHECK:               func.call @some_work(%[[VAL_8]], %[[VAL_2]], %[[VAL_20]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:               %[[VAL_23:.*]] = arith.constant 0 : i32
// CHECK:               aie.use_lock(%[[VAL_10]], Release, %[[VAL_23]])
// CHECK:               %[[VAL_24:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_5]], Release, %[[VAL_24]])
// CHECK:               %[[VAL_25:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_26:.*]] = arith.muli %[[VAL_17]], %[[VAL_25]] : index
// CHECK:               %[[VAL_27:.*]] = arith.addi %[[VAL_20]], %[[VAL_26]] : index
// CHECK:               %[[VAL_28:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_11]], Acquire, %[[VAL_28]])
// CHECK:               %[[VAL_29:.*]] = arith.constant 0 : i32
// CHECK:               aie.use_lock(%[[VAL_6]], Acquire, %[[VAL_29]])
// CHECK:               func.call @some_work(%[[VAL_9]], %[[VAL_3]], %[[VAL_27]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:               %[[VAL_30:.*]] = arith.constant 0 : i32
// CHECK:               aie.use_lock(%[[VAL_11]], Release, %[[VAL_30]])
// CHECK:               %[[VAL_31:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_6]], Release, %[[VAL_31]])
// CHECK:               %[[VAL_32:.*]] = arith.constant 2 : index
// CHECK:               %[[VAL_33:.*]] = arith.muli %[[VAL_17]], %[[VAL_32]] : index
// CHECK:               %[[VAL_34:.*]] = arith.addi %[[VAL_20]], %[[VAL_33]] : index
// CHECK:               %[[VAL_35:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_10]], Acquire, %[[VAL_35]])
// CHECK:               %[[VAL_36:.*]] = arith.constant 0 : i32
// CHECK:               aie.use_lock(%[[VAL_7]], Acquire, %[[VAL_36]])
// CHECK:               func.call @some_work(%[[VAL_8]], %[[VAL_4]], %[[VAL_34]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:               %[[VAL_37:.*]] = arith.constant 0 : i32
// CHECK:               aie.use_lock(%[[VAL_10]], Release, %[[VAL_37]])
// CHECK:               %[[VAL_38:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_7]], Release, %[[VAL_38]])
// CHECK:               %[[VAL_39:.*]] = arith.constant 3 : index
// CHECK:               %[[VAL_40:.*]] = arith.muli %[[VAL_17]], %[[VAL_39]] : index
// CHECK:               %[[VAL_41:.*]] = arith.addi %[[VAL_20]], %[[VAL_40]] : index
// CHECK:               %[[VAL_42:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_11]], Acquire, %[[VAL_42]])
// CHECK:               %[[VAL_43:.*]] = arith.constant 0 : i32
// CHECK:               aie.use_lock(%[[VAL_5]], Acquire, %[[VAL_43]])
// CHECK:               func.call @some_work(%[[VAL_9]], %[[VAL_2]], %[[VAL_41]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:               %[[VAL_44:.*]] = arith.constant 0 : i32
// CHECK:               aie.use_lock(%[[VAL_11]], Release, %[[VAL_44]])
// CHECK:               %[[VAL_45:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_5]], Release, %[[VAL_45]])
// CHECK:               %[[VAL_46:.*]] = arith.constant 4 : index
// CHECK:               %[[VAL_47:.*]] = arith.muli %[[VAL_17]], %[[VAL_46]] : index
// CHECK:               %[[VAL_48:.*]] = arith.addi %[[VAL_20]], %[[VAL_47]] : index
// CHECK:               %[[VAL_49:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_10]], Acquire, %[[VAL_49]])
// CHECK:               %[[VAL_50:.*]] = arith.constant 0 : i32
// CHECK:               aie.use_lock(%[[VAL_6]], Acquire, %[[VAL_50]])
// CHECK:               func.call @some_work(%[[VAL_8]], %[[VAL_3]], %[[VAL_48]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:               %[[VAL_51:.*]] = arith.constant 0 : i32
// CHECK:               aie.use_lock(%[[VAL_10]], Release, %[[VAL_51]])
// CHECK:               %[[VAL_52:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_6]], Release, %[[VAL_52]])
// CHECK:               %[[VAL_53:.*]] = arith.constant 5 : index
// CHECK:               %[[VAL_54:.*]] = arith.muli %[[VAL_17]], %[[VAL_53]] : index
// CHECK:               %[[VAL_55:.*]] = arith.addi %[[VAL_20]], %[[VAL_54]] : index
// CHECK:               %[[VAL_56:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_11]], Acquire, %[[VAL_56]])
// CHECK:               %[[VAL_57:.*]] = arith.constant 0 : i32
// CHECK:               aie.use_lock(%[[VAL_7]], Acquire, %[[VAL_57]])
// CHECK:               func.call @some_work(%[[VAL_9]], %[[VAL_4]], %[[VAL_55]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:               %[[VAL_58:.*]] = arith.constant 0 : i32
// CHECK:               aie.use_lock(%[[VAL_11]], Release, %[[VAL_58]])
// CHECK:               %[[VAL_59:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_7]], Release, %[[VAL_59]])
// CHECK:             }
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
