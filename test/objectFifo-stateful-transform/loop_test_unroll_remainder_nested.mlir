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

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           memref.global "public" @of_2 : memref<16xi32>
// CHECK:           memref.global "public" @of_1 : memref<16xi32>
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
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_1_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_1]], 0) {init = 0 : i32, sym_name = "of_1_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "of_1_lock_1"}
// CHECK:           %[[VAL_13:.*]] = aie.lock(%[[VAL_1]], 2) {init = 0 : i32, sym_name = "of_1_lock_2"}
// CHECK:           func.func @some_work(%[[VAL_14:.*]]: memref<16xi32>, %[[VAL_15:.*]]: memref<16xi32>, %[[VAL_16:.*]]: index, %[[VAL_17:.*]]: index) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_19:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_20:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_21:.*]] = arith.constant 8 : index
// CHECK:             %[[VAL_22:.*]] = arith.constant 9 : index
// CHECK:             %[[VAL_23:.*]] = arith.constant 3 : index
// CHECK:             scf.for %[[VAL_24:.*]] = %[[VAL_19]] to %[[VAL_22]] step %[[VAL_23]] {
// CHECK:               aie.use_lock(%[[VAL_5]], Acquire, 0)
// CHECK:               %[[VAL_25:.*]] = arith.constant 6 : index
// CHECK:               %[[VAL_26:.*]] = arith.constant 3 : index
// CHECK:               scf.for %[[VAL_27:.*]] = %[[VAL_19]] to %[[VAL_25]] step %[[VAL_26]] {
// CHECK:                 aie.use_lock(%[[VAL_11]], Acquire, 1)
// CHECK:                 func.call @some_work(%[[VAL_8]], %[[VAL_2]], %[[VAL_24]], %[[VAL_27]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                 aie.use_lock(%[[VAL_11]], Release, 0)
// CHECK:                 %[[VAL_28:.*]] = arith.constant 1 : index
// CHECK:                 %[[VAL_29:.*]] = arith.muli %[[VAL_20]], %[[VAL_28]] : index
// CHECK:                 %[[VAL_30:.*]] = arith.addi %[[VAL_27]], %[[VAL_29]] : index
// CHECK:                 aie.use_lock(%[[VAL_12]], Acquire, 1)
// CHECK:                 func.call @some_work(%[[VAL_9]], %[[VAL_2]], %[[VAL_24]], %[[VAL_30]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                 aie.use_lock(%[[VAL_12]], Release, 0)
// CHECK:                 %[[VAL_31:.*]] = arith.constant 2 : index
// CHECK:                 %[[VAL_32:.*]] = arith.muli %[[VAL_20]], %[[VAL_31]] : index
// CHECK:                 %[[VAL_33:.*]] = arith.addi %[[VAL_27]], %[[VAL_32]] : index
// CHECK:                 aie.use_lock(%[[VAL_13]], Acquire, 1)
// CHECK:                 func.call @some_work(%[[VAL_10]], %[[VAL_2]], %[[VAL_24]], %[[VAL_33]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                 aie.use_lock(%[[VAL_13]], Release, 0)
// CHECK:               }
// CHECK:               %[[VAL_34:.*]] = arith.constant 2 : index
// CHECK:               aie.use_lock(%[[VAL_11]], Acquire, 1)
// CHECK:               func.call @some_work(%[[VAL_8]], %[[VAL_2]], %[[VAL_24]], %[[VAL_25]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_11]], Release, 0)
// CHECK:               %[[VAL_35:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_36:.*]] = arith.muli %[[VAL_20]], %[[VAL_35]] : index
// CHECK:               %[[VAL_37:.*]] = arith.addi %[[VAL_25]], %[[VAL_36]] : index
// CHECK:               aie.use_lock(%[[VAL_12]], Acquire, 1)
// CHECK:               func.call @some_work(%[[VAL_9]], %[[VAL_2]], %[[VAL_24]], %[[VAL_37]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_12]], Release, 0)
// CHECK:               aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:               %[[VAL_38:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_39:.*]] = arith.muli %[[VAL_20]], %[[VAL_38]] : index
// CHECK:               %[[VAL_40:.*]] = arith.addi %[[VAL_24]], %[[VAL_39]] : index
// CHECK:               aie.use_lock(%[[VAL_6]], Acquire, 0)
// CHECK:               %[[VAL_41:.*]] = arith.constant 6 : index
// CHECK:               %[[VAL_42:.*]] = arith.constant 3 : index
// CHECK:               scf.for %[[VAL_43:.*]] = %[[VAL_19]] to %[[VAL_41]] step %[[VAL_42]] {
// CHECK:                 aie.use_lock(%[[VAL_13]], Acquire, 1)
// CHECK:                 func.call @some_work(%[[VAL_10]], %[[VAL_3]], %[[VAL_40]], %[[VAL_43]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                 aie.use_lock(%[[VAL_13]], Release, 0)
// CHECK:                 %[[VAL_44:.*]] = arith.constant 1 : index
// CHECK:                 %[[VAL_45:.*]] = arith.muli %[[VAL_20]], %[[VAL_44]] : index
// CHECK:                 %[[VAL_46:.*]] = arith.addi %[[VAL_43]], %[[VAL_45]] : index
// CHECK:                 aie.use_lock(%[[VAL_11]], Acquire, 1)
// CHECK:                 func.call @some_work(%[[VAL_8]], %[[VAL_3]], %[[VAL_40]], %[[VAL_46]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                 aie.use_lock(%[[VAL_11]], Release, 0)
// CHECK:                 %[[VAL_47:.*]] = arith.constant 2 : index
// CHECK:                 %[[VAL_48:.*]] = arith.muli %[[VAL_20]], %[[VAL_47]] : index
// CHECK:                 %[[VAL_49:.*]] = arith.addi %[[VAL_43]], %[[VAL_48]] : index
// CHECK:                 aie.use_lock(%[[VAL_12]], Acquire, 1)
// CHECK:                 func.call @some_work(%[[VAL_9]], %[[VAL_3]], %[[VAL_40]], %[[VAL_49]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                 aie.use_lock(%[[VAL_12]], Release, 0)
// CHECK:               }
// CHECK:               %[[VAL_50:.*]] = arith.constant 2 : index
// CHECK:               aie.use_lock(%[[VAL_13]], Acquire, 1)
// CHECK:               func.call @some_work(%[[VAL_10]], %[[VAL_3]], %[[VAL_40]], %[[VAL_41]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_13]], Release, 0)
// CHECK:               %[[VAL_51:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_52:.*]] = arith.muli %[[VAL_20]], %[[VAL_51]] : index
// CHECK:               %[[VAL_53:.*]] = arith.addi %[[VAL_41]], %[[VAL_52]] : index
// CHECK:               aie.use_lock(%[[VAL_11]], Acquire, 1)
// CHECK:               func.call @some_work(%[[VAL_8]], %[[VAL_3]], %[[VAL_40]], %[[VAL_53]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_11]], Release, 0)
// CHECK:               aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:               %[[VAL_54:.*]] = arith.constant 2 : index
// CHECK:               %[[VAL_55:.*]] = arith.muli %[[VAL_20]], %[[VAL_54]] : index
// CHECK:               %[[VAL_56:.*]] = arith.addi %[[VAL_24]], %[[VAL_55]] : index
// CHECK:               aie.use_lock(%[[VAL_7]], Acquire, 0)
// CHECK:               %[[VAL_57:.*]] = arith.constant 6 : index
// CHECK:               %[[VAL_58:.*]] = arith.constant 3 : index
// CHECK:               scf.for %[[VAL_59:.*]] = %[[VAL_19]] to %[[VAL_57]] step %[[VAL_58]] {
// CHECK:                 aie.use_lock(%[[VAL_12]], Acquire, 1)
// CHECK:                 func.call @some_work(%[[VAL_9]], %[[VAL_4]], %[[VAL_56]], %[[VAL_59]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                 aie.use_lock(%[[VAL_12]], Release, 0)
// CHECK:                 %[[VAL_60:.*]] = arith.constant 1 : index
// CHECK:                 %[[VAL_61:.*]] = arith.muli %[[VAL_20]], %[[VAL_60]] : index
// CHECK:                 %[[VAL_62:.*]] = arith.addi %[[VAL_59]], %[[VAL_61]] : index
// CHECK:                 aie.use_lock(%[[VAL_13]], Acquire, 1)
// CHECK:                 func.call @some_work(%[[VAL_10]], %[[VAL_4]], %[[VAL_56]], %[[VAL_62]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                 aie.use_lock(%[[VAL_13]], Release, 0)
// CHECK:                 %[[VAL_63:.*]] = arith.constant 2 : index
// CHECK:                 %[[VAL_64:.*]] = arith.muli %[[VAL_20]], %[[VAL_63]] : index
// CHECK:                 %[[VAL_65:.*]] = arith.addi %[[VAL_59]], %[[VAL_64]] : index
// CHECK:                 aie.use_lock(%[[VAL_11]], Acquire, 1)
// CHECK:                 func.call @some_work(%[[VAL_8]], %[[VAL_4]], %[[VAL_56]], %[[VAL_65]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:                 aie.use_lock(%[[VAL_11]], Release, 0)
// CHECK:               }
// CHECK:               %[[VAL_66:.*]] = arith.constant 2 : index
// CHECK:               aie.use_lock(%[[VAL_12]], Acquire, 1)
// CHECK:               func.call @some_work(%[[VAL_9]], %[[VAL_4]], %[[VAL_56]], %[[VAL_57]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_12]], Release, 0)
// CHECK:               %[[VAL_67:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_68:.*]] = arith.muli %[[VAL_20]], %[[VAL_67]] : index
// CHECK:               %[[VAL_69:.*]] = arith.addi %[[VAL_57]], %[[VAL_68]] : index
// CHECK:               aie.use_lock(%[[VAL_13]], Acquire, 1)
// CHECK:               func.call @some_work(%[[VAL_10]], %[[VAL_4]], %[[VAL_56]], %[[VAL_69]]) : (memref<16xi32>, memref<16xi32>, index, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_13]], Release, 0)
// CHECK:               aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

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
