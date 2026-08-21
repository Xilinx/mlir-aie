//===- unroll_factor_multiple_objectfifos.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL: module {
// CHECK:         aie.device(xcvc1902) {
// CHECK-DAG:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK-DAG:           %[[VAL_1:.*]] = aie.tile(1, 3)
// CHECK-DAG:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of_2_buff_0"} : memref<16xi32>
// CHECK-DAG:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of_2_buff_1"} : memref<16xi32>
// CHECK-DAG:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of_2_buff_2"} : memref<16xi32>
// CHECK-DAG:           %[[VAL_5:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of_2_lock_0"}
// CHECK-DAG:           %[[VAL_6:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of_2_lock_1"}
// CHECK-DAG:           %[[VAL_7:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of_2_lock_2"}
// CHECK-DAG:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_1_buff_0"} : memref<16xi32>
// CHECK-DAG:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_1_buff_1"} : memref<16xi32>
// CHECK-DAG:           %[[VAL_10:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "of_1_lock_0"}
// CHECK-DAG:           %[[VAL_11:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "of_1_lock_1"}
// CHECK:           func.func @some_work(%[[VAL_12:.*]]: memref<16xi32>, %[[VAL_13:.*]]: memref<16xi32>, %[[VAL_14:.*]]: index) {
// CHECK:             return
// CHECK:           }
// unroll factor = lcm(of_1 depth 2, of_2 depth 3) = 6: one rolled loop over 6
// unrolled sub-iterations walking of_1 buff0/1 and of_2 buff0/1/2 in lockstep.
// Binary-lock polarity: of_1 acquire=1/release=0, of_2 acquire=0/release=1.
// CHECK:           %[[VAL_15:.*]] = aie.core(%[[VAL_0]]) {
// CHECK-DAG:             %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:             %[[C12:.*]] = arith.constant 12 : index
// CHECK-DAG:             %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:             %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:             %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:             %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:             %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:             %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:             %[[L0:.*]] = arith.constant 0 : i32
// CHECK-DAG:             %[[L1:.*]] = arith.constant 1 : i32
// CHECK:             scf.for %[[IV:.*]] = %[[C0]] to %[[C12]] step %[[C6]] {
// CHECK:               aie.use_lock(%[[VAL_10]], Acquire, %[[L1]])
// CHECK:               aie.use_lock(%[[VAL_5]], Acquire, %[[L0]])
// CHECK:               func.call @some_work(%[[VAL_8]], %[[VAL_2]], %[[IV]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_10]], Release, %[[L0]])
// CHECK:               aie.use_lock(%[[VAL_5]], Release, %[[L1]])
// CHECK:               %[[I1:.*]] = arith.addi %[[IV]], %[[C1]] : index
// CHECK:               aie.use_lock(%[[VAL_11]], Acquire, %[[L1]])
// CHECK:               aie.use_lock(%[[VAL_6]], Acquire, %[[L0]])
// CHECK:               func.call @some_work(%[[VAL_9]], %[[VAL_3]], %[[I1]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_11]], Release, %[[L0]])
// CHECK:               aie.use_lock(%[[VAL_6]], Release, %[[L1]])
// CHECK:               %[[I2:.*]] = arith.addi %[[IV]], %[[C2]] : index
// CHECK:               aie.use_lock(%[[VAL_10]], Acquire, %[[L1]])
// CHECK:               aie.use_lock(%[[VAL_7]], Acquire, %[[L0]])
// CHECK:               func.call @some_work(%[[VAL_8]], %[[VAL_4]], %[[I2]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_10]], Release, %[[L0]])
// CHECK:               aie.use_lock(%[[VAL_7]], Release, %[[L1]])
// CHECK:               %[[I3:.*]] = arith.addi %[[IV]], %[[C3]] : index
// CHECK:               aie.use_lock(%[[VAL_11]], Acquire, %[[L1]])
// CHECK:               aie.use_lock(%[[VAL_5]], Acquire, %[[L0]])
// CHECK:               func.call @some_work(%[[VAL_9]], %[[VAL_2]], %[[I3]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_11]], Release, %[[L0]])
// CHECK:               aie.use_lock(%[[VAL_5]], Release, %[[L1]])
// CHECK:               %[[I4:.*]] = arith.addi %[[IV]], %[[C4]] : index
// CHECK:               aie.use_lock(%[[VAL_10]], Acquire, %[[L1]])
// CHECK:               aie.use_lock(%[[VAL_6]], Acquire, %[[L0]])
// CHECK:               func.call @some_work(%[[VAL_8]], %[[VAL_3]], %[[I4]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_10]], Release, %[[L0]])
// CHECK:               aie.use_lock(%[[VAL_6]], Release, %[[L1]])
// CHECK:               %[[I5:.*]] = arith.addi %[[IV]], %[[C5]] : index
// CHECK:               aie.use_lock(%[[VAL_11]], Acquire, %[[L1]])
// CHECK:               aie.use_lock(%[[VAL_7]], Acquire, %[[L0]])
// CHECK:               func.call @some_work(%[[VAL_9]], %[[VAL_4]], %[[I5]]) : (memref<16xi32>, memref<16xi32>, index) -> ()
// CHECK:               aie.use_lock(%[[VAL_11]], Release, %[[L0]])
// CHECK:               aie.use_lock(%[[VAL_7]], Release, %[[L1]])
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
        %elemIn = aie.objectfifo.acquire @of_1(Consume) : memref<16xi32>
        %elemOut = aie.objectfifo.acquire @of_2(Produce) : memref<16xi32>
        func.call @some_work(%elemIn, %elemOut, %indexInHeight) : (memref<16xi32>, memref<16xi32>, index) -> ()
        aie.objectfifo.release @of_1(Consume) [1]
        aie.objectfifo.release @of_2(Produce) [1]
      }

      aie.end
    }
  }
}
