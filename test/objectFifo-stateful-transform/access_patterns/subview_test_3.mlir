//===- subview_test_3.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: November 19th 2021
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK-DAG:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK-DAG:           %[[VAL_1:.*]] = aie.tile(1, 3)
// CHECK-DAG:           aie.buffer @[[VAL_2:of2_buff_0]](%[[VAL_1]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_3:of2_buff_1]](%[[VAL_1]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_4:of2_buff_2]](%[[VAL_1]]) : memref<16xi32>
// CHECK-DAG:           aie.lock @[[VAL_5:of2_lock_0]](%[[VAL_1]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @[[VAL_6:of2_lock_1]](%[[VAL_1]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @[[VAL_7:of2_lock_2]](%[[VAL_1]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @[[VAL_8:of_buff_0]](%[[VAL_0]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_9:of_buff_1]](%[[VAL_0]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_10:of_buff_2]](%[[VAL_0]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_11:of_buff_3]](%[[VAL_0]]) : memref<16xi32>
// CHECK-DAG:           aie.lock @[[VAL_12:of_lock_0]](%[[VAL_0]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @[[VAL_13:of_lock_1]](%[[VAL_0]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @[[VAL_14:of_lock_2]](%[[VAL_0]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @[[VAL_15:of_lock_3]](%[[VAL_0]]) {init = 0 : i32}
// CHECK:           func.func @some_work(%[[VAL_16:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = aie.core(%[[VAL_0]]) {
// CHECK-DAG:         %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:         %[[C1:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(@[[VAL_12]], Acquire, %[[C0]])
// CHECK:             aie.use_lock(@[[VAL_13]], Acquire, %[[C0]])
// CHECK:             func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(@[[VAL_5]], Acquire, %[[C1]])
// CHECK:             func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(@[[VAL_12]], Release, %[[C1]])
// CHECK:             aie.use_lock(@[[VAL_14]], Acquire, %[[C0]])
// CHECK:             aie.use_lock(@[[VAL_15]], Acquire, %[[C0]])
// CHECK:             func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(@[[VAL_13]], Release, %[[C1]])
// CHECK:             aie.use_lock(@[[VAL_14]], Release, %[[C1]])
// CHECK:             aie.use_lock(@[[VAL_15]], Release, %[[C1]])
// CHECK:             aie.use_lock(@[[VAL_5]], Release, %[[C0]])
// CHECK:             aie.use_lock(@[[VAL_6]], Acquire, %[[C1]])
// CHECK:             aie.use_lock(@[[VAL_7]], Acquire, %[[C1]])
// CHECK:             func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(@[[VAL_6]], Release, %[[C0]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = aie.core(%[[VAL_1]]) {
// CHECK-DAG:         %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:         %[[C1:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(@[[VAL_12]], Acquire, %[[C1]])
// CHECK:             func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(@[[VAL_5]], Acquire, %[[C0]])
// CHECK:             aie.use_lock(@[[VAL_6]], Acquire, %[[C0]])
// CHECK:             func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(@[[VAL_5]], Release, %[[C1]])
// CHECK:             aie.use_lock(@[[VAL_6]], Release, %[[C1]])
// CHECK:             aie.use_lock(@[[VAL_12]], Release, %[[C0]])
// CHECK:             aie.use_lock(@[[VAL_13]], Acquire, %[[C1]])
// CHECK:             aie.use_lock(@[[VAL_14]], Acquire, %[[C1]])
// CHECK:             func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(@[[VAL_13]], Release, %[[C0]])
// CHECK:             aie.use_lock(@[[VAL_14]], Release, %[[C0]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

// two objectfifos, two cores, producers and consumers are different for each objectfifo,
// interaction between acquire / release calls to both objectfifos on both cores, AIE1

module @multiCoreMixedFifo {
    aie.device(xcvc1902) {
        %tile12 = aie.tile(1, 2)
        %tile13 = aie.tile(1, 3)

        aie.objectfifo @of (%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @of2 (%tile13, {%tile12}, 3 : i32) : !aie.objectfifo<memref<16xi32>>

        func.func @some_work(%line_in:memref<16xi32>) -> () {
            return
        }

        %core11 = aie.core(%tile12) {
            %elem00, %elem01 = aie.objectfifo.acquire @of (Produce, 2) : memref<16xi32>, memref<16xi32>
            func.call @some_work(%elem00) : (memref<16xi32>) -> ()
            func.call @some_work(%elem01) : (memref<16xi32>) -> ()

            %elem002 = aie.objectfifo.acquire @of2 (Consume, 1) : memref<16xi32>
            func.call @some_work(%elem002) : (memref<16xi32>) -> ()

            aie.objectfifo.release @of (Produce, 1)
            %elem10, %elem11, %elem12 = aie.objectfifo.acquire @of (Produce, 3) : memref<16xi32>, memref<16xi32>, memref<16xi32>
            func.call @some_work(%elem10) : (memref<16xi32>) -> ()
            func.call @some_work(%elem11) : (memref<16xi32>) -> ()
            func.call @some_work(%elem12) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of (Produce, 3)

            aie.objectfifo.release @of2 (Consume, 1)
            %elem102, %elem112 = aie.objectfifo.acquire @of2 (Consume, 2) : memref<16xi32>, memref<16xi32>
            func.call @some_work(%elem102) : (memref<16xi32>) -> ()
            func.call @some_work(%elem112) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of2 (Consume, 1)

            aie.end
        }

        %core12 = aie.core(%tile13) {
            %elem00 = aie.objectfifo.acquire @of (Consume, 1) : memref<16xi32>
            func.call @some_work(%elem00) : (memref<16xi32>) -> ()

            %elem002, %elem012 = aie.objectfifo.acquire @of2 (Produce, 2) : memref<16xi32>, memref<16xi32>
            func.call @some_work(%elem002) : (memref<16xi32>) -> ()
            func.call @some_work(%elem012) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of2 (Produce, 2)

            aie.objectfifo.release @of (Consume, 1)
            %elem10, %elem11 = aie.objectfifo.acquire @of (Consume, 2) : memref<16xi32>, memref<16xi32>
            func.call @some_work(%elem10) : (memref<16xi32>) -> ()
            func.call @some_work(%elem11) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of (Consume, 2)

            aie.end
        }
    }
}
