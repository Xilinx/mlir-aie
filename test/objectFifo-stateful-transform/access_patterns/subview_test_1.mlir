//===- subview_test_1.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: October 26th 2021
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK-DAG:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK-DAG:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "objfifo_buff_0"} : memref<16xi32>
// CHECK-DAG:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "objfifo_buff_1"} : memref<16xi32>
// CHECK-DAG:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "objfifo_buff_2"} : memref<16xi32>
// CHECK-DAG:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "objfifo_buff_3"} : memref<16xi32>
// CHECK-DAG:           %[[VAL_6:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "objfifo_lock_0"}
// CHECK-DAG:           %[[VAL_7:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "objfifo_lock_1"}
// CHECK-DAG:           %[[VAL_8:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "objfifo_lock_2"}
// CHECK-DAG:           %[[VAL_9:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "objfifo_lock_3"}
// CHECK:           func.func @some_work(%[[VAL_10:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_11:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : i32
// CHECK:             %[[C1:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], Acquire, %[[C0]])
// CHECK:             aie.use_lock(%[[VAL_7]], Acquire, %[[C0]])
// CHECK:             func.call @some_work(%[[VAL_2]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_3]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[VAL_8]], Acquire, %[[C0]])
// CHECK:             func.call @some_work(%[[VAL_2]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_3]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_4]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[C1]])
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[C1]])
// CHECK:             aie.use_lock(%[[VAL_9]], Acquire, %[[C0]])
// CHECK:             func.call @some_work(%[[VAL_4]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_5]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_4]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_5]]) : (memref<16xi32>) -> ()
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

// single objectfifo (two different end points), single core, interaction between acquire / release calls, AIE1

module @singleFifo {
    aie.device(xcvc1902) {
        %tile12 = aie.tile(1, 2)
        %tile13 = aie.tile(1, 3)

        aie.objectfifo @objfifo (%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<16xi32>>

        func.func @some_work(%line_in:memref<16xi32>) -> () {
            return
        }

        %core12 = aie.core(%tile12) {
            // this acquires 2 elements
            %elem00, %elem01 = aie.objectfifo.acquire @objfifo(Produce) : memref<16xi32>, memref<16xi32>
            func.call @some_work(%elem00) : (memref<16xi32>) -> ()
            func.call @some_work(%elem01) : (memref<16xi32>) -> ()

            // this should only acquire one new element, previous two are still acquired
            %elem10, %elem11, %elem12 = aie.objectfifo.acquire @objfifo(Produce) : memref<16xi32>, memref<16xi32>, memref<16xi32>
            func.call @some_work(%elem10) : (memref<16xi32>) -> ()
            func.call @some_work(%elem11) : (memref<16xi32>) -> ()
            func.call @some_work(%elem12) : (memref<16xi32>) -> ()

            // one new acquire should take place
            aie.objectfifo.release @objfifo(Produce) [1]
            aie.objectfifo.release @objfifo(Produce) [1]
            %elem20, %elem21 = aie.objectfifo.acquire @objfifo(Produce) : memref<16xi32>, memref<16xi32>
            func.call @some_work(%elem20) : (memref<16xi32>) -> ()
            func.call @some_work(%elem21) : (memref<16xi32>) -> ()

            // no new acquires should take place, elem30 should be third element of objFifo (with index 2)
            %elem30, %elem31 = aie.objectfifo.acquire @objfifo(Produce) : memref<16xi32>, memref<16xi32>
            //%elem32 = aie.subview.access %subview3[2] : !aie.subview<memref<16xi32>> -> memref<16xi32> // expected to fail if this line is uncommented
            func.call @some_work(%elem30) : (memref<16xi32>) -> ()
            func.call @some_work(%elem31) : (memref<16xi32>) -> ()

            aie.end
        }
    }
}
