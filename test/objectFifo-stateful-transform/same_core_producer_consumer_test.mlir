//===- same_core_producer_consumer_test.mlir --------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: August 2nd 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   AIE.device(xcve2302) {
// CHECK:           memref.global "public" @of : memref<16xi32>
// CHECK:           %[[VAL_0:.*]] = AIE.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "of_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_2:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "of_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "of_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = AIE.lock(%[[VAL_0]], 0) {init = 3 : i32, sym_name = "of_prod_lock"}
// CHECK:           %[[VAL_5:.*]] = AIE.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "of_cons_lock"}
// CHECK:           func.func @some_work(%[[VAL_6:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_7:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:             AIE.useLock(%[[VAL_4]], AcquireGreaterEqual, 2)
// CHECK:             func.call @some_work(%[[VAL_1]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_2]]) : (memref<16xi32>) -> ()
// CHECK:             AIE.useLock(%[[VAL_5]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:             func.call @some_work(%[[VAL_1]]) : (memref<16xi32>) -> ()
// CHECK:             AIE.useLock(%[[VAL_4]], Release, 1)
// CHECK:             func.call @some_work(%[[VAL_2]]) : (memref<16xi32>) -> ()
// CHECK:             AIE.useLock(%[[VAL_5]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:             func.call @some_work(%[[VAL_2]]) : (memref<16xi32>) -> ()
// CHECK:             AIE.useLock(%[[VAL_4]], Release, 1)
// CHECK:             AIE.end
// CHECK:           }
// CHECK:         }

module @same_core {
    AIE.device(xcve2302) {
        %tile12 = AIE.tile(1, 2)

        AIE.objectfifo @of (%tile12, {%tile12}, 3 : i32) : !AIE.objectfifo<memref<16xi32>>

        func.func @some_work(%line_in:memref<16xi32>) -> () {
            return
        }

        %core12 = AIE.core(%tile12) {
            // this acquires 2 elements
            %subview0 = AIE.objectfifo.acquire @of (Produce, 2) : !AIE.objectfifosubview<memref<16xi32>>
            %elem00 = AIE.objectfifo.subview.access %subview0[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem01 = AIE.objectfifo.subview.access %subview0[1] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem00) : (memref<16xi32>) -> ()
            func.call @some_work(%elem01) : (memref<16xi32>) -> ()
            AIE.objectfifo.release @of (Produce, 1)

            %subview1 = AIE.objectfifo.acquire @of (Consume, 1) : !AIE.objectfifosubview<memref<16xi32>>
            %elem10 = AIE.objectfifo.subview.access %subview1[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem10) : (memref<16xi32>) -> ()
            AIE.objectfifo.release @of (Consume, 1)

            %subview2 = AIE.objectfifo.acquire @of (Produce, 1) : !AIE.objectfifosubview<memref<16xi32>>
            %elem20 = AIE.objectfifo.subview.access %subview2[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem20) : (memref<16xi32>) -> ()
            AIE.objectfifo.release @of (Produce, 1)

            %subview3 = AIE.objectfifo.acquire @of (Consume, 1) : !AIE.objectfifosubview<memref<16xi32>>
            %elem30 = AIE.objectfifo.subview.access %subview3[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem30) : (memref<16xi32>) -> ()
            AIE.objectfifo.release @of (Consume, 1)

            AIE.end
        }
    }
}
