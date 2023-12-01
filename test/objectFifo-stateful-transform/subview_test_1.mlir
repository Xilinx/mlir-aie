//===- subview_test_1.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
// Date: October 26th 2021
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = AIE.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(1, 3)
// CHECK:           %[[VAL_2:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "objfifo_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "objfifo_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "objfifo_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "objfifo_buff_3"} : memref<16xi32>
// CHECK:           %[[VAL_6:.*]] = AIE.lock(%[[VAL_0]], 0) {init = 0 : i32, sym_name = "objfifo_lock_0"}
// CHECK:           %[[VAL_7:.*]] = AIE.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "objfifo_lock_1"}
// CHECK:           %[[VAL_8:.*]] = AIE.lock(%[[VAL_0]], 2) {init = 0 : i32, sym_name = "objfifo_lock_2"}
// CHECK:           %[[VAL_9:.*]] = AIE.lock(%[[VAL_0]], 3) {init = 0 : i32, sym_name = "objfifo_lock_3"}
// CHECK:           func.func @some_work(%[[VAL_10:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_11:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:             AIE.useLock(%[[VAL_6]], Acquire, 0)
// CHECK:             AIE.useLock(%[[VAL_7]], Acquire, 0)
// CHECK:             func.call @some_work(%[[VAL_2]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_3]]) : (memref<16xi32>) -> ()
// CHECK:             AIE.useLock(%[[VAL_8]], Acquire, 0)
// CHECK:             func.call @some_work(%[[VAL_2]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_3]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_4]]) : (memref<16xi32>) -> ()
// CHECK:             AIE.useLock(%[[VAL_6]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_7]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_9]], Acquire, 0)
// CHECK:             func.call @some_work(%[[VAL_4]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_5]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_4]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_5]]) : (memref<16xi32>) -> ()
// CHECK:             AIE.end
// CHECK:           }
// CHECK:         }

module @singleFifo {
    AIE.device(xcvc1902) {
        %tile12 = AIE.tile(1, 2)
        %tile13 = AIE.tile(1, 3)

        AIE.objectfifo @objfifo (%tile12, {%tile13}, 4 : i32) : !AIE.objectfifo<memref<16xi32>>

        func.func @some_work(%line_in:memref<16xi32>) -> () {
            return
        }

        %core12 = AIE.core(%tile12) {
            // this acquires 2 elements
            %subview0 = AIE.objectfifo.acquire @objfifo (Produce, 2) : !AIE.objectfifosubview<memref<16xi32>>
            %elem00 = AIE.objectfifo.subview.access %subview0[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem01 = AIE.objectfifo.subview.access %subview0[1] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem00) : (memref<16xi32>) -> ()
            func.call @some_work(%elem01) : (memref<16xi32>) -> ()

            // this should only acquire one new element, previous two are still acquired
            %subview1 = AIE.objectfifo.acquire @objfifo (Produce, 3) : !AIE.objectfifosubview<memref<16xi32>>
            %elem10 = AIE.objectfifo.subview.access %subview1[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem11 = AIE.objectfifo.subview.access %subview1[1] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem12 = AIE.objectfifo.subview.access %subview1[2] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem10) : (memref<16xi32>) -> ()
            func.call @some_work(%elem11) : (memref<16xi32>) -> ()
            func.call @some_work(%elem12) : (memref<16xi32>) -> ()

            // one new acquire should take place
            AIE.objectfifo.release @objfifo (Produce, 1)
            AIE.objectfifo.release @objfifo (Produce, 1)
            %subview2 = AIE.objectfifo.acquire @objfifo (Produce, 2) : !AIE.objectfifosubview<memref<16xi32>>
            %elem20 = AIE.objectfifo.subview.access %subview2[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem21 = AIE.objectfifo.subview.access %subview2[1] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem20) : (memref<16xi32>) -> ()
            func.call @some_work(%elem21) : (memref<16xi32>) -> ()

            // no new acquires should take place, elem30 should be third element of objFifo (with index 2)
            %subview3 = AIE.objectfifo.acquire @objfifo (Produce, 2) : !AIE.objectfifosubview<memref<16xi32>>
            %elem30 = AIE.objectfifo.subview.access %subview3[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem31 = AIE.objectfifo.subview.access %subview3[1] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            //%elem32 = AIE.subview.access %subview3[2] : !AIE.subview<memref<16xi32>> -> memref<16xi32> // expected to fail if this line is uncommented
            func.call @some_work(%elem30) : (memref<16xi32>) -> ()
            func.call @some_work(%elem31) : (memref<16xi32>) -> ()

            AIE.end
        }
    }
}
