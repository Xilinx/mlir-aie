//===- subview_test_3.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
// Date: November 19th 2021
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = AIE.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(1, 3)
// CHECK:           %[[VAL_2:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "of2_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "of2_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "of2_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = AIE.lock(%[[VAL_1]], 0) {init = 0 : i32, sym_name = "of2_lock_0"}
// CHECK:           %[[VAL_6:.*]] = AIE.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "of2_lock_1"}
// CHECK:           %[[VAL_7:.*]] = AIE.lock(%[[VAL_1]], 2) {init = 0 : i32, sym_name = "of2_lock_2"}
// CHECK:           %[[VAL_8:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "of_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_9:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "of_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_10:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "of_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_11:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "of_buff_3"} : memref<16xi32>
// CHECK:           %[[VAL_12:.*]] = AIE.lock(%[[VAL_0]], 0) {init = 0 : i32, sym_name = "of_lock_0"}
// CHECK:           %[[VAL_13:.*]] = AIE.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "of_lock_1"}
// CHECK:           %[[VAL_14:.*]] = AIE.lock(%[[VAL_0]], 2) {init = 0 : i32, sym_name = "of_lock_2"}
// CHECK:           %[[VAL_15:.*]] = AIE.lock(%[[VAL_0]], 3) {init = 0 : i32, sym_name = "of_lock_3"}
// CHECK:           func.func @some_work(%[[VAL_16:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:             AIE.useLock(%[[VAL_12]], Acquire, 0)
// CHECK:             AIE.useLock(%[[VAL_13]], Acquire, 0)
// CHECK:             func.call @some_work(%[[VAL_8]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_9]]) : (memref<16xi32>) -> ()
// CHECK:             AIE.useLock(%[[VAL_5]], Acquire, 1)
// CHECK:             func.call @some_work(%[[VAL_2]]) : (memref<16xi32>) -> ()
// CHECK:             AIE.useLock(%[[VAL_12]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_14]], Acquire, 0)
// CHECK:             AIE.useLock(%[[VAL_15]], Acquire, 0)
// CHECK:             func.call @some_work(%[[VAL_9]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_10]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_11]]) : (memref<16xi32>) -> ()
// CHECK:             AIE.useLock(%[[VAL_13]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_14]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_15]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_5]], Release, 0)
// CHECK:             AIE.useLock(%[[VAL_6]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_7]], Acquire, 1)
// CHECK:             func.call @some_work(%[[VAL_3]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_4]]) : (memref<16xi32>) -> ()
// CHECK:             AIE.useLock(%[[VAL_6]], Release, 0)
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = AIE.core(%[[VAL_1]]) {
// CHECK:             AIE.useLock(%[[VAL_12]], Acquire, 1)
// CHECK:             func.call @some_work(%[[VAL_8]]) : (memref<16xi32>) -> ()
// CHECK:             AIE.useLock(%[[VAL_5]], Acquire, 0)
// CHECK:             AIE.useLock(%[[VAL_6]], Acquire, 0)
// CHECK:             func.call @some_work(%[[VAL_2]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_3]]) : (memref<16xi32>) -> ()
// CHECK:             AIE.useLock(%[[VAL_5]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_6]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_12]], Release, 0)
// CHECK:             AIE.useLock(%[[VAL_13]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_14]], Acquire, 1)
// CHECK:             func.call @some_work(%[[VAL_9]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_10]]) : (memref<16xi32>) -> ()
// CHECK:             AIE.useLock(%[[VAL_13]], Release, 0)
// CHECK:             AIE.useLock(%[[VAL_14]], Release, 0)
// CHECK:             AIE.end
// CHECK:           }
// CHECK:         }

module @multiCoreMixedFifo {
    AIE.device(xcvc1902) {
        %tile12 = AIE.tile(1, 2)
        %tile13 = AIE.tile(1, 3)

        AIE.objectfifo @of (%tile12, {%tile13}, 4 : i32) : !AIE.objectfifo<memref<16xi32>>
        AIE.objectfifo @of2 (%tile13, {%tile12}, 3 : i32) : !AIE.objectfifo<memref<16xi32>>

        func.func @some_work(%line_in:memref<16xi32>) -> () {
            return
        }

        %core11 = AIE.core(%tile12) {
            %subview0 = AIE.objectfifo.acquire @of (Produce, 2) : !AIE.objectfifosubview<memref<16xi32>>
            %elem00 = AIE.objectfifo.subview.access %subview0[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem01 = AIE.objectfifo.subview.access %subview0[1] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem00) : (memref<16xi32>) -> ()
            func.call @some_work(%elem01) : (memref<16xi32>) -> ()

            %subview02 = AIE.objectfifo.acquire @of2 (Consume, 1) : !AIE.objectfifosubview<memref<16xi32>>
            %elem002 = AIE.objectfifo.subview.access %subview02[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem002) : (memref<16xi32>) -> ()

            AIE.objectfifo.release @of (Produce, 1)
            %subview1 = AIE.objectfifo.acquire @of (Produce, 3) : !AIE.objectfifosubview<memref<16xi32>>
            %elem10 = AIE.objectfifo.subview.access %subview1[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem11 = AIE.objectfifo.subview.access %subview1[1] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem12 = AIE.objectfifo.subview.access %subview1[2] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem10) : (memref<16xi32>) -> ()
            func.call @some_work(%elem11) : (memref<16xi32>) -> ()
            func.call @some_work(%elem12) : (memref<16xi32>) -> ()
            AIE.objectfifo.release @of (Produce, 3)
            
            AIE.objectfifo.release @of2 (Consume, 1)
            %subview12 = AIE.objectfifo.acquire @of2 (Consume, 2) : !AIE.objectfifosubview<memref<16xi32>>
            %elem102 = AIE.objectfifo.subview.access %subview12[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem112 = AIE.objectfifo.subview.access %subview12[1] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem102) : (memref<16xi32>) -> ()
            func.call @some_work(%elem112) : (memref<16xi32>) -> ()
            AIE.objectfifo.release @of2 (Consume, 1)
            
            AIE.end
        }

        %core12 = AIE.core(%tile13) {
            %subview0 = AIE.objectfifo.acquire @of (Consume, 1) : !AIE.objectfifosubview<memref<16xi32>>
            %elem00 = AIE.objectfifo.subview.access %subview0[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem00) : (memref<16xi32>) -> ()

            %subview02 = AIE.objectfifo.acquire @of2 (Produce, 2) : !AIE.objectfifosubview<memref<16xi32>>
            %elem002 = AIE.objectfifo.subview.access %subview02[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem012 = AIE.objectfifo.subview.access %subview02[1] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem002) : (memref<16xi32>) -> ()
            func.call @some_work(%elem012) : (memref<16xi32>) -> ()
            AIE.objectfifo.release @of2 (Produce, 2)

            AIE.objectfifo.release @of (Consume, 1)
            %subview1 = AIE.objectfifo.acquire @of (Consume, 2) : !AIE.objectfifosubview<memref<16xi32>>
            %elem10 = AIE.objectfifo.subview.access %subview1[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem11 = AIE.objectfifo.subview.access %subview1[1] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem10) : (memref<16xi32>) -> ()
            func.call @some_work(%elem11) : (memref<16xi32>) -> ()
            AIE.objectfifo.release @of (Consume, 2)

            AIE.end
        }
    }
}
