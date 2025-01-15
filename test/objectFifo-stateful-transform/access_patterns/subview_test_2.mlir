//===- subview_test_2.mlir --------------------------*- MLIR -*-===//
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

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           memref.global "public" @of2 : memref<16xi32>
// CHECK:           memref.global "public" @of : memref<16xi32>
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of2_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of2_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of2_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_0]], 4) {init = 0 : i32, sym_name = "of2_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_0]], 5) {init = 0 : i32, sym_name = "of2_lock_1"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_0]], 6) {init = 0 : i32, sym_name = "of2_lock_2"}
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_11:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of_buff_3"} : memref<16xi32>
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_0]], 0) {init = 0 : i32, sym_name = "of_lock_0"}
// CHECK:           %[[VAL_13:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "of_lock_1"}
// CHECK:           %[[VAL_14:.*]] = aie.lock(%[[VAL_0]], 2) {init = 0 : i32, sym_name = "of_lock_2"}
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_0]], 3) {init = 0 : i32, sym_name = "of_lock_3"}
// CHECK:           func.func @some_work(%[[VAL_16:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             aie.use_lock(%[[VAL_12]], Acquire, 0)
// CHECK:             aie.use_lock(%[[VAL_13]], Acquire, 0)
// CHECK:             func.call @some_work(%[[VAL_8]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_9]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[VAL_5]], Acquire, 0)
// CHECK:             func.call @some_work(%[[VAL_2]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[VAL_12]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_14]], Acquire, 0)
// CHECK:             aie.use_lock(%[[VAL_15]], Acquire, 0)
// CHECK:             func.call @some_work(%[[VAL_9]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_10]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_11]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[VAL_13]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_14]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_15]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_6]], Acquire, 0)
// CHECK:             aie.use_lock(%[[VAL_7]], Acquire, 0)
// CHECK:             func.call @some_work(%[[VAL_3]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_4]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:             aie.use_lock(%[[VAL_12]], Acquire, 1)
// CHECK:             func.call @some_work(%[[VAL_8]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[VAL_5]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_6]], Acquire, 1)
// CHECK:             func.call @some_work(%[[VAL_2]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_3]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[VAL_5]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_13]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_14]], Acquire, 1)
// CHECK:             func.call @some_work(%[[VAL_9]]) : (memref<16xi32>) -> ()
// CHECK:             func.call @some_work(%[[VAL_10]]) : (memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[VAL_13]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_14]], Release, 0)
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @multiFifo {
    aie.device(xcvc1902) {
        %tile12 = aie.tile(1, 2)
        %tile13 = aie.tile(1, 3)

        aie.objectfifo @of (%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @of2 (%tile12, {%tile13}, 3 : i32) : !aie.objectfifo<memref<16xi32>>

        func.func @some_work(%line_in:memref<16xi32>) -> () {
            return
        }

        %core12 = aie.core(%tile12) {
            %subview0 = aie.objectfifo.acquire @of (Produce, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elem00 = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem01 = aie.objectfifo.subview.access %subview0[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem00) : (memref<16xi32>) -> ()
            func.call @some_work(%elem01) : (memref<16xi32>) -> ()

            %subview02 = aie.objectfifo.acquire @of2 (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
            %elem002 = aie.objectfifo.subview.access %subview02[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem002) : (memref<16xi32>) -> ()

            aie.objectfifo.release @of (Produce, 1)
            %subview1 = aie.objectfifo.acquire @of (Produce, 3) : !aie.objectfifosubview<memref<16xi32>>
            %elem10 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem11 = aie.objectfifo.subview.access %subview1[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem12 = aie.objectfifo.subview.access %subview1[2] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem10) : (memref<16xi32>) -> ()
            func.call @some_work(%elem11) : (memref<16xi32>) -> ()
            func.call @some_work(%elem12) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of (Produce, 3)
            
            aie.objectfifo.release @of2 (Produce, 1)
            %subview12 = aie.objectfifo.acquire @of2 (Produce, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elem102 = aie.objectfifo.subview.access %subview12[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem112 = aie.objectfifo.subview.access %subview12[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem102) : (memref<16xi32>) -> ()
            func.call @some_work(%elem112) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of2 (Produce, 1)
            
            aie.end
        }

        %core13 = aie.core(%tile13) {
            %subview0 = aie.objectfifo.acquire @of (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
            %elem00 = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem00) : (memref<16xi32>) -> ()

            %subview02 = aie.objectfifo.acquire @of2 (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elem002 = aie.objectfifo.subview.access %subview02[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem012 = aie.objectfifo.subview.access %subview02[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem002) : (memref<16xi32>) -> ()
            func.call @some_work(%elem012) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of2 (Consume, 2)

            aie.objectfifo.release @of (Consume, 1)
            %subview1 = aie.objectfifo.acquire @of (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elem10 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %elem11 = aie.objectfifo.subview.access %subview1[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem10) : (memref<16xi32>) -> ()
            func.call @some_work(%elem11) : (memref<16xi32>) -> ()
            aie.objectfifo.release @of (Consume, 2)

            aie.end
        }
    }
}
