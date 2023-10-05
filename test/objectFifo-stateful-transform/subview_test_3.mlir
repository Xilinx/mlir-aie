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

//CHECK: module @multiCoreMixedFifo {
//CHECK:   AIE.device(xcvc1902) {
//CHECK:     %0 = AIE.tile(1, 2)
//CHECK:     %1 = AIE.tile(1, 3)
//CHECK:     %2 = AIE.buffer(%1) {sym_name = "of2_buff_0"} : memref<16xi32>
//CHECK:     %3 = AIE.buffer(%1) {sym_name = "of2_buff_1"} : memref<16xi32>
//CHECK:     %4 = AIE.buffer(%1) {sym_name = "of2_buff_2"} : memref<16xi32>
//CHECK:     %5 = AIE.lock(%1, 0) {init = 0 : i32, sym_name = "of2_lock_0"}
//CHECK:     %6 = AIE.lock(%1, 1) {init = 0 : i32, sym_name = "of2_lock_1"}
//CHECK:     %7 = AIE.lock(%1, 2) {init = 0 : i32, sym_name = "of2_lock_2"}
//CHECK:     %8 = AIE.buffer(%0) {sym_name = "of_buff_0"} : memref<16xi32>
//CHECK:     %9 = AIE.buffer(%0) {sym_name = "of_buff_1"} : memref<16xi32>
//CHECK:     %10 = AIE.buffer(%0) {sym_name = "of_buff_2"} : memref<16xi32>
//CHECK:     %11 = AIE.buffer(%0) {sym_name = "of_buff_3"} : memref<16xi32>
//CHECK:     %12 = AIE.lock(%0, 0) {init = 0 : i32, sym_name = "of_lock_0"}
//CHECK:     %13 = AIE.lock(%0, 1) {init = 0 : i32, sym_name = "of_lock_1"}
//CHECK:     %14 = AIE.lock(%0, 2) {init = 0 : i32, sym_name = "of_lock_2"}
//CHECK:     %15 = AIE.lock(%0, 3) {init = 0 : i32, sym_name = "of_lock_3"}
//CHECK:     func.func @some_work(%arg0: memref<16xi32>) {
//CHECK:       return
//CHECK:     }
//CHECK:     %16 = AIE.core(%0) {
//CHECK:       AIE.useLock(%12, Acquire, 0)
//CHECK:       AIE.useLock(%13, Acquire, 0)
//CHECK:       func.call @some_work(%8) : (memref<16xi32>) -> ()
//CHECK:       func.call @some_work(%9) : (memref<16xi32>) -> ()
//CHECK:       AIE.useLock(%5, Acquire, 1)
//CHECK:       func.call @some_work(%2) : (memref<16xi32>) -> ()
//CHECK:       AIE.useLock(%12, Release, 1)
//CHECK:       AIE.useLock(%14, Acquire, 0)
//CHECK:       AIE.useLock(%15, Acquire, 0)
//CHECK:       func.call @some_work(%9) : (memref<16xi32>) -> ()
//CHECK:       func.call @some_work(%10) : (memref<16xi32>) -> ()
//CHECK:       func.call @some_work(%11) : (memref<16xi32>) -> ()
//CHECK:       AIE.useLock(%13, Release, 1)
//CHECK:       AIE.useLock(%14, Release, 1)
//CHECK:       AIE.useLock(%15, Release, 1)
//CHECK:       AIE.useLock(%5, Release, 0)
//CHECK:       AIE.useLock(%6, Acquire, 1)
//CHECK:       AIE.useLock(%7, Acquire, 1)
//CHECK:       func.call @some_work(%3) : (memref<16xi32>) -> ()
//CHECK:       func.call @some_work(%4) : (memref<16xi32>) -> ()
//CHECK:       AIE.useLock(%6, Release, 0)
//CHECK:       AIE.end
//CHECK:     }
//CHECK:     %17 = AIE.core(%1) {
//CHECK:       AIE.useLock(%12, Acquire, 1)
//CHECK:       func.call @some_work(%8) : (memref<16xi32>) -> ()
//CHECK:       AIE.useLock(%5, Acquire, 0)
//CHECK:       AIE.useLock(%6, Acquire, 0)
//CHECK:       func.call @some_work(%2) : (memref<16xi32>) -> ()
//CHECK:       func.call @some_work(%3) : (memref<16xi32>) -> ()
//CHECK:       AIE.useLock(%5, Release, 1)
//CHECK:       AIE.useLock(%6, Release, 1)
//CHECK:       AIE.useLock(%12, Release, 0)
//CHECK:       AIE.useLock(%13, Acquire, 1)
//CHECK:       AIE.useLock(%14, Acquire, 1)
//CHECK:       func.call @some_work(%9) : (memref<16xi32>) -> ()
//CHECK:       func.call @some_work(%10) : (memref<16xi32>) -> ()
//CHECK:       AIE.useLock(%13, Release, 0)
//CHECK:       AIE.useLock(%14, Release, 0)
//CHECK:       AIE.end
//CHECK:     }
//CHECK:   }
//CHECK: }

module @multiCoreMixedFifo {
    AIE.device(xcvc1902) {
        %tile12 = AIE.tile(1, 2)
        %tile13 = AIE.tile(1, 3)

        AIE.objectFifo @of (%tile12, {%tile13}, 4 : i32) : !AIE.objectFifo<memref<16xi32>>
        AIE.objectFifo @of2 (%tile13, {%tile12}, 3 : i32) : !AIE.objectFifo<memref<16xi32>>

        func.func @some_work(%line_in:memref<16xi32>) -> () {
            return
        }

        %core11 = AIE.core(%tile12) {
            %subview0 = AIE.objectFifo.acquire @of (Produce, 2) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem00 = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            %elem01 = AIE.objectFifo.subview.access %subview0[1] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem00) : (memref<16xi32>) -> ()
            func.call @some_work(%elem01) : (memref<16xi32>) -> ()

            %subview02 = AIE.objectFifo.acquire @of2 (Consume, 1) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem002 = AIE.objectFifo.subview.access %subview02[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem002) : (memref<16xi32>) -> ()

            AIE.objectFifo.release @of (Produce, 1)
            %subview1 = AIE.objectFifo.acquire @of (Produce, 3) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem10 = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            %elem11 = AIE.objectFifo.subview.access %subview1[1] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            %elem12 = AIE.objectFifo.subview.access %subview1[2] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem10) : (memref<16xi32>) -> ()
            func.call @some_work(%elem11) : (memref<16xi32>) -> ()
            func.call @some_work(%elem12) : (memref<16xi32>) -> ()
            AIE.objectFifo.release @of (Produce, 3)
            
            AIE.objectFifo.release @of2 (Consume, 1)
            %subview12 = AIE.objectFifo.acquire @of2 (Consume, 2) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem102 = AIE.objectFifo.subview.access %subview12[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            %elem112 = AIE.objectFifo.subview.access %subview12[1] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem102) : (memref<16xi32>) -> ()
            func.call @some_work(%elem112) : (memref<16xi32>) -> ()
            AIE.objectFifo.release @of2 (Consume, 1)
            
            AIE.end
        }

        %core12 = AIE.core(%tile13) {
            %subview0 = AIE.objectFifo.acquire @of (Consume, 1) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem00 = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem00) : (memref<16xi32>) -> ()

            %subview02 = AIE.objectFifo.acquire @of2 (Produce, 2) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem002 = AIE.objectFifo.subview.access %subview02[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            %elem012 = AIE.objectFifo.subview.access %subview02[1] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem002) : (memref<16xi32>) -> ()
            func.call @some_work(%elem012) : (memref<16xi32>) -> ()
            AIE.objectFifo.release @of2 (Produce, 2)

            AIE.objectFifo.release @of (Consume, 1)
            %subview1 = AIE.objectFifo.acquire @of (Consume, 2) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem10 = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            %elem11 = AIE.objectFifo.subview.access %subview1[1] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem10) : (memref<16xi32>) -> ()
            func.call @some_work(%elem11) : (memref<16xi32>) -> ()
            AIE.objectFifo.release @of (Consume, 2)

            AIE.end
        }
    }
}
