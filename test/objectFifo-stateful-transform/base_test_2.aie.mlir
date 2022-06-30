//===- base_test_2.aie.mlir --------------------------*- MLIR -*-===//
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

// CHECK: module @multiFifo  {
// CHECK:    %0 = AIE.tile(1, 2)
// CHECK:    %1 = AIE.tile(1, 3)
// CHECK:    %2 = AIE.buffer(%0) {sym_name = "buff0"} : memref<16xi32>
// CHECK:    %3 = AIE.lock(%0, 0)
// CHECK:    %4 = AIE.buffer(%0) {sym_name = "buff1"} : memref<16xi32>
// CHECK:    %5 = AIE.lock(%0, 1)
// CHECK:    %6 = AIE.buffer(%0) {sym_name = "buff2"} : memref<16xi32>
// CHECK:    %7 = AIE.lock(%0, 2)
// CHECK:    %8 = AIE.buffer(%0) {sym_name = "buff3"} : memref<16xi32>
// CHECK:    %9 = AIE.lock(%0, 3)
// CHECK:    %10 = AIE.buffer(%0) {sym_name = "buff4"} : memref<16xi32>
// CHECK:    %11 = AIE.lock(%0, 4)
// CHECK:    %12 = AIE.buffer(%0) {sym_name = "buff5"} : memref<16xi32>
// CHECK:    %13 = AIE.lock(%0, 5)
// CHECK:    %14 = AIE.buffer(%0) {sym_name = "buff6"} : memref<16xi32>
// CHECK:    %15 = AIE.lock(%0, 6)
// CHECK:    func @some_work(%arg0: memref<16xi32>) {
// CHECK:      return
// CHECK:    }
// CHECK:    %16 = AIE.core(%0)  {
// CHECK:      AIE.useLock(%3, Acquire, 0)
// CHECK:      AIE.useLock(%5, Acquire, 0)
// CHECK:      call @some_work(%2) : (memref<16xi32>) -> ()
// CHECK:      call @some_work(%4) : (memref<16xi32>) -> ()
// CHECK:      AIE.useLock(%11, Acquire, 0)
// CHECK:      call @some_work(%10) : (memref<16xi32>) -> ()
// CHECK:      AIE.useLock(%3, Release, 1)
// CHECK:      AIE.useLock(%7, Acquire, 0)
// CHECK:      AIE.useLock(%9, Acquire, 0)
// CHECK:      call @some_work(%4) : (memref<16xi32>) -> ()
// CHECK:      call @some_work(%6) : (memref<16xi32>) -> ()
// CHECK:      call @some_work(%8) : (memref<16xi32>) -> ()
// CHECK:      AIE.useLock(%5, Release, 1)
// CHECK:      AIE.useLock(%7, Release, 1)
// CHECK:      AIE.useLock(%9, Release, 1)
// CHECK:      AIE.useLock(%11, Release, 1)
// CHECK:      AIE.useLock(%13, Acquire, 0)
// CHECK:      AIE.useLock(%15, Acquire, 0)
// CHECK:      call @some_work(%12) : (memref<16xi32>) -> ()
// CHECK:      call @some_work(%14) : (memref<16xi32>) -> ()
// CHECK:      AIE.useLock(%13, Release, 1)
// CHECK:      AIE.end
// CHECK:    }
// CHECK:    %17 = AIE.core(%1)  {
// CHECK:      AIE.useLock(%3, Acquire, 1)
// CHECK:      call @some_work(%2) : (memref<16xi32>) -> ()
// CHECK:      AIE.useLock(%11, Acquire, 1)
// CHECK:      AIE.useLock(%13, Acquire, 1)
// CHECK:      call @some_work(%10) : (memref<16xi32>) -> ()
// CHECK:      call @some_work(%12) : (memref<16xi32>) -> ()
// CHECK:      AIE.useLock(%11, Release, 0)
// CHECK:      AIE.useLock(%13, Release, 0)
// CHECK:      AIE.useLock(%3, Release, 0)
// CHECK:      AIE.useLock(%5, Acquire, 1)
// CHECK:      AIE.useLock(%7, Acquire, 1)
// CHECK:      call @some_work(%4) : (memref<16xi32>) -> ()
// CHECK:      call @some_work(%6) : (memref<16xi32>) -> ()
// CHECK:      AIE.useLock(%5, Release, 0)
// CHECK:      AIE.useLock(%7, Release, 0)
// CHECK:      AIE.end
// CHECK:    }
// CHECK:  }

module @multiFifo {
    %tile12 = AIE.tile(1, 2)
    %tile13 = AIE.tile(1, 3)

    %objFifo = AIE.objectFifo.createObjectFifo(%tile12, %tile13, 4) : !AIE.objectFifo<memref<16xi32>>
    %objFifo2 = AIE.objectFifo.createObjectFifo(%tile12, %tile13, 3) : !AIE.objectFifo<memref<16xi32>>

    func @some_work(%line_in:memref<16xi32>) -> () {
        return
    }

    %core12 = AIE.core(%tile12) {
        %subview0 = AIE.objectFifo.acquire{ port = "produce" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 2) : !AIE.objectFifoSubview<memref<16xi32>>
        %elem00 = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        %elem01 = AIE.objectFifo.subview.access %subview0[1] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        call @some_work(%elem00) : (memref<16xi32>) -> ()
        call @some_work(%elem01) : (memref<16xi32>) -> ()

        %subview02 = AIE.objectFifo.acquire{ port = "produce" }(%objFifo2 : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
        %elem002 = AIE.objectFifo.subview.access %subview02[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        call @some_work(%elem002) : (memref<16xi32>) -> ()

        AIE.objectFifo.release{ port = "produce" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)
        %subview1 = AIE.objectFifo.acquire{ port = "produce" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 3) : !AIE.objectFifoSubview<memref<16xi32>>
        %elem10 = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        %elem11 = AIE.objectFifo.subview.access %subview1[1] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        %elem12 = AIE.objectFifo.subview.access %subview1[2] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        call @some_work(%elem10) : (memref<16xi32>) -> ()
        call @some_work(%elem11) : (memref<16xi32>) -> ()
        call @some_work(%elem12) : (memref<16xi32>) -> ()
        AIE.objectFifo.release{ port = "produce" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 3)
        
        AIE.objectFifo.release{ port = "produce" }(%objFifo2 : !AIE.objectFifo<memref<16xi32>>, 1)
        %subview12 = AIE.objectFifo.acquire{ port = "produce" }(%objFifo2 : !AIE.objectFifo<memref<16xi32>>, 2) : !AIE.objectFifoSubview<memref<16xi32>>
        %elem102 = AIE.objectFifo.subview.access %subview12[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        %elem112 = AIE.objectFifo.subview.access %subview12[1] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        call @some_work(%elem102) : (memref<16xi32>) -> ()
        call @some_work(%elem112) : (memref<16xi32>) -> ()
        AIE.objectFifo.release{ port = "produce" }(%objFifo2 : !AIE.objectFifo<memref<16xi32>>, 1)
        
        AIE.end
    }

    %core13 = AIE.core(%tile13) {
        %subview0 = AIE.objectFifo.acquire{ port = "consume" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
        %elem00 = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        call @some_work(%elem00) : (memref<16xi32>) -> ()

        %subview02 = AIE.objectFifo.acquire{ port = "consume" }(%objFifo2 : !AIE.objectFifo<memref<16xi32>>, 2) : !AIE.objectFifoSubview<memref<16xi32>>
        %elem002 = AIE.objectFifo.subview.access %subview02[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        %elem012 = AIE.objectFifo.subview.access %subview02[1] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        call @some_work(%elem002) : (memref<16xi32>) -> ()
        call @some_work(%elem012) : (memref<16xi32>) -> ()
        AIE.objectFifo.release{ port = "consume" }(%objFifo2 : !AIE.objectFifo<memref<16xi32>>, 2)

        AIE.objectFifo.release{ port = "consume" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)
        %subview1 = AIE.objectFifo.acquire{ port = "consume" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 2) : !AIE.objectFifoSubview<memref<16xi32>>
        %elem10 = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        %elem11 = AIE.objectFifo.subview.access %subview1[1] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        call @some_work(%elem10) : (memref<16xi32>) -> ()
        call @some_work(%elem11) : (memref<16xi32>) -> ()
        AIE.objectFifo.release{ port = "consume" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 2)

        AIE.end
    }
}