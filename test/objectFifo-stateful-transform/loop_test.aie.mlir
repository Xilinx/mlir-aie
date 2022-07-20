//===- loop_test.aie.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
// Date: February 9th 2022
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @loop {
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
// CHECK:    func.func @some_work(%arg0: memref<16xi32>) {
// CHECK:        return
// CHECK:    }
// CHECK:    %10 = AIE.core(%0) {
// CHECK:        %c1 = arith.constant 1 : index
// CHECK:        %c10 = arith.constant 10 : index
// CHECK:        AIE.useLock(%3, Acquire, 0)
// CHECK:        func.call @some_work(%2) : (memref<16xi32>) -> ()
// CHECK:        AIE.useLock(%3, Release, 1)
// CHECK:        %c9 = arith.constant 9 : index
// CHECK:        %c4 = arith.constant 4 : index
// CHECK:        scf.for %arg0 = %c1 to %c9 step %c4 {
// CHECK:          AIE.useLock(%5, Acquire, 0)
// CHECK:          func.call @some_work(%4) : (memref<16xi32>) -> ()
// CHECK:          AIE.useLock(%5, Release, 1)
// CHECK:          AIE.useLock(%7, Acquire, 0)
// CHECK:          func.call @some_work(%6) : (memref<16xi32>) -> ()
// CHECK:          AIE.useLock(%7, Release, 1)
// CHECK:          AIE.useLock(%9, Acquire, 0)
// CHECK:          func.call @some_work(%8) : (memref<16xi32>) -> ()
// CHECK:          AIE.useLock(%9, Release, 1)
// CHECK:          AIE.useLock(%3, Acquire, 0)
// CHECK:          func.call @some_work(%2) : (memref<16xi32>) -> ()
// CHECK:          AIE.useLock(%3, Release, 1)
// CHECK:        }
// CHECK:        AIE.useLock(%5, Acquire, 0)
// CHECK:        func.call @some_work(%4) : (memref<16xi32>) -> ()
// CHECK:        AIE.useLock(%5, Release, 1)
// CHECK:        AIE.end
// CHECK:    }
// CHECK:  }

module @loop  {
    %tile12 = AIE.tile(1, 2)
    %tile13 = AIE.tile(1, 3)

    %objFifo = AIE.objectFifo.createObjectFifo(%tile12, %tile13, 4) : !AIE.objectFifo<memref<16xi32>>

    func.func @some_work(%line_in:memref<16xi32>) -> () {
        return
    }

    %core12 = AIE.core(%tile12) {
        %c1 = arith.constant 1 : index
        %height = arith.constant 10 : index

        %subviewTop0 = AIE.objectFifo.acquire{ port = "produce" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
        %elemTop0 = AIE.objectFifo.subview.access %subviewTop0[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        func.call @some_work(%elemTop0) : (memref<16xi32>) -> ()
        AIE.objectFifo.release{ port = "produce" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)

        scf.for %indexInHeight = %c1 to %height step %c1 { 
            %subview0 = AIE.objectFifo.acquire{ port = "produce" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem00 = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem00) : (memref<16xi32>) -> ()
            AIE.objectFifo.release{ port = "produce" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)
        }
        
        AIE.end
    }
}