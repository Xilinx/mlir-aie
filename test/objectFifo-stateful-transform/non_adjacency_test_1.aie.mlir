//===- non_adjacency_test_1.aie.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
// Date: February 10th 2022
// 
//===----------------------------------------------------------------------===//

// REQUIRES: andrab
// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @non_adjacency {
// CHECK:    %0 = AIE.tile(1, 2)
// CHECK:    %1 = AIE.tile(3, 3)
// CHECK:    AIE.flow(%0, DMA : 0, %1, DMA : 1)
// CHECK:    %2 = AIE.buffer(%0) {sym_name = "buff0"} : memref<16xi32>
// CHECK:    %3 = AIE.lock(%0, 0)
// CHECK:    %4 = AIE.buffer(%0) {sym_name = "buff1"} : memref<16xi32>
// CHECK:    %5 = AIE.lock(%0, 1)
// CHECK:    %6 = AIE.mem(%0) {
// CHECK:      %14 = AIE.dmaStart(MM2S0, ^bb1, ^bb3)
// CHECK:    ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:      AIE.useLock(%3, Acquire, 1)
// CHECK:      AIE.dmaBd(<%2 : memref<16xi32>, 0, 16>, 0)
// CHECK:      AIE.useLock(%3, Release, 0)
// CHECK:      cf.br ^bb2
// CHECK:    ^bb2:  // pred: ^bb1
// CHECK:      AIE.useLock(%5, Acquire, 1)
// CHECK:      AIE.dmaBd(<%4 : memref<16xi32>, 0, 16>, 0)
// CHECK:      AIE.useLock(%5, Release, 0)
// CHECK:      cf.br ^bb1
// CHECK:    ^bb3:  // pred: ^bb0
// CHECK:      AIE.end
// CHECK:    }
// CHECK:    %7 = AIE.buffer(%1) {sym_name = "buff2"} : memref<16xi32>
// CHECK:    %8 = AIE.lock(%1, 0)
// CHECK:    %9 = AIE.buffer(%1) {sym_name = "buff3"} : memref<16xi32>
// CHECK:    %10 = AIE.lock(%1, 1)
// CHECK:    %11 = AIE.mem(%1) {
// CHECK:      %14 = AIE.dmaStart(S2MM1, ^bb1, ^bb3)
// CHECK:    ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:      AIE.useLock(%8, Acquire, 0)
// CHECK:      AIE.dmaBd(<%7 : memref<16xi32>, 0, 16>, 0)
// CHECK:      AIE.useLock(%8, Release, 1)
// CHECK:      cf.br ^bb2
// CHECK:    ^bb2:  // pred: ^bb1
// CHECK:      AIE.useLock(%10, Acquire, 0)
// CHECK:      AIE.dmaBd(<%9 : memref<16xi32>, 0, 16>, 0)
// CHECK:      AIE.useLock(%10, Release, 1)
// CHECK:      cf.br ^bb1
// CHECK:    ^bb3:  // pred: ^bb0
// CHECK:      AIE.end
// CHECK:    }
// CHECK:    func @some_work(%arg0: memref<16xi32>) {
// CHECK:      return
// CHECK:    }
// CHECK:    %12 = AIE.core(%0) {
// CHECK:      %c0 = arith.constant 0 : index
// CHECK:      %c2 = arith.constant 2 : index
// CHECK:      %c12 = arith.constant 12 : index
// CHECK:      scf.for %arg0 = %c0 to %c12 step %c2 {
// CHECK:        AIE.useLock(%3, Acquire, 0)
// CHECK:        call @some_work(%2) : (memref<16xi32>) -> ()
// CHECK:        AIE.useLock(%3, Release, 1)
// CHECK:        AIE.useLock(%5, Acquire, 0)
// CHECK:        call @some_work(%4) : (memref<16xi32>) -> ()
// CHECK:        AIE.useLock(%5, Release, 1)
// CHECK:      }
// CHECK:      AIE.end
// CHECK:    }
// CHECK:    %13 = AIE.core(%1) {
// CHECK:      %c0 = arith.constant 0 : index
// CHECK:      %c2 = arith.constant 2 : index
// CHECK:      %c12 = arith.constant 12 : index
// CHECK:      scf.for %arg0 = %c0 to %c12 step %c2 {
// CHECK:        AIE.useLock(%8, Acquire, 1)
// CHECK:        call @some_work(%7) : (memref<16xi32>) -> ()
// CHECK:        AIE.useLock(%8, Release, 0)
// CHECK:        AIE.useLock(%10, Acquire, 1)
// CHECK:        call @some_work(%9) : (memref<16xi32>) -> ()
// CHECK:        AIE.useLock(%10, Release, 0)
// CHECK:      }
// CHECK:      AIE.end
// CHECK:    }
// CHECK:  }

module @non_adjacency {
    %tile12 = AIE.tile(1, 2)
    %tile33 = AIE.tile(3, 3)

    %objFifo = AIE.objectFifo.createObjectFifo(%tile12, %tile33, 2) : !AIE.objectFifo<memref<16xi32>>

    func @some_work(%lineOut : memref<16xi32>) -> () {
        return
    }

    %core12 = AIE.core(%tile12) {
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %height = arith.constant 12 : index

        scf.for %indexInHeight = %c0 to %height step %c2 {
            %subview0 = AIE.objectFifo.acquire{ port = "produce" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem00 = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            call @some_work(%elem00) : (memref<16xi32>) -> ()
            AIE.objectFifo.release{ port = "produce" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)

            %subview1 = AIE.objectFifo.acquire{ port = "produce" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem10 = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            call @some_work(%elem10) : (memref<16xi32>) -> ()
            AIE.objectFifo.release{ port = "produce" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)
        }
        
        AIE.end
    }

    %core33 = AIE.core(%tile33) {
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %height = arith.constant 12 : index

        scf.for %indexInHeight = %c0 to %height step %c2 { 
            %subview0 = AIE.objectFifo.acquire{ port = "consume" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem00 = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            call @some_work(%elem00) : (memref<16xi32>) -> ()
            AIE.objectFifo.release{ port = "consume" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)

            %subview1 = AIE.objectFifo.acquire{ port = "consume" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem10 = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            call @some_work(%elem10) : (memref<16xi32>) -> ()
            AIE.objectFifo.release{ port = "consume" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)
        }
        
        AIE.end
    }
}