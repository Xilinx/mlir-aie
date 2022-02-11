//===- ping_pong_test.aie.mlir --------------------------*- MLIR -*-===//
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

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL: module @non_adjacency {
// CHECK-NEXT:    %0 = AIE.tile(1, 2)
// CHECK-NEXT:    %1 = AIE.tile(3, 3)
// CHECK-NEXT:    AIE.flow(%0, DMA : 0, %1, DMA : 1)
// CHECK-NEXT:    %2 = AIE.buffer(%0) {sym_name = "buff0"} : memref<16xi32>
// CHECK-NEXT:    %3 = AIE.lock(%0, 0)
// CHECK-NEXT:    %4 = AIE.buffer(%0) {sym_name = "buff1"} : memref<16xi32>
// CHECK-NEXT:    %5 = AIE.lock(%0, 1)
// CHECK-NEXT:    %6 = AIE.mem(%0) {
// CHECK-NEXT:      %14 = AIE.dmaStart(MM2S0, ^bb1, ^bb3)
// CHECK-NEXT:    ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:      AIE.useLock(%3, Acquire, 1)
// CHECK-NEXT:      AIE.dmaBd(<%2 : memref<16xi32>, 0, 16>, 0)
// CHECK-NEXT:      AIE.useLock(%3, Release, 0)
// CHECK-NEXT:      br ^bb2
// CHECK-NEXT:    ^bb2:  // pred: ^bb1
// CHECK-NEXT:      AIE.useLock(%5, Acquire, 1)
// CHECK-NEXT:      AIE.dmaBd(<%4 : memref<16xi32>, 0, 16>, 0)
// CHECK-NEXT:      AIE.useLock(%5, Release, 0)
// CHECK-NEXT:      br ^bb1
// CHECK-NEXT:    ^bb3:  // pred: ^bb0
// CHECK-NEXT:      AIE.end
// CHECK-NEXT:    }
// CHECK-NEXT:    %7 = AIE.buffer(%1) {sym_name = "buff2"} : memref<16xi32>
// CHECK-NEXT:    %8 = AIE.lock(%1, 0)
// CHECK-NEXT:    %9 = AIE.buffer(%1) {sym_name = "buff3"} : memref<16xi32>
// CHECK-NEXT:    %10 = AIE.lock(%1, 1)
// CHECK-NEXT:    %11 = AIE.mem(%1) {
// CHECK-NEXT:      %14 = AIE.dmaStart(S2MM1, ^bb1, ^bb3)
// CHECK-NEXT:    ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:      AIE.useLock(%8, Acquire, 0)
// CHECK-NEXT:      AIE.dmaBd(<%7 : memref<16xi32>, 0, 16>, 0)
// CHECK-NEXT:      AIE.useLock(%8, Release, 1)
// CHECK-NEXT:      br ^bb2
// CHECK-NEXT:    ^bb2:  // pred: ^bb1
// CHECK-NEXT:      AIE.useLock(%10, Acquire, 0)
// CHECK-NEXT:      AIE.dmaBd(<%9 : memref<16xi32>, 0, 16>, 0)
// CHECK-NEXT:      AIE.useLock(%10, Release, 1)
// CHECK-NEXT:      br ^bb1
// CHECK-NEXT:    ^bb3:  // pred: ^bb0
// CHECK-NEXT:      AIE.end
// CHECK-NEXT:    }
// CHECK-NEXT:    func @some_work(%arg0: memref<16xi32>) {
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    %12 = AIE.core(%0) {
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      %c12 = arith.constant 12 : index
// CHECK-NEXT:      scf.for %arg0 = %c0 to %c12 step %c1 {
// CHECK-NEXT:        AIE.useLock(%3, Acquire, 0)
// CHECK-NEXT:        call @some_work(%2) : (memref<16xi32>) -> ()
// CHECK-NEXT:        AIE.useLock(%3, Release, 1)
// CHECK-NEXT:      }
// CHECK-NEXT:      AIE.end
// CHECK-NEXT:    }
// CHECK-NEXT:    %13 = AIE.core(%1) {
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      %c12 = arith.constant 12 : index
// CHECK-NEXT:      scf.for %arg0 = %c0 to %c12 step %c1 {
// CHECK-NEXT:        AIE.useLock(%8, Acquire, 1)
// CHECK-NEXT:        call @some_work(%7) : (memref<16xi32>) -> ()
// CHECK-NEXT:        AIE.useLock(%8, Release, 0)
// CHECK-NEXT:      }
// CHECK-NEXT:      AIE.end
// CHECK-NEXT:    }
// CHECK-NEXT:  }

module @non_adjacency {
    %tile12 = AIE.tile(1, 2)
    %tile33 = AIE.tile(3, 3)

    %objFifo = AIE.objectFifo.createObjectFifo(%tile12, %tile33, 2) : !AIE.objectFifo<memref<16xi32>>

    func @some_work(%lineOut : memref<16xi32>) -> () {
        return
    }

    %core12 = AIE.core(%tile12) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        scf.for %indexInHeight = %c0 to %height step %c1 {
            %subview = AIE.objectFifo.acquire{ port = "produce" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem0 = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            call @some_work(%elem0) : (memref<16xi32>) -> ()
            AIE.objectFifo.release{ port = "produce" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)
        }
        
        AIE.end
    }

    %core33 = AIE.core(%tile33) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        scf.for %indexInHeight = %c0 to %height step %c1 { 
            %subview = AIE.objectFifo.acquire{ port = "consume" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem0 = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            call @some_work(%elem0) : (memref<16xi32>) -> ()
            AIE.objectFifo.release{ port = "consume" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)
        }
        
        AIE.end
    }
}