//===- tileDMA_test_1.aie.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
// Date: September 22nd 2022
// 
//===----------------------------------------------------------------------===//

// REQUIRES: andrab
// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @non_adjacency {
// CHECK:    %0 = AIE.tile(1, 2)
// CHECK:    %1 = AIE.tile(3, 3)
// CHECK:    AIE.multicast(%0, DMA : 0) {
// CHECK:      AIE.multi_dest<%1, DMA : 0>
// CHECK:    }
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
// CHECK:    func.func @some_work(%arg0: memref<16xi32>) {
// CHECK:      return
// CHECK:    }
// CHECK:    %12 = AIE.core(%0) {
// CHECK:      %c0 = arith.constant 0 : index
// CHECK:      %c1 = arith.constant 1 : index
// CHECK:      %c12 = arith.constant 12 : index
// CHECK:      %c2 = arith.constant 2 : index
// CHECK:      scf.for %arg0 = %c0 to %c12 step %c2 {
// CHECK:        AIE.useLock(%3, Acquire, 0)
// CHECK:        func.call @some_work(%2) : (memref<16xi32>) -> ()
// CHECK:        AIE.useLock(%3, Release, 1)
// CHECK:        AIE.useLock(%5, Acquire, 0)
// CHECK:        func.call @some_work(%4) : (memref<16xi32>) -> ()
// CHECK:        AIE.useLock(%5, Release, 1)
// CHECK:      }
// CHECK:      AIE.end
// CHECK:    }
// CHECK:    %13 = AIE.core(%1) {
// CHECK:      %c0 = arith.constant 0 : index
// CHECK:      %c1 = arith.constant 1 : index
// CHECK:      %c12 = arith.constant 12 : index
// CHECK:      %c2 = arith.constant 2 : index
// CHECK:      scf.for %arg0 = %c0 to %c12 step %c2 {
// CHECK:        AIE.useLock(%8, Acquire, 1)
// CHECK:        func.call @some_work(%7) : (memref<16xi32>) -> ()
// CHECK:        AIE.useLock(%8, Release, 0)
// CHECK:        AIE.useLock(%10, Acquire, 1)
// CHECK:        func.call @some_work(%9) : (memref<16xi32>) -> ()
// CHECK:        AIE.useLock(%10, Release, 0)
// CHECK:      }
// CHECK:      AIE.end
// CHECK:    }
// CHECK:  }

module @non_adjacency {
    %tile12 = AIE.tile(1, 2)
    %tile33 = AIE.tile(3, 3)

    %buff0 = AIE.buffer(%tile12) : memref<16xi32>
    %lock0 = AIE.lock(%tile12, 0)
    %buff1 = AIE.buffer(%tile12) : memref<16xi32>
    %lock1 = AIE.lock(%tile12, 1)

    %objFifo = AIE.objectFifo.createObjectFifo(%tile12, {%tile33}, 2) : !AIE.objectFifo<memref<16xi32>>

    func.func @some_work(%lineOut : memref<16xi32>) -> () {
        return
    }

    %core12 = AIE.core(%tile12) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        scf.for %indexInHeight = %c0 to %height step %c1 {
            %subview = AIE.objectFifo.acquire<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem0 = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            AIE.objectFifo.release<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)
        }
        
        AIE.end
    }

    %mem12 = AIE.mem(%tile12) {
        %14 = AIE.dmaStart(MM2S0, ^bb1, ^bb3)
    ^bb1:
        AIE.useLock(%lock0, Acquire, 1)
        AIE.dmaBd(<%buff0 : memref<16xi32>, 0, 16>, 0)
        AIE.useLock(%lock0, Release, 0)
        cf.br ^bb2
    ^bb2:
        AIE.useLock(%lock1, Acquire, 1)
        AIE.dmaBd(<%buff1 : memref<16xi32>, 0, 16>, 0)
        AIE.useLock(%lock1, Release, 0)
        cf.br ^bb1
    ^bb3:
        AIE.end
    }
}