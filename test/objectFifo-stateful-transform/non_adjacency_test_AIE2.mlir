//===- non_adjacency_test_AIE2.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Xilinx Inc.
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: May 9th 2023
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @non_adjacency_AIE2 {
// CHECK:   AIE.device(xcve2302) {
// CHECK:     %0 = AIE.tile(1, 2)
// CHECK:     %1 = AIE.tile(3, 3)
// CHECK:     AIE.flow(%0, DMA : 0, %1, DMA : 0)
// CHECK:     %2 = AIE.buffer(%0) {sym_name = "of_0_buff_0"} : memref<16xi32>
// CHECK:     %3 = AIE.buffer(%0) {sym_name = "of_0_buff_1"} : memref<16xi32>
// CHECK:     %4 = AIE.lock(%0, 0) {init = 2 : i32, sym_name = "of_0_prod_lock"}
// CHECK:     %5 = AIE.lock(%0, 1) {init = 0 : i32, sym_name = "of_0_cons_lock"}
// CHECK:     %6 = AIE.buffer(%1) {sym_name = "of_1_buff_0"} : memref<16xi32>
// CHECK:     %7 = AIE.buffer(%1) {sym_name = "of_1_buff_1"} : memref<16xi32>
// CHECK:     %8 = AIE.lock(%1, 0) {init = 2 : i32, sym_name = "of_1_prod_lock"}
// CHECK:     %9 = AIE.lock(%1, 1) {init = 0 : i32, sym_name = "of_1_cons_lock"}
// CHECK:     func.func @some_work(%arg0: memref<16xi32>) {
// CHECK:       return
// CHECK:     }
// CHECK:     %10 = AIE.core(%0) {
// CHECK:       %c0 = arith.constant 0 : index
// CHECK:       %c1 = arith.constant 1 : index
// CHECK:       %c12 = arith.constant 12 : index
// CHECK:       %c2 = arith.constant 2 : index
// CHECK:       scf.for %arg0 = %c0 to %c12 step %c2 {
// CHECK:         AIE.useLock(%4, AcquireGreaterEqual, 1)
// CHECK:         func.call @some_work(%2) : (memref<16xi32>) -> ()
// CHECK:         AIE.useLock(%5, Release, 1)
// CHECK:         AIE.useLock(%4, AcquireGreaterEqual, 1)
// CHECK:         func.call @some_work(%3) : (memref<16xi32>) -> ()
// CHECK:         AIE.useLock(%5, Release, 1)
// CHECK:       }
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     %11 = AIE.core(%1) {
// CHECK:       %c0 = arith.constant 0 : index
// CHECK:       %c1 = arith.constant 1 : index
// CHECK:       %c12 = arith.constant 12 : index
// CHECK:       %c2 = arith.constant 2 : index
// CHECK:       scf.for %arg0 = %c0 to %c12 step %c2 {
// CHECK:         AIE.useLock(%9, AcquireGreaterEqual, 1)
// CHECK:         func.call @some_work(%6) : (memref<16xi32>) -> ()
// CHECK:         AIE.useLock(%8, Release, 1)
// CHECK:         AIE.useLock(%9, AcquireGreaterEqual, 1)
// CHECK:         func.call @some_work(%7) : (memref<16xi32>) -> ()
// CHECK:         AIE.useLock(%8, Release, 1)
// CHECK:       }
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     %12 = AIE.mem(%0) {
// CHECK:       %14 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       AIE.useLock(%5, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%2 : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%4, Release, 1)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%5, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%3 : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%4, Release, 1)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     %13 = AIE.mem(%1) {
// CHECK:       %14 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       AIE.useLock(%8, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%6 : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%9, Release, 1)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%8, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%7 : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%9, Release, 1)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       AIE.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @non_adjacency_AIE2 {
 AIE.device(xcve2302) {
    %tile12 = AIE.tile(1, 2)
    %tile33 = AIE.tile(3, 3)

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

    %core33 = AIE.core(%tile33) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        scf.for %indexInHeight = %c0 to %height step %c1 { 
            %subview = AIE.objectFifo.acquire<Consume>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem0 = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            AIE.objectFifo.release<Consume>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)
        }
        
        AIE.end
    }
 }
}
