//===- non_adjacency_test_2.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
// Date: May 24th 2022
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

//CHECK: module @non_adjacency {
//CHECK:   AIE.device(xcvc1902) {
//CHECK:     memref.global "public" @objfifo_cons : memref<16xi32>
//CHECK:     memref.global "public" @objfifo : memref<16xi32>
//CHECK:     %0 = AIE.tile(1, 2)
//CHECK:     %1 = AIE.tile(3, 3)
//CHECK:     %2 = AIE.buffer(%1) {sym_name = "objfifo_cons_buff_0"} : memref<16xi32>
//CHECK:     %3 = AIE.buffer(%1) {sym_name = "objfifo_cons_buff_1"} : memref<16xi32>
//CHECK:     %4 = AIE.buffer(%1) {sym_name = "objfifo_cons_buff_2"} : memref<16xi32>
//CHECK:     %5 = AIE.buffer(%1) {sym_name = "objfifo_cons_buff_3"} : memref<16xi32>
//CHECK:     %6 = AIE.lock(%1, 0) {init = 0 : i32, sym_name = "objfifo_cons_lock_0"}
//CHECK:     %7 = AIE.lock(%1, 1) {init = 0 : i32, sym_name = "objfifo_cons_lock_1"}
//CHECK:     %8 = AIE.lock(%1, 2) {init = 0 : i32, sym_name = "objfifo_cons_lock_2"}
//CHECK:     %9 = AIE.lock(%1, 3) {init = 0 : i32, sym_name = "objfifo_cons_lock_3"}
//CHECK:     %10 = AIE.buffer(%0) {sym_name = "objfifo_buff_0"} : memref<16xi32>
//CHECK:     %11 = AIE.buffer(%0) {sym_name = "objfifo_buff_1"} : memref<16xi32>
//CHECK:     %12 = AIE.lock(%0, 0) {init = 0 : i32, sym_name = "objfifo_lock_0"}
//CHECK:     %13 = AIE.lock(%0, 1) {init = 0 : i32, sym_name = "objfifo_lock_1"}
//CHECK:     AIE.flow(%0, DMA : 0, %1, DMA : 0)
//CHECK:     func.func @some_work(%arg0: memref<16xi32>) {
//CHECK:       return
//CHECK:     }
//CHECK:     %14 = AIE.core(%0) {
//CHECK:       %c0 = arith.constant 0 : index
//CHECK:       %c1 = arith.constant 1 : index
//CHECK:       %c12 = arith.constant 12 : index
//CHECK:       %c2 = arith.constant 2 : index
//CHECK:       scf.for %arg0 = %c0 to %c12 step %c2 {
//CHECK:         AIE.useLock(%12, Acquire, 0)
//CHECK:         func.call @some_work(%10) : (memref<16xi32>) -> ()
//CHECK:         AIE.useLock(%12, Release, 1)
//CHECK:         AIE.useLock(%13, Acquire, 0)
//CHECK:         func.call @some_work(%11) : (memref<16xi32>) -> ()
//CHECK:         AIE.useLock(%13, Release, 1)
//CHECK:       }
//CHECK:       AIE.end
//CHECK:     }
//CHECK:     %15 = AIE.core(%1) {
//CHECK:       %c0 = arith.constant 0 : index
//CHECK:       %c1 = arith.constant 1 : index
//CHECK:       %c12 = arith.constant 12 : index
//CHECK:       %c4 = arith.constant 4 : index
//CHECK:       scf.for %arg0 = %c0 to %c12 step %c4 {
//CHECK:         AIE.useLock(%6, Acquire, 1)
//CHECK:         AIE.useLock(%7, Acquire, 1)
//CHECK:         AIE.useLock(%8, Acquire, 1)
//CHECK:         func.call @some_work(%2) : (memref<16xi32>) -> ()
//CHECK:         AIE.useLock(%6, Release, 0)
//CHECK:         AIE.useLock(%9, Acquire, 1)
//CHECK:         func.call @some_work(%3) : (memref<16xi32>) -> ()
//CHECK:         AIE.useLock(%7, Release, 0)
//CHECK:         AIE.useLock(%6, Acquire, 1)
//CHECK:         func.call @some_work(%4) : (memref<16xi32>) -> ()
//CHECK:         AIE.useLock(%8, Release, 0)
//CHECK:         AIE.useLock(%7, Acquire, 1)
//CHECK:         func.call @some_work(%5) : (memref<16xi32>) -> ()
//CHECK:         AIE.useLock(%9, Release, 0)
//CHECK:       }
//CHECK:       AIE.end
//CHECK:     }
//CHECK:     %16 = AIE.mem(%0) {
//CHECK:       %18 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
//CHECK:       AIE.useLock(%12, Acquire, 1)
//CHECK:       AIE.dmaBd(<%10 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%12, Release, 0)
//CHECK:       AIE.nextBd ^bb2
//CHECK:     ^bb2:  // pred: ^bb1
//CHECK:       AIE.useLock(%13, Acquire, 1)
//CHECK:       AIE.dmaBd(<%11 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%13, Release, 0)
//CHECK:       AIE.nextBd ^bb1
//CHECK:     ^bb3:  // pred: ^bb0
//CHECK:       AIE.end
//CHECK:     }
//CHECK:     %17 = AIE.mem(%1) {
//CHECK:       %18 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb5)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb4
//CHECK:       AIE.useLock(%6, Acquire, 0)
//CHECK:       AIE.dmaBd(<%2 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%6, Release, 1)
//CHECK:       AIE.nextBd ^bb2
//CHECK:     ^bb2:  // pred: ^bb1
//CHECK:       AIE.useLock(%7, Acquire, 0)
//CHECK:       AIE.dmaBd(<%3 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%7, Release, 1)
//CHECK:       AIE.nextBd ^bb3
//CHECK:     ^bb3:  // pred: ^bb2
//CHECK:       AIE.useLock(%8, Acquire, 0)
//CHECK:       AIE.dmaBd(<%4 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%8, Release, 1)
//CHECK:       AIE.nextBd ^bb4
//CHECK:     ^bb4:  // pred: ^bb3
//CHECK:       AIE.useLock(%9, Acquire, 0)
//CHECK:       AIE.dmaBd(<%5 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%9, Release, 1)
//CHECK:       AIE.nextBd ^bb1
//CHECK:     ^bb5:  // pred: ^bb0
//CHECK:       AIE.end
//CHECK:     }
//CHECK:   }
//CHECK: }

module @non_adjacency {
    AIE.device(xcvc1902) {
        %tile12 = AIE.tile(1, 2)
        %tile33 = AIE.tile(3, 3)

        AIE.objectFifo @objfifo (%tile12, {%tile33}, 2 : i32) : !AIE.objectFifo<memref<16xi32>>

        func.func @some_work(%lineOut : memref<16xi32>) -> () {
            return
        }

        %core12 = AIE.core(%tile12) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %height = arith.constant 12 : index

            scf.for %indexInHeight = %c0 to %height step %c1 {
                %subview = AIE.objectFifo.acquire @objfifo (Produce, 1) : !AIE.objectFifoSubview<memref<16xi32>>
                %elem0 = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
                func.call @some_work(%elem0) : (memref<16xi32>) -> ()
                AIE.objectFifo.release @objfifo (Produce, 1)
            }
            
            AIE.end
        }

        %core33 = AIE.core(%tile33) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %height = arith.constant 12 : index

            scf.for %indexInHeight = %c0 to %height step %c1 {
                %subview = AIE.objectFifo.acquire @objfifo (Consume, 3) : !AIE.objectFifoSubview<memref<16xi32>>
                %elem0 = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
                %elem1 = AIE.objectFifo.subview.access %subview[1] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
                %elem2 = AIE.objectFifo.subview.access %subview[2] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
                func.call @some_work(%elem0) : (memref<16xi32>) -> ()
                AIE.objectFifo.release @objfifo (Consume, 1)
            }
            
            AIE.end
        }
    }
}
