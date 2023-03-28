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
// CHECK:   %0 = AIE.tile(1, 2)
// CHECK:   %1 = AIE.tile(1, 3)
// CHECK:   %2 = AIE.buffer(%0) {sym_name = "of_0_buff_0"} : memref<16xi32>
// CHECK:   %3 = AIE.lock(%0, 0) {sym_name = "of_0_lock_0"}
// CHECK:   %4 = AIE.buffer(%0) {sym_name = "of_0_buff_1"} : memref<16xi32>
// CHECK:   %5 = AIE.lock(%0, 1) {sym_name = "of_0_lock_1"}
// CHECK:   %6 = AIE.buffer(%0) {sym_name = "of_0_buff_2"} : memref<16xi32>
// CHECK:   %7 = AIE.lock(%0, 2) {sym_name = "of_0_lock_2"}
// CHECK:   %8 = AIE.buffer(%0) {sym_name = "of_0_buff_3"} : memref<16xi32>
// CHECK:   %9 = AIE.lock(%0, 3) {sym_name = "of_0_lock_3"}
// CHECK:   func.func @some_work(%arg0: memref<16xi32>, %arg1: index) {
// CHECK:     return
// CHECK:   }
// CHECK:   %10 = AIE.core(%0) {
// CHECK:     %c0 = arith.constant 0 : index
// CHECK:     %c1 = arith.constant 1 : index
// CHECK:     %c2 = arith.constant 2 : index
// CHECK:     %c4 = arith.constant 4 : index
// CHECK:     %c21 = arith.constant 21 : index
// CHECK:     AIE.useLock(%3, Acquire, 0)
// CHECK:     func.call @some_work(%2, %c0) : (memref<16xi32>, index) -> ()
// CHECK:     AIE.useLock(%3, Release, 1)
// CHECK:     %c16 = arith.constant 16 : index
// CHECK:     %c8 = arith.constant 8 : index
// CHECK:     scf.for %arg0 = %c1 to %c16 step %c8 {
// CHECK:       AIE.useLock(%5, Acquire, 0)
// CHECK:       func.call @some_work(%4, %arg0) : (memref<16xi32>, index) -> ()
// CHECK:       AIE.useLock(%5, Release, 1)
// CHECK:       AIE.useLock(%7, Acquire, 0)
// CHECK:       %c2_5 = arith.constant 2 : index
// CHECK:       %14 = arith.addi %arg0, %c2_5 : index
// CHECK:       func.call @some_work(%6, %14) : (memref<16xi32>, index) -> ()
// CHECK:       AIE.useLock(%7, Release, 1)
// CHECK:       AIE.useLock(%9, Acquire, 0)
// CHECK:       %c4_6 = arith.constant 4 : index
// CHECK:       %15 = arith.addi %arg0, %c4_6 : index
// CHECK:       func.call @some_work(%8, %15) : (memref<16xi32>, index) -> ()
// CHECK:       AIE.useLock(%9, Release, 1)
// CHECK:       AIE.useLock(%3, Acquire, 0)
// CHECK:       %c6 = arith.constant 6 : index
// CHECK:       %16 = arith.addi %arg0, %c6 : index
// CHECK:       func.call @some_work(%2, %16) : (memref<16xi32>, index) -> ()
// CHECK:       AIE.useLock(%3, Release, 1)
// CHECK:     }
// CHECK:     AIE.useLock(%5, Acquire, 0)
// CHECK:     %c0_0 = arith.constant 0 : index
// CHECK:     func.call @some_work(%4, %c16) : (memref<16xi32>, index) -> ()
// CHECK:     AIE.useLock(%5, Release, 1)
// CHECK:     AIE.useLock(%7, Acquire, 0)
// CHECK:     %c2_1 = arith.constant 2 : index
// CHECK:     %11 = arith.addi %c16, %c2_1 : index
// CHECK:     func.call @some_work(%6, %11) : (memref<16xi32>, index) -> ()
// CHECK:     AIE.useLock(%7, Release, 1)
// CHECK:     AIE.useLock(%9, Acquire, 0)
// CHECK:     %c0_2 = arith.constant 0 : index
// CHECK:     func.call @some_work(%8, %c1) : (memref<16xi32>, index) -> ()
// CHECK:     AIE.useLock(%9, Release, 1)
// CHECK:     AIE.useLock(%3, Acquire, 0)
// CHECK:     %c1_3 = arith.constant 1 : index
// CHECK:     %12 = arith.addi %c1, %c1_3 : index
// CHECK:     func.call @some_work(%2, %12) : (memref<16xi32>, index) -> ()
// CHECK:     AIE.useLock(%3, Release, 1)
// CHECK:     AIE.useLock(%5, Acquire, 0)
// CHECK:     %c2_4 = arith.constant 2 : index
// CHECK:     %13 = arith.addi %c1, %c2_4 : index
// CHECK:     func.call @some_work(%4, %13) : (memref<16xi32>, index) -> ()
// CHECK:     AIE.useLock(%5, Release, 1)
// CHECK:     AIE.end
// CHECK:   }
// CHECK: }

module @loop  {
 AIE.device(xcvc1902) {
    %tile12 = AIE.tile(1, 2)
    %tile13 = AIE.tile(1, 3)

    %objFifo = AIE.objectFifo.createObjectFifo(%tile12, {%tile13}, 4) : !AIE.objectFifo<memref<16xi32>>

    func.func @some_work(%line_in:memref<16xi32>, %index:index) -> () {
        return
    }

    %core12 = AIE.core(%tile12) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c4 = arith.constant 4 : index
        %c21 = arith.constant 21 : index

        %subviewTop0 = AIE.objectFifo.acquire<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
        %elemTop0 = AIE.objectFifo.subview.access %subviewTop0[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        func.call @some_work(%elemTop0, %c0) : (memref<16xi32>,index) -> ()
        AIE.objectFifo.release<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)

        scf.for %indexInHeight = %c1 to %c21 step %c2 { 
            %subview = AIE.objectFifo.acquire<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem0 = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0,%indexInHeight) : (memref<16xi32>,index) -> ()
            AIE.objectFifo.release<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)
        }

        scf.for %indexInHeight = %c1 to %c4 step %c1 { 
            %subview = AIE.objectFifo.acquire<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem0 = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0,%indexInHeight) : (memref<16xi32>,index) -> ()
            AIE.objectFifo.release<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)
        }
        
        AIE.end
    }
 }
}
