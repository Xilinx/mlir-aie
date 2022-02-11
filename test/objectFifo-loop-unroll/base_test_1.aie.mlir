//===- base_test_1.aie.mlir --------------------------*- MLIR -*-===//
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

// RUN: aie-opt --aie-unroll-objectFifos %s | FileCheck %s

// CHECK-LABEL: module @loopUnroll {
// CHECK-NEXT:    %0 = AIE.tile(1, 2)
// CHECK-NEXT:    %1 = AIE.tile(1, 3)
// CHECK-NEXT:    %2 = AIE.objectFifo.createObjectFifo(%0, %1, 2) : !AIE.objectFifo<memref<16xi32>>
// CHECK-NEXT:    func @some_work(%arg0: memref<16xi32>) {
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    %3 = AIE.core(%0) {
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      %c10 = arith.constant 10 : index
// CHECK-NEXT:      %c2 = arith.constant 2 : index
// CHECK-NEXT:      scf.for %arg0 = %c0 to %c10 step %c2 {
// CHECK-NEXT:        %4 = AIE.objectFifo.acquire {port = "produce"}(%2 : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
// CHECK-NEXT:        %5 = AIE.objectFifo.subview.access %4[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
// CHECK-NEXT:        call @some_work(%5) : (memref<16xi32>) -> ()
// CHECK-NEXT:        AIE.objectFifo.release {port = "produce"}(%2 : !AIE.objectFifo<memref<16xi32>>, 1)
// CHECK-NEXT:        %6 = AIE.objectFifo.acquire {port = "produce"}(%2 : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
// CHECK-NEXT:        %7 = AIE.objectFifo.subview.access %6[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
// CHECK-NEXT:        call @some_work(%7) : (memref<16xi32>) -> ()
// CHECK-NEXT:        AIE.objectFifo.release {port = "produce"}(%2 : !AIE.objectFifo<memref<16xi32>>, 1)
// CHECK-NEXT:      }
// CHECK-NEXT:      AIE.end
// CHECK-NEXT:    }
// CHECK-NEXT:  }

module @loopUnroll  {
    %tile12 = AIE.tile(1, 2)
    %tile13 = AIE.tile(1, 3)

    %objFifo = AIE.objectFifo.createObjectFifo(%tile12, %tile13, 2) : !AIE.objectFifo<memref<16xi32>>

    func @some_work(%line_in:memref<16xi32>) -> () {
        return
    }

    %core12 = AIE.core(%tile12) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 10 : index

        scf.for %indexInHeight = %c0 to %height step %c1 { 
            %subview = AIE.objectFifo.acquire{ port = "produce" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem0 = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            call @some_work(%elem0) : (memref<16xi32>) -> ()
            AIE.objectFifo.release{ port = "produce" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)
        }
        
        AIE.end
    }
}