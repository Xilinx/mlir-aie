//===- base_test_2.aie.mlir --------------------------*- MLIR -*-===//
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

// RUN: aie-opt --aie-register-objectFifos %s | FileCheck %s

// CHECK: module @registerPatterns {
// CHECK:    %0 = AIE.tile(1, 2)
// CHECK:    %1 = AIE.tile(1, 3)
// CHECK:    %2 = AIE.objectFifo.createObjectFifo(%0, {%1}, 4) : !AIE.objectFifo<memref<16xi32>>
// CHECK:    %cst = arith.constant dense<[2, 3, 3, 2]> : tensor<4xi32>
// CHECK:    %cst_0 = arith.constant dense<[0, 1, 1, 2]> : tensor<4xi32>
// CHECK:    %c10 = arith.constant 10 : index
// CHECK:    func.func @producer_work() {
// CHECK:      return
// CHECK:    }
// CHECK:    %3 = AIE.core(%0) {
// CHECK:      %4 = AIE.objectFifo.acquire<Produce> (%2 : !AIE.objectFifo<memref<16xi32>>, 2) : !AIE.objectFifoSubview<memref<16xi32>>
// CHECK:      %5 = AIE.objectFifo.subview.access %4[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
// CHECK:      %6 = AIE.objectFifo.subview.access %4[1] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
// CHECK:      func.call @producer_work() : () -> ()
// CHECK:      %c0 = arith.constant 0 : index
// CHECK:      %c2 = arith.constant 2 : index
// CHECK:      %c1 = arith.constant 1 : index
// CHECK:      scf.for %arg0 = %c0 to %c2 step %c1 {
// CHECK:        %10 = AIE.objectFifo.acquire<Produce> (%2 : !AIE.objectFifo<memref<16xi32>>, 3) : !AIE.objectFifoSubview<memref<16xi32>>
// CHECK:        %11 = AIE.objectFifo.subview.access %10[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
// CHECK:        %12 = AIE.objectFifo.subview.access %10[1] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
// CHECK:        %13 = AIE.objectFifo.subview.access %10[2] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
// CHECK:        func.call @producer_work() : () -> ()
// CHECK:        AIE.objectFifo.release<Produce> (%2 : !AIE.objectFifo<memref<16xi32>>, 1)
// CHECK:      }
// CHECK:      %7 = AIE.objectFifo.acquire<Produce> (%2 : !AIE.objectFifo<memref<16xi32>>, 2) : !AIE.objectFifoSubview<memref<16xi32>>
// CHECK:      %8 = AIE.objectFifo.subview.access %7[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
// CHECK:      %9 = AIE.objectFifo.subview.access %7[1] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
// CHECK:      func.call @producer_work() : () -> ()
// CHECK:      AIE.objectFifo.release<Produce> (%2 : !AIE.objectFifo<memref<16xi32>>, 2)
// CHECK:      AIE.end
// CHECK:    }
// CHECK:  }

module @registerPatterns  {
 AIE.device(xcvc1902) {
    %tile12 = AIE.tile(1, 2)
    %tile13 = AIE.tile(1, 3)

    %objFifo = AIE.objectFifo.createObjectFifo(%tile12, {%tile13}, 4) : !AIE.objectFifo<memref<16xi32>>

    %acquirePattern = arith.constant dense<[2,3,3,2]> : tensor<4xi32>
    %releasePattern = arith.constant dense<[0,1,1,2]> : tensor<4xi32>
    %length = arith.constant 10 : index
    func.func @producer_work() -> () { 
        return
    }

    AIE.objectFifo.registerProcess<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, %acquirePattern : tensor<4xi32>, %releasePattern : tensor<4xi32>, @producer_work, %length) 
 }
}
