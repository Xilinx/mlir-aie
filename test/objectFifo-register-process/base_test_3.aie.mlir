//===- base_test_3.aie.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
// Date: June 1st 2022
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-register-objectFifos %s | FileCheck %s

// CHECK: module @registerPatterns {
// CHECK:   %0 = AIE.tile(1, 2)
// CHECK:   %1 = AIE.tile(1, 3)
// CHECK:   %2 = AIE.tile(3, 3)
// CHECK:   %3 = AIE.objectFifo.createObjectFifo(%0, {%2, %1}, 2) : !AIE.objectFifo<memref<16xi32>>
// CHECK:   %cst = arith.constant dense<1> : tensor<1xi32>
// CHECK:   %cst_0 = arith.constant dense<1> : tensor<1xi32>
// CHECK:   %c10 = arith.constant 10 : index
// CHECK:   func.func @producer_work() {
// CHECK:     return
// CHECK:   }
// CHECK:   %cst_1 = arith.constant dense<1> : tensor<1xi32>
// CHECK:   %cst_2 = arith.constant dense<1> : tensor<1xi32>
// CHECK:   %c10_3 = arith.constant 10 : index
// CHECK:   func.func @consumer_work1() {
// CHECK:     return
// CHECK:   }
// CHECK:   %cst_4 = arith.constant dense<1> : tensor<1xi32>
// CHECK:   %cst_5 = arith.constant dense<1> : tensor<1xi32>
// CHECK:   %c10_6 = arith.constant 10 : index
// CHECK:   func.func @consumer_work2() {
// CHECK:     return
// CHECK:   }
// CHECK:   %4 = AIE.core(%0) {
// CHECK:     %c0 = arith.constant 0 : index
// CHECK:     %c10_7 = arith.constant 10 : index
// CHECK:     %c1 = arith.constant 1 : index
// CHECK:     scf.for %arg0 = %c0 to %c10_7 step %c1 {
// CHECK:       %7 = AIE.objectFifo.acquire<Produce> (%3 : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
// CHECK:       %8 = AIE.objectFifo.subview.access %7[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
// CHECK:       func.call @producer_work() : () -> ()
// CHECK:       AIE.objectFifo.release<Produce> (%3 : !AIE.objectFifo<memref<16xi32>>, 1)
// CHECK:     }
// CHECK:     AIE.end
// CHECK:   }
// CHECK:   %5 = AIE.core(%2) {
// CHECK:     %c0 = arith.constant 0 : index
// CHECK:     %c10_7 = arith.constant 10 : index
// CHECK:     %c1 = arith.constant 1 : index
// CHECK:     scf.for %arg0 = %c0 to %c10_7 step %c1 {
// CHECK:       %7 = AIE.objectFifo.acquire<Consume> (%3 : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
// CHECK:       %8 = AIE.objectFifo.subview.access %7[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
// CHECK:       func.call @consumer_work1() : () -> ()
// CHECK:       AIE.objectFifo.release<Consume> (%3 : !AIE.objectFifo<memref<16xi32>>, 1)
// CHECK:     }
// CHECK:     AIE.end
// CHECK:   }
// CHECK:   %6 = AIE.core(%1) {
// CHECK:     %c0 = arith.constant 0 : index
// CHECK:     %c10_7 = arith.constant 10 : index
// CHECK:     %c1 = arith.constant 1 : index
// CHECK:     scf.for %arg0 = %c0 to %c10_7 step %c1 {
// CHECK:       %7 = AIE.objectFifo.acquire<Consume> (%3 : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
// CHECK:       %8 = AIE.objectFifo.subview.access %7[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
// CHECK:       func.call @consumer_work2() : () -> ()
// CHECK:       AIE.objectFifo.release<Consume> (%3 : !AIE.objectFifo<memref<16xi32>>, 1)
// CHECK:     }
// CHECK:     AIE.end
// CHECK:   }
// CHECK: }


module @registerPatterns  {
 AIE.device(xcvc1902) {
    %tile12 = AIE.tile(1, 2)
    %tile13 = AIE.tile(1, 3)
    %tile33 = AIE.tile(3, 3)

    %objFifo = AIE.objectFifo.createObjectFifo(%tile12, {%tile33, %tile13}, 2) : !AIE.objectFifo<memref<16xi32>>

    %prodAcqPattern = arith.constant dense<[1]> : tensor<1xi32>
    %prodRelPattern = arith.constant dense<[1]> : tensor<1xi32>
    %prodLength = arith.constant 10 : index
    func.func @producer_work() -> () {
        return
    }

    %consAcqPattern1 = arith.constant dense<[1]> : tensor<1xi32>
    %consRelPattern1 = arith.constant dense<[1]> : tensor<1xi32>
    %consLength1 = arith.constant 10 : index
    func.func @consumer_work1() -> () {
        return
    }

    %consAcqPattern2 = arith.constant dense<[1]> : tensor<1xi32>
    %consRelPattern2 = arith.constant dense<[1]> : tensor<1xi32>
    %consLength2 = arith.constant 10 : index
    func.func @consumer_work2() -> () {
        return
    }

    AIE.objectFifo.registerProcess<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, %prodAcqPattern : tensor<1xi32>, %prodRelPattern : tensor<1xi32>, @producer_work, %prodLength) 
    AIE.objectFifo.registerProcess<Consume>(%objFifo : !AIE.objectFifo<memref<16xi32>>, %consAcqPattern1 : tensor<1xi32>, %consRelPattern1 : tensor<1xi32>, @consumer_work1, %consLength1) 
    AIE.objectFifo.registerProcess<Consume>(%objFifo : !AIE.objectFifo<memref<16xi32>>, %consAcqPattern2 : tensor<1xi32>, %consRelPattern2 : tensor<1xi32>, @consumer_work2, %consLength2) 
 }
}
