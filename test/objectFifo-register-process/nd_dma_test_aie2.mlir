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

// RUN: aie-opt --aie-register-objectFifos %s | FileCheck %s

// CHECK: module @registerPatterns {
// CHECK:   AIE.device(xcve2302) {
// CHECK:     %[[tile_1_2:.*]] = AIE.tile(1, 2)
// CHECK:     %[[tile_1_3:.*]] = AIE.tile(1, 3)
// CHECK:     AIE.objectFifo @objfifo(%[[tile_1_2]] toStream [<1, 2>], {%[[tile_1_3]] fromStream [<3, 4>]}, 4 : i32) : !AIE.objectFifo<memref<16xi32>>
// CHECK:     %cst = arith.constant dense<1> : tensor<1xi32>
// CHECK:     %cst_0 = arith.constant dense<1> : tensor<1xi32>
// CHECK:     %c10 = arith.constant 10 : index
// CHECK:     func.func @producer_work() {
// CHECK:       return
// CHECK:     }
// CHECK:     %2 = AIE.core(%[[tile_1_2]]) {
// CHECK:       %c0 = arith.constant 0 : index
// CHECK:       %c10_1 = arith.constant 10 : index
// CHECK:       %c1 = arith.constant 1 : index
// CHECK:       scf.for %arg0 = %c0 to %c10_1 step %c1 {
// CHECK:         %3 = AIE.objectFifo.acquire @objfifo(Produce, 1) : !AIE.objectFifoSubview<memref<16xi32>>
// CHECK:         %4 = AIE.objectFifo.subview.access %3[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
// CHECK:         func.call @producer_work() : () -> ()
// CHECK:         AIE.objectFifo.release @objfifo(Produce, 1)
// CHECK:       }
// CHECK:       AIE.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @registerPatterns  {
    AIE.device(xcve2302) {
        %tile12 = AIE.tile(1, 2)
        %tile13 = AIE.tile(1, 3)

        AIE.objectFifo @objfifo (%tile12 toStream [<1, 2>], {%tile13 fromStream [<3, 4>]}, 4 : i32) : !AIE.objectFifo<memref<16xi32>>

        %acquirePattern = arith.constant dense<[1]> : tensor<1xi32>
        %releasePattern = arith.constant dense<[1]> : tensor<1xi32>
        %length = arith.constant 10 : index
        func.func @producer_work() -> () { 
            return
        }

        AIE.objectFifo.registerProcess @objfifo (Produce, %acquirePattern : tensor<1xi32>, %releasePattern : tensor<1xi32>, @producer_work, %length) 
    }
}
