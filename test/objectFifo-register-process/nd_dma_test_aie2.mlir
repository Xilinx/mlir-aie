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

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(1, 3)
// CHECK:           aie.objectfifo @objfifo(%[[VAL_0]] dimensionsToStream [<size = 1, stride = 2>], {%[[VAL_1]] fromStream [<size = 3, stride = 4>]}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK:           %[[VAL_3:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK:           %[[VAL_4:.*]] = arith.constant 10 : index
// CHECK:           func.func @producer_work() {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_5:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_7:.*]] = arith.constant 10 : index
// CHECK:             %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK:             scf.for %[[VAL_9:.*]] = %[[VAL_6]] to %[[VAL_7]] step %[[VAL_8]] {
// CHECK:               %[[VAL_10:.*]] = aie.objectfifo.acquire @objfifo(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
// CHECK:               %[[VAL_11:.*]] = aie.objectfifo.subview.access %[[VAL_10]][0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
// CHECK:               func.call @producer_work() : () -> ()
// CHECK:               aie.objectfifo.release @objfifo(Produce, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @registerPatterns  {
    aie.device(xcve2302) {
        %tile12 = aie.tile(1, 2)
        %tile13 = aie.tile(1, 3)

        aie.objectfifo @objfifo (%tile12 dimensionsToStream [<size = 1, stride = 2>], {%tile13 dimensionsFromStream [<size = 3, stride = 4>]}, 4 : i32) : !aie.objectfifo<memref<16xi32>>

        %acquirePattern = arith.constant dense<[1]> : tensor<1xi32>
        %releasePattern = arith.constant dense<[1]> : tensor<1xi32>
        %length = arith.constant 10 : index
        func.func @producer_work() -> () {
            return
        }

        aie.objectfifo.register_process @objfifo (Produce, %acquirePattern : tensor<1xi32>, %releasePattern : tensor<1xi32>, @producer_work, %length)
    }
}
