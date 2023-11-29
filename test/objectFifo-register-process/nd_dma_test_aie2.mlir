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

// CHECK-LABEL:   AIE.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = AIE.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(1, 3)
// CHECK:           AIE.objectfifo @objfifo(%[[VAL_0]] toStream [<1, 2>], {%[[VAL_1]] fromStream [<3, 4>]}, 4 : i32) : !AIE.objectfifo<memref<16xi32>>
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK:           %[[VAL_3:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK:           %[[VAL_4:.*]] = arith.constant 10 : index
// CHECK:           func.func @producer_work() {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_5:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_7:.*]] = arith.constant 10 : index
// CHECK:             %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK:             scf.for %[[VAL_9:.*]] = %[[VAL_6]] to %[[VAL_7]] step %[[VAL_8]] {
// CHECK:               %[[VAL_10:.*]] = AIE.objectfifo.acquire @objfifo(Produce, 1) : !AIE.objectfifosubview<memref<16xi32>>
// CHECK:               %[[VAL_11:.*]] = AIE.objectfifo.subview.access %[[VAL_10]][0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
// CHECK:               func.call @producer_work() : () -> ()
// CHECK:               AIE.objectfifo.release @objfifo(Produce, 1)
// CHECK:             }
// CHECK:             AIE.end
// CHECK:           }
// CHECK:         }

module @registerPatterns  {
    AIE.device(xcve2302) {
        %tile12 = AIE.tile(1, 2)
        %tile13 = AIE.tile(1, 3)

        AIE.objectfifo @objfifo (%tile12 toStream [<1, 2>], {%tile13 fromStream [<3, 4>]}, 4 : i32) : !AIE.objectfifo<memref<16xi32>>

        %acquirePattern = arith.constant dense<[1]> : tensor<1xi32>
        %releasePattern = arith.constant dense<[1]> : tensor<1xi32>
        %length = arith.constant 10 : index
        func.func @producer_work() -> () {
            return
        }

        AIE.objectfifo.register_process @objfifo (Produce, %acquirePattern : tensor<1xi32>, %releasePattern : tensor<1xi32>, @producer_work, %length)
    }
}
