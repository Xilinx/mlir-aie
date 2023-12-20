//===- base_test_5.aie.mlir --------------------------*- MLIR -*-===//
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

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(1, 3)
// CHECK:           aie.objectfifo @objfifo(%[[VAL_0]], {%[[VAL_1]]}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK:           %[[VAL_3:.*]] = arith.constant dense<[0, 1, 1, 1, 2]> : tensor<5xi32>
// CHECK:           %[[VAL_4:.*]] = arith.constant 5 : index
// CHECK:           func.func @producer_work() {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_5:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_6:.*]] = aie.objectfifo.acquire @objfifo(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
// CHECK:             %[[VAL_7:.*]] = aie.objectfifo.subview.access %[[VAL_6]][0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
// CHECK:             func.call @producer_work() : () -> ()
// CHECK:             %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_9:.*]] = arith.constant 3 : index
// CHECK:             %[[VAL_10:.*]] = arith.constant 1 : index
// CHECK:             scf.for %[[VAL_11:.*]] = %[[VAL_8]] to %[[VAL_9]] step %[[VAL_10]] {
// CHECK:               %[[VAL_12:.*]] = aie.objectfifo.acquire @objfifo(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
// CHECK:               %[[VAL_13:.*]] = aie.objectfifo.subview.access %[[VAL_12]][0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
// CHECK:               func.call @producer_work() : () -> ()
// CHECK:               aie.objectfifo.release @objfifo(Produce, 1)
// CHECK:             }
// CHECK:             %[[VAL_14:.*]] = aie.objectfifo.acquire @objfifo(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
// CHECK:             %[[VAL_15:.*]] = aie.objectfifo.subview.access %[[VAL_14]][0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
// CHECK:             func.call @producer_work() : () -> ()
// CHECK:             aie.objectfifo.release @objfifo(Produce, 2)
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @registerPatterns  {
    aie.device(xcvc1902) {
        %tile12 = aie.tile(1, 2)
        %tile13 = aie.tile(1, 3)

        aie.objectfifo @objfifo (%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<16xi32>>

        %acquirePattern = arith.constant dense<[1]> : tensor<1xi32>
        %releasePattern = arith.constant dense<[0,1,1,1,2]> : tensor<5xi32>
        %length = arith.constant 5 : index
        func.func @producer_work() -> () {
            return
        }

        aie.objectfifo.register_process @objfifo (Produce, %acquirePattern : tensor<1xi32>, %releasePattern : tensor<5xi32>, @producer_work, %length)
    }
}
