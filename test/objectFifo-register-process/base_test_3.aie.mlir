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

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_2:.*]] = aie.tile(3, 3)
// CHECK:           aie.objectfifo @objfifo(%[[VAL_0]], {%[[VAL_2]], %[[VAL_1]]}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
// CHECK:           %[[VAL_3:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK:           %[[VAL_4:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK:           %[[VAL_5:.*]] = arith.constant 10 : index
// CHECK:           func.func @producer_work() {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_6:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK:           %[[VAL_7:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK:           %[[VAL_8:.*]] = arith.constant 10 : index
// CHECK:           func.func @consumer_work1() {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_9:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK:           %[[VAL_10:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK:           %[[VAL_11:.*]] = arith.constant 10 : index
// CHECK:           func.func @consumer_work2() {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_13:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_14:.*]] = arith.constant 10 : index
// CHECK:             %[[VAL_15:.*]] = arith.constant 1 : index
// CHECK:             scf.for %[[VAL_16:.*]] = %[[VAL_13]] to %[[VAL_14]] step %[[VAL_15]] {
// CHECK:               %[[VAL_17:.*]] = aie.objectfifo.acquire @objfifo(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
// CHECK:               %[[VAL_18:.*]] = aie.objectfifo.subview.access %[[VAL_17]][0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
// CHECK:               func.call @producer_work() : () -> ()
// CHECK:               aie.objectfifo.release @objfifo(Produce, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_19:.*]] = aie.core(%[[VAL_2]]) {
// CHECK:             %[[VAL_20:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_21:.*]] = arith.constant 10 : index
// CHECK:             %[[VAL_22:.*]] = arith.constant 1 : index
// CHECK:             scf.for %[[VAL_23:.*]] = %[[VAL_20]] to %[[VAL_21]] step %[[VAL_22]] {
// CHECK:               %[[VAL_24:.*]] = aie.objectfifo.acquire @objfifo(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
// CHECK:               %[[VAL_25:.*]] = aie.objectfifo.subview.access %[[VAL_24]][0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
// CHECK:               func.call @consumer_work1() : () -> ()
// CHECK:               aie.objectfifo.release @objfifo(Consume, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_26:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:             %[[VAL_27:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_28:.*]] = arith.constant 10 : index
// CHECK:             %[[VAL_29:.*]] = arith.constant 1 : index
// CHECK:             scf.for %[[VAL_30:.*]] = %[[VAL_27]] to %[[VAL_28]] step %[[VAL_29]] {
// CHECK:               %[[VAL_31:.*]] = aie.objectfifo.acquire @objfifo(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
// CHECK:               %[[VAL_32:.*]] = aie.objectfifo.subview.access %[[VAL_31]][0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
// CHECK:               func.call @consumer_work2() : () -> ()
// CHECK:               aie.objectfifo.release @objfifo(Consume, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }


module @registerPatterns  {
    aie.device(xcvc1902) {
        %tile12 = aie.tile(1, 2)
        %tile13 = aie.tile(1, 3)
        %tile33 = aie.tile(3, 3)

        aie.objectfifo @objfifo (%tile12, {%tile33, %tile13}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

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

        aie.objectfifo.register_process @objfifo (Produce, %prodAcqPattern : tensor<1xi32>, %prodRelPattern : tensor<1xi32>, @producer_work, %prodLength)
        aie.objectfifo.register_process @objfifo (Consume, %consAcqPattern1 : tensor<1xi32>, %consRelPattern1 : tensor<1xi32>, @consumer_work1, %consLength1)
        aie.objectfifo.register_process @objfifo (Consume, %consAcqPattern2 : tensor<1xi32>, %consRelPattern2 : tensor<1xi32>, @consumer_work2, %consLength2)
    }
}
