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

// CHECK-LABEL: module @registerPatterns {
// CHECK-NEXT:    %0 = AIE.tile(1, 2)

module @registerPatterns  {
    %tile12 = AIE.tile(1, 2)
    %tile13 = AIE.tile(1, 3)

    %objFifo = AIE.objectFifo.createObjectFifo(%tile12, %tile13, 4) : !AIE.objectFifo<memref<16xi32>>

    %acquirePattern = arith.constant dense<[1]> : tensor<1xi32>
    %releasePattern = arith.constant dense<[0,1,1,1,2]> : tensor<5xi32>
    %length = arith.constant 5 : index
    func @producer_work() -> () { 
        return
    }

    AIE.objectFifo.registerProcess{ port = "produce" }(%objFifo : !AIE.objectFifo<memref<16xi32>>, %acquirePattern : tensor<1xi32>, %releasePattern : tensor<5xi32>, @producer_work, %length) 
}