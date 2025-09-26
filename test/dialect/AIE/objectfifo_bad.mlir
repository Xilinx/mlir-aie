//===- objectfifo_bad.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: not aie-opt -split-input-file %s 2>&1 | FileCheck %s

// CHECK: 'aie.objectfifo' op does not have enough depths specified for producer and for each consumer.

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile13 = aie.tile(1, 3)
   %tile23 = aie.tile(2, 3)

   aie.objectfifo @of_0 (%tile12, {%tile13, %tile23}, [2, 2]) : !aie.objectfifo<memref<16xi32>>
}

// -----

// CHECK: custom op 'aie.objectfifo' initial values should initialize all objects

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile23 = aie.tile(2, 3)

   aie.objectfifo @of0 (%tile12, {%tile23}, 3 : i32) : !aie.objectfifo<memref<4xi32>> = [dense<[0, 1, 2, 3]> : memref<4xi32>]
}

// -----

// CHECK: custom op 'aie.objectfifo' initial value should be an elements attribute

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile23 = aie.tile(2, 3)

   aie.objectfifo @of0 (%tile12, {%tile23}, 1 : i32) : !aie.objectfifo<memref<4xi32>> = [[0, 1, 2, 3]]
}

// -----

// CHECK: inferred shape of elements literal ({{\[}}2, 3]) does not match type ({{\[}}2, 2])

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile23 = aie.tile(2, 3)

   aie.objectfifo @of0 (%tile12, {%tile23}, 2 : i32) : !aie.objectfifo<memref<2x2xi32>> = [dense<[[4, 5], [6, 7]]> : memref<2x2xi32>, 
                                                                                          dense<[[0, 1, 2], [3, 4, 5]]> : memref<2x2xi32>]
}
