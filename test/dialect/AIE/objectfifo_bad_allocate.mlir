//===- objectfifo_bad_allocate.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: not aie-opt -split-input-file %s 2>&1 | FileCheck %s

// CHECK: can only be used in 1-to-1 object FIFOs

aie.device(xcve2302) {
   %tile20 = aie.tile(2, 0)
   %tile13 = aie.tile(1, 3)
   %tile23 = aie.tile(2, 3)

   aie.objectfifo @of_0 (%tile20, {%tile13, %tile23}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
   aie.objectfifo.allocate @of_0 (%tile13)
}

// -----

// CHECK: cannot allocate a shared memory module to objectfifo with set `via_DMA` attribute

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile13 = aie.tile(1, 3)

   aie.objectfifo @of_0 (%tile12, {%tile13}, 2 : i32) {via_DMA = true} : !aie.objectfifo<memref<16xi32>>
   aie.objectfifo.allocate @of_0 (%tile13)
}

// -----

// CHECK: cannot allocate a shared memory module to objectfifo with set `repeat_count` attribute

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile13 = aie.tile(1, 3)

   aie.objectfifo @of_0 (%tile12, {%tile13}, 2 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16xi32>>
   aie.objectfifo.allocate @of_0 (%tile13)
}

// -----

// CHECK: cannot allocate a shared memory module to objectfifo with set dimensions attribute

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile13 = aie.tile(1, 3)

   aie.objectfifo @of_0 (%tile12 dimensionsToStream [<size = 1, stride = 1>, <size = 1, stride = 1>], {%tile13}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
   aie.objectfifo.allocate @of_0 (%tile13)
}
