//===- objectfifo_bad_allocate.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// CHECK: cannot allocate a shared memory module to an objectfifo that moves by DMA: it asks for a dma transport

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile13 = aie.tile(1, 3)

   aie.objectfifo @of_0 (%tile12, {%tile13}, 2 : i32) {transport = #aie.transport<dma>} : !aie.objectfifo<memref<16xi32>>
   aie.objectfifo.allocate @of_0 (%tile13)
}

// -----

// CHECK: cannot allocate a shared memory module to an objectfifo that moves by DMA: it repeats objects (`repeat_count`)

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile13 = aie.tile(1, 3)

   aie.objectfifo @of_0 (%tile12, {%tile13}, 2 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16xi32>>
   aie.objectfifo.allocate @of_0 (%tile13)
}

// -----

// CHECK: cannot allocate a shared memory module to an objectfifo that moves by DMA: its producer reshapes the data (`dimensionsToStream`)

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile13 = aie.tile(1, 3)

   aie.objectfifo @of_0 (%tile12 dimensionsToStream [<size = 1, stride = 1>, <size = 1, stride = 1>], {%tile13}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
   aie.objectfifo.allocate @of_0 (%tile13)
}

// -----

// The fifo's consumer takes a different element type, which only the DMA can
// convert.
// CHECK: cannot allocate a shared memory module to an objectfifo that moves by DMA: its consumer takes a different element type

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile13 = aie.tile(1, 3)

   aie.objectfifo @of_0 (%tile12, {%tile13}, 2 : i32) : !aie.objectfifo<memref<16xi32>> -> !aie.objectfifo<memref<8xi32>>
   aie.objectfifo.allocate @of_0 (%tile13)
}

// -----

// The consumer reshapes what it receives, which only the DMA can do.
// CHECK: cannot allocate a shared memory module to an objectfifo that moves by DMA: a consumer reshapes the data (`dimensionsFromStream`)

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile13 = aie.tile(1, 3)

   aie.objectfifo @of_0 (%tile12, {%tile13 dimensionsFromStream [<size = 4, stride = 4>, <size = 4, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
   aie.objectfifo.allocate @of_0 (%tile13)
}
