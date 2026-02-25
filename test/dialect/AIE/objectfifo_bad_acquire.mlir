//===- objectfifo_bad_acquire.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: not aie-opt -split-input-file %s 2>&1 | FileCheck %s

// CHECK: 'aie.objectfifo.acquire' op must be called from inside a CoreOp

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile23 = aie.tile(2, 3)

   aie.objectfifo @of_0 (%tile12, {%tile23}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

   %subview = aie.objectfifo.acquire @of_0 (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
}

// -----

// CHECK: 'aie.objectfifo.acquire' op ObjectFifo element and ObjectFifoSubview element must match.

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile23 = aie.tile(2, 3)

   aie.objectfifo @of_0 (%tile12, {%tile23}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

   %core12 = aie.core(%tile12) {
      %subview = aie.objectfifo.acquire @of_0 (Produce, 1) : !aie.objectfifosubview<memref<2x2xi32>>
         
      aie.end
   }
}

// -----

// CHECK: 'aie.core' op producer port of objectFifo accessed by core running on non-producer tile

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile23 = aie.tile(2, 3)

   aie.objectfifo @of_0 (%tile12, {%tile23}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

   %core23 = aie.core(%tile23) {
      %subview = aie.objectfifo.acquire @of_0 (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
         
      aie.end
   }
}

// -----

// CHECK: 'aie.core' op consumer port of objectFifo accessed by core running on non-consumer tile

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile23 = aie.tile(2, 3)

   aie.objectfifo @of_0 (%tile12, {%tile23}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

   %core12 = aie.core(%tile12) {
      %subview = aie.objectfifo.acquire @of_0 (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
         
      aie.end
   }
}
