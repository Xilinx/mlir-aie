//===- objectfifo_bad_release.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: not aie-opt -split-input-file %s 2>&1 | FileCheck %s

// CHECK: 'aie.objectfifo.release' op must be called from inside a CoreOp

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile23 = aie.tile(2, 3)

   aie.objectfifo @of_0 (%tile12, {%tile23}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

   aie.objectfifo.release @of_0 (Produce, 1)
}

// -----

// CHECK: 'aie.core' op producer port of objectFifo accessed by core running on non-producer tile

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile23 = aie.tile(2, 3)

   aie.objectfifo @of_0 (%tile12, {%tile23}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

   %core23 = aie.core(%tile23) {
      aie.objectfifo.release @of_0 (Produce, 1)
         
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
      aie.objectfifo.release @of_0 (Consume, 1)
         
      aie.end
   }
}
