//===- objectfifo_bad_acquire.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt -split-input-file %s 2>&1 | FileCheck %s

// CHECK: 'aie.objectfifo.acquire' op must be called from inside a CoreOp

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile23 = aie.tile(2, 3)

   aie.objectfifo @of_0 (%tile12, {%tile23}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

   %elem = aie.objectfifo.acquire @of_0 (Produce) : memref<16xi32>
}

// -----

// CHECK: 'aie.objectfifo.acquire' op acquired object type 'memref<2x2xi32>' does not match the objectFifo's 'memref<16xi32>'

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile23 = aie.tile(2, 3)

   aie.objectfifo @of_0 (%tile12, {%tile23}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

   %core12 = aie.core(%tile12) {
      %elem = aie.objectfifo.acquire @of_0 (Produce) : memref<2x2xi32>

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
      %elem = aie.objectfifo.acquire @of_0 (Produce) : memref<16xi32>

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
      %elem = aie.objectfifo.acquire @of_0 (Consume) : memref<16xi32>

      aie.end
   }
}

// -----

// The pretty form cannot spell an acquire with no results, but a builder can.

// CHECK: 'aie.objectfifo.acquire' op must acquire at least one object

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile23 = aie.tile(2, 3)

   aie.objectfifo @of_0 (%tile12, {%tile23}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

   %core12 = aie.core(%tile12) {
      "aie.objectfifo.acquire"() <{objFifo_name = @of_0, port = 0 : i32}> : () -> ()

      aie.end
   }
}
