// RUN: aie-opt --aie-objectfifo-split %s -o %t1.mlir
// RUN: aie-opt --aie-objectfifo-split %t1.mlir -o %t2.mlir
// RUN: diff %t1.mlir %t2.mlir

// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Splitting IR that carries pools and endpoints leaves it alone: the pass only
// consumes what it has not yet lowered.

module @idempotent {
  aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @shared (%tile22, {%tile23}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in0 (%tile22, {%tile21}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in1 (%tile23, {%tile21}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out (%tile21, {%tile20}, 2 : i32) : !aie.objectfifo<memref<48xi32>>

    aie.objectfifo.link [@in0, @in1] -> [@out] ([0, 16][])

    %core22 = aie.core(%tile22) {
      %elem = aie.objectfifo.acquire @shared(Produce) : memref<16xi32>
      aie.objectfifo.release @shared(Produce) [1]
      aie.end
    }
  }
}
