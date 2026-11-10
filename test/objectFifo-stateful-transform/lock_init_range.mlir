//===- lock_init_range.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --verify-diagnostics %s

// The shared pool of in0 takes its producer's 64 objects, and its producer
// lock would start at 64, past what an AIE2 lock holds.

module {
  aie.device(npu2_1col) {
    %a = aie.tile(0, 2)
    %l = aie.tile(0, 3)
    // expected-error@+1 {{init value 64 is outside the lock's range [0, 63]}}
    aie.objectfifo @in0(%a, {%l}, [64, 2]) : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
    aie.core(%l) {
      %0 = aie.objectfifo.acquire @in0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in0(Consume, 1)
      aie.end
    }
  }
}
