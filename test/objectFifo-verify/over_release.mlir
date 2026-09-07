//===- over_release.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A loop body that releases more than it acquires underflows the held count as
// it repeats, whatever the trip count.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-objectfifo-split --aie-objectfifo-verify %s

module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @fifo_in(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        // expected-error@+1 {{cannot release more elements than are already acquired}}
        %1 = aie.objectfifo.acquire @fifo_in (Consume, 1) : memref<32x32xi32>
        aie.objectfifo.release @fifo_in (Consume, 1)
        aie.objectfifo.release @fifo_in (Consume, 1)
      }
      aie.end
    }
  }
}

// -----

// Acquiring as many as are released each trip is the ordinary case.

module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @fifo_in(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %1 = aie.objectfifo.acquire @fifo_in (Consume, 1) : memref<32x32xi32>
        aie.objectfifo.release @fifo_in (Consume, 1)
      }
      aie.end
    }
  }
}
