//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A shim BD on the circuit-switched path (under aie.shim_dma) supports 3 ND
// access dimensions -- getBDMaxDims(ShimNOCTile). This differs from the
// runtime-sequence task path, where the leading tap dimension is hoisted into
// the shim iteration register and 4 dimensions are allowed
// (see test/dialect/AIEX/dma_task_bd_too_many_dims.mlir).

// RUN: aie-opt --verify-diagnostics %s

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma(%tile_0_0) {
      %srcDma = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      %buf = aie.external_buffer { sym_name = "buf00" } : memref<128xi32>
      // expected-error@+1 {{Cannot give more than 3 dimensions}}
      aie.dma_bd(%buf : memref<128xi32> offset = 0 len = 128 sizes = [1, 1, 1, 1] strides = [1, 1, 1, 1])
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
