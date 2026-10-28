//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --split-input-file %s

// AIE1 must not bypass ND DMA validation. These cases exercise the target
// model's shim limits without requiring an AIE board or an NPU lowering pass.

// AIE1 shim wrap maximum remains valid.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 0)
    aie.shim_dma_allocation @dma (%tile, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<?xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 2, 1023][0, 0, 2048, 1]) {id = 0 : i64, metadata = @dma} : memref<?xi32>
    }
  }
}

// -----

// Reject a wrap one past the 10-bit shim limit.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 0)
    aie.shim_dma_allocation @dma (%tile, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<?xi32>) {
      // expected-error@+1 {{Size 0 is 1024 (encoded as 1024), which exceeds the [0:1023] range.}}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 2, 1024][0, 0, 2048, 1]) {id = 0 : i64, metadata = @dma} : memref<?xi32>
    }
  }
}

// -----

// AIE1 shim step maximum remains valid (encoded as step minus one).
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 0)
    aie.shim_dma_allocation @dma (%tile, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<?xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 2, 8][0, 0, 1048576, 1]) {id = 0 : i64, metadata = @dma} : memref<?xi32>
    }
  }
}

// -----

// Reject a step one past the 20-bit shim limit.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 0)
    aie.shim_dma_allocation @dma (%tile, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<?xi32>) {
      // expected-error@+1 {{Stride 1 is 1048577 (encoded as 1048576), which exceeds the [1:1048576] range.}}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 2, 8][0, 0, 1048577, 1]) {id = 0 : i64, metadata = @dma} : memref<?xi32>
    }
  }
}

// -----

// AIE1 iteration maximum remains valid.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 0)
    aie.shim_dma_allocation @dma (%tile, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<?xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][64, 1, 2, 8][4096, 0, 16, 1]) {id = 0 : i64, metadata = @dma} : memref<?xi32>
    }
  }
}

// -----

// Reject iteration counts beyond the AIE1 6-bit field.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 0)
    aie.shim_dma_allocation @dma (%tile, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<?xi32>) {
      // expected-error@+1 {{iteration count 65 exceeds the [1:64] range.}}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][65, 1, 2, 8][4096, 0, 16, 1]) {id = 0 : i64, metadata = @dma} : memref<?xi32>
    }
  }
}

// -----

// Reject non-positive dimensions.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 0)
    aie.shim_dma_allocation @dma (%tile, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<?xi32>) {
      // expected-error@+1 {{Size 1 must be a positive integer.}}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 0, 8][0, 0, 16, 1]) {id = 0 : i64, metadata = @dma} : memref<?xi32>
    }
  }
}

// -----

// Linear transfers still bypass the ND wrap limit.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 0)
    aie.shim_dma_allocation @dma (%tile, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<?xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0, 1]) {id = 0 : i64, metadata = @dma} : memref<?xi32>
    }
  }
}

// -----

// Do not introduce the NPU-only byte-offset alignment restriction on AIE1.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 0)
    aie.shim_dma_allocation @dma (%tile, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<?xi8>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1][1, 1, 1, 8][0, 0, 0, 1]) {id = 0 : i64, metadata = @dma} : memref<?xi8>
    }
  }
}

// -----

// Coordinate-dependent AIE1 checks wait for logical tile placement.
module {
  aie.device(xcvc1902) {
    %tile = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.shim_dma_allocation @dma (%tile, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<?xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 2, 8][0, 0, 16, 1]) {id = 0 : i64, metadata = @dma} : memref<?xi32>
    }
  }
}
