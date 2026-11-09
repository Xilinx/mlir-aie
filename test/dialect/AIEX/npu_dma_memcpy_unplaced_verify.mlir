//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --split-input-file %s

// Static and SSA-sized transfers must defer coordinate-dependent checks until
// logical tiles are placed, on every architecture.
module {
  aie.device(xcvc1902) {
    %tile = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.shim_dma_allocation @dma (%tile, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<?xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 2, 8][0, 0, 16, 1]) {id = 0 : i64, metadata = @dma} : memref<?xi32>
    }
  }
}

// -----

module {
  aie.device(xcvc1902) {
    %tile = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.shim_dma_allocation @dma (%tile, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<?xi32>, %size: i64) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, %size, 8][0, 0, 16, 1]) {id = 0 : i64, metadata = @dma} : memref<?xi32>
    }
  }
}

// -----

module {
  aie.device(npu1) {
    %tile = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.shim_dma_allocation @dma (%tile, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<?xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 2, 8][0, 0, 16, 1]) {id = 0 : i64, metadata = @dma} : memref<?xi32>
    }
  }
}

// -----

module {
  aie.device(npu1) {
    %tile = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.shim_dma_allocation @dma (%tile, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<?xi32>, %size: i64) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, %size, 8][0, 0, 16, 1]) {id = 0 : i64, metadata = @dma} : memref<?xi32>
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %tile = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.shim_dma_allocation @dma (%tile, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<?xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 2, 8][0, 0, 16, 1]) {id = 0 : i64, metadata = @dma} : memref<?xi32>
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %tile = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.shim_dma_allocation @dma (%tile, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<?xi32>, %size: i64) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, %size, 8][0, 0, 16, 1]) {id = 0 : i64, metadata = @dma} : memref<?xi32>
    }
  }
}

// -----

// A placed tile still receives the runtime transfer's coordinate checks.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @dma (%tile, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<?xi32>, %size: i64) {
      // expected-error@+1 {{runtime sizes/strides are only supported for shim NOC tile DMAs.}}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, %size, 8][0, 0, 16, 1]) {id = 0 : i64, metadata = @dma} : memref<?xi32>
    }
  }
}
