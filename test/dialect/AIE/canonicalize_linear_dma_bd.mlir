//===- canonicalize_linear_dma_bd.mlir -------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Tests for DMABDOp canonicalization: LinearizeContiguousBDTransfer folds
// contiguous row-major ND access patterns inside ShimDMAOp bodies into
// canonical linear form (no dimensions attribute).
//
// The same contiguous condition as NpuDmaMemcpyNdOp applies:
//   innermost stride == 1, each outer stride == product of inner sizes.
//
// Note: ShimDMAOp allows at most 3 dimensions; stride == 0 is invalid in
// aie.dma_bd regardless of the corresponding size value.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --canonicalize --split-input-file %s | FileCheck %s

// -----

// Basic 2D fold inside a ShimDMAOp: [2, 512][512, 1] -> linear (no dims).

// CHECK-LABEL: aie.device(npu1)
// CHECK:       aie.shim_dma
// CHECK:         aie.dma_bd
// CHECK-NOT:       dimensions
// CHECK-NOT:       [<
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %buf = aie.external_buffer { sym_name = "buf0" } : memref<1024xi32>
    aie.shim_dma(%tile_0_0) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%buf : memref<1024xi32>, 0, 1024,
          [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// 3D fold: [4, 8, 32][256, 32, 1] -> linear (no dims).

// CHECK-LABEL: aie.device(npu1)
// CHECK:       aie.shim_dma
// CHECK:         aie.dma_bd
// CHECK-NOT:       dimensions
// CHECK-NOT:       [<
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %buf = aie.external_buffer { sym_name = "buf1" } : memref<1024xi32>
    aie.shim_dma(%tile_0_0) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%buf : memref<1024xi32>, 0, 1024,
          [<size = 4, stride = 256>, <size = 8, stride = 32>, <size = 32, stride = 1>])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Already linear (no dims): must NOT be touched (idempotent).

// CHECK-LABEL: aie.device(npu1)
// CHECK:       aie.shim_dma
// CHECK:         aie.dma_bd
// CHECK-NOT:       dimensions
// CHECK-NOT:       [<
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %buf = aie.external_buffer { sym_name = "buf2" } : memref<4096xi32>
    aie.shim_dma(%tile_0_0) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%buf : memref<4096xi32>, 0, 4096)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Regression: packet attribute must be preserved when folding to linear form.
// A BD with packet routing info that is contiguous must still carry its
// packet header after canonicalization.  Before the fix, replaceOpWithNewOp
// was called with only (buffer, offset, len), silently dropping the packet
// attribute and breaking packet-switched designs at runtime.

// CHECK-LABEL: aie.device(npu1)
// CHECK:       aie.shim_dma
// CHECK:         aie.dma_bd
// CHECK-NOT:       dimensions
// CHECK-SAME:      packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %buf = aie.external_buffer { sym_name = "buf_pkt" } : memref<1024xi32>
    aie.shim_dma(%tile_0_0) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%buf : memref<1024xi32>, 0, 1024,
          [<size = 2, stride = 512>, <size = 512, stride = 1>]) {
            packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Non-contiguous (stride != size product): must NOT be folded.

// CHECK-LABEL: aie.device(npu1)
// CHECK:       aie.shim_dma
// CHECK:         aie.dma_bd
// CHECK-SAME:      [<size = 2, stride = 513>
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %buf = aie.external_buffer { sym_name = "buf3" } : memref<1200xi32>
    aie.shim_dma(%tile_0_0) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        // stride 513 != product 512: genuinely strided, must not linearize.
        // max index = 1*513 + 511*1 = 1024 < 1200 (in bounds).
        aie.dma_bd(%buf : memref<1200xi32>, 0, 1024,
          [<size = 2, stride = 513>, <size = 512, stride = 1>])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Large image: d0=1920 and d1=1080 both exceed the 1023 ND wrap limit.
// The access is contiguous (1080 rows x 1920 i32 words per row) so it
// folds to linear form, making both limits irrelevant.
//
// This is the motivating case for color_detect / 8k vision passthrough:
// a 1920x1080 RGBA frame expressed in i32 words.

// CHECK-LABEL: aie.device(npu1)
// CHECK:       aie.shim_dma
// CHECK:         aie.dma_bd
// CHECK-NOT:       dimensions
// CHECK-NOT:       [<
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %buf = aie.external_buffer { sym_name = "buf4" } : memref<2073600xi32>
    aie.shim_dma(%tile_0_0) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        // 1080 x 1920: d0=1920 > 1023, d1=1080 > 1023, but contiguous so
        // LinearizeContiguousBDTransfer folds it to linear mode.
        aie.dma_bd(%buf : memref<2073600xi32>, 0, 2073600,
          [<size = 1080, stride = 1920>, <size = 1920, stride = 1>])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
