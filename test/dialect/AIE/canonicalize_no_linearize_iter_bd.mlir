//===- canonicalize_no_linearize_iter_bd.mlir -------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Regression tests for LinearizeContiguousBDTransfer: BDs with an iteration
// dimension (outermost dim) whose explicit len covers only the inner dims
// must NOT be linearized.  Linearizing would remove the iteration stride,
// causing the hardware to repeatedly transfer only the first `len` elements
// from offset 0 instead of sweeping across the full buffer.
//
// The failing case was exposed by IRON transpose[M_2048-N_64-m_64-n_64-s_8]:
// the input fill BD has a contiguous 4D pattern where sizes[0] (outermost)
// controls the iteration count and len = product of inner 3 sizes.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --canonicalize --split-input-file %s | FileCheck %s

// -----

// 4D contiguous BD with explicit per-iteration len (4096) != product of all
// dims (32*1*64*64 = 131072).  The outermost dim is a BD iteration dimension.
// Must NOT be linearized — the iteration stride (4096) would be lost.
//
// This is the input fill pattern from transpose[M_2048-N_64-m_64-n_64-s_8].

// CHECK-LABEL: aie.device(npu1)
// CHECK:       aiex.dma_configure_task
// CHECK:         aie.dma_bd
// CHECK-SAME:      [<size = 32, stride = 4096>, <size = 1, stride = 64>, <size = 64, stride = 64>, <size = 64, stride = 1>]
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence @iter_bd_no_linearize(%arg0 : memref<131072xbf16>) {
      %t = aiex.dma_configure_task_for(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<131072xbf16>, 0, 4096,
          [<size = 32, stride = 4096>, <size = 1, stride = 64>,
           <size = 64, stride = 64>, <size = 64, stride = 1>]) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 31 : i32}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
    aie.shim_dma_allocation @of_fromMem (%tile_0_0, MM2S, 0)
  }
}

// -----

// Same pattern inside a ShimDMAOp: 4D contiguous with per-iteration len.
// Must NOT be linearized.

// CHECK-LABEL: aie.device(npu1)
// CHECK:       aie.shim_dma
// CHECK:         aie.dma_bd
// CHECK-SAME:      [<size = 32, stride = 4096>, <size = 1, stride = 64>, <size = 64, stride = 64>, <size = 64, stride = 1>]
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %buf = aie.external_buffer { sym_name = "buf_iter" } : memref<131072xbf16>
    aie.shim_dma(%tile_0_0) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%buf : memref<131072xbf16>, 0, 4096,
          [<size = 32, stride = 4096>, <size = 1, stride = 64>,
           <size = 64, stride = 64>, <size = 64, stride = 1>])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Positive regression: a 4D contiguous BD where len == product of all dims
// (len = 131072 = 32*1*64*64).  This SHOULD still be linearized.

// CHECK-LABEL: aie.device(npu1)
// CHECK:       aie.shim_dma
// CHECK:         aie.dma_bd
// CHECK-NOT:       dimensions
// CHECK-NOT:       [<
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %buf = aie.external_buffer { sym_name = "buf_full" } : memref<131072xbf16>
    aie.shim_dma(%tile_0_0) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%buf : memref<131072xbf16>, 0, 131072,
          [<size = 32, stride = 4096>, <size = 1, stride = 64>,
           <size = 64, stride = 64>, <size = 64, stride = 1>])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Positive regression: a 4D contiguous BD with NO explicit len.
// This SHOULD still be linearized (len computed from product).

// CHECK-LABEL: aie.device(npu1)
// CHECK:       aie.shim_dma
// CHECK:         aie.dma_bd
// CHECK-NOT:       dimensions
// CHECK-NOT:       [<
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %buf = aie.external_buffer { sym_name = "buf_nolen" } : memref<131072xbf16>
    aie.shim_dma(%tile_0_0) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%buf : memref<131072xbf16>, 0,
          [<size = 32, stride = 4096>, <size = 1, stride = 64>,
           <size = 64, stride = 64>, <size = 64, stride = 1>])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Positive regression: 2D contiguous BD from the original PR #2924 test set.
// Ensures existing linearization behavior is preserved for simple cases.

// CHECK-LABEL: aie.device(npu1)
// CHECK:       aie.shim_dma
// CHECK:         aie.dma_bd
// CHECK-NOT:       dimensions
// CHECK-NOT:       [<
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %buf = aie.external_buffer { sym_name = "buf_2d" } : memref<1024xi32>
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
