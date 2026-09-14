//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// verifyStridesWraps must reject tiles with no DMA hardware instead of
// silently deriving BD field widths for them. xcvc1902 row 0 outside the
// nocColumns set is ShimPLTile (AIETargetModel.h: "tiles with connections to
// the PL, no ShimDMA"), and --aie-dma-tasks-to-npu is one of the call sites
// with no AIE1 architecture gate.

// RUN: aie-opt --verify-diagnostics --split-input-file --aie-dma-tasks-to-npu %s

module {
  aie.device(xcvc1902) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error@+1 {{Unsupported tile type at (0, 0) Must be ShimNOC, Mem or Core.}}
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64 sizes = [1, 1, 1, 64] strides = [0, 0, 0, 1]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}
