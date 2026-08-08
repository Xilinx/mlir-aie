//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --verify-diagnostics --aie-dma-tasks-to-npu %s

// This pass emits absolute tile coordinates, so it cannot lower a task whose
// tile is still an aie.logical_tile. Before the tryGetTileOp guard this call
// reached TileElement::getTileOp() and aborted the process with
// "LLVM ERROR: Calling getTileOp requires TileOp." instead of diagnosing.

module {
  aie.device(npu1) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)

    aie.runtime_sequence(%arg0: memref<32xi8>) {
      // expected-error@+1 {{Cannot lower a DMA task whose tile is not placed}}
      %t = aiex.dma_configure_task(%shim, MM2S, 0) {
          aie.dma_bd(%arg0 : memref<32xi8> offset = 0 len = 32) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}
