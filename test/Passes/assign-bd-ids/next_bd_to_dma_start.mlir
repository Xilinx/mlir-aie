//===- next_bd_to_dma_start.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-bd-ids --verify-diagnostics %s

// A BD chain can go on to another BD or end; it cannot run into the next
// channel's dma_start.

module {
  aie.device(npu1) {
    %tile_0_1 = aie.tile(0, 1)
    %buf0 = aie.buffer(%tile_0_1) : memref<16xi32>
    %buf1 = aie.buffer(%tile_0_1) : memref<16xi32>
    %mem = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^out)
    ^bd0:
      // expected-error@+1 {{must be followed by a block with a dma_bd or by a block with only aie.end}}
      aie.dma_bd(%buf0 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^out
    ^out:
      %1 = aie.dma_start(MM2S, 0, ^bd1, ^end)
    ^bd1:
      aie.dma_bd(%buf1 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
