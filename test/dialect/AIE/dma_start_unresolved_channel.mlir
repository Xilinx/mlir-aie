//===- dma_start_unresolved_channel.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A dma_start naming a route endpoint has no channel index until
// aie-objectfifo-allocate assigns one. The passes and translations that need
// the index reject such a start instead of reading one that is not there.

// RUN: aie-opt --aie-assign-bd-ids --verify-diagnostics %s
// RUN: aie-opt --aie-create-pathfinder-flows --verify-diagnostics %s
// RUN: aie-opt --convert-aie-to-transaction --verify-diagnostics %s
// RUN: aie-translate --aie-generate-xaie --verify-diagnostics %s
// RUN: aie-translate --aie-mlir-to-shim-solution --verify-diagnostics %s

aie.device(npu2) {
  %mt = aie.tile(0, 1)
  %b = aie.buffer(%mt) {sym_name = "b"} : memref<64xi32>
  aie.route_endpoint @out(%mt) DMA
  aie.memtile_dma(%mt) {
    // expected-error@+1 {{names route endpoint @out in place of a channel index; run --aie-objectfifo-allocate to assign one}}
    aie.dma_start(MM2S, @out, ^bd0, ^end)
  ^bd0:
    aie.dma_bd(%b : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^bd0
  ^end:
    aie.end
  }
}
