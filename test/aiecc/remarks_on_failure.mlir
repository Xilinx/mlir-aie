//===- remarks_on_failure.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A compile that fails prints the remarks it held back: @b_out pins a channel
// a DMA program already starts, and aie-objectfifo-allocate's remark on the
// allocation it fell back to comes with the error.

// RUN: not %aiecc -n --tmpdir %t --get-npu-insts --npu-insts-name=%t.bin %s 2>&1 | FileCheck %s

// CHECK: error: 'aie.route_endpoint' op pinned MM2S DMA channel 0 is out of range or already in use on this tile
// CHECK: remark: could not find a spill-aware allocation
// CHECK: aiecc: pipeline failed

module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mt = aie.tile(0, 1)
    aie.buffer @b(%mt) : memref<64xi32>
    aie.route_endpoint @b_out(%mt) DMA {channelIndex = 0 : i32}
    aie.route_endpoint @b_dst(%shim) DMA {fifoName = "b"}
    aie.route from @b_out to [@b_dst]
    aie.memtile_dma(%mt) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      %b = aie.get_buffer @b : memref<64xi32>
      aie.dma_bd(%b : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
  }
}
