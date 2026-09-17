// RUN: aie-opt --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --aie-objectfifo-allocate --aie-place-tiles --aie-objectfifo-lower-dmas --aie-assign-lock-ids --aie-assign-buffer-addresses %s -o /dev/null

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Known equal coordinates guarantee locality. Placement resolves the two SSA
// tile references before the DMA verifier requires a canonical tile operand.
module {
  aie.device(npu2) {
    %home = aie.tile(0, 1)
    %reader = aie.logical_tile<MemTile>(0, 1)
    aie.objectfifo.pool @shared(%home) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @reader(%reader) drains @shared {channelIndex = 5 : i32}
  }
}
// CHECK: %[[HOME:.*]] = aie.tile(0, 1)
// CHECK: aie.buffer(%[[HOME]]) {sym_name = "shared_buff_0"}
// CHECK: @reader({{.*}}) drains @shared {channelIndex = 5 : i32}
