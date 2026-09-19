// RUN: aie-opt --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --aie-objectfifo-allocate --aie-place-tiles %s -o /dev/null

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Both physical and logical shim allocation operands seed the same hardware
// reservation table as endpoints on coordinate-equivalent tile references.
module {
  aie.device(npu2) {
    %home = aie.tile(0, 0)
    %alias = aie.logical_tile<ShimNOCTile>(0, 0)
    aie.shim_dma_allocation @read(%home, MM2S, 0)
    aie.shim_dma_allocation @write(%alias, S2MM, 0)
    aie.objectfifo.pool @p(%home) {depth = 0 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @reader(%alias) drains @p
    aie.objectfifo.dma_endpoint @writer(%home) fills @p
  }
}
// CHECK: @reader({{.*}}) drains @p {channelIndex = 1 : i32}
// CHECK: @writer({{.*}}) fills @p {channelIndex = 1 : i32}
