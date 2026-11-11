//===- test_place_pinned_on_claimed.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A shim allocation holds MM2S 0 of shim (0, 0) by index; a logical shim's
// circuit-switched fifo pins MM2S 0. Allocation refuses that pin where the
// allocation sits, so both placers put the logical shim on column 1.

// RUN: aie-opt --verify-diagnostics --aie-place-tiles --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=1' --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=2' --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s

// CHECK: aie.shim_dma_allocation @a__shim_alloc(%{{.*}}tile_1_0, MM2S, 0)

module @pinned_on_claimed {
  aie.device(npu2_2col) {
    %home = aie.tile(0, 0)
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %c = aie.tile(0, 2)
    aie.shim_dma_allocation @reserved(%home, MM2S, 0)
    aie.objectfifo @a(%s, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      %0 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
  }
}
