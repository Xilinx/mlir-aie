//===- test_place_pinned_channel_clash_diag.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two logical shims each pin MM2S 0 to a circuit-switched fifo on a device
// with one shim: the second finds the first's pin there. The sequential
// placer names that clash, not a merge setting.

// RUN: not aie-opt --aie-place-tiles %s 2>&1 | FileCheck %s

// CHECK: error: no ShimNOCTile with DMA channels to spare can hold this tile's pinned ends beside the ends pinned there
// CHECK: note: @b pins MM2S channel 0, which @a pins on (0, 0) too, but a shared channel carries both at once, which only packet-switched routes tell apart; 1 ShimNOCTile(s) hold such a pin
// CHECK-NOT: merge-logical-tiles

module @pinned_clash_diag {
  aie.device(npu2_1col) {
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    %d = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%s0, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%s1, {%d}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
    aie.core(%d) {
      %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
  }
}
