//===- test_place_pinned_channel_clash.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two logical shims pinned to (0, 0) send circuit fifos on pinned MM2S 0, which
// they cannot share wherever the rest goes: the sequential placer reports it
// and the SA charges it.

// RUN: not aie-opt --aie-place-tiles %s 2>&1 | FileCheck %s --check-prefix=SEQ
// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=1' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=SA

// SEQ: error: pins MM2S channel 0 of tile (0, 0), which @a pins there too, but a shared channel carries both at once, which only packet-switched routes tell apart
// SA: warning: SA placer's resource estimate is over capacity

module @pinned_clash {
  aie.device(npu2) {
    %s0 = aie.logical_tile<ShimNOCTile>(0, 0)
    %s1 = aie.logical_tile<ShimNOCTile>(0, 0)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%s0, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%s1, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
  }
}
