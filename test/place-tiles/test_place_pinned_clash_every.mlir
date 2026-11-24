//===- test_place_pinned_clash_every.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles %s

// Shim tiles %x and %y are both pinned to (0, 0). @f1 and @f2 pin %x's MM2S
// 0 and 1, and @g1 and @g2 pin %y's MM2S 0 and 1 to circuit-switched fifos,
// which cannot share a channel. The sequential placer reports both of %y's
// clashes, whichever of its fifos is declared first.

module @g1_first {
  aie.device(npu2) {
    %x = aie.logical_tile<ShimNOCTile>(0, 0)
    %y = aie.logical_tile<ShimNOCTile>(0, 0)
    %c = aie.logical_tile<CoreTile>(0, 2)
    %d = aie.logical_tile<CoreTile>(0, 3)
    aie.objectfifo @f1(%x, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f2(%x, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 1>} : !aie.objectfifo<memref<16xi32>>
    // expected-error@+1 {{pins MM2S channel 0 of tile (0, 0), which @f1 pins there too, but a shared channel carries both at once, which only packet-switched routes tell apart}}
    aie.objectfifo @g1(%y, {%d}, 3 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<8xi32>>
    // expected-error@+1 {{pins MM2S channel 1 of tile (0, 0), which @f2 pins there too, but a shared channel carries both at once, which only packet-switched routes tell apart}}
    aie.objectfifo @g2(%y, {%d}, 3 : i32) {prod_port = #aie.end_port<DMA : 1>} : !aie.objectfifo<memref<8xi32>>
    aie.core(%c) {
      %a = aie.objectfifo.acquire @f1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f1(Consume, 1)
      %b = aie.objectfifo.acquire @f2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f2(Consume, 1)
      aie.end
    }
    aie.core(%d) {
      %a = aie.objectfifo.acquire @g1(Consume, 1) : memref<8xi32>
      aie.objectfifo.release @g1(Consume, 1)
      %b = aie.objectfifo.acquire @g2(Consume, 1) : memref<8xi32>
      aie.objectfifo.release @g2(Consume, 1)
      aie.end
    }
  }
}

// -----

module @g2_first {
  aie.device(npu2) {
    %x = aie.logical_tile<ShimNOCTile>(0, 0)
    %y = aie.logical_tile<ShimNOCTile>(0, 0)
    %c = aie.logical_tile<CoreTile>(0, 2)
    %d = aie.logical_tile<CoreTile>(0, 3)
    aie.objectfifo @f1(%x, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f2(%x, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 1>} : !aie.objectfifo<memref<16xi32>>
    // expected-error@+1 {{pins MM2S channel 1 of tile (0, 0), which @f2 pins there too, but a shared channel carries both at once, which only packet-switched routes tell apart}}
    aie.objectfifo @g2(%y, {%d}, 3 : i32) {prod_port = #aie.end_port<DMA : 1>} : !aie.objectfifo<memref<8xi32>>
    // expected-error@+1 {{pins MM2S channel 0 of tile (0, 0), which @f1 pins there too, but a shared channel carries both at once, which only packet-switched routes tell apart}}
    aie.objectfifo @g1(%y, {%d}, 3 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<8xi32>>
    aie.core(%c) {
      %a = aie.objectfifo.acquire @f1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f1(Consume, 1)
      %b = aie.objectfifo.acquire @f2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f2(Consume, 1)
      aie.end
    }
    aie.core(%d) {
      %a = aie.objectfifo.acquire @g1(Consume, 1) : memref<8xi32>
      aie.objectfifo.release @g1(Consume, 1)
      %b = aie.objectfifo.acquire @g2(Consume, 1) : memref<8xi32>
      aie.objectfifo.release @g2(Consume, 1)
      aie.end
    }
  }
}
