//===- test_place_core_pin_clash_none.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Every compute tile holds a circuit-switched end on pinned MM2S 0, so a
// logical core whose fifo pins MM2S 0 fits none: the sequential placer names
// the clash at the first tile it fits otherwise.

// RUN: aie-opt --verify-diagnostics --aie-place-tiles %s

module @no_tile {
  aie.device(npu2_1col) {
    %t2 = aie.tile(0, 2)
    %t3 = aie.tile(0, 3)
    %t4 = aie.tile(0, 4)
    %t5 = aie.tile(0, 5)
    // expected-error@+2 {{no available compute tiles for placement}}
    // expected-note@+1 {{on (0, 2), the first compute tile it fits otherwise, @q pins MM2S channel 0, which @p2 pins there too, but a shared channel carries both at once, which only packet-switched routes tell apart}}
    %l = aie.logical_tile<CoreTile>(?, ?)
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @p2(%t2, {%s}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @p3(%t3, {%s}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @p4(%t4, {%s}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @p5(%t5, {%s}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @q(%l, {%s}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.core(%l) {
      %0 = aie.objectfifo.acquire @q(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @q(Produce, 1)
      aie.end
    }
  }
}
