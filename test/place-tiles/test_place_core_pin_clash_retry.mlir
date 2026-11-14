//===- test_place_core_pin_clash_retry.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A logical core whose fifo pins MM2S 0 skips a compute tile where another
// circuit-switched end pins MM2S 0 and takes the next one: (0, 2) holds such
// a pin, so the core goes to (0, 3), as the SA puts it. When every compute
// tile holds one: test_place_core_pin_clash_none.mlir.

// RUN: aie-opt --verify-diagnostics --aie-place-tiles %s | FileCheck %s
// RUN: aie-opt --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=1' %s | FileCheck %s

// CHECK: %[[T:.*]] = aie.tile(0, 3)
// CHECK: aie.objectfifo @q(%[[T]], {

module @next_tile {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    %l = aie.logical_tile<CoreTile>(?, ?)
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @p(%t, {%s0}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @q(%l, {%s1}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.core(%l) {
      %0 = aie.objectfifo.acquire @q(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @q(Produce, 1)
      aie.end
    }
  }
}
