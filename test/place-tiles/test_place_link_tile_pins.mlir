//===- test_place_link_tile_pins.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two logical mem tiles each link a chain whose source fifo pins the mem
// tile's S2MM 0 in order: they cannot share one mem tile. Both placers keep
// them apart, and the split and allocation go through.

// RUN: aie-opt --verify-diagnostics --aie-place-tiles --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=1' --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s

// CHECK-DAG: aie.tile(0, 1)
// CHECK-DAG: aie.tile(1, 1)

module @link_tile_pins {
  aie.device(npu2_2col) {
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %m0 = aie.logical_tile<MemTile>(?, ?)
    %m1 = aie.logical_tile<MemTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in0(%s0, {%m0}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>]} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out0(%m0, {%c0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in0] -> [@out0] ([] [])
    aie.objectfifo @in1(%s1, {%m1}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>]} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out1(%m1, {%c1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in1] -> [@out1] ([] [])
    aie.core(%c0) {
      %0 = aie.objectfifo.acquire @out0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @out0(Consume, 1)
      aie.end
    }
    aie.core(%c1) {
      %0 = aie.objectfifo.acquire @out1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @out1(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @run(%x : memref<16xi32>, %y : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in0, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%y[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in1, id = 1 : i64} : memref<16xi32>
    }
  }
}
