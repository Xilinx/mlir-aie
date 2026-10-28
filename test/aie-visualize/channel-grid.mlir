//===- channel-grid.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Contracting the hops away leaves arrows between tiles that are neither in
// the same row nor the same column. Such an arrow runs down the column
// channel of the tile it leaves, across a row channel, and up the column
// channel of the tile it reaches, so every straight piece of it lies in a
// corridor and takes a track there. Drawn as one straight run it would cross
// whatever lies between the two tiles.

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot --arrows=channel --contract=port,dma,buffer %s | FileCheck %s

// Two column channels and one row channel, so the arrow turns four times: it
// repeats one x, then one y, then another x.
// CHECK: buffer_0 -> buffer_1
// CHECK-SAME: pos="{{[^"]*}}[[A:[-0-9.]+]],{{[-0-9.]+}} [[A]],{{[^"]*}},[[B:[-0-9.]+]] {{[-0-9.]+}},[[B]]{{[^"]*}}[[C:[-0-9.]+]],{{[-0-9.]+}} [[C]],

module {
  aie.device(npu1) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t11 = aie.tile(1, 1)
    %t21 = aie.tile(2, 1)
    %t22 = aie.tile(2, 2)

    %out = aie.buffer(%t03) {sym_name = "out"} : memref<16xi32>
    %in = aie.buffer(%t22) {sym_name = "in"} : memref<16xi32>

    aie.mem(%t03) {
      %0 = aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd(%out : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd
    ^end:
      aie.end
    }

    aie.mem(%t22) {
      %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd(%in : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd
    ^end:
      aie.end
    }

    aie.flow(%t03, DMA : 0, %t22, DMA : 0)
      via (%t03 : DMA : 0 -> South : 0,
           %t02 : North : 0 -> South : 0,
           %t01 : North : 0 -> East : 0,
           %t11 : West : 0 -> East : 0,
           %t21 : West : 0 -> North : 0,
           %t22 : South : 0 -> DMA : 0)
  }
}
