//===- channel-arrows.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Channel arrows run in the gaps between tiles: ports move out of the tile
// onto a ring around it, each arrow gets its own geometry, and the graph is
// handed to graphviz already laid out.

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot %s | FileCheck %s --check-prefix=STRAIGHT
// RUN: aie-visualize --emit-dot --arrows=channel %s | FileCheck %s --check-prefix=CHANNEL
// RUN: not aie-visualize --emit-dot --arrows=diagonal %s 2>&1 | FileCheck %s --check-prefix=BADSTYLE

// Straight arrows leave the geometry to graphviz and pin nodes in inches.
// STRAIGHT: graph [layout=neato
// STRAIGHT: pos="{{[-0-9.]+}},{{[-0-9.]+}}!"
// STRAIGHT-NOT: -> {{.*}}pos="

// Channel arrows carry their own geometry, which only nop2 leaves alone, so
// positions switch to the points that engine reads.
// CHANNEL: graph [layout=nop2
// CHANNEL-NOT: pos="{{[-0-9.]+}},{{[-0-9.]+}}!"
// CHANNEL: -> {{.*}}pos="

// BADSTYLE: --arrows expects 'straight' or 'channel'

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %produced = aie.buffer(%t02) : memref<16xi32>
    %consumed = aie.buffer(%t03) : memref<16xi32>

    aie.mem(%t02) {
      %0 = aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd(%produced : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^end:
      aie.end
    }

    aie.mem(%t03) {
      %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd(%consumed : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^end:
      aie.end
    }

    aie.flow(%t02, DMA : 0, %t03, DMA : 0)
      via (%t02 : DMA : 0 -> North : 0,
           %t03 : South : 0 -> DMA : 0)
  }
}
