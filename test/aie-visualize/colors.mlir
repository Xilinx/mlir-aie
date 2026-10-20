//===- colors.mlir ---------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A colour scheme names every colour the view draws in, so a dark page gets
// light labels and a tile outline that steps off the background rather than
// ruling a line across it.

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot %s | FileCheck %s --check-prefix=LIGHT
// RUN: aie-visualize --emit-dot --colors=mako %s | FileCheck %s --check-prefix=MAKO
// RUN: not aie-visualize --emit-dot --colors=puce %s 2>&1 | FileCheck %s --check-prefix=BAD

// LIGHT: bgcolor="white"
// LIGHT: tile_0_2 [shape=box{{.*}}fillcolor="#eef7ee"
// LIGHT-SAME: fontcolor="#555555"

// MAKO: bgcolor="#06080c"
// MAKO: tile_0_2 [shape=box{{.*}}style="rounded,filled"
// MAKO-SAME: fillcolor="#101722"
// MAKO-SAME: color="#1b2430"
// MAKO-SAME: fontcolor="#dfe8ee"

// BAD: --colors expects 'light' or 'mako'

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
