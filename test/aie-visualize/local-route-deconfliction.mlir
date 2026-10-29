//===- local-route-deconfliction.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot --arrows=avoidance --contract=port %s -o %t.dot
// RUN: %python %S/Inputs/check-arrow-separation.py %t.dot | FileCheck %s --check-prefix=DOT
// RUN: neato -n2 -Gsplines=line -Tsvg %t.dot -o %t.svg
// RUN: %python %S/Inputs/check-svg-render-geometry.py %t.svg --dot %t.dot --allow-defects | FileCheck %s --check-prefix=SVG

// DOT: missing_geometry=0 collinear_overlaps=0 narrow_gaps=0 diagonals=0 crossings=4
// SVG: edges=24 diagonals=0 curved_bends={{[0-9]+}} node_intersections=0 collinear_overlaps=0 narrow_gaps=0

module {
  aie.device(npu1_2col) {
    %t0_0 = aie.tile(0, 0)
    %t0_1 = aie.tile(0, 1)
    %t0_2 = aie.tile(0, 2)
    %t0_3 = aie.tile(0, 3)
    %t1_0 = aie.tile(1, 0)
    %t1_1 = aie.tile(1, 1)
    %t1_2 = aie.tile(1, 2)
    %t1_3 = aie.tile(1, 3)

    aie.flow(%t0_0, DMA : 1, %t0_1, DMA : 0)
    aie.flow(%t0_1, DMA : 1, %t0_2, DMA : 1)
    aie.flow(%t0_2, DMA : 0, %t1_0, DMA : 0)
    aie.flow(%t0_3, DMA : 0, %t1_1, DMA : 1)
    aie.flow(%t1_0, DMA : 1, %t1_3, DMA : 0)
    aie.flow(%t1_2, DMA : 1, %t0_0, DMA : 1)
    aie.flow(%t1_3, DMA : 0, %t0_2, DMA : 0)
    aie.flow(%t0_2, DMA : 1, %t1_3, DMA : 1)
  }
}