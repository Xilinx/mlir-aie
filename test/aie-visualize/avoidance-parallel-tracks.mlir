//===- avoidance-parallel-tracks.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot --arrows=avoidance --contract=port %s -o %t
// RUN: %python %S/Inputs/check-arrow-separation.py %t | FileCheck %s --check-prefix=CLEAN
// RUN: env AIE_VISUALIZE_GRAPH_STATS=1 aie-visualize --emit-dot --arrows=avoidance --contract=port %s -o %t 2>&1 | FileCheck %s --check-prefix=STATS
// RUN: not %python %S/Inputs/check-arrow-separation.py %S/Inputs/overlapping-arrows.dot 2>&1 | FileCheck %s --check-prefix=OVERLAP
// RUN: not %python %S/Inputs/check-arrow-separation.py %S/Inputs/narrow-gap-arrows.dot 2>&1 | FileCheck %s --check-prefix=NARROW

// CLEAN: collinear_overlaps=0 narrow_gaps=0 diagonals=0
// STATS: graph nodes=10 edges=9 contracted=9
// STATS: avoidance requests=9 obstacles=10 grid=23x92 states=8464 route_calls={{[0-9]+}} expanded={{[0-9]+}} grid_ms={{[0-9.e+-]+}} initial_ms={{[0-9.e+-]+}} conflict_ms={{[0-9.e+-]+}}
// OVERLAP: collinear_overlaps=1 narrow_gaps=2 diagonals=0 crossings=3
// NARROW: collinear_overlaps=0 narrow_gaps=1 diagonals=0 crossings=0

module {
  aie.device(npu1_1col) {
    %t0 = aie.tile(0, 0)
    %t1 = aie.tile(0, 1)
    %t2 = aie.tile(0, 2)
    %t3 = aie.tile(0, 3)

    aie.flow(%t0, DMA : 0, %t3, DMA : 0)
    aie.flow(%t0, DMA : 1, %t3, DMA : 1)
    aie.flow(%t1, DMA : 0, %t2, DMA : 0)
  }
}