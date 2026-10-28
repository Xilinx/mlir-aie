//===- avoidance-parallel-tracks.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot --arrows=avoidance --contract=port %s -o %t
// RUN: %python %S/Inputs/check-arrow-separation.py %t
// RUN: not %python %S/Inputs/check-arrow-separation.py %S/Inputs/overlapping-arrows.dot 2>&1 | FileCheck %s --check-prefix=OVERLAP

// OVERLAP: collinear_overlaps=1 narrow_gaps=2 crossings=3

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