//===- requires-vias.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aie-visualize
// RUN: not aie-visualize --emit-dot %s 2>&1 | FileCheck %s
// RUN: aie-visualize --emit-dot --topology-only %s | FileCheck %s --check-prefix=TOPOLOGY

// CHECK: error: 'aie.flow' op requires vias; run aie-find-flows with emit-vias=true

// TOPOLOGY: graph [layout=neato, overlap=true, splines=curved
// TOPOLOGY: tile_0_2 {{.*}} pos="0.000000,6.000000!"
// TOPOLOGY: tile_0_3 {{.*}} pos="0.000000,9.000000!"
// TOPOLOGY: tile_0_2 -> tile_0_3
// TOPOLOGY-NOT: p_0_2_1_0_m

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.flow(%t02, DMA : 0, %t03, DMA : 0)
  }
}
