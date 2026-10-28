//===- requires-vias.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Physical views require route vias, while contracted semantic views use endpoints.

// REQUIRES: aie-visualize
// RUN: not aie-visualize --emit-dot --scale=1 %s 2>&1 | FileCheck %s
// RUN: aie-visualize --emit-dot --scale=1 --contract=port,buffer %s | FileCheck %s --check-prefix=TOPOLOGY

// CHECK: error: flow carries no vias, so its route cannot be drawn

// TOPOLOGY: graph [layout=neato, overlap=true, splines=curved
// TOPOLOGY: tile_0_2 {{.*}} pos="0.000000,6.000000!"
// TOPOLOGY: tile_0_3 {{.*}} pos="0.000000,9.000000!"
// Contracting ports keeps the DMA channels a flow leaves and arrives on.
// TOPOLOGY-DAG: tile_0_2 -> p_0_2_1_0_m
// TOPOLOGY-DAG: p_0_2_1_0_m -> p_0_3_1_0_s
// TOPOLOGY-DAG: p_0_3_1_0_s -> tile_0_3

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.flow(%t02, DMA : 0, %t03, DMA : 0)
  }
}
