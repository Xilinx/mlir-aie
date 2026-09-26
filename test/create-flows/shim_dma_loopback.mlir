//===- shim_dma_loopback.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows %s | FileCheck %s

// A circuit flow from a shim DMA back into the same shim's S2MM passes both
// ends through the shim mux.

// CHECK-LABEL: module
// CHECK:         aie.switchbox(%shim_noc_tile_1_0) {
// CHECK-NEXT:      aie.connect<South : 7, South : 2>
// CHECK-NEXT:    }
// CHECK:         aie.shim_mux(%shim_noc_tile_1_0) {
// CHECK-DAG:       aie.connect<DMA : 1, North : 7>
// CHECK-DAG:       aie.connect<North : 2, DMA : 0>

module {
  aie.device(npu1) {
    %t10 = aie.tile(1, 0)
    aie.flow(%t10, DMA : 1, %t10, DMA : 0)
  }
}

// -----

// The same, fanning out to a core as well.

// CHECK-LABEL: module
// CHECK:         aie.switchbox(%shim_noc_tile_1_0) {
// CHECK-DAG:       aie.connect<South : 3, South : 3>
// CHECK-DAG:       aie.connect<South : 3, North : {{[0-5]}}>
// CHECK:         }
// CHECK:         aie.shim_mux(%shim_noc_tile_1_0) {
// CHECK-DAG:       aie.connect<DMA : 0, North : 3>
// CHECK-DAG:       aie.connect<North : 3, DMA : 1>
// CHECK:         aie.connect<{{.*}}, DMA : 0>

module {
  aie.device(npu1) {
    %t10 = aie.tile(1, 0)
    %t12 = aie.tile(1, 2)
    aie.flow(%t10, DMA : 0, %t10, DMA : 1)
    aie.flow(%t10, DMA : 0, %t12, DMA : 0)
  }
}
