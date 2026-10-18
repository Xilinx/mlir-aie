//===- shim_dma_behind_fixed_masterset.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-create-pathfinder-flows %s 2>&1 | FileCheck %s

// Shim DMA:1 is reached on South:3 through the shim mux, and a fixed masterset
// already owns South:3, so no flow can reach DMA:1. Reduced from
// router_properties.py seed 772.

// CHECK: error: Unable to find a legal routing: no path leads from (0, 3) DMA:0 to (0, 0) DMA:1

module {
  aie.device(npu1_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_3 = aie.tile(0, 3)
    %sb_0_0 = aie.switchbox(%t_0_0) {
      %a = aie.amsel<0> (0)
      %m = aie.masterset(South : 3, %a)
    }
    aie.packet_flow(4) { aie.packet_source<%t_0_3, DMA : 0> aie.packet_dest<%t_0_0, DMA : 1> }
  }
}

// -----

// DMA:0 is on South:2, which is still free.

// CHECK-LABEL: aie.device(npu1_1col)
// CHECK:         aie.shim_mux(%{{.*}}) {
// CHECK:           aie.connect<North : 2, DMA : 0>
// CHECK:         aie.switchbox(%{{.*}}) {
// CHECK-DAG:       aie.masterset(South : 3,
// CHECK-DAG:       aie.masterset(South : 2,

module {
  aie.device(npu1_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_3 = aie.tile(0, 3)
    %sb_0_0 = aie.switchbox(%t_0_0) {
      %a = aie.amsel<0> (0)
      %m = aie.masterset(South : 3, %a)
    }
    aie.packet_flow(4) { aie.packet_source<%t_0_3, DMA : 0> aie.packet_dest<%t_0_0, DMA : 0> }
  }
}
