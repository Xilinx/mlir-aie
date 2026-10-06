//===- shim_dma_existing_rules.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Shim DMA 0 enters the switchbox on South 3 through the shim mux. Flow 20
// starts there too, so its rule joins the rules already on South 3 and the
// shim mux keeps its one connect.

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK:      aie.shim_mux
// CHECK-NEXT:   aie.connect<DMA : 0, North : 3>
// CHECK-NEXT: }
// CHECK:      aie.switchbox(%{{.*}}shim_noc_tile_0_0)
// CHECK-NOT:  aie.packet_rules
// CHECK:        aie.packet_rules(South : 3) {
// CHECK-NEXT:     aie.rule(31, 15, %{{.*}})
// CHECK-NEXT:     aie.rule(31, 20, %{{.*}})
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NOT:  aie.packet_flow

module {
  aie.device(npu1_1col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %mux00 = aie.shim_mux(%t00) {
      aie.connect<DMA : 0, North : 3>
    }
    %sb00 = aie.switchbox(%t00) {
      %a0 = aie.amsel<0> (0)
      %m0 = aie.masterset(North : 0, %a0)
      aie.packet_rules(South : 3) {
        aie.rule(31, 15, %a0)
      }
    }
    %sb01 = aie.switchbox(%t01) {
      %a0 = aie.amsel<0> (0)
      %m0 = aie.masterset(DMA : 0, %a0)
      aie.packet_rules(South : 0) {
        aie.rule(31, 15, %a0)
      }
    }
    aie.packet_flow(20) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t01, DMA : 1>
    }
  }
}
