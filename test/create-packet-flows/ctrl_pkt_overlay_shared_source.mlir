//===- ctrl_pkt_overlay_shared_source.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Packet flow 3 leaves shim (0,0) by DMA:1, which also sends the control
// packets for (0,3)-(0,5). The control overlay's ids keep the master port,
// amsel and first rule slot they take in the standalone overlay; id 3 takes
// the slot after them, the only one a control-packet reload writes on that
// port. Routing them together moved the overlay off North:1 and left its rule
// in a block the reload skips whole, so the design hung on the NPU (#3837).

// RUN: aie-opt --pass-pipeline="builtin.module(aie-generate-column-control-overlay{route-shim-to-tile-ctrl=true emit-standalone-overlay=true},aie.device(aie-create-pathfinder-flows))" %s -o %t
// RUN: sed -n '/@design {/,/^  }/p' %t | FileCheck %s --check-prefixes=CHECK,DESIGN
// RUN: sed -n '/@ctrl_pkt_overlay {/,/^  }/p' %t | FileCheck %s
// RUN: aie-opt --pass-pipeline="builtin.module(aie-materialize-runtime-sequences,aie-expand-load-pdi{ctrl-pkt=true})" %t | FileCheck %s --check-prefix=RELOAD

// CHECK-LABEL: aie.switchbox(%shim_noc_tile_0_0) {
// CHECK-DAG:     %[[A52:.+]] = aie.amsel<5> (2)
// CHECK-DAG:     aie.masterset(North : 1, %[[A52]]) {aie.is_ctrl_pkt_overlay}
// CHECK:         aie.packet_rules(South : 7) {
// CHECK-NEXT:      aie.rule(28, 28, %[[A52]])
// DESIGN-SAME:       is_ctrl_pkt_overlay
// DESIGN-NEXT:     aie.rule(31, 3, %{{.+}})
// DESIGN-NEXT:   }
// DESIGN-NOT:    is_ctrl_pkt_overlay
// DESIGN-LABEL: aie.switchbox(%mem_tile_0_1)

// RELOAD: aiex.npu.load_pdi {device_ref = @ctrl_pkt_overlay

module {
  aie.device(npu2) @main {
    aie.runtime_sequence @sequence(%arg0 : memref<16xi32>) {
      aiex.configure @design {
      }
    }
  }
  aie.device(npu2) @design {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t10 = aie.tile(1, 0)
    %t11 = aie.tile(1, 1)
    %t12 = aie.tile(1, 2)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    aie.flow(%t10, DMA : 1, %t01, DMA : 0)
    aie.flow(%t01, DMA : 0, %t10, DMA : 0)
    aie.flow(%t01, DMA : 1, %t02, DMA : 0)
    aie.flow(%t02, DMA : 0, %t01, DMA : 1)
    aie.flow(%t01, DMA : 2, %t03, DMA : 0)
    aie.flow(%t03, DMA : 0, %t01, DMA : 2)
    aie.flow(%t01, DMA : 3, %t04, DMA : 0)
    aie.flow(%t04, DMA : 0, %t01, DMA : 3)
    aie.flow(%t01, DMA : 4, %t05, DMA : 0)
    aie.flow(%t05, DMA : 0, %t01, DMA : 4)
    aie.flow(%t01, DMA : 5, %t02, DMA : 1)
    aie.packet_flow(3) {
      aie.packet_source<%t00, DMA : 1>
      aie.packet_dest<%t11, DMA : 0>
    }
  }
}
