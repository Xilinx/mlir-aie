//===- packet_flow_mask_same_id_conflict.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two flows from one source state different masks for id 0x8 and head north
// together, and a third flow from that source carries 0xc south. Only mask
// 0x1b claims 0xc, so the source port would need two routes for it, whichever
// order the flows come in. Routing reports the pair.

// RUN: not aie-opt --split-input-file --aie-create-pathfinder-flows %s 2>&1 | FileCheck %s

// CHECK-COUNT-2: error: Unable to find a legal routing: at tile ({{[0-9]+}}, {{[0-9]+}}), packet flows through {{[A-Za-z]+[0-9]+}} claim rule (mask 0x1B, id 0x8) and rule (mask 0x1F, id 0xC), which both match id 0xC

module {
  aie.device(npu1_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)

    aie.packet_flow(8, mask = 27) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t03, DMA : 0>
    }
    aie.packet_flow(8, mask = 28) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t04, DMA : 0>
    }
    aie.packet_flow(12) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t00, DMA : 0>
    }
  }
}

// -----

module {
  aie.device(npu1_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)

    aie.packet_flow(8, mask = 28) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t04, DMA : 0>
    }
    aie.packet_flow(8, mask = 27) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t03, DMA : 0>
    }
    aie.packet_flow(12) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t00, DMA : 0>
    }
  }
}
