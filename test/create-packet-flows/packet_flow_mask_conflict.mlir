//===- packet_flow_mask_conflict.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two claims on one slave port that share an id.
//
// The first flow claims 0x8 through 0xb. The second carries 0x9, which sits
// inside that range, so the port would need two routes for one id. Routing
// reports the pair rather than handing the id to whichever rule the arbiter
// checks first.

// RUN: not aie-opt --aie-create-pathfinder-flows %s 2>&1 | FileCheck %s

// CHECK: error: packet flows through DMA0 claim rule (mask 0x1F, id 0x9) and rule (mask 0x1C, id 0x8), which both match id 0x9

module {
  aie.device(npu1_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)

    aie.packet_flow(8 mask 28) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t00, DMA : 0>
    }

    aie.packet_flow(9) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t03, DMA : 0>
    }
  }
}
