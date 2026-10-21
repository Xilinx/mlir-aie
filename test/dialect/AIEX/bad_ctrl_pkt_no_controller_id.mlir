//===- bad_ctrl_pkt_no_controller_id.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-translate --aie-ctrlpkt-to-bin -aie-output-binary=false %s 2>&1 | FileCheck %s

// A control packet's stream header routes it by its tile's controller_id.
// Tile (0, 2) has none and tile (0, 3) is not declared, so neither packet has
// a route.

// CHECK: error: 'aiex.control_packet' op targets tile (0, 2), which has no controller_id to route it by
module {
  aie.device(npu2_1col) {
    %tile_0_2 = aie.tile(0, 2)
    aie.runtime_sequence() {
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}
