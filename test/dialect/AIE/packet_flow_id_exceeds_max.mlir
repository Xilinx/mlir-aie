//===- packet_flow_id_exceeds_max.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A packet header carries five ID bits, so an ID past the target's maximum is
// rejected on the flow that asks for it, at that flow's location.

// RUN: not aie-opt %s 2>&1 | FileCheck %s

// CHECK: design.py:10:4: error: 'aie.packet_flow' op packet id 40 exceeds the maximum of 31

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.packet_flow(40) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t03, DMA : 0>
    } loc("iron:act_in"("design.py":10:4))
  }
}
