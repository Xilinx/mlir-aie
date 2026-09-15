//===- packet_flow_mask_bad_id.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A slave port accepts a packet when `incoming & mask == ID`. ID 0x3 sets a bit
// the mask 0x1 clears, so the test never holds and the flow carries nothing.

// RUN: not aie-opt %s 2>&1 | FileCheck %s

// CHECK: error: 'aie.packet_flow' op has ID 0x3 outside mask 0x1, which no packet can match

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    aie.packet_flow(3 mask 1) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t02, DMA : 0>
    }
  }
}
