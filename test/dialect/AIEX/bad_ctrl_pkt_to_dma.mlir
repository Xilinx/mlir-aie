//===- bad_ctrl_pkt_to_dma.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-ctrl-packet-to-dma -verify-diagnostics

// A control packet to a column without a shim DMA allocation marked
// is_ctrl_overlay has no channel to go through, whatever the column's
// allocations are named.

aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0)
  aie.runtime_sequence() {
    // expected-error@+1 {{'aiex.control_packet' op column 0 has no control packet shim DMA allocation; run -aie-generate-column-control-overlay with route-shim-to-tile-ctrl=true first}}
    aiex.control_packet {address = 0 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
  }
}
