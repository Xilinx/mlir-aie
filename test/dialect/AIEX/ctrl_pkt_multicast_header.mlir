//===- ctrl_pkt_multicast_header.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-ctrl-packet-to-dma | FileCheck %s
// RUN: aie-translate --aie-ctrlpkt-to-bin -aie-output-binary=false %s | FileCheck %s --check-prefix=BIN

// A packet stamped aiex.ctrl_pkt_multicast_id goes on that id, so its header
// carries 28 (0x1C, odd parity) in place of tile (0, 2)'s controller id 27
// (0x1B). A transfer is routed by its first header, so the multicast packets
// and the tile's own go in separate transfers.

// CHECK:       aiex.npu.dma_memcpy_nd({{.*}}[0, 0, 0, 0][1, 1, 1, 6]
// CHECK:       aiex.npu.dma_memcpy_nd({{.*}}[0, 0, 0, 6][1, 1, 1, 3]
// CHECK-NOT:   aiex.npu.dma_memcpy_nd

// BIN:      0000001C
// BIN-NEXT: 0002C000
// BIN-NEXT: 00000001
// BIN-NEXT: 0000001C
// BIN-NEXT: 8002C004
// BIN-NEXT: 00000002
// BIN-NEXT: 8000001B
// BIN-NEXT: 0003F004
// BIN-NEXT: 00000003
module {
  aie.device(npu2_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0)
    aie.runtime_sequence() {
      aiex.control_packet {address = 2277376 : ui32, aiex.ctrl_pkt_multicast_id = 28 : i32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2277380 : ui32, aiex.ctrl_pkt_multicast_id = 28 : i32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2355204 : ui32, data = array<i32: 3>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}
