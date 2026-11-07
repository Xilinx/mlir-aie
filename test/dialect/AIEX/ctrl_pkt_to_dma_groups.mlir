//===- ctrl_pkt_to_dma_groups.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-ctrl-packet-to-dma | FileCheck %s

// Adjacent groups of one kind and device share a batch; a change of kind or
// device, and a control packet outside groups, start their own.

// CHECK-LABEL: aie.device(npu2) {
// CHECK-NOT:   aiex.control_packet_group
// CHECK:       aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 9][0, 0, 0, 1])
// CHECK:       aiex.npu.sync
// CHECK:       aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 9][1, 1, 1, 3][0, 0, 0, 1])
// CHECK:       aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 12][1, 1, 1, 6][0, 0, 0, 1])
// CHECK-NOT:   aiex.npu.dma_memcpy_nd

module {
  aie.device(npu2) @cfg {
  }
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence() {
      aiex.control_packet_group @cfg(Config) {
        aiex.control_packet {address = 0 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 4 : ui32, data = array<i32: 200>, opcode = 0 : i32, stream_id = 0 : i32}
      }
      aiex.control_packet_group @cfg(Config) {
        aiex.control_packet {address = 8 : ui32, data = array<i32: 300>, opcode = 0 : i32, stream_id = 0 : i32}
      }
      aiex.control_packet_group @cfg(Teardown) {
        aiex.control_packet {address = 12 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      }
      aiex.control_packet {address = 16 : ui32, data = array<i32: 500>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 20 : ui32, data = array<i32: 600>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
  }
}
