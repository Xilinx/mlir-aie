//===- dedup_ctrl_packet_multicast.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-dedup-ctrl-packet-multicast --split-input-file | FileCheck %s

// Tiles (0, 2), (0, 3) and (0, 5) of @cfg share multicast id 25. Each tile's
// first two writes (a core reset and a program word) are the same apart from
// the address's column and row; the third (a switch) differs. Those two stay on
// (0, 2), stamped with the id, and the others' copies go; the switches stay per
// tile. The enable phase is compared on its own, and the three enables go once.

// CHECK-LABEL: @dedup
// CHECK-NEXT:  address = 2301952 : ui32, aiex.ctrl_pkt_device = @cfg, aiex.ctrl_pkt_multicast_id = 25 : i32, data = array<i32: 2>
// CHECK-NEXT:  address = 2228224 : ui32, aiex.ctrl_pkt_device = @cfg, aiex.ctrl_pkt_multicast_id = 25 : i32, data = array<i32: 7, 8>
// CHECK-NEXT:  address = 2355204 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 1>
// CHECK-NEXT:  address = 3403780 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 2>
// CHECK-NEXT:  address = 5500932 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 3>
// CHECK-NEXT:  address = 2301952 : ui32, aiex.ctrl_pkt_device = @cfg, aiex.ctrl_pkt_enable_phase, aiex.ctrl_pkt_multicast_id = 25 : i32, data = array<i32: 1>
// CHECK-NOT:   control_packet
module {
  aie.device(npu2) @cfg {
    %t02 = aie.tile(0, 2) {aie.ctrl_pkt_multicast_id = 25 : i32}
    %t03 = aie.tile(0, 3) {aie.ctrl_pkt_multicast_id = 25 : i32}
    %t05 = aie.tile(0, 5) {aie.ctrl_pkt_multicast_id = 25 : i32}
  }
  aie.device(npu2) {
    aie.runtime_sequence @dedup() {
      aiex.control_packet {address = 2301952 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 7, 8>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2355204 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3350528 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276800 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 7, 8>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3403780 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 5447680 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 5373952 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 7, 8>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 5500932 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 3>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, aiex.ctrl_pkt_device = @cfg, aiex.ctrl_pkt_enable_phase, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3350528 : ui32, aiex.ctrl_pkt_device = @cfg, aiex.ctrl_pkt_enable_phase, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 5447680 : ui32, aiex.ctrl_pkt_device = @cfg, aiex.ctrl_pkt_enable_phase, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}

// -----

// The pass leaves alone a group whose tile (0, 5) has no packets in the run
// (the multicast would reach it too), packets of a device with no multicast
// (@plain) or naming none, and a common write that comes first on another tile
// than the lowest.

// CHECK-LABEL: @untouched
// CHECK-NOT:   ctrl_pkt_multicast_id = 25
module {
  aie.device(npu2) @cfg {
    %t02 = aie.tile(0, 2) {aie.ctrl_pkt_multicast_id = 25 : i32}
    %t03 = aie.tile(0, 3) {aie.ctrl_pkt_multicast_id = 25 : i32}
    %t05 = aie.tile(0, 5) {aie.ctrl_pkt_multicast_id = 25 : i32}
  }
  aie.device(npu2) @plain {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
  }
  aie.device(npu2) {
    aie.runtime_sequence @untouched() {
      %zero = arith.constant 0 : i32
      aiex.control_packet {address = 2301952 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3350528 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.npu.write32(%zero, %zero) : i32, i32
      aiex.control_packet {address = 2301952 : ui32, aiex.ctrl_pkt_device = @plain, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3350528 : ui32, aiex.ctrl_pkt_device = @plain, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.npu.write32(%zero, %zero) : i32, i32
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3350528 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.npu.write32(%zero, %zero) : i32, i32
      aiex.control_packet {address = 5447680 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3350528 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}
