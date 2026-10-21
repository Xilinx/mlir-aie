//===- dedup_ctrl_packet_multicast.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-dedup-ctrl-packet-multicast --split-input-file | FileCheck %s
// RUN: aie-opt %s -aie-dedup-ctrl-packet-multicast=drain-merge=false --split-input-file | FileCheck %s --check-prefix=NOMERGE
// What the dedup emits lands in order under parallel shims.
// RUN: aie-opt %s -aie-dedup-ctrl-packet-multicast --split-input-file -aie-check-ctrl-packet-delivery=parallel-shims=true -o /dev/null

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
// CHECK-NOT:   aiex.ctrl_pkt_multicast_id = 25
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

// -----

// A group across columns 0 and 1 (width 2). In @conflict, tile (1, 2) writes
// the core reset register again after the shared packets, so they are stamped
// aiex.ctrl_pkt_base_phase and parallel shims send them before column 1's. In
// @commute no other packet writes their registers, so they stay in the phase,
// unless drain-merge is off.

// CHECK-LABEL: @conflict
// CHECK-NEXT:  address = 2301952 : ui32, aiex.ctrl_pkt_base_phase, aiex.ctrl_pkt_device = @cfg, aiex.ctrl_pkt_multicast_id = 25 : i32, data = array<i32: 2>
// CHECK-NEXT:  address = 2355204 : ui32
// CHECK-NEXT:  address = 35856384 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 0>
// CHECK-LABEL: @commute
// CHECK-NEXT:  address = 2301952 : ui32, aiex.ctrl_pkt_device = @cfg, aiex.ctrl_pkt_multicast_id = 25 : i32, data = array<i32: 2>
// CHECK-NEXT:  address = 2355204 : ui32
// CHECK-NEXT:  address = 35909636 : ui32

// In @enable, tile (1, 2)'s remaining packet (outside the enable phase, as a
// gated enable is) enables its core, which must follow the program the shared
// packets carry, so they are stamped too, though no address is shared.
// CHECK-LABEL: @enable
// CHECK-NEXT:  aiex.ctrl_pkt_base_phase, aiex.ctrl_pkt_device = @cfg, aiex.ctrl_pkt_multicast_id = 25 : i32

// NOMERGE-LABEL: @commute
// NOMERGE-NEXT:  aiex.ctrl_pkt_base_phase, aiex.ctrl_pkt_device = @cfg, aiex.ctrl_pkt_multicast_id = 25 : i32
module {
  aie.device(npu2) @cfg {
    %t02 = aie.tile(0, 2) {aie.ctrl_pkt_multicast_id = 25 : i32}
    %t12 = aie.tile(1, 2) {aie.ctrl_pkt_multicast_id = 25 : i32}
  }
  aie.device(npu2) {
    aie.runtime_sequence @conflict() {
      aiex.control_packet {address = 2301952 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2355204 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35856384 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35856384 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    aie.runtime_sequence @commute() {
      aiex.control_packet {address = 2301952 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2355204 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35856384 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35909636 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    aie.runtime_sequence @enable() {
      aiex.control_packet {address = 2228224 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2355204 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35782656 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35856384 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}
