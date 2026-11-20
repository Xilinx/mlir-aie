//===- txn_to_pkt.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-txn-to-ctrl-packet | FileCheck %s
// RUN: aie-opt %s -aie-insert-ctrl-packet-barriers -aie-coalesce-control-packets | FileCheck %s --check-prefix=MERGE
// RUN: aie-opt %s -aie-insert-ctrl-packet-barriers -aie-reorder-ctrl-packets-by-tile -aie-coalesce-control-packets | FileCheck %s --check-prefix=ORDER
// RUN: aie-opt %s -aie-insert-ctrl-packet-barriers -aie-reorder-ctrl-packets-by-tile -aie-dedup-ctrl-packet-multicast -aie-coalesce-control-packets | FileCheck %s --check-prefix=MULTI
// CHECK: aiex.control_packet {address = 65432 : ui32, data = array<i32: 123, 456, 789, 43981, -2147483648, -1, -2, -3>

module {
  aie.device(npu2_1col) {
    memref.global "private" constant @blockwrite_data_0 : memref<8xi32> = dense<[123, 456, 789, 0xabcd, 0x80000000, -1, -2, -3]>
    aie.runtime_sequence @run() {
      %0 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%0) {address = 65432 : ui32} : memref<8xi32>
    }
  }

// MERGE-LABEL: @adjacent
// MERGE: address = 2228224 : ui32, data = array<i32: 1, 2, 3, 4>
// MERGE-NEXT: aiex.control_packet {address = 2228240 : ui32, data = array<i32: 5>
// MERGE-NEXT: aiex.control_packet_barrier {compiler_only}
// MERGE-NEXT: aiex.control_packet {address = 2301952 : ui32
// MERGE-NEXT: aiex.control_packet_barrier {compiler_only}
// MERGE-NEXT: aiex.control_packet {address = 2301956 : ui32
// MERGE-NEXT: aiex.control_packet_barrier {compiler_only}
  aie.device(npu2) @merge {
    aie.runtime_sequence @adjacent() {
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228228 : ui32, data = array<i32: 2, 3>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228236 : ui32, data = array<i32: 4>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228240 : ui32, data = array<i32: 5>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301956 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
  aie.device(npu2) @normalize {
    // ORDER-LABEL: @overwritten
    // ORDER: address = 2228224 : ui32, data = array<i32: 1, 7, 8, 4>
    // ORDER-NEXT: aiex.control_packet_barrier {compiler_only}
    // ORDER-NEXT: aiex.control_packet {address = 2301952 : ui32
    // ORDER-NEXT: aiex.control_packet_barrier {compiler_only}
    // ORDER-NEXT: aiex.control_packet {address = 2228224 : ui32, data = array<i32: 9>
    aie.runtime_sequence @overwritten() {
      aiex.control_packet {address = 2228236 : ui32, data = array<i32: 4>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1, 2, 3>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228228 : ui32, data = array<i32: 7, 8>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
  aie.device(npu2) @overlay {
  } {aie.ctrl_pkt_multicast = [{column = 0 : i32, id = 28 : i32, tiles = [#aie.tile<0, 2>, #aie.tile<0, 3>]}], aie.is_ctrl_pkt_overlay}
  aie.device(npu2) @cfg {
  } {overlay = @overlay}
  aie.device(npu2) @multicast {
    // MULTI-LABEL: @matching
    // MULTI-NEXT: aiex.set_device @cfg
    // MULTI-NEXT: aiex.control_packet_barrier
    // MULTI-NEXT: aiex.control_packet {address = 2228224 : ui32, data = array<i32: 7, 9>, multicast_id = 28 : i32
    // MULTI-NEXT: aiex.control_packet_barrier
    // MULTI-NOT: multicast_id
    // MULTI: aiex.control_packet {address = 2301952 : ui32
    // MULTI: aiex.control_packet {address = 3350528 : ui32
    aie.runtime_sequence @matching() {
      aiex.set_device @cfg
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228228 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276800 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276804 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3350528 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    // MULTI-LABEL: @mixed_headers
    // MULTI-NOT: multicast_id
    // MULTI: address = 3276800 : ui32, data = array<i32: 1>
    // MULTI-NEXT: aiex.control_packet {address = 3276800 : ui32, data = array<i32: 2>
    // MULTI-NEXT: aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1>
    // MULTI-NEXT: {{^ *}}}
    aie.runtime_sequence @mixed_headers() {
      aiex.set_device @cfg
      aiex.control_packet {address = 3276800 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276800 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 1 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}
