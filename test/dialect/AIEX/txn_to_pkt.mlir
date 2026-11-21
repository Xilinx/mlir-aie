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
// RUN: aie-opt %s -aie-insert-ctrl-packet-barriers -aie-reorder-ctrl-packets-by-tile="address-order=false" | FileCheck %s --check-prefix=DSE
// RUN: aie-opt %s -aie-insert-ctrl-packet-barriers -aie-reorder-ctrl-packets-by-tile="eliminate-dead-writes=false" | FileCheck %s --check-prefix=SORT
// RUN: aie-opt %s -aie-insert-ctrl-packet-barriers -aie-reorder-ctrl-packets-by-tile="address-order=false eliminate-dead-writes=false" | FileCheck %s --check-prefix=RETAIN
// RUN: aie-opt %s -aie-insert-ctrl-packet-barriers -aie-dedup-ctrl-packet-multicast | FileCheck %s --check-prefix=RAW-MULTI
// RUN: aie-opt %s -aie-expand-load-pdi="ctrl-pkt=true teardown=true" -aie-legalize-ctrl-packet -aie-schedule-ctrl-packet-activation -aie-insert-ctrl-packet-barriers -aie-reorder-ctrl-packets-by-tile -aie-dedup-ctrl-packet-multicast -aie-coalesce-control-packets -aie-check-ctrl-packet-delivery | FileCheck %s --check-prefix=RAW-ORDER
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
    // RAW-ORDER-LABEL: @raw_activation
    // RAW-ORDER: aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>
    // RAW-ORDER-NEXT: aiex.control_packet_barrier {compiler_only}
    // RAW-ORDER-NEXT: aiex.control_packet {address = 2301952 : ui32, data = array<i32: 0>
    // RAW-ORDER-NOT: data = array<i32: 1>
    // RAW-ORDER: {{^ *}}}
    aie.runtime_sequence @raw_activation() {
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    // ORDER-LABEL: @overwritten
    // ORDER: address = 2228224 : ui32, data = array<i32: 1, 7, 8, 4>
    // ORDER-NEXT: aiex.control_packet_barrier {compiler_only}
    // ORDER-NEXT: aiex.control_packet {address = 2301952 : ui32
    // ORDER-NEXT: aiex.control_packet_barrier {compiler_only}
    // ORDER-NEXT: aiex.control_packet {address = 2228224 : ui32, data = array<i32: 9>
    // DSE-LABEL: @overwritten
    // DSE-NEXT: address = 2228236 : ui32, data = array<i32: 4>
    // DSE-NEXT: address = 2228224 : ui32, data = array<i32: 1>
    // DSE-NEXT: address = 2228228 : ui32, data = array<i32: 7, 8>
    // SORT-LABEL: @overwritten
    // SORT-NEXT: address = 2228224 : ui32, data = array<i32: 1, 2, 3>
    // SORT-NEXT: address = 2228236 : ui32, data = array<i32: 4>
    // SORT-NEXT: address = 2228228 : ui32, data = array<i32: 7, 8>
    // RETAIN-LABEL: @overwritten
    // RETAIN-NEXT: address = 2228236 : ui32, data = array<i32: 4>
    // RETAIN-NEXT: address = 2228224 : ui32, data = array<i32: 1, 2, 3>
    // RETAIN-NEXT: address = 2228228 : ui32, data = array<i32: 7, 8>
    aie.runtime_sequence @overwritten() {
      aiex.control_packet {address = 2228236 : ui32, data = array<i32: 4>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1, 2, 3>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228228 : ui32, data = array<i32: 7, 8>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    // DSE-LABEL: @reverse_overlap
    // DSE-NEXT: address = 2228232 : ui32, data = array<i32: 2>
    // DSE-NEXT: address = 2228224 : ui32, data = array<i32: 3, 4>
    // DSE-NEXT: {{^ *}}}
    // SORT-LABEL: @reverse_overlap
    // SORT-NEXT: address = 2228228 : ui32, data = array<i32: 1, 2>
    // SORT-NEXT: address = 2228224 : ui32, data = array<i32: 3, 4>
    // SORT-NEXT: {{^ *}}}
    aie.runtime_sequence @reverse_overlap() {
      aiex.control_packet {address = 2228228 : ui32, data = array<i32: 1, 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 3, 4>, opcode = 0 : i32, stream_id = 0 : i32}
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
    // RAW-MULTI-LABEL: @interleaved_overwrite
    // RAW-MULTI-NEXT: aiex.set_device @cfg
    // RAW-MULTI-NEXT: address = 3276800 : ui32, data = array<i32: 7>, opcode
    // RAW-MULTI-NEXT: address = 3276800 : ui32, data = array<i32: 9>, opcode
    // RAW-MULTI-NEXT: address = 2228224 : ui32, data = array<i32: 7>, opcode
    // RAW-MULTI-NEXT: {{^ *}}}
    aie.runtime_sequence @interleaved_overwrite() {
      aiex.set_device @cfg
      aiex.control_packet {address = 3276800 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276800 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}
