//===- dedup_ctrl_packet_multicast.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file --aie-generate-column-control-overlay="configuration=true adaptive=false multicast-span=1" --aie-insert-ctrl-packet-barriers --aie-reorder-ctrl-packets-by-tile --aie-dedup-ctrl-packet-multicast --aie-check-ctrl-packet-delivery | FileCheck %s --check-prefixes=CHECK,SPAN1
// RUN: aie-opt %s --split-input-file --aie-generate-column-control-overlay="configuration=true adaptive=false multicast-span=2" --aie-insert-ctrl-packet-barriers --aie-reorder-ctrl-packets-by-tile --aie-dedup-ctrl-packet-multicast --aie-check-ctrl-packet-delivery | FileCheck %s --check-prefixes=CHECK,SPAN2
// RUN: aie-opt %s --split-input-file --aie-generate-column-control-overlay="configuration=true adaptive=false multicast-span=2" --aie-insert-ctrl-packet-barriers --aie-reorder-ctrl-packets-by-tile --aie-dedup-ctrl-packet-multicast=parallel-shims=true --aie-check-ctrl-packet-delivery=parallel-shims=true | FileCheck %s --check-prefixes=CHECK,SPAN2

// Core disables and resets remain separate actions on every destination.

// CHECK-LABEL: aie.runtime_sequence @seq()
// CHECK-NEXT:  aiex.set_device @cfg
// SPAN1:         address = 2301952 : ui32, data = array<i32: 0>, multicast_id = 28 : i32
// SPAN2:         address = 2301952 : ui32, data = array<i32: 0>, opcode
// SPAN2:         address = 3350528 : ui32, data = array<i32: 0>, opcode
// CHECK:       aiex.control_packet_barrier{{$}}
// SPAN1:         address = 2301952 : ui32, data = array<i32: 2>, multicast_id = 28 : i32
// SPAN1-NEXT:    aiex.control_packet_barrier {compiler_only, packet_effect, scope = {core = [#aie.tile<0, 2>, #aie.tile<0, 3>]}}
// SPAN1-NEXT:    address = 2228224 : ui32, data = array<i32: 7>, multicast_id = 28 : i32
// SPAN2:         address = 2301952 : ui32, data = array<i32: 2>, opcode
// SPAN2:         address = 2228224 : ui32, data = array<i32: 7>, opcode
// CHECK:         address = 2355204 : ui32, data = array<i32: 1>, opcode
// CHECK:         address = 3403780 : ui32, data = array<i32: 2>, opcode
// CHECK:         address = 3350528 : ui32, data = array<i32: 1>, opcode

module {
  aie.device(npu2) @cfg {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t12 = aie.tile(1, 2)
    %t13 = aie.tile(1, 3)
    %c02 = aie.core(%t02) { aie.end }
    %c03 = aie.core(%t03) { aie.end }
    %c12 = aie.core(%t12) { aie.end }
    %c13 = aie.core(%t13) { aie.end }
  }
  aie.device(npu2) @main {
    aie.runtime_sequence @seq() {
      aiex.set_device @cfg {quiescent}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3350528 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet_barrier
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2355204 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3350528 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276800 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3403780 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3350528 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}

// -----

// Core enables bound the program-write runs for both serial and parallel shims.

// CHECK-LABEL: aie.runtime_sequence @seq()
// CHECK-NEXT:  aiex.set_device @cfg
// CHECK-NEXT:    address = 2228224 : ui32, data = array<i32: 7>, opcode
// CHECK:         address = 2301952 : ui32, data = array<i32: 1>, opcode
// CHECK:         address = 35782656 : ui32, data = array<i32: 7>, opcode
// CHECK:         address = 35856384 : ui32, data = array<i32: 1>, opcode
// CHECK:         address = 3276800 : ui32, data = array<i32: 9>, opcode

module {
  aie.device(npu2) @cfg {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t12 = aie.tile(1, 2)
    %c02 = aie.core(%t02) { aie.end }
    %c03 = aie.core(%t03) { aie.end }
    %c12 = aie.core(%t12) { aie.end }
  }
  aie.device(npu2) @main {
    aie.runtime_sequence @seq() {
      aiex.set_device @cfg {quiescent}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276800 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35782656 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35856384 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276800 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
    }

    // SPAN1-LABEL: aie.runtime_sequence @ordinary_prefix()
    // SPAN1-NEXT:  aiex.set_device @cfg
    // SPAN1-NEXT:    address = 2228224 : ui32, data = array<i32: 7>, multicast_id = 28 : i32
    // SPAN1-NEXT:    address = 2228228 : ui32, data = array<i32: 9>, multicast_id = 28 : i32
    // SPAN1-NEXT:    address = 35782656 : ui32, data = array<i32: 7>, opcode
    // SPAN1-NEXT:    address = 35782660 : ui32, data = array<i32: 9>, opcode
    // SPAN1-NEXT:  {{^ *}}}
    // SPAN2-LABEL: aie.runtime_sequence @ordinary_prefix()
    // SPAN2-NEXT:  aiex.set_device @cfg
    // SPAN2-NEXT:    address = 2228224 : ui32, data = array<i32: 7>, multicast_id = 28 : i32
    // SPAN2-NEXT:    address = 2228228 : ui32, data = array<i32: 9>, multicast_id = 28 : i32
    // SPAN2-NEXT:  {{^ *}}}
    aie.runtime_sequence @ordinary_prefix() {
      aiex.set_device @cfg {quiescent}
      aiex.control_packet {address = 3276804 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35782660 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228228 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 3>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35782656 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276800 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}

// -----

// Every tile would answer one read, so matching reads (opcode 1) stay apart.

// CHECK-LABEL: aie.runtime_sequence @reads()
// CHECK-NEXT:  aiex.set_device @cfg
// CHECK-NEXT:  aiex.control_packet_barrier {compiler_only, packet_effect}
// CHECK-NEXT:    address = 2301952 : ui32, length = 1 : i32, opcode = 1 : i32
// CHECK-NEXT:  aiex.control_packet_barrier {compiler_only, packet_effect}
// CHECK-NEXT:    address = 3350528 : ui32, length = 1 : i32, opcode = 1 : i32
// CHECK-NEXT:  aiex.control_packet_barrier {compiler_only, packet_effect}
// CHECK-NEXT:  {{^ *}}}

module {
  aie.device(npu2) @cfg {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %c02 = aie.core(%t02) { aie.end }
    %c03 = aie.core(%t03) { aie.end }
  }
  aie.device(npu2) @main {
    aie.runtime_sequence @reads() {
      aiex.set_device @cfg
      aiex.control_packet {address = 2301952 : ui32, length = 1 : i32, opcode = 1 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3350528 : ui32, length = 1 : i32, opcode = 1 : i32, stream_id = 0 : i32}
    }
  }
}
