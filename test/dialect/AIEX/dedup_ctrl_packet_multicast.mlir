//===- dedup_ctrl_packet_multicast.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file --aie-generate-column-control-overlay="configuration=true adaptive=false multicast-span=1" --aie-dedup-ctrl-packet-multicast | FileCheck %s --check-prefix=SPAN1
// RUN: aie-opt %s --split-input-file --aie-generate-column-control-overlay="configuration=true adaptive=false multicast-span=2" --aie-dedup-ctrl-packet-multicast | FileCheck %s --check-prefix=SPAN2
// RUN: aie-opt %s --split-input-file --aie-generate-column-control-overlay="configuration=true adaptive=false multicast-span=2" --aie-dedup-ctrl-packet-multicast="parallel-shims=true" | FileCheck %s --check-prefix=PAR2

// Tiles (0, 2) and (0, 3) are column 0's multicast (id 28). The teardown and
// the configuration are runs of their own, so each dedups its own leading
// packets: the teardown's core disable, then the configuration's core reset
// and program word. Tile (0, 3)'s enable does not commute with them, but it
// leaves by the same control channel along the multicast's trail, so the
// shared packets stay in place. At span 2 the multicast also takes column 1's
// cores, which have no packets here, so nothing goes.

// SPAN1-LABEL: aie.runtime_sequence @seq()
// SPAN1-NEXT:  aiex.control_packet_group @cfg(Teardown) {
// SPAN1-NEXT:    address = 2301952 : ui32, data = array<i32: 0>, multicast_id = 28 : i32
// SPAN1-NEXT:  {{^ *}}}
// SPAN1-NEXT:  aiex.control_packet_group @cfg(Config) {
// SPAN1-NEXT:    address = 2301952 : ui32, data = array<i32: 2>, multicast_id = 28 : i32
// SPAN1-NEXT:    address = 2228224 : ui32, data = array<i32: 7>, multicast_id = 28 : i32
// SPAN1-NEXT:    address = 2355204 : ui32, data = array<i32: 1>, opcode
// SPAN1-NEXT:    address = 3403780 : ui32, data = array<i32: 2>, opcode
// SPAN1-NEXT:    address = 3350528 : ui32, data = array<i32: 1>, opcode
// SPAN1-NEXT:  {{^ *}}}

// SPAN2-LABEL: aie.runtime_sequence @seq()
// SPAN2-NOT:   multicast_id
// SPAN2:       {{^ *}}}

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
      aiex.control_packet_group @cfg(Teardown) {
        aiex.control_packet {address = 2301952 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 3350528 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      }
      aiex.control_packet_group @cfg(Config) {
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
}

// -----

// At span 2, tiles (0, 2), (0, 3) and (1, 2) are one multicast from column 0.
// Tile (1, 2)'s own packets leave by column 1's channel. Serially that channel
// is awaited after column 0's, so the program word stays in place; with
// parallel shims they land alongside it, and its enable does not commute with
// the word, so the word goes first, behind a barrier. The control packet
// outside groups stays as it is.

// SPAN2-LABEL: aie.runtime_sequence @seq()
// SPAN2-NEXT:  aiex.control_packet_group @cfg(Config) {
// SPAN2-NEXT:    address = 2228224 : ui32, data = array<i32: 7>, multicast_id = 28 : i32
// SPAN2-NEXT:    address = 2301952 : ui32, data = array<i32: 1>, opcode
// SPAN2-NEXT:    address = 35856384 : ui32, data = array<i32: 1>, opcode
// SPAN2-NEXT:  {{^ *}}}
// SPAN2-NEXT:  address = 3276800 : ui32, data = array<i32: 9>, opcode

// PAR2-LABEL: // -----
// PAR2:       aie.runtime_sequence @seq()
// PAR2-NEXT:  aiex.control_packet_group @cfg(Config) {
// PAR2-NEXT:    address = 2228224 : ui32, data = array<i32: 7>, multicast_id = 28 : i32
// PAR2-NEXT:  {{^ *}}}
// PAR2-NEXT:  aiex.control_packet_barrier
// PAR2-NEXT:  aiex.control_packet_group @cfg(Config) {
// PAR2-NEXT:    address = 2301952 : ui32, data = array<i32: 1>, opcode
// PAR2-NEXT:    address = 35856384 : ui32, data = array<i32: 1>, opcode
// PAR2-NEXT:  {{^ *}}}
// PAR2-NEXT:  address = 3276800 : ui32, data = array<i32: 9>, opcode

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
      aiex.control_packet_group @cfg(Config) {
        aiex.control_packet {address = 2228224 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 3276800 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 35782656 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 35856384 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      }
      aiex.control_packet {address = 3276800 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}

// -----

// Every tile would answer one read, so matching reads (opcode 1) stay apart.

// SPAN1-LABEL: aie.runtime_sequence @reads()
// SPAN1-NOT:   multicast_id
// SPAN1:       {{^ *}}}

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
      aiex.control_packet_group @cfg(Config) {
        aiex.control_packet {address = 2301952 : ui32, length = 1 : i32, opcode = 1 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 3350528 : ui32, length = 1 : i32, opcode = 1 : i32, stream_id = 0 : i32}
      }
    }
  }
}
