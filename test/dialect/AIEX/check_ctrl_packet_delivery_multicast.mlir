//===- check_ctrl_packet_delivery_multicast.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file --aie-generate-column-control-overlay="configuration=true adaptive=false multicast-span=2" -aie-check-ctrl-packet-delivery=parallel-shims=true -verify-diagnostics
// RUN: not aie-opt %s --split-input-file --aie-generate-column-control-overlay="configuration=true adaptive=false multicast-span=2" -aie-check-ctrl-packet-delivery 2>&1 | FileCheck %s --check-prefix=SERIAL

// Serially column 0's transfers are awaited before column 1's.
// SERIAL-NOT: op may land
// SERIAL:     multicast id 9
// SERIAL-NOT: op may land

// Tiles (0, 2) and (1, 2) are one multicast (id 28), from column 0. A packet
// on it and a later enable of core (1, 2) on column 1 go in one phase on two
// channels, so the enable may land before the program word it must follow.
// Serially column 0's transfer is awaited first.
module {
  aie.device(npu2) @cfg {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t12 = aie.tile(1, 2)
    %c02 = aie.core(%t02) { aie.end }
    %c12 = aie.core(%t12) { aie.end }
  }
  aie.device(npu2) {
    aie.runtime_sequence @order() {
      aiex.control_packet_group @cfg(Config) {
        // expected-note @+1 {{the earlier packet}}
        aiex.control_packet {address = 2228224 : ui32, data = array<i32: 7>, multicast_id = 28 : i32, opcode = 0 : i32, stream_id = 0 : i32}
        // expected-error @+1 {{may land on tile (1, 2) before an earlier control packet it does not commute with}}
        aiex.control_packet {address = 35856384 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}

// -----

// With a barrier after it, the multicast packet goes in an earlier phase.
module {
  aie.device(npu2) @cfg {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t12 = aie.tile(1, 2)
    %c02 = aie.core(%t02) { aie.end }
    %c12 = aie.core(%t12) { aie.end }
  }
  aie.device(npu2) {
    aie.runtime_sequence @base_phase() {
      aiex.control_packet_group @cfg(Config) {
        aiex.control_packet {address = 2228224 : ui32, data = array<i32: 7>, multicast_id = 28 : i32, opcode = 0 : i32, stream_id = 0 : i32}
      }
      aiex.control_packet_barrier
      aiex.control_packet_group @cfg(Config) {
        aiex.control_packet {address = 35856384 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}

// -----

// A switch write of (1, 2) commutes with the multicast program word.
module {
  aie.device(npu2) @cfg {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t12 = aie.tile(1, 2)
    %c02 = aie.core(%t02) { aie.end }
    %c12 = aie.core(%t12) { aie.end }
  }
  aie.device(npu2) {
    aie.runtime_sequence @commute() {
      aiex.control_packet_group @cfg(Config) {
        aiex.control_packet {address = 2228224 : ui32, data = array<i32: 7>, multicast_id = 28 : i32, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 35909636 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}

// -----

// A two-word read of (1, 2) covers the word the later multicast program word
// writes, which may land first.
module {
  aie.device(npu2) @cfg {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t12 = aie.tile(1, 2)
    %c02 = aie.core(%t02) { aie.end }
    %c12 = aie.core(%t12) { aie.end }
  }
  aie.device(npu2) {
    aie.runtime_sequence @read() {
      aiex.control_packet_group @cfg(Config) {
        // expected-note @+1 {{the earlier packet}}
        aiex.control_packet {address = 35782652 : ui32, length = 2 : i32, opcode = 1 : i32, stream_id = 0 : i32}
        // expected-error @+1 {{may land on tile (1, 2) before an earlier control packet it does not commute with}}
        aiex.control_packet {address = 2228224 : ui32, data = array<i32: 7>, multicast_id = 28 : i32, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}

// -----

// A packet on a multicast id the overlay does not list.
module {
  aie.device(npu2) @cfg {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t12 = aie.tile(1, 2)
    %c02 = aie.core(%t02) { aie.end }
    %c12 = aie.core(%t12) { aie.end }
  }
  aie.device(npu2) {
    aie.runtime_sequence @unknown() {
      aiex.control_packet_group @cfg(Config) {
        // expected-error @+1 {{is sent on multicast id 9 from tile (0, 2), which is not the lowest tile of a multicast on that id in the overlay of its device}}
        aiex.control_packet {address = 2228224 : ui32, data = array<i32: 7>, multicast_id = 9 : i32, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}

// -----

// A packet on multicast id 28 addressed to tile (0, 3), which is not its lowest
// tile.
module {
  aie.device(npu2) @cfg {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t12 = aie.tile(1, 2)
    %c02 = aie.core(%t02) { aie.end }
    %c12 = aie.core(%t12) { aie.end }
  }
  aie.device(npu2) {
    aie.runtime_sequence @outside() {
      aiex.control_packet_group @cfg(Config) {
        // expected-error @+1 {{is sent on multicast id 28 from tile (0, 3), which is not the lowest tile of a multicast on that id}}
        aiex.control_packet {address = 3276800 : ui32, data = array<i32: 7>, multicast_id = 28 : i32, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}

// -----

// A packet on multicast id 28 addressed to tile (1, 2), which is in it but
// not its lowest tile.
module {
  aie.device(npu2) @cfg {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t12 = aie.tile(1, 2)
    %c02 = aie.core(%t02) { aie.end }
    %c12 = aie.core(%t12) { aie.end }
  }
  aie.device(npu2) {
    aie.runtime_sequence @not_lowest() {
      aiex.control_packet_group @cfg(Config) {
        // expected-error @+1 {{is sent on multicast id 28 from tile (1, 2), which is not the lowest tile of a multicast on that id}}
        aiex.control_packet {address = 35782656 : ui32, data = array<i32: 7>, multicast_id = 28 : i32, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}
