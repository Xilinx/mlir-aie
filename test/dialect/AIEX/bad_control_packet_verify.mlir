//===- bad_control_packet_verify.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The control-packet opcode is a 2-bit field: 0 write, 1 read, 2 write with
// response, 3 reserved. Writes carry `data`; reads carry only `length`, which
// must fit one packet (at most 4 words).

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(npu2_1col) {
  aie.runtime_sequence @reserved_opcode() {
    // expected-error@+1 {{opcode 3 is reserved; expected 0 (write), 1 (read), or 2 (write with response)}}
    aiex.control_packet {address = 0x1f000 : ui32, opcode = 3 : i32, stream_id = 0 : i32, data = array<i32: 1>}
  }
}

// -----

// A wider opcode would also overwrite the stream ID.
aie.device(npu2_1col) {
  aie.runtime_sequence @wide_opcode() {
    // expected-error@+1 {{opcode 5 is reserved}}
    aiex.control_packet {address = 0x1f000 : ui32, opcode = 5 : i32, stream_id = 0 : i32, data = array<i32: 1>}
  }
}

// -----

aie.device(npu2_1col) {
  aie.runtime_sequence @read_with_data() {
    // expected-error@+1 {{read control packet must not carry a data payload}}
    aiex.control_packet {address = 0x1f000 : ui32, opcode = 1 : i32, stream_id = 0 : i32, data = array<i32: 7, 8>}
  }
}

// -----

aie.device(npu2_1col) {
  aie.runtime_sequence @write_without_data() {
    // expected-error@+1 {{control packet with a write opcode requires a data payload}}
    aiex.control_packet {address = 0x1f000 : ui32, opcode = 0 : i32, stream_id = 0 : i32, length = 2 : i32}
  }
}

// -----

aie.device(npu2_1col) {
  aie.runtime_sequence @write_with_response_without_data() {
    // expected-error@+1 {{control packet with a write opcode requires a data payload}}
    aiex.control_packet {address = 0x1f000 : ui32, opcode = 2 : i32, stream_id = 0 : i32}
  }
}

// -----

aie.device(npu1_1col) {
  aie.runtime_sequence @long_read() {
    // expected-error@+1 {{read control packet length 8 exceeds the maximum of 4 32-bit words}}
    aiex.control_packet {address = 0x1f000 : ui32, opcode = 1 : i32, stream_id = 0 : i32, length = 8 : i32}
  }
}

// -----

aie.device(npu2_1col) {
  aie.runtime_sequence @empty_write() {
    // expected-error@+1 {{control packet payload must be at least 1 word}}
    aiex.control_packet {address = 0x1f000 : ui32, opcode = 0 : i32, stream_id = 0 : i32, data = array<i32>}
  }
}

// -----

// The largest read, and a write longer than one packet: legal here, since
// aie-legalize-control-packet splits writes.
aie.device(npu2_1col) {
  aie.runtime_sequence @valid() {
    aiex.control_packet {address = 0x1f000 : ui32, opcode = 1 : i32, stream_id = 0 : i32, length = 4 : i32}
    aiex.control_packet {address = 0x1f000 : ui32, opcode = 2 : i32, stream_id = 0 : i32, data = array<i32: 1, 2, 3, 4, 5, 6>}
  }
}
