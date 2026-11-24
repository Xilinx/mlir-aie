//===- bad_control_packet_empty.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A control packet with neither `data` nor `length` set would encode
// beats = 0 - 1, which underflows and corrupts the address the same way an
// over-4-word payload does. The op verifier rejects it.

// RUN: aie-opt --verify-diagnostics %s

module {
  aie.device(npu2_1col) {
    aie.runtime_sequence @no_data_no_length() {
      // expected-error @+1 {{read control packet requires a length attribute}}
      aiex.control_packet {address = 0x1f000 : ui32, opcode = 1 : i32, stream_id = 0 : i32}
    }
  }
}
