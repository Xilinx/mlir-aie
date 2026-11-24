//===- bad_control_packet_zero_length.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A read control packet (no `data`) with `length = 0` would encode
// beats = 0 - 1. The op verifier rejects it.

// RUN: aie-opt --verify-diagnostics %s

module {
  aie.device(npu2_1col) {
    aie.runtime_sequence @zero_length() {
      // expected-error @+1 {{control packet payload must be at least 1 word}}
      aiex.control_packet {address = 0x1f000 : ui32, length = 0 : i32, opcode = 1 : i32, stream_id = 0 : i32}
    }
  }
}
