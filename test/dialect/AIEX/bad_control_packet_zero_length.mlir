//===- bad_control_packet_zero_length.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A read control packet (no `data`) with `length = 0` still encodes
// beats = 0 - 1, so the diagnostic must name `length`, not `payload`, since
// `length` is what the op actually carries.

// RUN: not aie-translate --aie-ctrlpkt-to-bin %s 2>&1 | FileCheck %s
// CHECK: length is empty; a control packet must carry at least 1 word on the wire

module {
  aie.device(npu2_1col) {
    aie.runtime_sequence @zero_length() {
      aiex.control_packet {address = 0x1f000 : ui32, length = 0 : i32, opcode = 1 : i32, stream_id = 0 : i32}
    }
  }
}
