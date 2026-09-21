//===- bad_control_packet_empty.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A control packet with neither `data` nor `length` set encodes beats = 0 - 1,
// which underflows and corrupts the address the same way an over-4-word
// payload does (see AIETranslateControlPacketsToUI32Vec).

// RUN: not aie-translate --aie-ctrlpkt-to-bin %s 2>&1 | FileCheck %s
// CHECK: payload is empty; a control packet must carry at least 1 word on the wire

module {
  aie.device(npu2_1col) {
    aie.runtime_sequence @no_data_no_length() {
      aiex.control_packet {address = 0x1f000 : ui32, opcode = 1 : i32, stream_id = 0 : i32}
    }
  }
}
