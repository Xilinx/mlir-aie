//===- bad_control_packet.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An over-4-word payload corrupts the packet's address instead of truncating
// (see AIETranslateControlPacketsToUI32Vec), so translating one straight to a
// binary without first running --aie-legalize-control-packet must fail.

// RUN: not aie-translate --aie-ctrlpkt-to-bin %s 2>&1 | FileCheck %s
// CHECK: payload is 8 words; a control packet carries at most 4 on the wire

module {
  aie.device(npu2_1col) {
    aie.runtime_sequence @too_much_data() {
      aiex.control_packet {address = 0x1f000 : ui32, opcode = 0 : i32, stream_id = 0 : i32, data = array<i32: 1, 2, 3, 4, 5, 6, 7, 8>}
    }
  }
}
