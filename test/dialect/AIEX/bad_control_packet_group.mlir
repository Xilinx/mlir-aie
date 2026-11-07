//===- bad_control_packet_group.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file --verify-diagnostics

module {
  aie.device(npu2) @cfg {
  }
  aie.device(npu2) {
    aie.runtime_sequence() {
      aiex.control_packet_group @cfg(Config) {
        // expected-error@+1 {{cannot be in an aiex.control_packet_group, which holds only aiex.control_packet ops}}
        %c0 = arith.constant 0 : i32
      }
    }
  }
}

// -----

module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      // expected-error@+1 {{@missing is not a device}}
      aiex.control_packet_group @missing(Teardown) {
        aiex.control_packet {address = 0 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}
