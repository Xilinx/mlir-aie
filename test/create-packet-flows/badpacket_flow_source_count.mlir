//===- badpacket_flow_source_count.mlir ------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --split-input-file %s

// Regression tests for issue #2583 verifier gap:
// PacketFlowOp must have at least one packet_source and at least one
// packet_dest. Verify that the verifier rejects zero-source and zero-dest
// flows. (Multi-source flows are valid; see multi_source_packet_flow.mlir.)

// Test 1: zero sources — must be rejected by verifier.
module {
  aie.device(xcvc1902) {
    %t11 = aie.tile(1, 1)
    // expected-error@+1 {{must have at least one aie.packet_source}}
    aie.packet_flow(0x0) {
      aie.packet_dest<%t11, Core : 0>
    }
  }
}

// -----

// Test 2: zero dests — must be rejected by verifier.
module {
  aie.device(xcvc1902) {
    %t11 = aie.tile(1, 1)
    // expected-error@+1 {{must have at least one aie.packet_dest}}
    aie.packet_flow(0x0) {
      aie.packet_source<%t11, West : 0>
    }
  }
}
