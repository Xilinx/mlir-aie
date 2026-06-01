//===- test_insert_trace_flows_verify_packet_id_overflow.mlir -*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Explicit aie.trace.packet ids reserve slots the auto-allocator must skip.
// With packet-id-start=30 and explicit reservations at 30 and 31, there is no
// free id left in the 5-bit (0..31) hardware range for the third (auto) trace.

// RUN: aie-opt %s -verify-diagnostics -aie-insert-trace-flows="packet-id-start=30"

module @auto_packet_id_overflow_with_reservations {
  // expected-error@+1 {{id(s) reserved by explicit aie.trace.packet}}
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)
    %tile04 = aie.tile(0, 4)
    aie.trace @pinned1(%tile02) {
      aie.trace.packet id=30 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.trace @pinned2(%tile03) {
      aie.trace.packet id=31 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.trace @auto(%tile04) {
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 65536
      aie.trace.start_config @pinned1
      aie.trace.start_config @pinned2
      aie.trace.start_config @auto
    }
  }
}
