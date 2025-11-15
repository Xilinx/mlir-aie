//===- test_trace_verify.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file -verify-diagnostics

// Test: Too many events (max 8)
module {
  aie.device(npu1_1col) {
    %tile = aie.tile(0, 2)
    
    // expected-error@+1 {{trace unit supports maximum 8 events}}
    aie.trace @test_too_many_events(%tile) {
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.event<"INSTR_EVENT_1">
      aie.trace.event<"INSTR_VECTOR">
      aie.trace.event<"LOCK_STALL">
      aie.trace.event<"MEMORY_STALL">
      aie.trace.event<"STREAM_STALL">
      aie.trace.event<"CASCADE_STALL">
      aie.trace.event<"ACTIVE">
      aie.trace.event<"DISABLED">
    }
  }
}

// -----

// Test: Packet ID out of range (too low)
module {
  aie.device(npu1_1col) {
    %tile = aie.tile(0, 2)
    
    aie.trace @test_packet_id_low(%tile) {
      // expected-error@+1 {{attribute 'id' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 1 whose maximum value is 31}}
      aie.trace.packet id=0 type="core"
    }
  }
}

// -----

// Test: Packet ID out of range (too high)
module {
  aie.device(npu1_1col) {
    %tile = aie.tile(0, 2)
    
    aie.trace @test_packet_id_high(%tile) {
      // expected-error@+1 {{attribute 'id' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 1 whose maximum value is 31}}
      aie.trace.packet id=32 type="core"
    }
  }
}

// -----

// Test: Start event - must specify broadcast or event
module {
  aie.device(npu1_1col) {
    %tile = aie.tile(0, 2)
    
    aie.trace @test_start_missing(%tile) {
      // expected-error@+1 {{must specify either broadcast or event}}
      aie.trace.start
    }
  }
}

// -----

// Test: Start event - cannot specify both
module {
  aie.device(npu1_1col) {
    %tile = aie.tile(0, 2)
    
    aie.trace @test_start_both(%tile) {
      // expected-error@+1 {{cannot specify both broadcast and event}}
      aie.trace.start broadcast=15 event=<"TRUE">
    }
  }
}

// -----

// Test: Valid trace configuration (should pass)
module {
  aie.device(npu1_1col) {
    %tile = aie.tile(0, 2)
    
    aie.trace @test_valid(%tile) {
      aie.trace.mode "Event-Time"
      aie.trace.packet id=1 type="core"
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.event<"INSTR_VECTOR">
      aie.trace.event<"LOCK_STALL">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
  }
}
