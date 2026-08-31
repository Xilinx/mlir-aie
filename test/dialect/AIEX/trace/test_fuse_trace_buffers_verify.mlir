//===- test_fuse_trace_buffers_verify.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file -verify-diagnostics -aie-insert-trace-flows -aie-trace-to-config -aie-trace-pack-reg-writes -aie-inline-trace-config -aie-fuse-trace-buffers

// Test: a callee that writes its trace into its own output buffer has no
// argument to slice.

module {
  aie.device(npu1_1col) @main {
    aie.runtime_sequence @main_seq(%arg0: memref<64xi32>) {
      aiex.configure @dev_a {
        // expected-error@+1 {{calls a design whose trace.host_config sets reuse_output_buffer=true}}
        aiex.run @seq_a (%arg0) : (memref<64xi32>)
      }
    }
  }
  aie.device(npu1_1col) @dev_a {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.trace @t(%core) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence @seq_a(%arg0: memref<64xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32, reuse_output_buffer = true}
      aie.trace.start_config @t
    }
  }
}

// -----

// Test: a caller cannot grow its output buffer to hold a callee's slice.

module {
  aie.device(npu1_1col) @main {
    %shim_m = aie.tile(0, 0)
    %core_m = aie.tile(0, 3)
    aie.trace @main_trace(%core_m) {
      aie.trace.packet id=5 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    // expected-error@+1 {{reuse_output_buffer=true cannot be combined with aiex.run calls into traced designs}}
    aie.runtime_sequence @main_seq(%arg0: memref<64xi32>) {
      aie.trace.host_config {buffer_size = 4096 : i32, reuse_output_buffer = true}
      aie.trace.start_config @main_trace
      aiex.configure @dev_a {
        aiex.run @seq_a (%arg0) : (memref<64xi32>)
      }
    }
  }
  aie.device(npu1_1col) @dev_a {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.trace @t(%core) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence @seq_a(%arg0: memref<64xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @t
    }
  }
}
