//===- conform_overlay_trace_buffer.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-materialize-runtime-sequences %s | FileCheck %s

// The --get-full-elf --reconfig-method=ctrlpkt flow wraps each config in a host sequence that
// forwards the config's args through `aiex.run @sequence(args)`. Trace lowering
// appends a trace buffer to a config's runtime_sequence (tagging it with
// aie.trace_buffer_arg) AFTER that wrapper is built, so the wrapper is left one
// operand short and would fail verifyRunOpsInConfigureOp. Materialize's conform
// threads the tagged trace buffer up into the wrapping host sequence + run --
// but ONLY for the config that traces. Here config_1 traces and config_2 does
// not.

module {
  // CHECK-LABEL: aie.device(npu2) @overlay_host
  aie.device(npu2) @overlay_host {
    // seq_1 wraps the TRACING config -> gains the trace buffer as a 4th arg.
    // CHECK: aie.runtime_sequence @seq_1(%{{[a-z0-9_]+}}: memref<4096xi32>, %{{[a-z0-9_]+}}: memref<1xi32>, %{{[a-z0-9_]+}}: memref<4096xi32>, %{{[a-z0-9_]+}}: memref<8192xi8>)
    aie.runtime_sequence @seq_1(%a0: memref<4096xi32>, %a1: memref<1xi32>, %a2: memref<4096xi32>) {
      aiex.configure @config_1 {
        aiex.run @sequence(%a0, %a1, %a2) : (memref<4096xi32>, memref<1xi32>, memref<4096xi32>)
      }
    }
    // seq_2 wraps the NON-tracing config -> stays at 3 args (trailing `) {`
    // means no 4th arg was threaded in).
    // CHECK: aie.runtime_sequence @seq_2(%{{[a-z0-9_]+}}: memref<4096xi32>, %{{[a-z0-9_]+}}: memref<1xi32>, %{{[a-z0-9_]+}}: memref<4096xi32>) {
    aie.runtime_sequence @seq_2(%a0: memref<4096xi32>, %a1: memref<1xi32>, %a2: memref<4096xi32>) {
      aiex.configure @config_2 {
        aiex.run @sequence(%a0, %a1, %a2) : (memref<4096xi32>, memref<1xi32>, memref<4096xi32>)
      }
    }
  }
  // Tracing config: its @sequence carries the trace buffer at arg 3, tagged.
  aie.device(npu2) @config_1 {
    aie.runtime_sequence @sequence(%a0: memref<4096xi32>, %a1: memref<1xi32>, %a2: memref<4096xi32>, %trace: memref<8192xi8>) attributes {aie.trace_buffer_arg = 3 : i32} {
    }
  }
  // Non-tracing config: no trace buffer, no tag -> conform leaves seq_2 alone.
  aie.device(npu2) @config_2 {
    aie.runtime_sequence @sequence(%a0: memref<4096xi32>, %a1: memref<1xi32>, %a2: memref<4096xi32>) {
    }
  }
}
