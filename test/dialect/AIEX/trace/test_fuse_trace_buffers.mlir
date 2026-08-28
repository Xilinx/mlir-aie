//===- test_fuse_trace_buffers.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows -aie-trace-to-config -aie-trace-pack-reg-writes -aie-inline-trace-config -aie-fuse-trace-buffers --aie-materialize-runtime-sequences -aie-resolve-address-patch-buffers | FileCheck %s

// Test: two calls of the same traced design get distinct regions.
// CHECK-LABEL: aie.runtime_sequence @main_seq
// CHECK-SAME: memref<16384xi8>
// CHECK-SAME: {device = "dev_a", offset = 0 : i64, sequence = "seq_a", size = 8192 : i64}
// CHECK-SAME: {device = "dev_a", offset = 8192 : i64, sequence = "seq_a", size = 8192 : i64}
module {
  aie.device(npu1_1col) @main {
    aie.runtime_sequence @main_seq(%arg0: memref<64xi32>) {
      aiex.configure @dev_a {
        // CHECK: aiex.npu.address_patch(%{{.*}} : i32) {{{.*}}arg_idx = 1 : i32}
        aiex.run @seq_a (%arg0) : (memref<64xi32>)
        // CHECK: %[[OFF1:.*]] = arith.constant 8192 : i32
        // CHECK: aiex.npu.address_patch(%[[OFF1]] : i32) {{{.*}}arg_idx = 1 : i32}
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

// -----

// Test: the caller's own traces take the front of the buffer, the callee's the
// tail. The caller's data arguments keep their indices.
// CHECK-LABEL: aie.runtime_sequence @main_seq
// CHECK-SAME: %{{.*}}: memref<64xi32>, %{{.*}}: memref<12288xi8>
// CHECK-SAME: {device = "main", offset = 0 : i64, sequence = "main_seq", size = 4096 : i64}
// CHECK-SAME: {device = "dev_a", offset = 4096 : i64, sequence = "seq_a", size = 8192 : i64}
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
    aie.runtime_sequence @main_seq(%arg0: memref<64xi32>) {
      aie.trace.host_config {buffer_size = 4096 : i32}
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

// -----

// Test: an untraced caller of an untraced callee gains no trace argument.
// CHECK-LABEL: aie.runtime_sequence @main_seq
// CHECK-NOT: aie.trace_slices
// CHECK-SAME: memref<64xi32>) {
module {
  aie.device(npu1_1col) @main {
    aie.runtime_sequence @main_seq(%arg0: memref<64xi32>) {
      aiex.configure @dev_a {
        aiex.run @seq_a (%arg0) : (memref<64xi32>)
      }
    }
  }
  aie.device(npu1_1col) @dev_a {
    %shim = aie.tile(0, 0)
    aie.runtime_sequence @seq_a(%arg0: memref<64xi32>) {
    }
  }
}
