//===- test_trace_configure_run.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-insert-trace-flows -aie-trace-to-config -aie-trace-pack-reg-writes -aie-inline-trace-config -aie-fuse-trace-buffers --aie-materialize-runtime-sequences -aie-resolve-address-patch-buffers | FileCheck %s

// Two designs reached through aiex.configure, each with its own trace. They
// share one trace buffer on the dispatched sequence, split by offset.

module {
  aie.device(npu1_1col) @main {
    // The fused buffer follows the data arguments, so their indices are
    // unchanged. 2 x 8192 bytes, one region per configured device.
    // CHECK-LABEL: aie.runtime_sequence @main_seq
    // CHECK-SAME: memref<64xi32>, %{{.*}}: memref<64xi32>, %{{.*}}: memref<16384xi8>
    // CHECK-SAME: trace_slices = [
    // CHECK-SAME: #aie.trace_slice<device = "dev_a", sequence = "seq_a", offset = 0, size = 8192>
    // CHECK-SAME: #aie.trace_slice<device = "dev_b", sequence = "seq_b", offset = 8192, size = 8192>
    aie.runtime_sequence @main_seq(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
      // Both patches name the fused buffer (host buffer 2). The slice offset
      // reaches the BD through arg_plus.
      // CHECK: aiex.npu.load_pdi {device_ref = @dev_a}
      // CHECK: aiex.npu.address_patch(%[[OFFA:.*]] : i32) {{{.*}}arg_idx = 2 : i32}
      aiex.configure @dev_a {
        aiex.run @seq_a (%arg0) : (memref<64xi32>)
      }
      // CHECK: aiex.npu.load_pdi {device_ref = @dev_b}
      // CHECK: %[[OFFB:.*]] = arith.constant 8192 : i32
      // CHECK: aiex.npu.address_patch(%[[OFFB]] : i32) {{{.*}}arg_idx = 2 : i32}
      aiex.configure @dev_b {
        aiex.run @seq_b (%arg1) : (memref<64xi32>)
      }
    }
  }

  aie.device(npu1_1col) @dev_a {
    %shim_a = aie.tile(0, 0)
    %core_a = aie.tile(0, 2)

    aie.trace @a_core_trace(%core_a) {
      aie.trace.mode "Event-Time"
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence @seq_a(%arg0: memref<64xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @a_core_trace
    }
  }

  aie.device(npu1_1col) @dev_b {
    %shim_b = aie.tile(0, 0)
    %core_b = aie.tile(0, 2)

    aie.trace @b_core_trace(%core_b) {
      aie.trace.mode "Event-Time"
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence @seq_b(%arg0: memref<64xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @b_core_trace
    }
  }
}
