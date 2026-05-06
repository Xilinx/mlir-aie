//===- test_insert_trace_flows_controller_id.mlir ------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows | FileCheck %s

// -----

// Test: Pass auto-assigns controller_id=15 for shim on <=6 row device
// CHECK-LABEL: module @ctrl_id_auto_assign
module @ctrl_id_auto_assign {
  aie.device(npu1_1col) {
    // CHECK: aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile00 = aie.tile(0, 0)
    %tile02 = aie.tile(0, 2)
    aie.trace @core_trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      // controller_id=15: value = 15 << 8 = 3840, mask = 0xFF00 = 65280
      // CHECK: aiex.npu.maskwrite32 {{{.*}}mask = 65280{{.*}}value = 3840{{.*}}}
      aie.trace.host_config buffer_size = 65536
      aie.trace.start_config @core_trace
    }
  }
}

// -----

// Test: Pass respects user-specified controller_id (does not overwrite)
// CHECK-LABEL: module @ctrl_id_user_specified
module @ctrl_id_user_specified {
  aie.device(npu1_1col) {
    // CHECK: aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
    %tile00 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
    %tile02 = aie.tile(0, 2)
    aie.trace @core_trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      // controller_id=5: value = 5 << 8 = 1280, mask = 0xFF00 = 65280
      // CHECK: aiex.npu.maskwrite32 {{{.*}}mask = 65280{{.*}}value = 1280{{.*}}}
      aie.trace.host_config buffer_size = 65536
      aie.trace.start_config @core_trace
    }
  }
}

// -----

// Test: Pass auto-assigns controller_id when shim tile is created (no shim in input)
// CHECK-LABEL: module @ctrl_id_created_shim
module @ctrl_id_created_shim {
  aie.device(npu1_1col) {
    // CHECK: aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile02 = aie.tile(0, 2)
    aie.trace @core_trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      // CHECK: aiex.npu.maskwrite32 {{{.*}}mask = 65280{{.*}}value = 3840{{.*}}}
      aie.trace.host_config buffer_size = 65536
      aie.trace.start_config @core_trace
    }
  }
}

// -----

// Test: NPU2 (AIE2P) device uses same register DB for controller_id encoding
// CHECK-LABEL: module @ctrl_id_npu2
module @ctrl_id_npu2 {
  aie.device(npu2_1col) {
    // CHECK: aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 10>}
    %tile00 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 10>}
    %tile02 = aie.tile(0, 2)
    aie.trace @core_trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      // controller_id=10: value = 10 << 8 = 2560, mask = 0xFF00 = 65280
      // CHECK: aiex.npu.maskwrite32 {{{.*}}mask = 65280{{.*}}value = 2560{{.*}}}
      aie.trace.host_config buffer_size = 65536
      aie.trace.start_config @core_trace
    }
  }
}
