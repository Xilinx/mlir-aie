//===- pinned_missing_control_source.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s --aie-pin-control-overlay 2>&1 | FileCheck %s

// A config control source absent from @ctrl_pkt_overlay's captured routing must
// HARD-FAIL: without a pinned route it would ship unpinned and could drift
// config-to-config, silently defeating the pinning for that source. Here @cfg's
// control source is (0,1) DMA:1 but the overlay only routes (0,1) DMA:0.

// CHECK: error: {{.*}}control source (0, 1) {{.*}}has no captured route in @ctrl_pkt_overlay
module {
  aie.device(npu2) {
    %t01 = aie.tile(0, 1)
    %t03 = aie.tile(0, 3)
    aie.packet_flow(1) {
      aie.packet_source<%t01, DMA : 1>
      aie.packet_dest<%t03, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
  } {sym_name = "cfg"}
  aie.device(npu2) {
    %o01 = aie.tile(0, 1)
    %o03 = aie.tile(0, 3)
    aie.packet_flow(1) {
      aie.packet_source<%o01, DMA : 0>
      aie.packet_dest<%o03, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
  } {sym_name = "ctrl_pkt_overlay"}
}
