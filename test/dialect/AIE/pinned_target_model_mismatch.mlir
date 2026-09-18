//===- pinned_target_model_mismatch.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s --aie-pin-control-overlay 2>&1 | FileCheck %s

// A captured control route encodes physical ports valid only for the overlay's
// target model; replaying it into a config declared on a DIFFERENT device model
// would be silently wrong. The pass must reject the mismatch. Here @cfg is npu1
// and @ctrl_pkt_overlay is npu2.

// CHECK: error: {{.*}}device model differs from @ctrl_pkt_overlay
module {
  aie.device(npu1) {
    %t01 = aie.tile(0, 1)
    %t03 = aie.tile(0, 3)
    aie.packet_flow(1) {
      aie.packet_source<%t01, DMA : 0>
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
