//===- column_control_overlay_priority_request.mlir -----------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file -aie-create-pathfinder-flows | aie-opt --split-input-file -aie-generate-column-control-overlay | FileCheck %s

// A marked DMA-to-TileControl request is not a routed response.
// CHECK-LABEL: aie.device(npu1_1col) {
// CHECK: aie.packet_rules(DMA : 0)
// CHECK: aie.packet_flow(15) {
// CHECK-NEXT: aie.packet_source<%{{.*}}, TileControl : 0>
// CHECK-NEXT: aie.packet_dest<%{{.*}}, South : 0>
aie.device(npu1_1col) {
  %shim = aie.tile(0, 0)
  %compute = aie.tile(0, 2)
  aie.packet_flow(3) {
    aie.packet_source<%shim, DMA : 0>
    aie.packet_dest<%compute, TileControl : 0>
  } {priority_route = true}
}

// -----

// A response in one column must not prevent generation in another column.
// CHECK-LABEL: aie.device(npu1_2col) {
// CHECK: aie.packet_rules(TileControl : 0)
// CHECK-NOT: aie.packet_flow
// CHECK: aie.packet_flow(15) {
// CHECK-NEXT: aie.packet_source<%[[SHIM:.*]], TileControl : 0>
// CHECK-NEXT: aie.packet_dest<%[[SHIM]], South : 0>
aie.device(npu1_2col) {
  %shim0 = aie.tile(0, 0)
  %compute0 = aie.tile(0, 2)
  %shim1 = aie.tile(1, 0)
  %compute1 = aie.tile(1, 2)
  aie.packet_flow(15) {
    aie.packet_source<%shim0, TileControl : 0>
    aie.packet_dest<%shim0, South : 0>
  } {keep_pkt_header = true, priority_route = true}
}
