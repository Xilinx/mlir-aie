//===- assign_controller_ids.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-assign-tile-controller-ids --split-input-file | FileCheck %s
// RUN: aie-opt %s -aie-assign-tile-controller-ids="column-wise-unique-ids=false" --split-input-file | FileCheck %s --check-prefix=GLOBAL

// assign controller ids to aie.tile_op, for control packets

// CHECK-LABEL: module {
// CHECK: aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
// CHECK: aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
// CHECK: aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
// CHECK: aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
// CHECK: aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
// CHECK: aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
// GLOBAL-LABEL: module {
// GLOBAL: aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
// GLOBAL: aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
// GLOBAL: aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
// GLOBAL: aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
// GLOBAL: aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
// GLOBAL: aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 6>}

aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
  %tile_0_2 = aie.tile(0, 2)
  %tile_0_3 = aie.tile(0, 3)
  %tile_0_4 = aie.tile(0, 4)
  %tile_0_5 = aie.tile(0, 5)
}

// -----

// CHECK-LABEL: module {
// CHECK: aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
// CHECK: aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
// CHECK: aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
// CHECK: aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
// CHECK: aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
// CHECK: aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
// CHECK: aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
// CHECK: aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
// CHECK: aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
// CHECK: aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
// CHECK: aie.tile(1, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
// CHECK: aie.tile(1, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
// GLOBAL-LABEL: module {
// GLOBAL: aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
// GLOBAL: aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
// GLOBAL: aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
// GLOBAL: aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
// GLOBAL: aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
// GLOBAL: aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
// GLOBAL: aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
// GLOBAL: aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 8>}
// GLOBAL: aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 9>}
// GLOBAL: aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 10>}
// GLOBAL: aie.tile(1, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 11>}
// GLOBAL: aie.tile(1, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 12>}

aie.device(npu1_2col) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
  %tile_0_2 = aie.tile(0, 2)
  %tile_0_3 = aie.tile(0, 3)
  %tile_0_4 = aie.tile(0, 4)
  %tile_0_5 = aie.tile(0, 5)
  %tile_1_0 = aie.tile(1, 0)
  %tile_1_1 = aie.tile(1, 1)
  %tile_1_2 = aie.tile(1, 2)
  %tile_1_3 = aie.tile(1, 3)
  %tile_1_4 = aie.tile(1, 4)
  %tile_1_5 = aie.tile(1, 5)
}
