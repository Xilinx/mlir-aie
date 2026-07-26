//===- subcube_cover_multihop.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// ids 26,28,29 from one source: 28 drops at tile(0,2) while {26,29} forward via
// rules that avoid 28 (the common-bits mask (24,24) would false-match it).

// CHECK-LABEL: aie.device(npu1_1col) {
// CHECK:         aie.switchbox({{.*}}) {
// CHECK:           aie.packet_rules(South : 1) {
// CHECK-DAG:         aie.rule(31, 28, %[[DROP:.*]])
// CHECK-DAG:         aie.rule(31, 26, %[[FWD:.*]])
// CHECK-DAG:         aie.rule(31, 29, %[[FWD]])
// CHECK:           }

aie.device(npu1_1col) {
  %03 = aie.tile(0, 3)
  %02 = aie.tile(0, 2)
  %00 = aie.tile(0, 0)
  aie.packet_flow(28) {
    aie.packet_source<%00, DMA : 0>
    aie.packet_dest<%02, TileControl : 0>
  }
  aie.packet_flow(29) {
    aie.packet_source<%00, DMA : 0>
    aie.packet_dest<%03, TileControl : 0>
  }
  aie.packet_flow(26) {
    aie.packet_source<%00, DMA : 0>
    aie.packet_dest<%03, DMA : 0>
  }
}
