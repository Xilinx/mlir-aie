//===- subcube_cover_overbudget.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-create-pathfinder-flows %s 2>&1 | FileCheck %s

// Five destinations on one slave port need five rules, over the 4-slot budget.

// CHECK: error{{.*}} slave port packet rules exceed the 4-slot limit

module @overbudget {
 aie.device(xcvc1902) {
  %t11 = aie.tile(1, 1)
  aie.packet_flow(0x0) { aie.packet_source<%t11, West : 0>  aie.packet_dest<%t11, Core : 0> }
  aie.packet_flow(0x1) { aie.packet_source<%t11, West : 0>  aie.packet_dest<%t11, Core : 1> }
  aie.packet_flow(0x2) { aie.packet_source<%t11, West : 0>  aie.packet_dest<%t11, DMA : 0> }
  aie.packet_flow(0x3) { aie.packet_source<%t11, West : 0>  aie.packet_dest<%t11, DMA : 1> }
  aie.packet_flow(0x4) { aie.packet_source<%t11, West : 0>  aie.packet_dest<%t11, TileControl : 0> }
 }
}
