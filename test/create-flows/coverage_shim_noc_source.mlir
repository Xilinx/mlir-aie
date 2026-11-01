//===- coverage_shim_noc_source.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// NOC0/NOC1 enter the shim switchbox on South 2/3, NOC2/NOC3 on South 6/7.

// CHECK: aie.switchbox(%{{.*}}tile_2_0) {
// CHECK-DAG: aie.connect<South : 2, North : {{[0-9]+}}>
// CHECK-DAG: aie.connect<South : 3, North : {{[0-9]+}}>
// CHECK-DAG: aie.connect<South : 6, North : {{[0-9]+}}>
// CHECK-DAG: aie.connect<South : 7, North : {{[0-9]+}}>
// CHECK: aie.shim_mux(%{{.*}}tile_2_0) {
// CHECK-DAG: aie.connect<NOC : 0, North : 2>
// CHECK-DAG: aie.connect<NOC : 1, North : 3>
// CHECK-DAG: aie.connect<NOC : 2, North : 6>
// CHECK-DAG: aie.connect<NOC : 3, North : 7>

module {
  aie.device(xcvc1902) {
    %t20 = aie.tile(2, 0)
    %t22 = aie.tile(2, 2)
    %t23 = aie.tile(2, 3)
    aie.flow(%t20, NOC : 0, %t22, DMA : 0)
    aie.flow(%t20, NOC : 1, %t22, DMA : 1)
    aie.flow(%t20, NOC : 2, %t23, DMA : 0)
    aie.flow(%t20, NOC : 3, %t23, DMA : 1)
  }
}
