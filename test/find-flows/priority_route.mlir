//===- priority_route.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The router may merge flow 9 into the rules of priority flow 1 on the hops the
// two share. Only the master set at a flow's destination says whether the flow
// is a priority_route one, so flow 9 is lifted without it.

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows --aie-create-pathfinder-flows --aie-find-flows %s | FileCheck %s

// CHECK-NOT: aie.switchbox
// CHECK:     aie.packet_flow(1) {
// CHECK-NEXT:  aie.packet_source<%{{.*}}, DMA : 1>
// CHECK-NEXT:  aie.packet_dest<%{{.*}}mem_tile_0_1, DMA : 0>
// CHECK-NEXT: } {priority_route = true}
// CHECK:     aie.packet_flow(9) {
// CHECK-NEXT:  aie.packet_source<%{{.*}}, DMA : 1>
// CHECK-NEXT:  aie.packet_dest<%{{.*}}tile_0_2, DMA : 1>
// CHECK-NEXT: }{{$}}
// CHECK-NOT: aie.switchbox

module {
  aie.device(npu1_1col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    aie.packet_flow(1) {
      aie.packet_source<%t00, DMA : 1>
      aie.packet_dest<%t01, DMA : 0>
    } {priority_route = true}
    aie.packet_flow(9) {
      aie.packet_source<%t00, DMA : 1>
      aie.packet_dest<%t02, DMA : 1>
    }
  }
}
