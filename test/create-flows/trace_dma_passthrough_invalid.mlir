//===- trace_dma_passthrough_invalid.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Regression test for https://github.com/Xilinx/mlir-aie/issues/2689.
//
// A flow whose source is a Trace unit and whose destination requires
// leaving the source tile eastward (Trace can only legally connect
// directly to DMA/FIFO/South) must not be routed through the source
// tile's DMA channel as if it were a pass-through crossbar arc: DMA
// S2MM (write-in) and MM2S (read-out) on the same channel index are
// independent hardware ports with no implicit link between them, so
// such a route silently drops the data instead of delivering it.

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

module {
 aie.device(npu1) {
  %tile_1_2 = aie.tile(1, 2)
  %tile_2_2 = aie.tile(2, 2)

  aie.flow(%tile_1_2, Trace : 0, %tile_2_2, DMA : 0)
 }
}

// The router must instead use a real physical path: Trace can only exit
// tile_1_2 southward, so it bounces through mem_tile_1_1 (via a legal
// North:North loopback connect, a real hardware capability) before
// heading east to tile_2_2's DMA:0.
// CHECK: aie.switchbox(%mem_tile_1_1) {
// CHECK-NEXT: aie.connect<North : {{[0-9]+}}, North : {{[0-9]+}}>
// CHECK-NEXT: }
// CHECK: aie.switchbox(%tile_1_2) {
// CHECK-NOT: DMA
// CHECK: aie.connect<Trace : 0, South : {{[0-9]+}}>
// CHECK: }
// CHECK: aie.switchbox(%tile_2_2) {
// CHECK-NEXT: aie.connect<West : {{[0-9]+}}, DMA : 0>
// CHECK-NEXT: }
