//===- pin_prod_cons_AIE2.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=false" --aie-assign-lock-ids %s | FileCheck %s

// A producer and consumer are both pinned to DMA channel 1. First-free
// assignment would pick channel 0 for each; the pins force channel 1 on both
// the MM2S (producer) and S2MM (consumer) sides and on the flow between them.

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           aie.flow(%{{.*}}tile_1_2, DMA : 1, %{{.*}}tile_3_3, DMA : 1)
// CHECK:           %mem_1_2 = aie.mem(%{{.*}}tile_1_2) {
// CHECK:             aie.dma_start(MM2S, 1,
// CHECK:           %mem_3_3 = aie.mem(%{{.*}}tile_3_3) {
// CHECK:             aie.dma_start(S2MM, 1,

module @dma_channel_pinning {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of_pin (%tile12, {%tile33}, 2 : i32) {prod_dma_channel = 1 : i32, cons_dma_channels = array<i32: 1>} : !aie.objectfifo<memref<16xi32>>
  }
}
