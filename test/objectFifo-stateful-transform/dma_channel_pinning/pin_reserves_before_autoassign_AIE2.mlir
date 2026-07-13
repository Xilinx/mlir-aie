//===- pin_reserves_before_autoassign_AIE2.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=false" %s | FileCheck %s

// Two fifos share the same producer and consumer tiles. @of_auto is declared
// first and would take channel 0 under first-free assignment. @of_pinned is
// pinned to channel 0, so reservePinnedChannels() must reserve it before
// auto-assignment runs, pushing @of_auto onto channel 1. This proves pins are
// reserved up front rather than losing to whichever fifo is processed first.

// The producer tile's DMA shows @of_auto pushed onto channel 1 while the
// pinned @of_pinned holds channel 0 it reserved.

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK-DAG:       aie.flow(%{{.*}}tile_1_2, DMA : 0, %{{.*}}tile_3_3, DMA : 0)
// CHECK-DAG:       aie.flow(%{{.*}}tile_1_2, DMA : 1, %{{.*}}tile_3_3, DMA : 1)
// CHECK:           %mem_1_2 = aie.mem(%{{.*}}tile_1_2) {
// CHECK:             aie.dma_start(MM2S, 1,
// CHECK:             aie.dma_bd(%of_auto_buff_0
// CHECK:             aie.dma_start(MM2S, 0,
// CHECK:             aie.dma_bd(%of_pinned_buff_0

module @dma_channel_pinning_reserve {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of_auto (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_pinned (%tile12, {%tile33}, 2 : i32) {prod_dma_channel = 0 : i32, cons_dma_channels = array<i32: 0>} : !aie.objectfifo<memref<16xi32>>
  }
}
