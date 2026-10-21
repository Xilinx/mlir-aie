//===- out_of_order_shared_channel_bad.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A channel its fifos receive in turn holds their order, so they cannot share
// it. Only an out-of-order channel holds none, each object naming the
// descriptor that takes it.

// RUN: not aie-opt --aie-objectfifo-split --aie-objectfifo-allocate %s 2>&1 | FileCheck %s

// CHECK: error: {{.*}}pinned S2MM DMA channel 1 is out of range or already in use
// CHECK: note: {{.*}}@a{{.*}}occupies channel 1
module @shared_needs_out_of_order {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @a(%tile_0_2, {%tile_0_5}, 2 : i32) {packet, packet_id = 3 : i8, out_of_order, cons_dma_channels = array<i32: 1>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%tile_0_3, {%tile_0_5}, 2 : i32) {packet, packet_id = 4 : i8, cons_dma_channels = array<i32: 1>} : !aie.objectfifo<memref<16xi32>>
  }
}
