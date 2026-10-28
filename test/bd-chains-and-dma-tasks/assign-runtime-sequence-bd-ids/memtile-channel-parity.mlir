//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

// A mem tile partitions its 48 buffer descriptors by channel PARITY, not by
// direction: an even channel can only submit ids 0-23, an odd channel only
// 24-47 (AIETargetModel::isBdChannelAccessible). Allocation used to pass
// channelIndex=0 unconditionally, under the belief that runtime sequences only
// ever touch shim and compute tiles -- which stopped being true once mem tile
// dma_configure_task landed (test/npu-xrt/memtile_dmas/dma_configure_task_*).
// An unpinned BD on an odd channel then got an id its channel cannot reach.

// CHECK-LABEL: @memtile_parity
// An even channel draws from the low half...
// CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 0)
// CHECK: aie.dma_bd({{.*}}) {bd_id = 0 : i32}
// ...and an odd channel from the high half.
// CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 1)
// CHECK: aie.dma_bd({{.*}}) {bd_id = 24 : i32}

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) {address = 0 : i32} : memref<1024xi32>
    aie.runtime_sequence @memtile_parity() {
      %even = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
      %odd = aiex.dma_configure_task(%tile_0_1, MM2S, 1) {
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
    }
  }
}
