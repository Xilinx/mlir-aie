// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-annotate-shim-sharing %s | FileCheck %s

// CHECK: aie.route_endpoint @a(
// CHECK-SAME: aie.shareable_with = []
// CHECK: aie.route_endpoint @b(
// CHECK-SAME: aie.shareable_with = []

module {
  aie.device(npu2_1col) @application {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.route_endpoint @a(%shim) DMA
    aie.route_endpoint @b(%shim) DMA
    aie.route_endpoint @source(%core) Core {channelIndex = 0 : i32}
    aie.route from @source to [@a, @b]
    aie.runtime_sequence @receive(%output : memref<16xi32>) {
      %first = aiex.dma_configure_task_for @a {
        aie.dma_bd(%output : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      %second = aiex.dma_configure_task_for @a {
        aie.dma_bd(%output : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%first)
      aiex.dma_start_task(%second)
      aiex.dma_await_task(%first)
      aiex.dma_free_task(%first)
      aiex.npu.dma_memcpy_nd(%output[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @b}
      aiex.dma_await_task(%second)
    }
  }
}
