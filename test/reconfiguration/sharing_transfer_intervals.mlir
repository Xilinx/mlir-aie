// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-annotate-shim-sharing %s | FileCheck %s
// RUN: sed '/@b(.*) DMA/s/DMA/DMA {aie.shareable_with = []}/' %s | aie-opt --aie-annotate-shim-sharing | FileCheck %s --check-prefix=EXPLICIT

// CHECK: aie.route_endpoint @a(
// CHECK-SAME: aie.shareable_with = [@c]
// CHECK: aie.route_endpoint @b(
// CHECK-SAME: aie.shareable_with = [@c]
// CHECK: aie.route_endpoint @c(
// CHECK-SAME: aie.shareable_with = [@a, @b]
// EXPLICIT: aie.route_endpoint @b(
// EXPLICIT-SAME: aie.shareable_with = []

module {
  aie.device(npu2_1col) @application {
    %shim = aie.tile(0, 0)
    %t2 = aie.tile(0, 2)
    %t3 = aie.tile(0, 3)
    aie.route_endpoint @a(%shim) DMA
    aie.route_endpoint @b(%shim) DMA
    aie.route_endpoint @c(%shim) DMA
    aie.route_endpoint @source_a(%t2) Core {channelIndex = 0 : i32}
    aie.route_endpoint @source_b(%t3) Core {channelIndex = 0 : i32}
    aie.route_endpoint @source_c(%t3) Core {channelIndex = 1 : i32}
    aie.route from @source_a to [@a]
    aie.route from @source_b to [@b]
    aie.route from @source_c to [@c]
    aie.runtime_sequence @receive(%output : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%output[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @a}
      aiex.npu.dma_memcpy_nd(%output[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @b}
      aiex.npu.dma_memcpy_nd(%output[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @a}
      aiex.npu.dma_memcpy_nd(%output[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @c, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @c}
    }
  }
}
