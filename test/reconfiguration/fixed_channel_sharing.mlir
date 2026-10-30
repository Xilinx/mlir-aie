// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s

// CHECK: aie.shim_dma_allocation @fixed(%{{.*}}, MM2S, 0)
// CHECK-SAME: aie.fixed
// CHECK: aie.packet_flow(31)
// CHECK-NEXT: aie.packet_source<%{{.*}}, DMA : 0>
// CHECK-DAG: aie.shim_dma_allocation @a_shim_alloc(%{{.*}}, MM2S, 0, <pkt_id = 0>)
// CHECK-DAG: aie.shim_dma_allocation @b_shim_alloc(%{{.*}}, MM2S, 0, <pkt_id = 1>)
// CHECK-DAG: aie.shim_dma_allocation @c_shim_alloc(%{{.*}}, MM2S, 0, <pkt_id = 2>)

module {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %t2 = aie.tile(0, 2)
    %t3 = aie.tile(0, 3)
    %t4 = aie.tile(0, 4)
    %t5 = aie.tile(0, 5)
    aie.shim_dma_allocation @fixed(%shim, MM2S, 0) {aie.fixed, aie.shareable_with = [@a, @b, @c]}
    aie.shim_dma_allocation @reserved(%shim, MM2S, 1)
    aie.packet_flow(31) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%t5, Core : 0>
    }
    aie.route_endpoint @a(%shim) DMA
    aie.route_endpoint @a_out(%t2) Core {channelIndex = 0 : i32}
    aie.route from @a to [@a_out]
    aie.runtime_sequence @run_a(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
    }
    aie.route_endpoint @b(%shim) DMA
    aie.route_endpoint @b_out(%t3) Core {channelIndex = 0 : i32}
    aie.route from @b to [@b_out]
    aie.runtime_sequence @run_b(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 0 : i64} : memref<16xi32>
    }
    aie.route_endpoint @c(%shim) DMA
    aie.route_endpoint @c_out(%t4) Core {channelIndex = 0 : i32}
    aie.route from @c to [@c_out]
    aie.runtime_sequence @run_c(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @c, id = 0 : i64} : memref<16xi32>
    }
  }
}
