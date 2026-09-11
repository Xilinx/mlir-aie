// RUN: aie-opt --aie-objectfifo-allocate %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocating the larger, low-demand pool first would spill both it and the
// four-input pool. Their five S2MM endpoints would then compete for the four
// channels that can reach adjacent MemTile memory.

module @memtile_spill_channel_priority {
  aie.device(npu2) {
    %mem0 = aie.tile(0, 1)
    %mem1 = aie.tile(1, 1)
    %reserved = aie.buffer(%mem0) {sym_name = "reserved"} : memref<89600xi32>

    aie.objectfifo.pool @low_demand(%mem0) {
      depth = 2 : i32
    } : memref<25600xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 25600 : i32}
    }
    aie.objectfifo.dma_endpoint @low(%mem0) fills @low_demand

    aie.objectfifo.pool @high_demand(%mem0) {
      depth = 2 : i32
    } : memref<10240xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 10240 : i32}
    }
    aie.objectfifo.dma_endpoint @high0(%mem0) fills @high_demand
    aie.objectfifo.dma_endpoint @high1(%mem0) fills @high_demand
    aie.objectfifo.dma_endpoint @high2(%mem0) fills @high_demand
    aie.objectfifo.dma_endpoint @high3(%mem0) fills @high_demand
  }
}

// CHECK-LABEL: @memtile_spill_channel_priority
// CHECK-DAG: %[[MEM0:.*]] = aie.tile(0, 1)
// CHECK-DAG: %[[MEM1:.*]] = aie.tile(1, 1)
// CHECK-DAG: aie.buffer(%[[MEM0]]) {sym_name = "high_demand_buff_0"} : memref<10240xi32>
// CHECK-DAG: aie.buffer(%[[MEM0]]) {sym_name = "high_demand_buff_1"} : memref<10240xi32>
// CHECK-DAG: aie.buffer(%[[MEM1]]) {sym_name = "low_demand_buff_0"} : memref<25600xi32>
// CHECK-DAG: aie.buffer(%[[MEM1]]) {sym_name = "low_demand_buff_1"} : memref<25600xi32>
// CHECK-DAG: aie.objectfifo.dma_endpoint @low({{.*}}) fills @low_demand {channelIndex = 0 : i32}
// CHECK-DAG: aie.objectfifo.dma_endpoint @high0({{.*}}) fills @high_demand {channelIndex = 1 : i32}
// CHECK-DAG: aie.objectfifo.dma_endpoint @high1({{.*}}) fills @high_demand {channelIndex = 2 : i32}
// CHECK-DAG: aie.objectfifo.dma_endpoint @high2({{.*}}) fills @high_demand {channelIndex = 3 : i32}
// CHECK-DAG: aie.objectfifo.dma_endpoint @high3({{.*}}) fills @high_demand {channelIndex = 4 : i32}
