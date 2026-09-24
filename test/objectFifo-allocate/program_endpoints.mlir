// RUN: aie-opt --split-input-file --aie-objectfifo-allocate %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A DMA channel the design programs itself, named by an aie.route_endpoint.
// Allocation picks its index after the objectFIFOs', resolves every
// aie.dma_start naming it to that index, and turns every runtime task naming it
// into an aiex.dma_configure_task on the endpoint's tile and channel.

// A MemTile's even channels use BDs 0-23 and its odd ones BDs 24-47. The odd
// half holds the heavy pool's 20 descriptors, so the program takes the even
// MM2S 2 over the first free channel, the odd MM2S 1.

// CHECK-LABEL: @parity
// CHECK: aie.route_endpoint @b_out(%[[MT:.*]]) DMA {channelIndex = 2 : i32}
// CHECK: aie.memtile_dma(%[[MT]]) {
// CHECK:   aie.dma_start(MM2S, 2, ^bb1, ^bb3)
module @parity {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mt = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.objectfifo.pool @heavy_pool(%mt) {depth = 10 : i32} : memref<16xi32> {
      aie.objectfifo.segment @hs {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @heavy_in(%mt) fills @heavy_pool {channelIndex = 1 : i32}
    aie.objectfifo.dma_endpoint @heavy_out(%mt) drains @heavy_pool {channelIndex = 3 : i32}
    aie.objectfifo.pool @light_pool(%mt) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @ls {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @light_in(%mt) fills @light_pool {channelIndex = 0 : i32}
    aie.objectfifo.dma_endpoint @light_out(%mt) drains @light_pool {channelIndex = 0 : i32}
    aie.route_endpoint @heavy_src(%shim) DMA {fifoName = "heavy"}
    aie.route_endpoint @heavy_dst(%shim) DMA {fifoName = "heavy_out"}
    aie.route_endpoint @light_src(%shim) DMA {fifoName = "light"}
    aie.route_endpoint @light_dst(%core) DMA
    aie.route from @heavy_src to [@heavy_in]
    aie.route from @heavy_out to [@heavy_dst]
    aie.route from @light_src to [@light_in]
    aie.route from @light_out to [@light_dst]

    %b = aie.buffer(%mt) {sym_name = "b"} : memref<64xi32>
    aie.route_endpoint @b_out(%mt) DMA
    aie.route_endpoint @b_dst(%shim) DMA {fifoName = "b"}
    aie.route from @b_out to [@b_dst]
    aie.memtile_dma(%mt) {
      aie.dma_start(MM2S, @b_out, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<64xi32> offset = 0 len = 32)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%b : memref<64xi32> offset = 32 len = 32)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
  }
}

// -----

// Both halves hold 22 BDs, so the larger program, @b_out, takes the lowest
// free channel, and its two BDs leave the smaller one room only on an odd
// channel. A core's BDs serve all its channels alike, so @b_core takes the
// lowest. The runtime task on @b_in is configured on its channel directly, and
// the one on the shim endpoint follows the shim allocation's rename.

// CHECK-LABEL: @programs_and_tasks
// CHECK-DAG: aie.route_endpoint @b_shim(%[[SHIM:.*]]) DMA {channelIndex = 1 : i32, fifoName = "b"}
// CHECK-DAG: aie.route_endpoint @b_in(%[[MT:.*]]) DMA {channelIndex = 3 : i32}
// CHECK-DAG: aie.route_endpoint @b_out(%[[MT]]) DMA {channelIndex = 2 : i32}
// CHECK-DAG: aie.route_endpoint @b_core(%[[CORE:.*]]) DMA {channelIndex = 0 : i32}
// CHECK-DAG: aie.flow(%[[SHIM]], DMA : 1, %[[MT]], DMA : 3)
// CHECK-DAG: aie.flow(%[[MT]], DMA : 2, %[[CORE]], DMA : 0)
// CHECK: aie.memtile_dma(%[[MT]]) {
// CHECK:   aie.dma_start(MM2S, 2, ^bb1, ^bb3)
// CHECK: aie.mem(%[[CORE]]) {
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK: aie.runtime_sequence
// CHECK:   %[[T0:.*]] = aiex.dma_configure_task_for @b_shim_alloc {
// CHECK:   aiex.dma_start_task(%[[T0]])
// CHECK:   %[[T1:.*]] = aiex.dma_configure_task(%[[MT]], S2MM, 3) {
// CHECK:     aie.dma_bd(%{{.*}} : memref<64xi32> offset = 0 len = 64)
// CHECK:   } {repeat_count = 3 : i32}
// CHECK:   aiex.dma_start_task(%[[T1]])
// CHECK:   aiex.dma_free_task(%[[T1]])
// CHECK: aie.shim_dma_allocation @b_shim_alloc(%[[SHIM]], MM2S, 1)
module @programs_and_tasks {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    %mt = aie.tile(0, 1)
    %core = aie.tile(0, 2)

    aie.objectfifo.pool @heavy_pool(%mt) {depth = 20 : i32} : memref<16xi32> {
      aie.objectfifo.segment @hs {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.pool @light_pool(%mt) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @ls {offset = 0 : i32, size = 16 : i32}
    }
    aie.route_endpoint @heavy_shim(%shim1) DMA {fifoName = "heavy"}
    aie.objectfifo.dma_endpoint @heavy_in(%mt) fills @heavy_pool {channelIndex = 1 : i32}
    aie.objectfifo.dma_endpoint @heavy_out(%mt) drains @heavy_pool
    aie.route_endpoint @heavy_sink(%shim1) DMA {fifoName = "heavy_out"}
    aie.route from @heavy_shim to [@heavy_in]
    aie.route from @heavy_out to [@heavy_sink]
    aie.route_endpoint @light_shim(%shim) DMA {fifoName = "light"}
    aie.objectfifo.dma_endpoint @light_in(%mt) fills @light_pool
    aie.objectfifo.dma_endpoint @light_out(%mt) drains @light_pool
    aie.route_endpoint @light_sink(%shim) DMA {fifoName = "light_out"}
    aie.route from @light_shim to [@light_in]
    aie.route from @light_out to [@light_sink]

    %b = aie.buffer(%mt) {sym_name = "b"} : memref<64xi32>
    %b_l1 = aie.buffer(%core) {sym_name = "b_l1"} : memref<32xi32>
    aie.route_endpoint @b_shim(%shim) DMA {fifoName = "b"}
    aie.route_endpoint @b_in(%mt) DMA
    aie.route_endpoint @b_out(%mt) DMA
    aie.route_endpoint @b_core(%core) DMA
    aie.route from @b_shim to [@b_in]
    aie.route from @b_out to [@b_core]

    aie.memtile_dma(%mt) {
      aie.dma_start(MM2S, @b_out, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<64xi32> offset = 0 len = 32)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%b : memref<64xi32> offset = 32 len = 32)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    aie.mem(%core) {
      aie.dma_start(S2MM, @b_core, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b_l1 : memref<32xi32> offset = 0 len = 32)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }

    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t0 = aiex.dma_configure_task_for @b_shim {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @b_in {
        aie.dma_bd(%b : memref<64xi32> offset = 0 len = 64)
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%t1)
      aiex.dma_free_task(%t1)
    }
  }
}

// -----

// The BDs of a runtime task count against its channel's half just as a static
// program's do: the pinned task on S2MM 2 takes three of the even half's BDs,
// so the program takes the odd S2MM 1 over the first free channel, S2MM 0.

// CHECK-LABEL: @pinned_task_demand
// CHECK: aie.route_endpoint @prog(%{{.*}}) DMA {channelIndex = 1 : i32}
module @pinned_task_demand {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mt = aie.tile(0, 1)
    %buf = aie.buffer(%mt) {sym_name = "buf"} : memref<64xi32>
    aie.route_endpoint @src(%shim) DMA {fifoName = "src"}
    aie.route_endpoint @prog(%mt) DMA
    aie.route from @src to [@prog]
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %big = aiex.dma_configure_task(%mt, S2MM, 2) {
        aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 4)
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%buf : memref<64xi32> offset = 4 len = 4)
        aie.next_bd ^bd2
      ^bd2:
        aie.dma_bd(%buf : memref<64xi32> offset = 8 len = 4)
        aie.end
      }
      aiex.dma_start_task(%big)
      aiex.dma_free_task(%big)
      %t = aiex.dma_configure_task_for @prog {
        aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_free_task(%t)
    }
  }
}

// -----

// A shim endpoint without a fifoName gets no shim allocation, so its runtime
// task is configured on the shim's channel directly too. The source of a
// packet-switched route stamps its header on the task it becomes.

// CHECK-LABEL: @shim_and_packet
// CHECK: aie.runtime_sequence
// CHECK:   aiex.dma_configure_task(%{{.*}}, MM2S, 0) {
// CHECK:   aiex.dma_configure_task(%{{.*}}, MM2S, 0, <pkt_type = 0, pkt_id = 0>) {
// CHECK-NOT: aie.shim_dma_allocation
module @shim_and_packet {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mt = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    %buf = aie.buffer(%mt) {sym_name = "buf"} : memref<64xi32>
    %l1 = aie.buffer(%core) {sym_name = "l1"} : memref<64xi32>
    aie.route_endpoint @host(%shim) DMA
    aie.route_endpoint @landing(%mt) DMA
    aie.route from @host to [@landing]
    aie.route_endpoint @spray(%mt) DMA
    aie.route_endpoint @catch(%core) DMA
    aie.route from @spray to [@catch] {packet}
    aie.mem(%core) {
      aie.dma_start(S2MM, @catch, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%l1 : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t0 = aiex.dma_configure_task_for @host {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @spray {
        aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%t1)
    }
  }
}

// -----

// A program whose BDs read a neighbor's buffer can only use MemTile channels
// 0-3. The heavy pool fills the odd half, so by BD room alone it would take the
// even MM2S 4; the restriction leaves it the odd MM2S 3.

// CHECK-LABEL: @adjacent_program
// CHECK: aie.route_endpoint @reach(%{{.*}}) DMA {channelIndex = 3 : i32}
// CHECK: aie.dma_start(MM2S, 3, ^bb{{.*}}, ^bb{{.*}})
module @adjacent_program {
  aie.device(npu2) {
    %shim = aie.tile(1, 0)
    %west = aie.tile(0, 1)
    %mt = aie.tile(1, 1)
    %far = aie.buffer(%west) {sym_name = "far"} : memref<64xi32>
    %near = aie.buffer(%mt) {sym_name = "near"} : memref<64xi32>
    aie.objectfifo.pool @heavy(%mt) {depth = 10 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @heavy_in(%mt) fills @heavy {channelIndex = 1 : i32}
    aie.objectfifo.dma_endpoint @heavy_out(%mt) drains @heavy {channelIndex = 1 : i32}
    aie.route_endpoint @heavy_src(%shim) DMA {fifoName = "heavy"}
    aie.route_endpoint @heavy_dst(%shim) DMA {fifoName = "heavy_out"}
    aie.route from @heavy_src to [@heavy_in]
    aie.route from @heavy_out to [@heavy_dst]
    aie.route_endpoint @reach(%mt) DMA
    aie.route_endpoint @reach_dst(%shim) DMA {fifoName = "reach"}
    aie.route from @reach to [@reach_dst]
    aie.memtile_dma(%mt) {
      aie.dma_start(MM2S, 0, ^bd0, ^start2)
    ^bd0:
      aie.dma_bd(%near : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^bd0
    ^start2:
      aie.dma_start(MM2S, 2, ^bd2, ^start3)
    ^bd2:
      aie.dma_bd(%near : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^bd2
    ^start3:
      aie.dma_start(MM2S, @reach, ^bd3, ^end)
    ^bd3:
      aie.dma_bd(%far : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^bd3
    ^end:
      aie.end
    }
  }
}
