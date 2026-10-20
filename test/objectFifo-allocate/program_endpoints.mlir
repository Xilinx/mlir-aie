// RUN: aie-opt --split-input-file --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s

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
    aie.objectfifo.dma_endpoint @heavy_out(%mt) drains @heavy_pool {channelIndex = 0 : i32}
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

// A runtime task's BDs do not steer the choice, so the device image does not
// depend on the runtime sequence: an instruction stream built for another
// sequence runs on it. The task pinned to S2MM 2 leaves the program the
// lowest free channel, S2MM 0.

// CHECK-LABEL: @pinned_task_demand
// CHECK: aie.route_endpoint @prog(%{{.*}}) DMA {channelIndex = 0 : i32}
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
// CHECK:   aiex.dma_configure_task(%{{.*}}, MM2S, 0, <pkt_id = 0>) {
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
    aie.route from @spray to [@catch] {packet = #aie.packet_info<>}
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

// -----

// The same image whatever the runtime sequence: the even half holds the heavy
// pool's twelve BDs, so @b_in and @b_out take the odd channels 3 whether their
// runtime tasks have one BD or eight. When the tasks steered the choice, eight
// BDs each moved @b_out to MM2S 2, and an image built with one-BD tasks no
// longer matched an instruction stream built with eight-BD ones.

// CHECK-LABEL: @task_size_one
// CHECK-DAG: aie.route_endpoint @b_in(%{{.*}}) DMA {channelIndex = 3 : i32}
// CHECK-DAG: aie.route_endpoint @b_out(%{{.*}}) DMA {channelIndex = 3 : i32}
module @task_size_one {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mt = aie.tile(0, 1)
    %shim1 = aie.tile(1, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo.pool @heavy_pool(%mt) {depth = 6 : i32} : memref<16xi32> {
      aie.objectfifo.segment @hs {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.pool @light_pool(%mt) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @ls {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @heavy_in(%mt) fills @heavy_pool {channelIndex = 0 : i32}
    aie.objectfifo.dma_endpoint @heavy_out(%mt) drains @heavy_pool {channelIndex = 0 : i32}
    aie.objectfifo.dma_endpoint @light_in(%mt) fills @light_pool {channelIndex = 1 : i32}
    aie.objectfifo.dma_endpoint @light_out(%mt) drains @light_pool {channelIndex = 1 : i32}
    aie.route_endpoint @heavy_src(%shim) DMA {fifoName = "heavy"}
    aie.route_endpoint @heavy_dst(%shim) DMA {fifoName = "heavy_out"}
    aie.route_endpoint @light_src(%shim) DMA {fifoName = "light"}
    aie.route_endpoint @light_dst(%core) DMA
    aie.route from @heavy_src to [@heavy_in]
    aie.route from @heavy_out to [@heavy_dst]
    aie.route from @light_src to [@light_in]
    aie.route from @light_out to [@light_dst]
    %b = aie.buffer(%mt) {sym_name = "b"} : memref<256xi32>
    aie.route_endpoint @b_shim(%shim1) DMA {fifoName = "b"}
    aie.route_endpoint @b_in(%mt) DMA
    aie.route_endpoint @b_out(%mt) DMA
    aie.route_endpoint @b_sink(%shim1) DMA {fifoName = "b_sink"}
    aie.route from @b_shim to [@b_in]
    aie.route from @b_out to [@b_sink]
    aie.runtime_sequence(%arg0: memref<256xi32>) {
      %b_in = aiex.dma_configure_task_for @b_in {
        aie.dma_bd(%b : memref<256xi32> offset = 0 len = 16)
        aie.end
      }
      aiex.dma_start_task(%b_in)
      aiex.dma_free_task(%b_in)
      %b_out = aiex.dma_configure_task_for @b_out {
        aie.dma_bd(%b : memref<256xi32> offset = 0 len = 16)
        aie.end
      }
      aiex.dma_start_task(%b_out)
      aiex.dma_free_task(%b_out)
    }
  }
}

// -----

// CHECK-LABEL: @task_size_eight
// CHECK-DAG: aie.route_endpoint @b_in(%{{.*}}) DMA {channelIndex = 3 : i32}
// CHECK-DAG: aie.route_endpoint @b_out(%{{.*}}) DMA {channelIndex = 3 : i32}
module @task_size_eight {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mt = aie.tile(0, 1)
    %shim1 = aie.tile(1, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo.pool @heavy_pool(%mt) {depth = 6 : i32} : memref<16xi32> {
      aie.objectfifo.segment @hs {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.pool @light_pool(%mt) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @ls {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @heavy_in(%mt) fills @heavy_pool {channelIndex = 0 : i32}
    aie.objectfifo.dma_endpoint @heavy_out(%mt) drains @heavy_pool {channelIndex = 0 : i32}
    aie.objectfifo.dma_endpoint @light_in(%mt) fills @light_pool {channelIndex = 1 : i32}
    aie.objectfifo.dma_endpoint @light_out(%mt) drains @light_pool {channelIndex = 1 : i32}
    aie.route_endpoint @heavy_src(%shim) DMA {fifoName = "heavy"}
    aie.route_endpoint @heavy_dst(%shim) DMA {fifoName = "heavy_out"}
    aie.route_endpoint @light_src(%shim) DMA {fifoName = "light"}
    aie.route_endpoint @light_dst(%core) DMA
    aie.route from @heavy_src to [@heavy_in]
    aie.route from @heavy_out to [@heavy_dst]
    aie.route from @light_src to [@light_in]
    aie.route from @light_out to [@light_dst]
    %b = aie.buffer(%mt) {sym_name = "b"} : memref<256xi32>
    aie.route_endpoint @b_shim(%shim1) DMA {fifoName = "b"}
    aie.route_endpoint @b_in(%mt) DMA
    aie.route_endpoint @b_out(%mt) DMA
    aie.route_endpoint @b_sink(%shim1) DMA {fifoName = "b_sink"}
    aie.route from @b_shim to [@b_in]
    aie.route from @b_out to [@b_sink]
    aie.runtime_sequence(%arg0: memref<256xi32>) {
      %b_in = aiex.dma_configure_task_for @b_in {
        aie.dma_bd(%b : memref<256xi32> offset = 0 len = 16)
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%b : memref<256xi32> offset = 16 len = 16)
        aie.next_bd ^bd2
      ^bd2:
        aie.dma_bd(%b : memref<256xi32> offset = 32 len = 16)
        aie.next_bd ^bd3
      ^bd3:
        aie.dma_bd(%b : memref<256xi32> offset = 48 len = 16)
        aie.next_bd ^bd4
      ^bd4:
        aie.dma_bd(%b : memref<256xi32> offset = 64 len = 16)
        aie.next_bd ^bd5
      ^bd5:
        aie.dma_bd(%b : memref<256xi32> offset = 80 len = 16)
        aie.next_bd ^bd6
      ^bd6:
        aie.dma_bd(%b : memref<256xi32> offset = 96 len = 16)
        aie.next_bd ^bd7
      ^bd7:
        aie.dma_bd(%b : memref<256xi32> offset = 112 len = 16)
        aie.end
      }
      aiex.dma_start_task(%b_in)
      aiex.dma_free_task(%b_in)
      %b_out = aiex.dma_configure_task_for @b_out {
        aie.dma_bd(%b : memref<256xi32> offset = 0 len = 16)
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%b : memref<256xi32> offset = 16 len = 16)
        aie.next_bd ^bd2
      ^bd2:
        aie.dma_bd(%b : memref<256xi32> offset = 32 len = 16)
        aie.next_bd ^bd3
      ^bd3:
        aie.dma_bd(%b : memref<256xi32> offset = 48 len = 16)
        aie.next_bd ^bd4
      ^bd4:
        aie.dma_bd(%b : memref<256xi32> offset = 64 len = 16)
        aie.next_bd ^bd5
      ^bd5:
        aie.dma_bd(%b : memref<256xi32> offset = 80 len = 16)
        aie.next_bd ^bd6
      ^bd6:
        aie.dma_bd(%b : memref<256xi32> offset = 96 len = 16)
        aie.next_bd ^bd7
      ^bd7:
        aie.dma_bd(%b : memref<256xi32> offset = 112 len = 16)
        aie.end
      }
      aiex.dma_start_task(%b_out)
      aiex.dma_free_task(%b_out)
    }
  }
}

// -----

// Packet routes naming one endpoint share its channel: two cores merge into
// the MemTile's one S2MM channel, and the MemTile sends both packet ids out of
// one MM2S channel. Endpoints with more BDs pick first, then by structural
// rank, not source order: @scatter's two BDs take MM2S 0 and use up room in
// the even half, so @side goes to the odd S2MM 1 and @gather to S2MM 3.
// keep_pkt_header carries over to the packet flows. A program may source a
// packet route whose id is pinned, since its BDs state the header themselves,
// so the shared @scatter gets no header stamped. Each packet flow takes the
// place of its route.

// CHECK-LABEL: @shared_packet_channels
// CHECK: aie.route_endpoint @gather(%[[MT:.*]]) DMA {channelIndex = 3 : i32}
// CHECK: aie.packet_flow(1) {
// CHECK:   aie.packet_dest<%[[MT]], DMA : 3>
// CHECK: } {keep_pkt_header = true}
// CHECK: aie.packet_flow(2) {
// CHECK:   aie.packet_dest<%[[MT]], DMA : 3>
// CHECK: } {keep_pkt_header = true}
// CHECK: aie.route_endpoint @side(%[[MT]]) DMA {channelIndex = 1 : i32}
// CHECK: aie.route_endpoint @scatter(%[[MT]]) DMA {channelIndex = 0 : i32}
// CHECK-NOT: packet = #aie.packet_info
// CHECK: aie.packet_flow(1) {
// CHECK:   aie.packet_source<%[[MT]], DMA : 0>
// CHECK: aie.packet_flow(2) {
// CHECK:   aie.packet_source<%[[MT]], DMA : 0>
// CHECK: aie.dma_start(MM2S, 0, ^bb{{.*}}, ^bb{{.*}})
// CHECK: aie.dma_start(S2MM, 3, ^bb{{.*}}, ^bb{{.*}})
// CHECK: aie.dma_start(S2MM, 1, ^bb{{.*}}, ^bb{{.*}})
module @shared_packet_channels {
  aie.device(npu2) {
    %mt = aie.tile(0, 1)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    %c = aie.tile(0, 4)
    %buf = aie.buffer(%mt) {sym_name = "buf"} : memref<64xi32>
    %la = aie.buffer(%a) {sym_name = "la"} : memref<64xi32>
    %lb = aie.buffer(%b) {sym_name = "lb"} : memref<64xi32>
    aie.route_endpoint @a_out(%a) DMA
    aie.route_endpoint @b_out(%b) DMA
    aie.route_endpoint @gather(%mt) DMA
    aie.route from @a_out to [@gather] {keep_pkt_header, packet = #aie.packet_info<pkt_id = 1>}
    aie.route from @b_out to [@gather] {keep_pkt_header, packet = #aie.packet_info<pkt_id = 2>}
    aie.route_endpoint @c_out(%c) DMA
    aie.route_endpoint @side(%mt) DMA
    aie.route from @c_out to [@side]
    aie.route_endpoint @scatter(%mt) DMA
    aie.route_endpoint @a_in(%a) DMA
    aie.route_endpoint @b_in(%b) DMA
    aie.route from @scatter to [@a_in] {packet = #aie.packet_info<pkt_id = 1>}
    aie.route from @scatter to [@b_in] {packet = #aie.packet_info<pkt_id = 2>}
    aie.memtile_dma(%mt) {
      aie.dma_start(MM2S, @scatter, ^bd0, ^recv)
    ^bd0:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 32) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%buf : memref<64xi32> offset = 32 len = 32) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.next_bd ^bd0
    ^recv:
      aie.dma_start(S2MM, @gather, ^bd2, ^side)
    ^bd2:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^bd2
    ^side:
      aie.dma_start(S2MM, @side, ^bd3, ^end)
    ^bd3:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^bd3
    ^end:
      aie.end
    }
  }
}

// -----

// A shared endpoint takes one channel however many flows name it: a compute
// tile has two S2MM channels, 0 and 1, and three flows into it fit on them,
// @own on the channel @merge leaves.

// CHECK-LABEL: @shared_channel_counted_once
// CHECK: aie.route_endpoint @merge(%{{.*}}) DMA {channelIndex = [[#M:]] : i32}
// CHECK: aie.route_endpoint @own(%{{.*}}) DMA {channelIndex = [[#1-M]] : i32}
// CHECK: aie.packet_dest<%{{.*}}, DMA : [[#M]]>
// CHECK: aie.packet_dest<%{{.*}}, DMA : [[#M]]>
module @shared_channel_counted_once {
  aie.device(npu2) {
    %mt = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    %l1 = aie.buffer(%core) {sym_name = "l1"} : memref<64xi32>
    aie.route_endpoint @x(%mt) DMA
    aie.route_endpoint @y(%mt) DMA
    aie.route_endpoint @z(%mt) DMA
    aie.route_endpoint @merge(%core) DMA
    aie.route_endpoint @own(%core) DMA
    aie.route from @x to [@merge] {packet = #aie.packet_info<pkt_id = 1>}
    aie.route from @y to [@merge] {packet = #aie.packet_info<pkt_id = 2>}
    aie.route from @z to [@own]
    aie.mem(%core) {
      aie.dma_start(S2MM, @merge, ^bd0, ^own)
    ^bd0:
      aie.dma_bd(%l1 : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^bd0
    ^own:
      aie.dma_start(S2MM, @own, ^bd1, ^end)
    ^bd1:
      aie.dma_bd(%l1 : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^bd1
    ^end:
      aie.end
    }
  }
}

// -----

// A runtime task on a shared source keeps the headers its BDs state, and its
// configure gets none of its own.

// CHECK-LABEL: @shared_source_task_headers
// CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, [[#]]) {
// CHECK:   aie.dma_bd({{.*}}) {packet = #aie.packet_info<pkt_id = 1>}
// CHECK:   aie.dma_bd({{.*}}) {packet = #aie.packet_info<pkt_id = 2>}
module @shared_source_task_headers {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mt = aie.tile(0, 1)
    aie.route_endpoint @host(%shim) DMA
    aie.route_endpoint @a(%mt) DMA
    aie.route_endpoint @b(%mt) DMA
    aie.route from @host to [@a] {packet = #aie.packet_info<pkt_id = 1>}
    aie.route from @host to [@b] {packet = #aie.packet_info<pkt_id = 2>}
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t = aiex.dma_configure_task_for @host {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 32) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%arg0 : memref<64xi32> offset = 32 len = 32) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// A program on a shared source may state each header with aie.dma_bd_packet.

// CHECK-LABEL: @shared_source_bd_packet_ops
// CHECK: aie.dma_start(MM2S, [[#]], ^bb{{.*}}, ^bb{{.*}})
// CHECK: aie.dma_bd_packet(0, 1)
// CHECK: aie.dma_bd_packet(0, 2)
module @shared_source_bd_packet_ops {
  aie.device(npu2) {
    %mt = aie.tile(0, 1)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    %buf = aie.buffer(%mt) {sym_name = "buf"} : memref<64xi32>
    aie.route_endpoint @scatter(%mt) DMA
    aie.route_endpoint @a_in(%a) DMA
    aie.route_endpoint @b_in(%b) DMA
    aie.route from @scatter to [@a_in] {packet = #aie.packet_info<pkt_id = 1>}
    aie.route from @scatter to [@b_in] {packet = #aie.packet_info<pkt_id = 2>}
    aie.memtile_dma(%mt) {
      aie.dma_start(MM2S, @scatter, ^bd0, ^end)
    ^bd0:
      aie.dma_bd_packet(0, 1)
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 32)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd_packet(0, 2)
      aie.dma_bd(%buf : memref<64xi32> offset = 32 len = 32)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
  }
}

// -----

// A dma_memcpy_nd on a shared shim source states its own header, and reaches
// the channel through the shim allocation, which carries none.

// CHECK-LABEL: @shared_source_memcpy_header
// CHECK: aiex.npu.dma_memcpy_nd({{.*}}packet = <pkt_id = 2>) {{{.*}}metadata = @host_shim_alloc}
// CHECK: aie.shim_dma_allocation @host_shim_alloc(%{{.*}}, MM2S, [[#]]){{$}}
module @shared_source_memcpy_header {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mt = aie.tile(0, 1)
    aie.route_endpoint @host(%shim) DMA {fifoName = "host"}
    aie.route_endpoint @a(%mt) DMA
    aie.route_endpoint @b(%mt) DMA
    aie.route from @host to [@a] {packet = #aie.packet_info<pkt_id = 1>}
    aie.route from @host to [@b] {packet = #aie.packet_info<pkt_id = 2>}
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1], packet = <pkt_id = 2, pkt_type = 0>) {id = 0 : i64, metadata = @host} : memref<64xi32>
    }
  }
}

// -----

// A BD chain started on a shared shim source states each BD's header, so it
// reaches the channel through the shim allocation like a task does.

// CHECK-LABEL: @shared_source_bd_chain_headers
// CHECK: aie.dma_bd({{.*}}) {packet = #aie.packet_info<pkt_id = 1>}
// CHECK: aie.dma_bd({{.*}}) {packet = #aie.packet_info<pkt_id = 2>}
// CHECK: aiex.dma_start_bd_chain_for @chain(%{{.*}}) : (memref<64xi32>) for @host_shim_alloc
// CHECK: aie.shim_dma_allocation @host_shim_alloc(%{{.*}}, MM2S, [[#]]){{$}}
module @shared_source_bd_chain_headers {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mt = aie.tile(0, 1)
    aie.route_endpoint @host(%shim) DMA {fifoName = "host"}
    aie.route_endpoint @a(%mt) DMA
    aie.route_endpoint @b(%mt) DMA
    aie.route from @host to [@a] {packet = #aie.packet_info<pkt_id = 1>}
    aie.route from @host to [@b] {packet = #aie.packet_info<pkt_id = 2>}
    aie.bd_chain @chain(%buf: memref<64xi32>) {
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 32) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.next_bd ^b1
    ^b1:
      aie.dma_bd(%buf : memref<64xi32> offset = 32 len = 32) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t = aiex.dma_start_bd_chain_for @chain(%arg0) : (memref<64xi32>) for @host
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// The pass runs once per device; a second device reusing the first's endpoint
// names sees only its own flows.

// CHECK-LABEL: @shared_source_in_two_devices
// CHECK: aie.device(npu2) @first
// CHECK: aie.shim_dma_allocation @host_shim_alloc(%{{.*}}, MM2S, [[#]]){{$}}
// CHECK: aie.device(npu2) @second
// CHECK: aie.shim_dma_allocation @host_shim_alloc(%{{.*}}, MM2S, [[#]]){{$}}
module @shared_source_in_two_devices {
  aie.device(npu2) @first {
    %shim = aie.tile(0, 0)
    %mt = aie.tile(0, 1)
    aie.route_endpoint @host(%shim) DMA {fifoName = "host"}
    aie.route_endpoint @a(%mt) DMA
    aie.route_endpoint @b(%mt) DMA
    aie.route from @host to [@a] {packet = #aie.packet_info<pkt_id = 1>}
    aie.route from @host to [@b] {packet = #aie.packet_info<pkt_id = 2>}
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1], packet = <pkt_id = 1, pkt_type = 0>) {id = 0 : i64, metadata = @host} : memref<64xi32>
    }
  }
  aie.device(npu2) @second {
    %shim = aie.tile(0, 0)
    %mt = aie.tile(0, 1)
    aie.route_endpoint @host(%shim) DMA {fifoName = "host"}
    aie.route_endpoint @a(%mt) DMA
    aie.route_endpoint @b(%mt) DMA
    aie.route from @host to [@a] {packet = #aie.packet_info<pkt_id = 1>}
    aie.route from @host to [@b] {packet = #aie.packet_info<pkt_id = 2>}
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1], packet = <pkt_id = 2, pkt_type = 0>) {id = 0 : i64, metadata = @host} : memref<64xi32>
    }
  }
}
