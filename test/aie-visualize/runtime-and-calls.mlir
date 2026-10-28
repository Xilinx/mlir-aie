//===- runtime-and-calls.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Runtime arguments and kernel calls connect to their DMA channels and buffers.

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot --scale=1 %s | FileCheck %s --check-prefix=ALL
// RUN: aie-visualize --emit-dot --scale=1 --skip-runtime %s | FileCheck %s --check-prefix=SKIP
// RUN: aie-visualize --emit-dot --scale=1 --contract=port,buffer %s | FileCheck %s --check-prefix=TOPOLOGY
// RUN: aie-visualize --emit-dot --scale=0.5 %s | FileCheck %s --check-prefix=HALF

// A label keeps its point size however small the drawing is, so a box that
// has to hold one keeps its size too and only the space between tiles
// shrinks. Tile (0, 2) sits at 6.175 inches drawn full size.
// HALF: tile_0_0 {{.*}} height=2.150000
// HALF: tile_0_2 {{.*}} height=2.500000{{.*}}pos="0.000000,5.325000!"

// ALL: tile_0_0 {{.*}} height=2.150000
// ALL: p_0_0_1_0_m {{.*}} label="MM2S0"{{.*}} class="port flow src-3 dst-3"
// ALL: p_0_0_1_1_s {{.*}} label="S2MM1"{{.*}} class="port flow src-3 dst-3"
// ALL: buffer_0 {{.*}} label="core_buffer"
// An argument the host writes is an origin, and one it reads is a fate.
// ALL: runtime_arg_0 {{.*}} label="%direct_input"{{.*}} class="runtime-argument flow src-5 dst-5"
// ALL: runtime_arg_1 {{.*}} label="%allocated_input"{{.*}} class="runtime-argument flow src-6 dst-3"
// ALL: runtime_arg_2 {{.*}} label="%endpoint_input"{{.*}} class="runtime-argument flow src-4 dst-3"
// ALL: runtime_arg_3 {{.*}} label="%output"{{.*}} class="runtime-argument flow src-3 dst-3"
// ALL: call_0 {{.*}} label="kernel"
// ALL-SAME: class="call detail-calls flow src-2"
// ALL: call_1 {{.*}} label="view_kernel"
// ALL-SAME: class="call detail-calls flow src-1"
// One argument feeding two shims keeps its origin on both links.
// ALL-DAG: runtime_arg_0 -> p_0_0_1_0_m {{.*}}class="runtime-link flow src-5 dst-3"
// ALL-DAG: runtime_arg_0 -> p_1_0_1_0_m {{.*}}class="runtime-link flow src-5 dst-4"
// ALL-DAG: runtime_arg_1 -> p_0_0_1_0_m
// ALL-DAG: runtime_arg_2 -> p_0_0_1_0_m
// A route endpoint states no direction, so the argument reaches both.
// ALL-DAG: p_0_0_1_0_s -> runtime_arg_2 {{.*}}class="runtime-link flow src-4 dst-3"
// ALL-DAG: p_0_0_1_1_s -> runtime_arg_3 {{.*}}class="runtime-link flow src-3 dst-3"
// A kernel that writes a buffer is what feeds the flow leaving it.
// ALL-DAG: call_0 -> buffer_0 {{.*}}class="call-link detail-calls detail-buffers flow src-2 dst-0"
// ALL-DAG: call_1 -> buffer_0 {{.*}}class="call-link detail-calls detail-buffers flow src-1 dst-0"
// ALL-DAG: call_1 -> buffer_1 {{.*}}class="call-link detail-calls detail-buffers flow src-1 dst-1"
// ALL-NOT: runtime_arg_4
// ALL-NOT: runtime_task_

// SKIP: label="MM2S0"
// SKIP: label="S2MM1"
// SKIP-NOT: runtime_arg_
// SKIP-NOT: runtime_task_

// TOPOLOGY: p_0_0_1_0_m {{.*}}label="MM2S0"{{.*}}class="port flow src-4 dst-2"
// TOPOLOGY: buffer_0 {{.*}}label="core_buffer"{{.*}}class="buffer detail-buffers flow src-0 dst-0"
// TOPOLOGY: runtime_arg_0 {{.*}} label="%direct_input"{{.*}} class="runtime-argument flow src-6 dst-5"
// TOPOLOGY: runtime_arg_3 {{.*}} label="%output"{{.*}} class="runtime-argument flow src-4 dst-2"
// TOPOLOGY: call_0 {{.*}} label="kernel"{{.*}} class="call detail-calls flow src-2"
// TOPOLOGY: call_1 {{.*}} label="view_kernel"{{.*}} class="call detail-calls flow src-3"
// A kernel writing a buffer is what feeds the flow that leaves it.
// TOPOLOGY-DAG: call_0 -> buffer_0 {{.*}}class="call-link detail-calls detail-buffers flow src-2 dst-0"
// TOPOLOGY-DAG: call_1 -> buffer_0 {{.*}}class="call-link detail-calls detail-buffers flow src-3 dst-0"
// A DMA channel stands between a buffer and the tile it leaves for.
// TOPOLOGY-DAG: buffer_0 -> p_0_2_1_0_m {{.*}}class="buffer-link detail-buffers flow src-0 dst-0"
// TOPOLOGY-DAG: p_0_2_1_0_m -> tile_0_3 {{.*}}class="route detail-packet-ids flow src-0 dst-0"
// Contracting a buffer away leaves an edge with no buffer on it, which
// hiding the buffers must leave alone.
// TOPOLOGY-DAG: call_1 -> p_0_2_1_1_m {{.*}}class="call-link detail-calls flow src-3 dst-3"
// TOPOLOGY-DAG: p_0_2_1_1_m -> tile_0_4 {{.*}}class="route detail-packet-ids flow src-3 dst-3"
// Three arguments feed the same shim channel, and each keeps its own origin.
// TOPOLOGY-DAG: runtime_arg_0 -> p_0_0_1_0_m {{.*}}class="runtime-link flow src-6 dst-2"
// TOPOLOGY-DAG: runtime_arg_1 -> p_0_0_1_0_m {{.*}}class="runtime-link flow src-7 dst-2"
// TOPOLOGY-DAG: runtime_arg_2 -> p_0_0_1_0_m {{.*}}class="runtime-link flow src-5 dst-2"
// TOPOLOGY-DAG: p_0_0_1_0_m -> p_0_0_1_1_s {{.*}}class="route detail-packet-ids flow src-4 dst-2"
// TOPOLOGY-DAG: runtime_arg_0 -> p_1_0_1_0_m {{.*}}class="runtime-link flow src-6 dst-4"
// TOPOLOGY-DAG: p_1_0_1_0_m -> tile_1_2 {{.*}}class="route detail-packet-ids flow src-6 dst-4"
// TOPOLOGY-DAG: p_0_0_1_1_s -> runtime_arg_3 {{.*}}class="runtime-link flow src-4 dst-2"
// TOPOLOGY-NOT: topology_junction_

module {
  aie.device(npu1_2col) {
    func.func private @kernel(memref<16xi32>)
    func.func private @view_kernel(memref<16xi32, strided<[1]>>)

    %shim = aie.tile(0, 0)
    %second_shim = aie.tile(1, 0)
    %core_tile = aie.tile(0, 2)
    %second_core = aie.tile(1, 2)
    %sink0 = aie.tile(0, 3)
    %sink1 = aie.tile(0, 4)
    %core_buffer = aie.buffer(%core_tile) {sym_name = "core_buffer"} : memref<16xi32>
    %alternate_buffer = aie.buffer(%core_tile) {sym_name = "alternate_buffer"} : memref<16xi32>

    aie.core(%core_tile) {
      %true = arith.constant true
      func.call @kernel(%core_buffer) : (memref<16xi32>) -> ()
      cf.cond_br %true, ^bb1(%core_buffer : memref<16xi32>), ^bb1(%alternate_buffer : memref<16xi32>)
    ^bb1(%selected: memref<16xi32>):
      %view = memref.subview %selected[0] [16] [1] : memref<16xi32> to memref<16xi32, strided<[1]>>
      func.call @view_kernel(%view) : (memref<16xi32, strided<[1]>>) -> ()
      aie.end
    }

    aie.mem(%core_tile) {
      %0 = aie.dma_start(MM2S, 0, ^core_bd, ^alternate_start)
    ^core_bd:
      aie.dma_bd(%core_buffer : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^core_bd
    ^alternate_start:
      %1 = aie.dma_start(MM2S, 1, ^alternate_bd, ^end)
    ^alternate_bd:
      aie.dma_bd(%alternate_buffer : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^alternate_bd
    ^end:
      aie.end
    }

    aie.flow(%shim, DMA : 0, %core_tile, Core : 0)
      via (%shim : DMA : 0 -> North : 0,
           %core_tile : South : 0 -> Core : 0)
    aie.flow(%core_tile, Core : 0, %shim, DMA : 1)
      via (%core_tile : Core : 0 -> South : 1,
           %shim : North : 1 -> DMA : 1)
        aie.flow(%core_tile, DMA : 0, %sink0, Core : 0)
          via (%core_tile : DMA : 0 -> North : 2,
               %sink0 : South : 2 -> Core : 0)
        aie.flow(%core_tile, DMA : 1, %sink1, Core : 0)
          via (%core_tile : DMA : 1 -> North : 3,
               %sink0 : South : 3 -> North : 3,
               %sink1 : South : 3 -> Core : 0)
    aie.flow(%second_shim, DMA : 0, %second_core, Core : 0)
      via (%second_shim : DMA : 0 -> North : 0,
           %second_core : South : 0 -> Core : 0)

    aie.shim_dma_allocation @input_alloc(%shim, MM2S, 0)
    aie.shim_dma_allocation @output_alloc(%shim, S2MM, 1)
    aie.route_endpoint @input_endpoint(%shim) DMA {channelIndex = 0 : i32}
    aie.runtime_sequence @run(
        %direct_input: memref<16xi32>,
        %allocated_input: memref<16xi32>,
      %endpoint_input: memref<16xi32>,
        %output: memref<16xi32>) attributes {
          runtime_arg_names = ["direct_input", "allocated_input", "endpoint_input", "output"]
        } {
      %direct = aiex.dma_configure_task(%shim, MM2S, 0) {
        aie.dma_bd(%direct_input : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      %second_direct = aiex.dma_configure_task(%second_shim, MM2S, 0) {
        aie.dma_bd(%direct_input : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      %allocated = aiex.dma_configure_task_for @input_alloc {
        aie.dma_bd(%allocated_input : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      %endpoint = aiex.dma_configure_task_for @input_endpoint {
        aie.dma_bd(%endpoint_input : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      aiex.npu.dma_memcpy_nd(%output[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {
        metadata = @output_alloc, id = 0 : i64
      } : memref<16xi32>
    }
  }
}
