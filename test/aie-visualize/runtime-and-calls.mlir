//===- runtime-and-calls.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Runtime arguments and kernel calls connect to their DMA channels and buffers.

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot %s | FileCheck %s --check-prefix=ALL
// RUN: aie-visualize --emit-dot --skip-runtime %s | FileCheck %s --check-prefix=SKIP
// RUN: aie-visualize --emit-dot --contract=port,buffer %s | FileCheck %s --check-prefix=TOPOLOGY

// ALL: tile_0_0 {{.*}} height=2.150000
// ALL: p_0_0_1_0_m {{.*}} label="MM2S0"{{.*}} class="port flow flow-0"
// ALL: p_0_0_1_1_s {{.*}} label="S2MM1"{{.*}} class="port flow flow-1"
// ALL: buffer_0 {{.*}} label="core_buffer"
// ALL: runtime_arg_0 {{.*}} label="%direct_input"{{.*}} class="runtime-argument flow flow-0 flow-4"
// ALL: runtime_arg_1 {{.*}} label="%allocated_input"{{.*}} class="runtime-argument flow flow-0"
// ALL: runtime_arg_2 {{.*}} label="%endpoint_input"{{.*}} class="runtime-argument flow flow-0"
// ALL: runtime_arg_3 {{.*}} label="%output"{{.*}} class="runtime-argument flow flow-1"
// ALL: runtime_arg_0 -> p_0_0_1_0_m {{.*}}class="runtime-link flow flow-0"
// ALL: runtime_arg_0 -> p_1_0_1_0_m {{.*}}class="runtime-link flow flow-4"
// ALL-NOT: runtime_arg_4
// ALL: runtime_arg_1 -> p_0_0_1_0_m
// ALL: runtime_arg_2 -> p_0_0_1_0_m
// ALL: p_0_0_1_0_s -> runtime_arg_2
// ALL: p_0_0_1_1_s -> runtime_arg_3 {{.*}}class="runtime-link flow flow-1"
// ALL-NOT: runtime_task_
// ALL: call_0 {{.*}} label="kernel"
// ALL-SAME: class="call detail-calls flow flow-2"
// ALL: buffer_0 -> call_0
// ALL: call_1 {{.*}} label="view_kernel"
// ALL-SAME: class="call detail-calls flow flow-2 flow-3"
// ALL: buffer_0 -> call_1 {{.*}}class="call-link detail-calls detail-buffers flow flow-2"
// ALL: buffer_1 -> call_1 {{.*}}class="call-link detail-calls detail-buffers flow flow-3"
// ALL: class="route flow flow-0"

// SKIP: label="MM2S0"
// SKIP: label="S2MM1"
// SKIP-NOT: runtime_arg_
// SKIP-NOT: runtime_task_

// TOPOLOGY: p_0_0_1_0_m {{.*}}label="MM2S0"{{.*}}class="port flow flow-0"
// TOPOLOGY: buffer_0 {{.*}}label="core_buffer"{{.*}}class="buffer detail-buffers flow flow-2"
// TOPOLOGY: runtime_arg_0 {{.*}} label="%direct_input"{{.*}} class="runtime-argument flow flow-0 flow-4"
// TOPOLOGY: runtime_arg_3 {{.*}} label="%output"{{.*}} class="runtime-argument flow flow-1"
// TOPOLOGY: call_0 {{.*}} label="kernel"{{.*}} class="call detail-calls flow flow-2"
// TOPOLOGY: call_1 {{.*}} label="view_kernel"{{.*}} class="call detail-calls flow flow-2 flow-3"
// TOPOLOGY-DAG: call_0 -> buffer_0 {{.*}}class="route detail-packet-ids detail-calls flow flow-2"
// TOPOLOGY-DAG: call_1 -> buffer_0 {{.*}}class="route detail-packet-ids detail-calls flow flow-2"
// TOPOLOGY-DAG: buffer_0 -> tile_0_3 {{.*}}class="route detail-packet-ids detail-calls flow flow-2"{{.*}}label=<<FONT COLOR="#1a9850">F2</FONT>>
// TOPOLOGY-DAG: call_1 -> tile_0_4 {{.*}}class="route detail-packet-ids detail-calls flow flow-3"
// TOPOLOGY-DAG: runtime_arg_0 -> p_0_0_1_0_m {{.*}}class="route detail-packet-ids flow flow-0"
// TOPOLOGY-DAG: runtime_arg_1 -> p_0_0_1_0_m {{.*}}class="route detail-packet-ids flow flow-0"
// TOPOLOGY-DAG: runtime_arg_2 -> p_0_0_1_0_m {{.*}}class="route detail-packet-ids flow flow-0"
// TOPOLOGY-DAG: p_0_0_1_0_m -> p_0_2_0_0 {{.*}}class="route detail-packet-ids flow flow-0"{{.*}}label=<<FONT COLOR="#d73027">F0</FONT>>
// TOPOLOGY-DAG: runtime_arg_0 -> tile_1_2 {{.*}}class="route detail-packet-ids flow flow-4"
// TOPOLOGY-DAG: p_0_2_0_0 -> runtime_arg_3 {{.*}}class="route detail-packet-ids flow flow-1"
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
