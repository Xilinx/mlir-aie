//===- follow-through-buffers.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Flow grouping crosses serial buffers but retains buffer fan-out and join nodes.

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot %s | FileCheck %s --check-prefix=ALL
// RUN: aie-visualize --emit-dot --only-flow=0 %s | FileCheck %s --check-prefix=GROUP
// RUN: aie-visualize --emit-dot --only-flow=1 %s | FileCheck %s --check-prefix=OTHER
// RUN: aie-visualize --emit-dot --only-flow=2 %s | FileCheck %s --check-prefix=FANOUT
// RUN: aie-visualize --emit-dot --only-flow=3 %s | FileCheck %s --check-prefix=JOIN
// RUN: aie-visualize --emit-dot --no-follow-buffers --only-flow=8 %s | FileCheck %s --check-prefix=UNGROUPED
// RUN: aie-visualize --emit-dot --contract=port,buffer %s | FileCheck %s --check-prefix=TOPOLOGY

// ALL-DAG: p_0_2_1_0_m -> p_0_2_5_0 [color="#d73027"
// ALL-DAG: p_0_2_5_0 -> p_0_3_3_0 {{.*}} label=<<FONT COLOR="#d73027">F0</FONT>>
// ALL-DAG: p_0_3_1_0_m -> p_0_3_5_1 [color="#d73027"
// ALL-DAG: p_0_3_1_1_m -> p_0_3_3_3 [color="#d73027"
// ALL-DAG: p_0_3_5_1 -> p_0_4_3_1 {{.*}} label=<<FONT COLOR="#d73027">F0</FONT>>
// ALL-DAG: p_0_3_5_2 -> p_0_3_1_1_s [color="#d73027"
// ALL-DAG: p_0_4_1_1_m -> p_0_4_5_0 [color="#4575b4"
// ALL-DAG: p_0_4_5_0 -> p_0_5_3_0 {{.*}} label=<<FONT COLOR="#4575b4">F1</FONT>>

// GROUP: p_0_3_1_0_s [shape=box, fixedsize=true,{{.*}} label="S2MM0"
// GROUP: p_0_3_1_0_m [shape=box, fixedsize=true,{{.*}} label="MM2S0"
// GROUP: label="bridge"
// GROUP-NOT: memref
// GROUP-DAG: p_0_2_1_0_m -> p_0_2_5_0
// GROUP-DAG: p_0_3_1_0_m -> p_0_3_5_1
// GROUP-DAG: p_0_3_1_1_m -> p_0_3_3_3
// GROUP-DAG: p_0_3_5_2 -> p_0_3_1_1_s
// GROUP-DAG: p_0_3_1_0_s -> buffer_0 {{.*}}class="buffer-link{{.*}}];
// GROUP-DAG: buffer_0 -> p_0_3_1_0_m {{.*}}class="buffer-link{{.*}}];
// GROUP-DAG: p_0_3_1_1_s -> buffer_0 {{.*}}class="buffer-link{{.*}}];
// GROUP-DAG: buffer_0 -> p_0_3_1_1_m {{.*}}class="buffer-link{{.*}}];
// GROUP-NOT: p_0_4_0_1

// OTHER-NOT: label="bridge"
// OTHER: p_0_4_1_1_m -> p_0_4_5_0
// OTHER: p_0_4_5_0 -> p_0_5_3_0 {{.*}} label=<<FONT COLOR="#4575b4">F1</FONT>>

// FANOUT: label="fanout"
// FANOUT-DAG: buffer_1 -> p_0_5_1_0_m {{.*}}class="buffer-link{{.*}}];
// FANOUT-DAG: buffer_1 -> p_0_5_1_1_m {{.*}}class="buffer-link{{.*}}];

// JOIN: label="join"
// JOIN-DAG: p_0_2_1_0_s -> buffer_2 {{.*}}class="buffer-link{{.*}}];
// JOIN-DAG: p_0_2_1_1_s -> buffer_2 {{.*}}class="buffer-link{{.*}}];

// UNGROUPED: F8

// TOPOLOGY-DAG: buffer_0 {{.*}}label="bridge"{{.*}}class="buffer detail-buffers flow flow-0"
// TOPOLOGY-DAG: buffer_1 {{.*}}label="fanout"{{.*}}class="buffer detail-buffers flow flow-2"
// TOPOLOGY-DAG: buffer_2 {{.*}}label="join"{{.*}}class="buffer detail-buffers flow flow-3"
// Contracting ports removes the switchbox hops a flow passes through, not
// the DMA channel it leaves or arrives on, so a buffer reaches a switchbox
// by way of its channel.
// TOPOLOGY-DAG: buffer_1 -> p_0_5_1_0_m {{.*}}class="buffer-link detail-buffers flow flow-2"
// TOPOLOGY-DAG: p_0_5_1_0_m -> p_0_3_0_0 {{.*}}class="route detail-packet-ids flow flow-2"
// TOPOLOGY-DAG: buffer_1 -> p_0_5_1_1_m {{.*}}class="buffer-link detail-buffers flow flow-2"
// TOPOLOGY-DAG: p_0_5_1_1_m -> p_0_4_0_0 {{.*}}class="route detail-packet-ids flow flow-2"
// TOPOLOGY-DAG: p_0_3_0_0 -> p_0_2_1_0_s {{.*}}flow-3
// TOPOLOGY-DAG: p_0_2_1_0_s -> buffer_2 {{.*}}class="buffer-link detail-buffers{{.*}}flow-3
// TOPOLOGY-DAG: p_0_5_0_0 -> p_0_2_1_1_s {{.*}}flow-3
// TOPOLOGY-DAG: p_0_2_1_1_s -> buffer_2 {{.*}}class="buffer-link detail-buffers{{.*}}flow-3
// TOPOLOGY-NOT: tile_0_5 -> buffer_1
// TOPOLOGY-NOT: buffer_2 -> tile_0_2
// TOPOLOGY-NOT: topology_junction_
// TOPOLOGY-NOT: tile_0_2 -> tile_0_2
// TOPOLOGY-NOT: tile_0_3 -> tile_0_3
// TOPOLOGY-NOT: tile_0_4 -> tile_0_4

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)

    %bridge = aie.buffer(%t03) {sym_name = "bridge"} : memref<16xi32>
    %fanout = aie.buffer(%t05) {sym_name = "fanout"} : memref<16xi32>
    %join = aie.buffer(%t02) {sym_name = "join"} : memref<16xi32>

    aie.mem(%t05) {
      %0 = aie.dma_start(MM2S, 0, ^mm2s0, ^mm2s1_start)
    ^mm2s0:
      aie.dma_bd(%fanout : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^mm2s0
    ^mm2s1_start:
      %1 = aie.dma_start(MM2S, 1, ^mm2s1, ^end)
    ^mm2s1:
      aie.dma_bd(%fanout : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^mm2s1
    ^end:
      aie.end
    }

    aie.mem(%t03) {
      %0 = aie.dma_start(S2MM, 0, ^s2mm0, ^s2mm2_start)
    ^s2mm0:
      aie.dma_bd(%bridge : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^s2mm0
    ^s2mm2_start:
      %1 = aie.dma_start(S2MM, 1, ^s2mm1, ^mm2s0_start)
    ^s2mm1:
      aie.dma_bd(%bridge : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^s2mm1
    ^mm2s0_start:
      %2 = aie.dma_start(MM2S, 0, ^mm2s0, ^mm2s3_start)
    ^mm2s0:
      aie.dma_bd(%bridge : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^mm2s0
    ^mm2s3_start:
      %3 = aie.dma_start(MM2S, 1, ^mm2s1, ^end)
    ^mm2s1:
      aie.dma_bd(%bridge : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^mm2s1
    ^end:
      aie.end
    }

    aie.mem(%t02) {
      %0 = aie.dma_start(S2MM, 0, ^s2mm0, ^s2mm1_start)
    ^s2mm0:
      aie.dma_bd(%join : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^s2mm0
    ^s2mm1_start:
      %1 = aie.dma_start(S2MM, 1, ^s2mm1, ^end)
    ^s2mm1:
      aie.dma_bd(%join : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^s2mm1
    ^end:
      aie.end
    }

    aie.flow(%t02, DMA : 0, %t03, DMA : 0)
      via (%t02 : DMA : 0 -> North : 0,
           %t03 : South : 0 -> DMA : 0)
    aie.flow(%t03, DMA : 0, %t04, DMA : 0)
      via (%t03 : DMA : 0 -> North : 1,
           %t04 : South : 1 -> DMA : 0)
    aie.flow(%t04, Core : 0, %t03, DMA : 1)
      via (%t04 : Core : 0 -> South : 2,
           %t03 : North : 2 -> DMA : 1)
    aie.flow(%t03, DMA : 1, %t02, Core : 0)
      via (%t03 : DMA : 1 -> South : 3,
           %t02 : North : 3 -> Core : 0)
    aie.flow(%t04, DMA : 1, %t05, Core : 0)
      via (%t04 : DMA : 1 -> North : 0,
           %t05 : South : 0 -> Core : 0)
    aie.flow(%t05, DMA : 0, %t03, Core : 0)
      via (%t05 : DMA : 0 -> South : 4,
           %t04 : North : 4 -> South : 4,
           %t03 : North : 4 -> Core : 0)
    aie.flow(%t05, DMA : 1, %t04, Core : 0)
      via (%t05 : DMA : 1 -> South : 3,
           %t04 : North : 3 -> Core : 0)
    aie.flow(%t03, Core : 0, %t02, DMA : 0)
      via (%t03 : Core : 0 -> South : 4,
           %t02 : North : 4 -> DMA : 0)
    aie.flow(%t05, Core : 0, %t02, DMA : 1)
      via (%t05 : Core : 0 -> South : 5,
           %t04 : North : 5 -> South : 5,
           %t03 : North : 5 -> South : 5,
           %t02 : North : 5 -> DMA : 1)
  }
}
