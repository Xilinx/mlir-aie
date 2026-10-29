//===- follow-through-buffers.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Provenance follows a stream through a buffer: what a buffer drains carries
// the origins that filled it, and a buffer two streams fill reports both.

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot %s | FileCheck %s --check-prefix=ALL
// RUN: aie-visualize --emit-dot --only-flow=0 %s | FileCheck %s --check-prefix=BRIDGE
// RUN: aie-visualize --emit-dot --only-flow=1 %s | FileCheck %s --check-prefix=FANOUT
// RUN: aie-visualize --emit-dot --contract=port,buffer %s | FileCheck %s --check-prefix=TOPOLOGY
// RUN: env AIE_VISUALIZE_GRAPH_STATS=1 aie-visualize --emit-dot --skip-verify --arrows=avoidance %s -o %t 2>&1 | FileCheck %s --check-prefix=LANES
// RUN: %python %S/Inputs/check-arrow-separation.py %t | FileCheck %s --check-prefix=CLEAN
// RUN: env AIE_VISUALIZE_GRAPH_STATS=1 MLIR_AIE_TEST_LANE_SEARCH_CAP=5 aie-visualize --emit-dot --skip-verify --arrows=avoidance %s -o %t.capped 2>&1 | FileCheck %s --check-prefix=CAPPED
// RUN: env MLIR_AIE_TEST_LANE_SEARCH_CAP=0 aie-visualize --emit-dot --skip-verify --arrows=avoidance %s -o %t.fallback
// RUN: cmp %t.capped %t.fallback

// LANES: lane_components={{[1-9][0-9]*}} lane_accepted={{[1-9][0-9]*}}
// LANES-SAME: lane_largest={{[1-9][0-9]*}} lane_corridors={{[1-9][0-9]*}} lane_corridor_largest={{[1-9][0-9]*}}
// LANES-SAME: lane_variables={{[1-9][0-9]*}} lane_assignments={{[1-9][0-9]*}} lane_capped=0
// CLEAN: missing_geometry=0 collinear_overlaps=0 narrow_gaps=0 diagonals=0
// CAPPED: lane_variables={{[1-9][0-9]*}} lane_assignments=5 lane_capped=1

// ALL-DAG: p_0_2_1_0_m -> p_0_2_5_0_out_r0
// ALL-DAG: p_0_2_5_0_out_r0 -> p_0_3_3_0_in_r0
// ALL-DAG: p_0_3_1_0_m -> p_0_3_5_1_out_r1
// ALL-DAG: p_0_3_1_1_m -> p_0_3_3_3_out_r3
// ALL-DAG: p_0_3_5_1_out_r1 -> p_0_4_3_1_in_r1
// ALL-DAG: p_0_3_5_2_in_r2 -> p_0_3_1_1_s
// ALL-DAG: p_0_4_1_1_m -> p_0_4_5_0_out_r4
// ALL-DAG: p_0_4_5_0_out_r4 -> p_0_5_3_0_in_r4

// Both channels the bridge feeds carry the origins that filled it, and differ
// only in where they end up.
// BRIDGE: label="bridge"
// BRIDGE-DAG: buffer_0 -> p_0_3_1_0_m {{.*}}class="buffer-link detail-buffers flow src-0 dst-5{{.*}}"
// BRIDGE-DAG: buffer_0 -> p_0_3_1_1_m {{.*}}class="buffer-link detail-buffers flow src-0 dst-3{{.*}}"
// BRIDGE-NOT: label="fanout"

// FANOUT: label="fanout"
// FANOUT-DAG: buffer_1 -> p_0_5_1_0_m {{.*}}class="buffer-link detail-buffers flow src-1 dst-4{{.*}}"
// FANOUT-DAG: buffer_1 -> p_0_5_1_1_m {{.*}}class="buffer-link detail-buffers flow src-1 dst-5{{.*}}"
// FANOUT-NOT: label="bridge"

// TOPOLOGY-DAG: buffer_0 {{.*}}label="bridge"{{.*}}class="buffer detail-buffers flow src-0 dst-0{{.*}}"
// TOPOLOGY-DAG: buffer_1 {{.*}}label="fanout"{{.*}}class="buffer detail-buffers flow src-1 dst-1{{.*}}"
// TOPOLOGY-DAG: buffer_2 {{.*}}label="join"{{.*}}class="buffer detail-buffers flow src-2 dst-2{{.*}}"
// Two streams fill the join buffer, and each link states its own origin.
// TOPOLOGY-DAG: p_0_2_1_0_s -> buffer_2 {{.*}}flow src-5 dst-2{{.*}}"
// TOPOLOGY-DAG: p_0_2_1_1_s -> buffer_2 {{.*}}flow src-6 dst-2{{.*}}"
// Arriving at a tile anchor is a different unknown from leaving it, so
// provenance does not pass through one.
// TOPOLOGY-DAG: p_0_4_1_0_s -> tile_0_4 {{.*}}flow src-0 dst-4{{.*}}"
// TOPOLOGY-DAG: tile_0_4 -> p_0_4_1_1_m {{.*}}flow src-7 dst-6{{.*}}"
// Contracting ports removes the switchbox hops a flow passes through, not
// the DMA channel it leaves or arrives on, so a buffer reaches the channel
// that drains it and the chain continues from there.
// TOPOLOGY-DAG: buffer_1 -> p_0_5_1_0_m {{.*}}class="buffer-link detail-buffers flow src-1 dst-7{{.*}}"
// TOPOLOGY-DAG: p_0_5_1_0_m -> tile_0_3 {{.*}}class="route detail-packet-ids flow src-1 dst-7{{.*}}"
// TOPOLOGY-DAG: buffer_1 -> p_0_5_1_1_m {{.*}}class="buffer-link detail-buffers flow src-1 dst-4{{.*}}"
// TOPOLOGY-DAG: p_0_5_1_1_m -> tile_0_4 {{.*}}class="route detail-packet-ids flow src-1 dst-4{{.*}}"
// TOPOLOGY-NOT: topology_junction_
// TOPOLOGY-NOT: tile_0_5 -> buffer_1
// TOPOLOGY-NOT: buffer_2 -> tile_0_2
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
