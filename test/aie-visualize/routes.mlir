//===- routes.mlir ---------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Route views render vias, packet metadata, arbitration, semantic topology, and selected flows.

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot %s | FileCheck %s --check-prefix=ALL
// RUN: aie-visualize --emit-dot %s | FileCheck %s --check-prefix=VIAS
// RUN: aie-visualize --emit-dot --contract=port,buffer %s | FileCheck %s --check-prefix=TOPOLOGY
// RUN: aie-visualize --emit-dot --contract=port,buffer --device=main %s | FileCheck %s --check-prefix=TOPOLOGY
// RUN: aie-visualize --emit-dot --contract=port,buffer --tile-width=3.2 %s | FileCheck %s --check-prefix=WIDE
// RUN: aie-visualize --emit-dot --only-flow=1 %s | FileCheck %s --check-prefix=ONLY
// RUN: not aie-visualize --emit-dot --only-flow=9 %s 2>&1 | FileCheck %s --check-prefix=BAD-ID
// RUN: not aie-visualize --emit-dot --contract=unknown %s 2>&1 | FileCheck %s --check-prefix=BAD-KIND

// ALL: digraph aie_routes
// ALL: graph [layout=neato, overlap=true, splines=curved, outputorder=nodesfirst
// ALL: tile_0_2 {{.*}} pos="0.000000,6.000000!"
// ALL: tile_0_3 {{.*}} pos="0.000000,9.000000!"
// ALL-NOT: tile_1_2
// ALL: buffer_0 {{.*}} label="source"
// ALL-SAME: class="buffer detail-buffers flow src-0 dst-0"
// ALL: buffer_1 {{.*}} label="dest"
// ALL-NOT: memref
// Two packet flows leave the same buffer and cross the same hop. The hop is
// named for the packet crossing it, so each is a place of its own and each
// is named once.
// ALL-DAG: p_0_2_5_0_k3 -> p_0_3_3_0_k3 {{.*}} label=<<FONT COLOR="#d73027">F0 pkt=3/31</FONT>>
// ALL-DAG: p_0_2_5_0_k4 -> p_0_3_3_0_k4 {{.*}} label=<<FONT COLOR="#d73027">F0 pkt=4/31</FONT>>
// ALL-DAG: p_0_3_1_0_m -> p_0_3_1_1_s {{.*}} label=<<FONT COLOR="#984ea3">F3 pkt=6/31</FONT>>
// ALL-DAG: p_0_4_0_0 -> p_0_4_3_2
// ALL-DAG: p_0_4_3_2 -> p_0_3_5_2 {{.*}} label=<<FONT COLOR="#1a9850">F2</FONT>>
// ALL-DAG: p_0_3_5_2 -> p_0_3_0_0
// ALL-DAG: buffer_0 -> p_0_2_1_0_m {{.*}}class="buffer-link{{.*}}];
// ALL-DAG: p_0_3_1_0_s -> buffer_1 {{.*}}class="buffer-link{{.*}}];
// ALL-NOT: p_0_2_0_0

// VIAS: p_0_2_5_0_k3 [shape=point, width=0.09, {{.*}} xlabel="N0"
// VIAS-SAME: class="port detail-vias
// VIAS: p_0_3_3_0_k3 [shape=point, width=0.09, {{.*}} xlabel="S0"
// VIAS: arbitration_0 {{.*}} label="r0 pkt=3/31 -> A2.M0 header"
// VIAS-SAME: tooltip="rule 0: packet ID & 31 = 3, arbiter 2, msel 0, keep packet header: yes, priority route: no, control overlay: no"
// VIAS-SAME: class="arbitration detail-arbitration detail-vias flow src-0 dst-0"
// VIAS: arbitration_1 {{.*}} label="r1 pkt=4/31 -> A2.M1 header"
// VIAS-SAME: tooltip="rule 1: packet ID & 31 = 4, arbiter 2, msel 1, keep packet header: yes, priority route: no, control overlay: no"
// VIAS-SAME: class="arbitration detail-arbitration detail-vias flow src-0 dst-0"
// VIAS: p_0_2_1_0_m -> p_0_2_5_0_k3
// VIAS-NOT: arrowhead=none

// TOPOLOGY: graph [layout=neato, overlap=true, splines=curved
// TOPOLOGY: tile_0_2 {{.*}} pos="0.000000,6.000000!"
// TOPOLOGY: tile_0_3 {{.*}} pos="0.000000,9.000000!"
// TOPOLOGY-NOT: tile_1_2
// TOPOLOGY: buffer_0 {{.*}}label="source"{{.*}}class="buffer detail-buffers flow src-0 dst-0"
// TOPOLOGY: buffer_1 {{.*}}label="dest"{{.*}}class="buffer detail-buffers flow src-0 dst-0"
// A switchbox hop that only passes things through is contracted away even
// when several flows share it: they enter it from the same place and leave
// it for the same place, so the edge that replaces it says what the hop
// said. The DMA channels either end stay.
// TOPOLOGY-DAG: buffer_0 -> p_0_2_1_0_m {{.*}}flow src-0 dst-0"
// TOPOLOGY-DAG: p_0_2_1_0_m -> p_0_3_1_0_s {{.*}}flow src-0 dst-0"
// TOPOLOGY-DAG: p_0_3_1_0_s -> buffer_1 {{.*}}flow src-0 dst-0"
// TOPOLOGY-NOT: p_0_2_5_0 ->
// TOPOLOGY-NOT: tile_0_2 -> buffer_0
// TOPOLOGY-NOT: buffer_1 -> tile_0_3
// TOPOLOGY: tile_0_4 -> tile_0_3

// WIDE: tile_0_2 {{.*}}width=3.200000
// WIDE: p_0_2_1_0_m {{.*}}width=0.82{{.*}}pos="-1.400000,
// WIDE: buffer_0 {{.*}}width=1.750000{{.*}}pos="0.000000,

// ONLY-NOT: F3 pkt=6
// ONLY: F1 pkt=5/31
// ONLY-NOT: F2

// BAD-ID: --only-flow references unknown flow 9; valid IDs are 0 through 5
// BAD-KIND: --contract references unknown node kind 'unknown'
// FLOW2-NOT: F3 pkt=6/31

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)

    %source = aie.buffer(%t02) {sym_name = "source"} : memref<16xi32>
    %dest = aie.buffer(%t03) {sym_name = "dest"} : memref<16xi32>
    %other = aie.buffer(%t04) {sym_name = "other"} : memref<16xi32>

    aie.mem(%t02) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%source : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    aie.mem(%t03) {
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%dest : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    aie.mem(%t04) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%other : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }

    aie.packet_flow(3, mask = 31) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t03, DMA : 0>
    } via (%t02 : DMA : 0 -> North : 0,
           %t03 : South : 0 -> DMA : 0)
    aie.packet_flow(4, mask = 31) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t03, DMA : 0>
    } via (%t02 : DMA : 0 -> North : 0,
           %t03 : South : 0 -> DMA : 0)
    aie.packet_flow(5, mask = 31) {
      aie.packet_source<%t04, DMA : 0>
      aie.packet_dest<%t03, DMA : 1>
    } via (%t04 : DMA : 0 -> South : 1,
           %t03 : North : 1 -> DMA : 1)
    aie.packet_flow(6, mask = 31) {
      aie.packet_source<%t03, DMA : 0>
      aie.packet_dest<%t03, DMA : 1>
    } via (%t03 : DMA : 0 -> DMA : 1)

    // This materialized connection differs from the pinned flow below. The
    // visualizer must use the flow's vias as its only route description.
    aie.switchbox(%t02) {
      aie.connect<Core : 0, DMA : 0>
      %a20 = aie.amsel<2> (0)
      %a21 = aie.amsel<2> (1)
      aie.masterset(North : 0, %a20, %a21) {keep_pkt_header = true}
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 3, %a20)
        aie.rule(31, 4, %a21)
      }
    }
    aie.flow(%t04, Core : 0, %t03, Core : 0)
      via (%t04 : Core : 0 -> South : 2,
           %t03 : North : 2 -> Core : 0)
  }
}
