//===- switchbox-wire.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// No op states the wire between two tiles. A stream whose next hop is a rule
// in the switchbox of the tile above therefore stops at the edge of its own
// tile unless the visualizer adds that wire. The flow here is lifted only as
// far as North 0 of tile (0, 2). The switchbox of tile (0, 3) holds the rest:
// packet 8 arrives on South 0 and goes to the channel that fills a buffer.

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot --split-packet-ports %s | FileCheck %s
// RUN: aie-visualize --emit-dot --split-packet-ports --contract=port,dma,buffer %s | FileCheck %s --check-prefix=TOPOLOGY

// CHECK-DAG: p_0_1_1_0_m -> p_0_1_5_0_k8
// CHECK-DAG: p_0_2_3_0_k8 -> p_0_2_5_0_k8
// The wire between tile (0, 2) and tile (0, 3).
// CHECK-DAG: p_0_2_5_0_k8 -> p_0_3_3_0_k8
// The rule in the switchbox of tile (0, 3).
// CHECK-DAG: p_0_3_3_0_k8 -> p_0_3_1_0_s
// The channel is now one the design runs, so its buffer joins the stream.
// CHECK-DAG: p_0_3_1_0_s -> buffer_0

// Contracting the hops away leaves what the stream does: it fills the buffer.
// TOPOLOGY: tile_0_1 -> buffer_0

module {
  aie.device(npu1_1col) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)

    %in = aie.buffer(%t03) {sym_name = "in"} : memref<16xi32>

    aie.mem(%t03) {
      %0 = aie.dma_start(S2MM, 0, ^s2mm0, ^end)
    ^s2mm0:
      aie.dma_bd(%in : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^s2mm0
    ^end:
      aie.end
    }

    aie.packet_flow(8, mask = 31) {
      aie.packet_source<%t01, DMA : 0>
      aie.packet_dest<%t02, North : 0>
    } via (%t01 : DMA : 0 -> North : 0,
           %t02 : South : 0 -> North : 0)

    aie.switchbox(%t03) {
      %a = aie.amsel<0> (0)
      aie.masterset(DMA : 0, %a)
      aie.packet_rules(South : 0) {
        aie.rule(31, 8, %a)
      }
    }
  }
}
