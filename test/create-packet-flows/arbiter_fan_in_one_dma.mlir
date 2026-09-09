//===- arbiter_fan_in_one_dma.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s

// Three flows converge on one DMA channel of the core at (1,2), one from each
// side, so (1,2) is the only switchbox any two of them meet in. A master port
// can only be tied to one arbiter, so they have to share there whatever else
// is true of them.
//
// Sharing costs nothing in this shape. All three are drained by the same DMA
// channel, one packet at a time, so the order the arbiter picks is the order
// they would have been taken in anyway, and none of them can be waiting on
// another. That holds even for flow 0, which sends 256 bytes per buffer
// descriptor into a channel receiving 64 at a time: on any other pairing that
// would keep the arbiter to itself.
//
// The exemption is deliberately no wider than this. Two flows sharing a
// master port that leaves the switchbox can still diverge downstream, and one
// of them can then be waiting on the other, so those are left to the rules.

module {
  aie.device(npu2) {
    %c12 = aie.tile(1, 2)
    %c02 = aie.tile(0, 2)
    %c22 = aie.tile(2, 2)
    %c13 = aie.tile(1, 3)

    aie.packet_flow(0) { aie.packet_source<%c02, DMA : 0>  aie.packet_dest<%c12, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%c22, DMA : 0>  aie.packet_dest<%c12, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%c13, DMA : 0>  aie.packet_dest<%c12, DMA : 0> }

    // 256 bytes a descriptor, into a channel taking 64: four receive
    // descriptors per packet, and the reason flow 0 would otherwise be given
    // an arbiter of its own.
    %b02 = aie.buffer(%c02) : memref<64xi32>
    aie.mem(%c02) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b02 : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }

    %b22 = aie.buffer(%c22) : memref<16xi32>
    aie.mem(%c22) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b22 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }

    %b13 = aie.buffer(%c13) : memref<16xi32>
    aie.mem(%c13) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b13 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }

    %b12 = aie.buffer(%c12) : memref<16xi32>
    aie.mem(%c12) {
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b12 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
  }
}

// NOWARN-NOT: warning

// One amsel, one master port, and all three arriving on it.

// CHECK-LABEL: aie.switchbox(%tile_1_2)
// CHECK:         %[[FANIN:.*]] = aie.amsel<0> (0)
// CHECK:         aie.masterset(DMA : 0, %[[FANIN]])
// CHECK:         aie.packet_rules(North : 1) {
// CHECK-NEXT:      aie.rule(31, 2, %[[FANIN]])
// CHECK:         aie.packet_rules(East : 3) {
// CHECK-NEXT:      aie.rule(31, 1, %[[FANIN]])
// CHECK:         aie.packet_rules(West : 3) {
// CHECK-NEXT:      aie.rule(31, 0, %[[FANIN]])
