//===- arbiter_pinned_master_port.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=WARN
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s

// A master port can only be tied to one arbiter, so the second flow to leave by
// one has no allocation left to choose. That is still worth reporting when the
// arbiter is one it can deadlock against: the fix is not a different amsel, it
// is giving the two flows separate master ports.
//
// The two flows here meet twice, and only one of the meetings is a hazard.
// arbiter_fan_in_one_dma.mlir is the same pair drawn so they meet only at the
// second.
//
//   (1,4) --.
//           |          both southbound on one link, and reported
//   (1,3) --+--.
//              |       both into one DMA channel, and exempt
//           (1,2)
//
// Flow 0 sends 256 bytes a descriptor into a channel taking 64, so it holds an
// arbiter across four of the consumer's descriptors. Where that arbiter also
// carries flow 1, flow 1 waits on a link it could otherwise have been using --
// at (1,3), where the two still have somewhere different to be. By (1,2) they
// do not: the same DMA drains both, in whatever order, so sharing there costs
// nothing and goes unreported.

module {
  aie.device(npu2) {
    %c12 = aie.tile(1, 2)
    %c13 = aie.tile(1, 3)
    %c14 = aie.tile(1, 4)

    aie.packet_flow(0) { aie.packet_source<%c13, DMA : 0>  aie.packet_dest<%c12, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%c14, DMA : 0>  aie.packet_dest<%c12, DMA : 0> }

    %b13 = aie.buffer(%c13) : memref<64xi32>
    aie.mem(%c13) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b13 : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }

    %b14 = aie.buffer(%c14) : memref<16xi32>
    aie.mem(%c14) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b14 : memref<16xi32> offset = 0 len = 16)
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

// One warning, naming the transit tile and the reason there was nothing to
// choose. Nothing is said about (1,2).

// WARN:      warning: at tile (1, 3), packet flow 1 shares arbiter 0 with packet flow 0, which it can deadlock against; a master port of this flow is already tied to that arbiter
// WARN-NOT:  warning

// Routing is unchanged by the warning: one arbiter at each tile, shared.

// CHECK-LABEL: aie.switchbox(%tile_1_3)
// CHECK:         %[[SOUTH:.*]] = aie.amsel<0> (0)
// CHECK:         aie.masterset(South : {{.*}}, %[[SOUTH]])
// CHECK:         aie.rule({{.*}}, %[[SOUTH]])
// CHECK:         aie.rule({{.*}}, %[[SOUTH]])
