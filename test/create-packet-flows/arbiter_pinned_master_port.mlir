//===- arbiter_pinned_master_port.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s

// A master port is tied to one arbiter, so the second flow to leave by it has
// nothing to choose. The two flows meet twice:
//
//   (1,4) --.
//           |          both southbound on one link
//   (1,3) --+--.
//              |       both into one DMA channel
//           (1,2)
//
// Flow 0 sends 256 bytes a descriptor into a channel taking 64, holding the
// arbiter across four of them, so at (1,3) flow 1 can wait on flow 0. But both
// end at one DMA channel, which takes their packets in whatever order they
// come: flow 1 waiting at (1,3) is no worse than waiting at (1,2), and neither
// meeting can deadlock. arbiter_fan_in_one_dma.mlir is just the second one.

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

// NOWARN-NOT: warning

// One arbiter at each tile, shared.

// CHECK-LABEL: aie.switchbox(%tile_1_3)
// CHECK:         %[[SOUTH:.*]] = aie.amsel<0> (0)
// CHECK:         aie.masterset(South : {{.*}}, %[[SOUTH]])
// CHECK:         aie.rule({{.*}}, %[[SOUTH]])
// CHECK:         aie.rule({{.*}}, %[[SOUTH]])
