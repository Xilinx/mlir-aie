//===- arbiter_multi_bd_receive.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Memtile (0,1) emits flows 0..3 south and receives flows 4..6 from the cores
// above, one arbiter short. Flow 6 is placed last and has to share.
//
// Flow 4 sends 256 bytes per descriptor into a DMA taking 64 at a time. One
// send descriptor is one packet and the grant is held until tlast, so it spans
// all four receive descriptors and any co-tenant waits out the whole sequence.
// Flow 5 sends and receives the same size, so flow 6 joins that arbiter.

module {
  aie.device(npu2) {
    %m  = aie.tile(0, 1)
    %s1 = aie.tile(1, 0)
    %s2 = aie.tile(2, 0)
    %s3 = aie.tile(3, 0)
    %s4 = aie.tile(4, 0)
    %c3 = aie.tile(0, 3)
    %c4 = aie.tile(0, 4)
    %c5 = aie.tile(0, 5)

    aie.packet_flow(0) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%s1, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%m, DMA : 1>  aie.packet_dest<%s2, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%m, DMA : 2>  aie.packet_dest<%s3, DMA : 0> }
    aie.packet_flow(3) { aie.packet_source<%m, DMA : 3>  aie.packet_dest<%s4, DMA : 0> }

    // 256 bytes out, 64 bytes in: four receive descriptors per packet.
    aie.packet_flow(4) { aie.packet_source<%c4, DMA : 0>  aie.packet_dest<%m, DMA : 0> }
    // 64 bytes out, 64 bytes in.
    aie.packet_flow(5) { aie.packet_source<%c3, DMA : 0>  aie.packet_dest<%m, DMA : 1> }
    // 64 bytes out, 64 bytes in.
    aie.packet_flow(6) { aie.packet_source<%c5, DMA : 0>  aie.packet_dest<%m, DMA : 2> }

    %b4 = aie.buffer(%c4) : memref<64xi32>
    aie.mem(%c4) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b4 : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }

    %b3 = aie.buffer(%c3) : memref<16xi32>
    aie.mem(%c3) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b3 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }

    %b5 = aie.buffer(%c5) : memref<16xi32>
    aie.mem(%c5) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b5 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }

    %bm = aie.buffer(%m) : memref<16xi32>
    aie.memtile_dma(%m) {
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^ch1)
    ^bd0:
      aie.dma_bd(%bm : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd0
    ^ch1:
      %1 = aie.dma_start(S2MM, 1, ^bd1, ^ch2)
    ^bd1:
      aie.dma_bd(%bm : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd1
    ^ch2:
      %2 = aie.dma_start(S2MM, 2, ^bd2, ^end)
    ^bd2:
      aie.dma_bd(%bm : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd2
    ^end:
      aie.end
    }
  }
}

// Flow 4 keeps arbiter 4 to itself; flow 6 wraps onto flow 5's arbiter.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[SPANS:.*]] = aie.amsel<4> (0)
// CHECK:         %[[SAME:.*]] = aie.amsel<5> (0)
// CHECK:         %[[SHARED:.*]] = aie.amsel<5> (1)
// CHECK:         aie.masterset(DMA : 0, %[[SPANS]])
// CHECK:         aie.masterset(DMA : 1, %[[SAME]])
// CHECK:         aie.masterset(DMA : 2, %[[SHARED]])
