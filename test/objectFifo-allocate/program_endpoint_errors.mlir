// RUN: aie-opt --split-input-file --aie-objectfifo-allocate --verify-diagnostics %s -o /dev/null

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The route decides which way an endpoint's channel points; a DMA program
// naming it must agree.
module @direction_mismatch {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mt = aie.tile(0, 1)
    %b = aie.buffer(%mt) {sym_name = "b"} : memref<64xi32>
    aie.route_endpoint @b_out(%mt) DMA
    aie.route_endpoint @b_dst(%shim) DMA {fifoName = "b"}
    aie.route from @b_out to [@b_dst]
    aie.memtile_dma(%mt) {
      // expected-error @+1 {{starts S2MM on @b_out, but its route makes it MM2S}}
      aie.dma_start(S2MM, @b_out, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
  }
}

// -----

// Allocation stamps a packet header on runtime tasks, not on a static program.
module @packet_source_program {
  aie.device(npu2) {
    %mt = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    %b = aie.buffer(%mt) {sym_name = "b"} : memref<64xi32>
    aie.route_endpoint @spray(%mt) DMA
    aie.route_endpoint @catch(%core) DMA
    aie.route from @spray to [@catch] {packet}
    aie.memtile_dma(%mt) {
      // expected-error @+1 {{names @spray, the source of a packet-switched route; a DMA program's BDs would not carry its header}}
      aie.dma_start(MM2S, @spray, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
  }
}

// -----

// A pinned endpoint may not take a channel a DMA program already starts.
module @pinned_clash {
  // expected-remark @+1 {{could not find a spill-aware allocation}}
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mt = aie.tile(0, 1)
    %b = aie.buffer(%mt) {sym_name = "b"} : memref<64xi32>
    // expected-error @+1 {{pinned MM2S DMA channel 0 is out of range or already in use on this tile}}
    aie.route_endpoint @b_out(%mt) DMA {channelIndex = 0 : i32}
    aie.route_endpoint @b_dst(%shim) DMA {fifoName = "b"}
    aie.route from @b_out to [@b_dst]
    aie.memtile_dma(%mt) {
      // expected-note @+1 {{pre-existing aie.dma_start reserves DMA channel 0}}
      aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
  }
}

// -----

// A compute tile has two S2MM channels.
module @no_channel_left {
  // expected-remark @+1 {{could not find a spill-aware allocation}}
  aie.device(npu2) {
    %shim0 = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    // expected-error @+1 {{number of input DMA channel exceeded! requires at least 3 S2MM channels, but capacity is 2}}
    %core = aie.tile(0, 2)
    aie.route_endpoint @s0(%shim0) DMA {fifoName = "s0"}
    aie.route_endpoint @s1(%shim0) DMA {fifoName = "s1"}
    aie.route_endpoint @s2(%shim1) DMA {fifoName = "s2"}
    // expected-note @+1 {{DMA endpoint @d0; occupies channel 0}}
    aie.route_endpoint @d0(%core) DMA
    // expected-note @+1 {{DMA endpoint @d1; occupies channel 1}}
    aie.route_endpoint @d1(%core) DMA
    // expected-note @+1 {{DMA endpoint @d2}}
    aie.route_endpoint @d2(%core) DMA
    aie.route from @s0 to [@d0]
    aie.route from @s1 to [@d1]
    aie.route from @s2 to [@d2]
  }
}

// -----

// Only MemTile channels 0-3 reach a neighbor's memory.
module @adjacent_exhausted {
  // expected-remark @+1 {{could not find a spill-aware allocation}}
  aie.device(npu2) {
    %shim = aie.tile(1, 0)
    %west = aie.tile(0, 1)
    // expected-error @+1 {{number of output DMA channel exceeded! requires at least 5 MM2S channels, but capacity is 4 for adjacent MemTile access}}
    %mt = aie.tile(1, 1)
    %far = aie.buffer(%west) {sym_name = "far"} : memref<64xi32>
    %near = aie.buffer(%mt) {sym_name = "near"} : memref<64xi32>
    // expected-note @+1 {{DMA endpoint @reach requires adjacent MemTile access}}
    aie.route_endpoint @reach(%mt) DMA
    aie.route_endpoint @reach_dst(%shim) DMA {fifoName = "reach"}
    aie.route from @reach to [@reach_dst]
    aie.memtile_dma(%mt) {
      // expected-note @+1 {{pre-existing aie.dma_start reserves DMA channel 0}}
      aie.dma_start(MM2S, 0, ^bd0, ^start1)
    ^bd0:
      aie.dma_bd(%near : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^bd0
    ^start1:
      // expected-note @+1 {{pre-existing aie.dma_start reserves DMA channel 1}}
      aie.dma_start(MM2S, 1, ^bd1, ^start2)
    ^bd1:
      aie.dma_bd(%near : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^bd1
    ^start2:
      // expected-note @+1 {{pre-existing aie.dma_start reserves DMA channel 2}}
      aie.dma_start(MM2S, 2, ^bd2, ^start3)
    ^bd2:
      aie.dma_bd(%near : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^bd2
    ^start3:
      // expected-note @+1 {{pre-existing aie.dma_start reserves DMA channel 3}}
      aie.dma_start(MM2S, 3, ^bd3, ^start4)
    ^bd3:
      aie.dma_bd(%near : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^bd3
    ^start4:
      aie.dma_start(MM2S, @reach, ^bd4, ^end)
    ^bd4:
      aie.dma_bd(%far : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^bd4
    ^end:
      aie.end
    }
  }
}
