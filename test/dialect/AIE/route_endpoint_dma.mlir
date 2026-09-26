// RUN: aie-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A DMA end may sit on any tile, and a tile's DMA program may name its channel
// by that endpoint rather than an index.

// CHECK-LABEL: @program_endpoints
// CHECK: aie.route_endpoint @b_in(%{{.*}}) DMA
// CHECK: aie.route_endpoint @b_core(%{{.*}}) DMA
// CHECK: aie.dma_start(MM2S, @b_out, ^bb1, ^bb2)
// CHECK: aie.dma_start(S2MM, @b_core, ^bb1, ^bb2)
module @program_endpoints {
  aie.device(npu2) {
    %mt = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    %b = aie.buffer(%mt) {sym_name = "b"} : memref<64xi32>
    %b_l1 = aie.buffer(%core) {sym_name = "b_l1"} : memref<64xi32>
    aie.route_endpoint @b_in(%mt) DMA
    aie.route_endpoint @b_out(%mt) DMA
    aie.route_endpoint @b_core(%core) DMA
    aie.route from @b_out to [@b_core]
    aie.memtile_dma(%mt) {
      aie.dma_start(MM2S, @b_out, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    aie.mem(%core) {
      aie.dma_start(S2MM, @b_core, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b_l1 : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
  }
}

// -----

aie.device(npu2) {
  %core = aie.tile(0, 2)
  // expected-error @+1 {{a PLIO end is on a shim}}
  aie.route_endpoint @p(%core) PLIO
}

// -----

aie.device(npu2) {
  %core = aie.tile(0, 2)
  %b = aie.buffer(%core) {sym_name = "b"} : memref<64xi32>
  aie.mem(%core) {
    // expected-error @+1 {{endpoint @b is not an aie.route_endpoint}}
    aie.dma_start(S2MM, @b, ^bd0, ^end)
  ^bd0:
    aie.dma_bd(%b : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^bd0
  ^end:
    aie.end
  }
}

// -----

aie.device(npu2) {
  %core = aie.tile(0, 2)
  %b = aie.buffer(%core) {sym_name = "b"} : memref<64xi32>
  aie.route_endpoint @port(%core) Core {channelIndex = 0 : i32}
  aie.mem(%core) {
    // expected-error @+1 {{endpoint @port names a Core port, not a DMA channel}}
    aie.dma_start(S2MM, @port, ^bd0, ^end)
  ^bd0:
    aie.dma_bd(%b : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^bd0
  ^end:
    aie.end
  }
}

// -----

aie.device(npu2) {
  %core = aie.tile(0, 2)
  %other = aie.tile(0, 3)
  %b = aie.buffer(%core) {sym_name = "b"} : memref<64xi32>
  aie.route_endpoint @elsewhere(%other) DMA
  aie.mem(%core) {
    // expected-error @+1 {{endpoint @elsewhere is on a different tile than this DMA program}}
    aie.dma_start(S2MM, @elsewhere, ^bd0, ^end)
  ^bd0:
    aie.dma_bd(%b : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^bd0
  ^end:
    aie.end
  }
}

// -----

aie.device(npu2) {
  %core = aie.tile(0, 2)
  %b = aie.buffer(%core) {sym_name = "b"} : memref<64xi32>
  aie.route_endpoint @ep(%core) DMA
  aie.mem(%core) {
    // expected-error @+1 {{names its channel by exactly one of an index and an endpoint}}
    aie.dma_start(S2MM, 0, ^bd0, ^end) {endpoint = @ep}
  ^bd0:
    aie.dma_bd(%b : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^bd0
  ^end:
    aie.end
  }
}

// -----

// Each endpoint a program names takes a channel of its own.
aie.device(npu2) {
  %core = aie.tile(0, 2)
  %b = aie.buffer(%core) {sym_name = "b"} : memref<64xi32>
  aie.route_endpoint @e0(%core) DMA
  aie.route_endpoint @e1(%core) DMA
  // expected-error @+1 {{uses more input channels than available on this tile}}
  aie.mem(%core) {
    aie.dma_start(S2MM, 0, ^bd0, ^s1)
  ^bd0:
    aie.dma_bd(%b : memref<64xi32> offset = 0 len = 16)
    aie.next_bd ^bd0
  ^s1:
    aie.dma_start(S2MM, @e0, ^bd1, ^s2)
  ^bd1:
    aie.dma_bd(%b : memref<64xi32> offset = 16 len = 16)
    aie.next_bd ^bd1
  ^s2:
    aie.dma_start(S2MM, @e1, ^bd2, ^end)
  ^bd2:
    aie.dma_bd(%b : memref<64xi32> offset = 32 len = 16)
    aie.next_bd ^bd2
  ^end:
    aie.end
  }
}
