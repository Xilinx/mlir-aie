// RUN: aie-opt --split-input-file --aie-objectfifo-allocate --verify-diagnostics %s -o /dev/null
// RUN: sed -e 's/MM2S/S2MM/g' -e 's/drains/fills/g' %s | aie-opt --split-input-file --aie-objectfifo-allocate --verify-diagnostics -o /dev/null

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Only active reservations in the failed channel range contribute. The pinned
// local endpoint on channel 0 does contribute, even though it needs no neighbor
// access. Channel 4, pending local endpoints, and later remote endpoints do not.
// Physical-coordinate aliases share both reservations and their provenance.
module @restricted_contributors {
  // expected-remark @+1 {{could not find a spill-aware allocation}}
  aie.device(npu2) {
    // expected-error @+1 {{requires at least 5 MM2S channels, but capacity is 4 for adjacent MemTile access}}
    %home = aie.tile(0, 1)
    %alias = aie.logical_tile<MemTile>(0, 1)
    %neighbor = aie.tile(1, 1)
    %local = aie.buffer(%home) {sym_name = "local"} : memref<16xi32>
    %remote = aie.buffer(%neighbor) {sym_name = "remote"} : memref<16xi32>
    aie.memtile_dma(%home) {
      // expected-note @+1 {{pre-existing aie.dma_start reserves DMA channel 1}}
      aie.dma_start(MM2S, 1, ^bd, ^end)
    ^bd:
      aie.dma_bd(%local : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd
    ^end:
      aie.end
    }
    aie.objectfifo.pool @local_pool(%home) {depth = 1 : i32, buffers = [@local]} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.pool @remote_pool(%neighbor) {depth = 1 : i32, buffers = [@remote]} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @pending_local(%home) drains @local_pool
    aie.objectfifo.dma_endpoint @pinned_high(%alias) drains @local_pool {channelIndex = 4 : i32}
    // expected-note @+1 {{DMA endpoint @pinned_low; occupies channel 0}}
    aie.objectfifo.dma_endpoint @pinned_low(%alias) drains @local_pool {channelIndex = 0 : i32}
    // expected-note @+1 {{DMA endpoint @remote0 requires adjacent MemTile access; occupies channel 2}}
    aie.objectfifo.dma_endpoint @remote0(%home) drains @remote_pool
    // expected-note @+1 {{DMA endpoint @remote1 requires adjacent MemTile access; occupies channel 3}}
    aie.objectfifo.dma_endpoint @remote1(%alias) drains @remote_pool
    // expected-note @+1 {{DMA endpoint @failing requires adjacent MemTile access}}
    aie.objectfifo.dma_endpoint @failing(%home) drains @remote_pool
    aie.objectfifo.dma_endpoint @pending_remote(%home) drains @remote_pool
  }
}

// -----

// Shim allocations are independent reservations, not anonymous FIFO users.
module @shim_contributors {
  // expected-remark @+1 {{could not find a spill-aware allocation}}
  aie.device(npu2) {
    // expected-error @+1 {{requires at least 3 MM2S channels, but capacity is 2}}
    %home = aie.tile(0, 0)
    %alias = aie.logical_tile<ShimNOCTile>(0, 0)
    // expected-note @+1 {{pre-existing aie.shim_dma_allocation reserves DMA channel 0}}
    aie.shim_dma_allocation @reserved(%alias, MM2S, 0)
    aie.objectfifo.pool @p(%home) {depth = 0 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    // expected-note @+1 {{DMA endpoint @active; occupies channel 1}}
    aie.objectfifo.dma_endpoint @active(%alias) drains @p
    // expected-note @+1 {{DMA endpoint @failing}}
    aie.objectfifo.dma_endpoint @failing(%home) drains @p
    aie.objectfifo.dma_endpoint @pending(%home) drains @p
  }
}

// -----

// A pinned collision points at both the request and the original reservation.
module @pinned_collision {
  // expected-remark @+1 {{could not find a spill-aware allocation}}
  aie.device(npu2) {
    %home = aie.tile(0, 0)
    // expected-note @+1 {{pre-existing aie.shim_dma_allocation reserves DMA channel 0}}
    aie.shim_dma_allocation @reserved(%home, MM2S, 0)
    aie.objectfifo.pool @p(%home) {depth = 0 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error @+1 {{pinned MM2S DMA channel 0 is out of range or already in use on this tile}}
    aie.objectfifo.dma_endpoint @failing(%home) drains @p {channelIndex = 0 : i32}
  }
}
