//===- logical_tile_op_withops_bad.mlir ------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file %s 2>&1 | FileCheck %s

// Test verification errors when using LogicalTileOp with TileElement ops

// Interconnect ops explicitly require placed TileOp
// CHECK: error{{.*}}'aie.switchbox' op requires a placed tile (aie.tile), not a logical tile
module @test_switchbox_with_logical_tile {
  aie.device(npu2) {
    %tile = aie.logical_tile<CoreTile>(?, ?)
    // Switchbox requires aie.tile, not aie.logical_tile
    aie.switchbox(%tile) {
      aie.end
    }
    aie.end
  }
}

// -----

// CHECK: error{{.*}}'aie.shim_mux' op requires a placed tile (aie.tile), not a logical tile
module @test_shim_mux_with_logical_tile {
  aie.device(npu2) {
    %tile = aie.logical_tile<ShimNOCTile>(?, ?)
    // ShimMux requires aie.tile, not aie.logical_tile
    aie.shim_mux(%tile) {
      aie.connect<North : 0, DMA : 0>
    }
    aie.end
  }
}

// -----

// MemOp with wrong tile type
// CHECK: error{{.*}}'aie.mem' op failed to verify that op exists in a core tile
module @test_mem_wrong_tile_type {
  aie.device(npu2) {
    // MemTile cannot have MemOp (use MemTileDMAOp instead)
    %mem_tile = aie.logical_tile<MemTile>(?, ?)
    %mem = aie.mem(%mem_tile) {
      aie.end
    }
    aie.end
  }
}

// -----

// MemTileDMAOp with wrong tile type
// CHECK: error{{.*}}'aie.memtile_dma' op failed to verify that op exists in a MemTile
module @test_memtile_dma_wrong_tile_type {
  aie.device(npu2) {
    // CoreTile cannot have MemTileDMAOp
    %core_tile = aie.logical_tile<CoreTile>(?, ?)
    %memtile_dma = aie.memtile_dma(%core_tile) {
      aie.end
    }
    aie.end
  }
}

// -----

// ShimDMAOp with wrong tile type
// CHECK: error{{.*}}'aie.shim_dma' op failed to verify that op exists in a shim tile with NOC connection
module @test_shim_dma_wrong_tile_type {
  aie.device(npu2) {
    // CoreTile cannot have ShimDMAOp
    %core_tile = aie.logical_tile<CoreTile>(?, ?)
    %shim_dma = aie.shim_dma(%core_tile) {
      aie.end
    }
    aie.end
  }
}

// -----

// BufferOp on tile without memory
// CHECK: error{{.*}}'aie.buffer' op failed to verify that op exists in a tile with local memory
module @test_buffer_on_shim_tile {
  aie.device(npu2) {
    // ShimNOCTile has no memory for buffers
    %shim_tile = aie.logical_tile<ShimNOCTile>(?, ?)
    %buf = aie.buffer(%shim_tile) : memref<256xi32>
    aie.end
  }
}

// -----

// ShimDMAAllocationOp with wrong tile type
// CHECK: error{{.*}}'aie.shim_dma_allocation' op tile must be a shim tile
module @test_shim_dma_allocation_wrong_tile_type {
  aie.device(npu2) {
    %core = aie.logical_tile<CoreTile>(?, ?)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @of_bad(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    // CoreTile cannot be used for ShimDMAAllocationOp
    aie.shim_dma_allocation @of_bad(%core, MM2S, 0)
    aie.end
  }
}

// -----

// HasValidDMAChannels trait verification with LogicalTileOp
// CHECK: error{{.*}}'aie.mem' op uses more output channels than available on this tile
module @test_has_valid_dma_channels {
  aie.device(npu2) {
    %tile = aie.logical_tile<CoreTile>(?, ?)
    // CoreTile only has 2 MM2S channels, try to use 3
    aie.mem(%tile) {
      %dma0 = aie.dma_start(MM2S, 0, ^bd0, ^dma1)
    ^dma1:
      %dma1_token = aie.dma_start(MM2S, 1, ^bd0, ^dma2)
    ^dma2:
      %dma2_token = aie.dma_start(MM2S, 2, ^bd0, ^end)
    ^bd0:
      aie.end
    ^end:
      aie.end
    }
    aie.end
  }
}

// -----

// HasValidBDs trait verification with LogicalTileOp
// CHECK: error{{.*}}'aie.mem' op has more than 16 blocks
module @test_has_valid_bds {
  aie.device(npu2) {
    %tile = aie.logical_tile<CoreTile>(?, ?)
    %buf = aie.buffer(%tile) : memref<256xi32>
    aie.mem(%tile) {
      %dma = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd4
    ^bd4:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd5
    ^bd5:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd6
    ^bd6:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd7
    ^bd7:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd8
    ^bd8:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd9
    ^bd9:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd10
    ^bd10:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd11
    ^bd11:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd12
    ^bd12:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd13
    ^bd13:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd14
    ^bd14:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd15
    ^bd15:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd16
    ^bd16:
      // This is the 17th BD, should fail
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    aie.end
  }
}
