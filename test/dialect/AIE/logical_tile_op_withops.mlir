//===- logical_tile_op_withops.mlir ----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file %s | FileCheck %s

// Test LogicalTileOp with expected ops

// Test CoreOp, MemOp, BufferOp, LockOp with LogicalTileOp<CoreTile>
// CHECK-LABEL: @test_core_tile_elements
// CHECK: %[[TILE:.*]] = aie.logical_tile<CoreTile>(?, ?)
// CHECK: aie.core(%[[TILE]])
// CHECK: aie.mem(%[[TILE]])
// CHECK: aie.buffer(%[[TILE]])
// CHECK: aie.lock(%[[TILE]])
module @test_core_tile_elements {
  aie.device(npu2) {
    %core_tile = aie.logical_tile<CoreTile>(?, ?)
    %core = aie.core(%core_tile) {
      aie.end
    }
    %mem = aie.mem(%core_tile) {
      aie.end
    }
    %buf = aie.buffer(%core_tile) : memref<256xi32>
    %lock = aie.lock(%core_tile)
    aie.end
  }
}

// -----

// Test MemTileDMAOp, BufferOp, LockOp with LogicalTileOp<MemTile>
// CHECK-LABEL: @test_mem_tile_elements
// CHECK: %[[TILE:.*]] = aie.logical_tile<MemTile>(?, ?)
// CHECK: aie.memtile_dma(%[[TILE]])
// CHECK: aie.buffer(%[[TILE]])
// CHECK: aie.lock(%[[TILE]])
module @test_mem_tile_elements {
  aie.device(npu2) {
    %mem_tile = aie.logical_tile<MemTile>(?, ?)
    %memtile_dma = aie.memtile_dma(%mem_tile) {
      aie.end
    }
    %buf = aie.buffer(%mem_tile) : memref<256xi32>
    %lock = aie.lock(%mem_tile)
    aie.end
  }
}

// -----

// Test ShimDMAOp, LockOp with LogicalTileOp<ShimNOCTile>
// CHECK-LABEL: @test_shim_noc_tile_elements
// CHECK: %[[TILE:.*]] = aie.logical_tile<ShimNOCTile>(?, ?)
// CHECK: aie.shim_dma(%[[TILE]])
// CHECK: aie.lock(%[[TILE]])
module @test_shim_noc_tile_elements {
  aie.device(npu2) {
    %shim_tile = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_dma = aie.shim_dma(%shim_tile) {
      aie.end
    }
    %lock = aie.lock(%shim_tile)
    aie.end
  }
}

// -----

// Test ObjectFifoCreateOp with LogicalTileOp
// CHECK-LABEL: @test_objectfifo_shim_to_core
// CHECK: %[[SHIM:.*]] = aie.logical_tile<ShimNOCTile>(?, ?)
// CHECK: %[[CORE:.*]] = aie.logical_tile<CoreTile>(?, ?)
// CHECK: aie.objectfifo @of2(%[[SHIM]], {%[[CORE]]}, 2 : i32)
module @test_objectfifo_shim_to_core {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %core = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @of2(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.end
  }
}

// -----

// Test ObjectFifoLinkOp with LogicalTileOp
// CHECK-LABEL: @test_objectfifo_link
// CHECK: %[[CORE:.*]] = aie.logical_tile<CoreTile>(?, ?)
// CHECK: %[[MEM:.*]] = aie.logical_tile<MemTile>(?, ?)
// CHECK: %[[CORE2:.*]] = aie.logical_tile<CoreTile>(?, ?)
// CHECK: aie.objectfifo @of_in(%[[CORE]], {%[[MEM]]}, 2 : i32)
// CHECK: aie.objectfifo @of_out(%[[MEM]], {%[[CORE2]]}, 2 : i32)
// CHECK: aie.objectfifo.link [@of_in] -> [@of_out]([] [])
module @test_objectfifo_link {
  aie.device(npu2) {
    %core1 = aie.logical_tile<CoreTile>(?, ?)
    %mem = aie.logical_tile<MemTile>(?, ?)
    %core2 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @of_in(%core1, {%mem}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @of_out(%mem, {%core2}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@of_in] -> [@of_out]([] [])
    aie.end
  }
}

// -----

// Test ShimDMAAllocationOp with LogicalTileOp<ShimNOCTile>
// CHECK-LABEL: @test_shim_dma_allocation
// CHECK: %[[SHIM:.*]] = aie.logical_tile<ShimNOCTile>(?, ?)
// CHECK: %[[CORE:.*]] = aie.logical_tile<CoreTile>(?, ?)
// CHECK: aie.objectfifo @of_alloc(%[[SHIM]], {%[[CORE]]}, 2 : i32)
// CHECK: aie.shim_dma_allocation @of_alloc_dma(%[[SHIM]], MM2S, 0)
module @test_shim_dma_allocation {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %core = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @of_alloc(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.shim_dma_allocation @of_alloc_dma(%shim, MM2S, 0)
    aie.end
  }
}

// -----

// Mixed LogicalTileOp and TileOp usage
// CHECK-LABEL: @test_mixed_tile_types
// CHECK: %[[TILE:.*]] = aie.tile(0, 2)
// CHECK: %[[LOGICAL:.*]] = aie.logical_tile<CoreTile>(?, ?)
// CHECK: aie.objectfifo @of_mixed(%[[TILE]], {%[[LOGICAL]]}, 2 : i32)
module @test_mixed_tile_types {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %logical = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @of_mixed(%tile, {%logical}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.end
  }
}

// -----

// Test DMAConfigureTaskOp with LogicalTileOp
module @test_dma_configure_task {
  aie.device(npu2) {
    %shim_tile = aie.logical_tile<ShimNOCTile>(?, ?)
    %buffer = aie.external_buffer {sym_name = "ext_buffer"} : memref<1024xi32>

    aie.runtime_sequence(%arg0: memref<1024xi32>) {
      %task = aiex.dma_configure_task(%shim_tile, MM2S, 0) {
        aie.dma_bd(%buffer : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%task)
    }
    aie.end
  }
}

// -----

// Test TileElement ops with LogicalTileOp (DMAConfigureTaskOp on MemTile)
module @test_dma_configure_task_memtile {
  aie.device(npu2) {
    %mem_tile = aie.logical_tile<MemTile>(?, ?)
    %buffer_in = aie.buffer(%mem_tile) {sym_name = "buf_in"} : memref<256xi32>

    aie.runtime_sequence(%arg0: memref<256xi32>) {
      %task = aiex.dma_configure_task(%mem_tile, S2MM, 0) {
        aie.dma_bd(%buffer_in : memref<256xi32>, 0, 256) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%task)
    }
    aie.end
  }
}
