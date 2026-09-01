//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A shim transfer feeds an objectFIFO whose far end receives whole objects.
// The runtime buffer descriptor is sized in the host buffer's element type and
// the fifo's object type reaches this side only through the allocation's
// elem_type, so aiex.dma_configure_task_for is where the two extents meet.

// RUN: aie-opt --verify-diagnostics --split-input-file %s

// 300 f32 = 1200 bytes into 256-byte objects: the last one never fills.
module {
  aie.device(npu2_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of (%tile_0_0, MM2S, 0) {elem_type = memref<64xi32>}
    aie.runtime_sequence(%arg0: memref<300xf32>) {
      // expected-error@+1 {{moves 1200 bytes through @of, which is not a whole number of that objectFIFO's 256-byte objects}}
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<300xf32> offset = 0 len = 300 sizes = [1, 1, 1, 300] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// 256 f32 = 1024 bytes is an exact 4 objects.
module {
  aie.device(npu2_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of (%tile_0_0, MM2S, 0) {elem_type = memref<64xi32>}
    aie.runtime_sequence(%arg0: memref<256xf32>) {
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<256xf32> offset = 0 len = 256 sizes = [1, 1, 1, 256] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// The host buffer's element type need not match the fifo's: only the extent is
// compared. 256 i8 is 256 bytes, one whole object of 64 i32.
module {
  aie.device(npu2_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of (%tile_0_0, MM2S, 0) {elem_type = memref<64xi32>}
    aie.runtime_sequence(%arg0: memref<256xi8>) {
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<256xi8> offset = 0 len = 256 sizes = [1, 1, 1, 256] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// The repeat count multiplies the transfer: 4 issues of 64 bytes is one object.
module {
  aie.device(npu2_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of (%tile_0_0, MM2S, 0) {elem_type = memref<64xi32>}
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 16 sizes = [1, 1, 4, 4] strides = [0, 0, 4, 1])
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// An allocation that records no object type is not checked: a control-overlay
// channel carries no objectFIFO, and a join, a split or a padding endpoint has
// no single object extent to compare against.
module {
  aie.device(npu2_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of (%tile_0_0, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<300xf32>) {
      %t = aiex.dma_configure_task_for @of {
        aie.dma_bd(%arg0 : memref<300xf32> offset = 0 len = 300 sizes = [1, 1, 1, 300] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}
