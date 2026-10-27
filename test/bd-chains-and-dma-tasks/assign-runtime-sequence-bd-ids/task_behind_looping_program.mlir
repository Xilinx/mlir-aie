//===- task_behind_looping_program.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A runtime task queued on a channel whose static BD chain loops never runs.
// A chain that ends, or an aiex.dma_channel_reset since the latest load, frees
// the channel; after a load, the loaded device's programs are the ones that run.

// RUN: aie-opt --verify-diagnostics --split-input-file --aie-assign-runtime-sequence-bd-ids %s

module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    aie.buffer @buf(%t) {address = 0 : i32} : memref<64xi32>
    aie.memtile_dma(%t) {
      // expected-note @+1 {{the program starts here}}
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      %buf = aie.get_buffer @buf : memref<64xi32>
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    aie.runtime_sequence @seq() {
      %task = aiex.dma_configure_task(%t, MM2S, 0) {
        %buf_1 = aie.get_buffer @buf : memref<64xi32>
        aie.dma_bd(%buf_1 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      // expected-error @+1 {{starts a task on a channel whose static DMA program loops forever, so the task never runs; end that program's BD chain, or reset the channel with aiex.dma_channel_reset first}}
      aiex.dma_start_task(%task)
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    aie.buffer @buf(%t) {address = 0 : i32} : memref<64xi32>
    aie.memtile_dma(%t) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^s2mm)
    ^bd0:
      %buf = aie.get_buffer @buf : memref<64xi32>
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64) {bd_id = 0 : i32}
      aie.end
    ^s2mm:
      %1 = aie.dma_start(S2MM, 0, ^bd1, ^end)
    ^bd1:
      %buf_1 = aie.get_buffer @buf : memref<64xi32>
      aie.dma_bd(%buf_1 : memref<64xi32> offset = 0 len = 64) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.next_bd ^bd1
    ^end:
      aie.end
    }
    aie.runtime_sequence @seq() {
      %out = aiex.dma_configure_task(%t, MM2S, 0) {
        %buf_2 = aie.get_buffer @buf : memref<64xi32>
        aie.dma_bd(%buf_2 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%out)
      aiex.dma_channel_reset(%t, S2MM, 0)
      %in = aiex.dma_configure_task(%t, S2MM, 0) {
        %buf_3 = aie.get_buffer @buf : memref<64xi32>
        aie.dma_bd(%buf_3 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%in)
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    aie.buffer @buf(%t) {address = 0 : i32} : memref<64xi32>
    aie.runtime_sequence @seq() {
      aiex.dma_channel_reset(%t, MM2S, 0)
      aiex.npu.load_pdi {device_ref = @looping}
      %task = aiex.dma_configure_task(%t, MM2S, 0) {
        %buf = aie.get_buffer @buf : memref<64xi32>
        aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      // expected-error @+1 {{starts a task on a channel whose static DMA program loops forever}}
      aiex.dma_start_task(%task)
    }
  }
  aie.device(npu2) @looping {
    %t = aie.tile(0, 1)
    aie.buffer @buf(%t) {address = 0 : i32} : memref<64xi32>
    aie.memtile_dma(%t) {
      // expected-note @+1 {{the program starts here}}
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      %buf = aie.get_buffer @buf : memref<64xi32>
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    aie.buffer @buf(%t) {address = 0 : i32} : memref<64xi32>
    aie.memtile_dma(%t) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      %buf = aie.get_buffer @buf : memref<64xi32>
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    aie.runtime_sequence @seq() {
      aiex.npu.load_pdi {device_ref = @quiet}
      %task = aiex.dma_configure_task(%t, MM2S, 0) {
        %buf_1 = aie.get_buffer @buf : memref<64xi32>
        aie.dma_bd(%buf_1 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%task)
    }
  }
  aie.device(npu2) @quiet {
    %t = aie.tile(0, 1)
  }
}
