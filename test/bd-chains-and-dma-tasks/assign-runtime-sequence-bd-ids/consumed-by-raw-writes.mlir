//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids='warn-unsafe-bd-reuse=true' \
// RUN:         --verify-diagnostics --split-input-file %s

// A raw register write onto the input's task queue (shim MM2S 0's, 0x1D214)
// pushes a task the counts cannot see, so no await on the output proves the
// input's pushes finished; one that may land anywhere counts the same. A
// write that stops short of the queue leaves the proof standing.

// A maskwrite32 onto the queue.
module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%tile_0_0, S2MM, 0)
    aie.runtime_sequence @rounds(%a: memref<512xi32>, %c: memref<512xi32>, %where: i32) {
      %in0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in0)
      %out0 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out0)
      aiex.dma_await_task(%out0)
      // expected-note@+1 {{released here}}
      aiex.dma_free_task(%in0)
      %in1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-warning@+1 {{reuses buffer descriptor ID 0 on tile (0,0) after it was released by an aiex.dma_free_task that had no completion guarantee}}
        aie.dma_bd(%a : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in1)
      %out1 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out1)
      aiex.dma_await_task(%out1)
      aiex.dma_free_task(%in1)
      %queue = arith.constant 119316 : i32
      %bd = arith.constant 15 : i32
      %mask = arith.constant 15 : i32
      // expected-note@+1 {{the other transfer}}
      aiex.npu.maskwrite32(%queue, %bd, %mask) {column = 0 : i32, row = 0 : i32} : i32, i32, i32
    }
  }
}

// -----

// A blockwrite whose second word is the queue.
module {
  aie.device(npu2) {
    memref.global "private" constant @words : memref<2xi32> = dense<[15, 0]>
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%tile_0_0, S2MM, 0)
    aie.runtime_sequence @rounds(%a: memref<512xi32>, %c: memref<512xi32>, %where: i32) {
      %in0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in0)
      %out0 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out0)
      aiex.dma_await_task(%out0)
      // expected-note@+1 {{released here}}
      aiex.dma_free_task(%in0)
      %in1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-warning@+1 {{reuses buffer descriptor ID 0 on tile (0,0) after it was released by an aiex.dma_free_task that had no completion guarantee}}
        aie.dma_bd(%a : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in1)
      %out1 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out1)
      aiex.dma_await_task(%out1)
      aiex.dma_free_task(%in1)
      %w = memref.get_global @words : memref<2xi32>
      // expected-note@+1 {{the other transfer}}
      aiex.npu.blockwrite(%w) {address = 119312 : ui32, column = 0 : i32, row = 0 : i32} : memref<2xi32>
    }
  }
}

// -----

// A write32 at an address the host passes in.
module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%tile_0_0, S2MM, 0)
    aie.runtime_sequence @rounds(%a: memref<512xi32>, %c: memref<512xi32>, %where: i32) {
      %in0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in0)
      %out0 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out0)
      aiex.dma_await_task(%out0)
      // expected-note@+1 {{released here}}
      aiex.dma_free_task(%in0)
      %in1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-warning@+1 {{reuses buffer descriptor ID 0 on tile (0,0) after it was released by an aiex.dma_free_task that had no completion guarantee}}
        aie.dma_bd(%a : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in1)
      %out1 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out1)
      aiex.dma_await_task(%out1)
      aiex.dma_free_task(%in1)
      %bd = arith.constant 15 : i32
      // expected-note@+1 {{the other transfer}}
      aiex.npu.write32(%where, %bd) : i32, i32
    }
  }
}

// -----

// A blockwrite of the two words before the queue stops short of it.
module {
  aie.device(npu2) {
    memref.global "private" constant @words : memref<2xi32> = dense<[15, 0]>
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%tile_0_0, S2MM, 0)
    aie.runtime_sequence @rounds(%a: memref<512xi32>, %c: memref<512xi32>, %where: i32) {
      %in0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in0)
      %out0 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out0)
      aiex.dma_await_task(%out0)
      aiex.dma_free_task(%in0)
      %in1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in1)
      %out1 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out1)
      aiex.dma_await_task(%out1)
      aiex.dma_free_task(%in1)
      %w = memref.get_global @words : memref<2xi32>
      aiex.npu.blockwrite(%w) {address = 119308 : ui32, column = 0 : i32, row = 0 : i32} : memref<2xi32>
    }
  }
}

// -----

// A maskwrite32 onto the same queue on another column.
module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%tile_0_0, S2MM, 0)
    aie.runtime_sequence @rounds(%a: memref<512xi32>, %c: memref<512xi32>, %where: i32) {
      %in0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in0)
      %out0 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out0)
      aiex.dma_await_task(%out0)
      aiex.dma_free_task(%in0)
      %in1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in1)
      %out1 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out1)
      aiex.dma_await_task(%out1)
      aiex.dma_free_task(%in1)
      %queue = arith.constant 119316 : i32
      %bd = arith.constant 15 : i32
      %mask = arith.constant 15 : i32
      aiex.npu.maskwrite32(%queue, %bd, %mask) {column = 1 : i32, row = 0 : i32} : i32, i32, i32
    }
  }
}
