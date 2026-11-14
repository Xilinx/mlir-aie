//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-lower-dynamic-bd-pool \
// RUN:         --aie-assign-runtime-sequence-bd-ids='warn-unsafe-bd-reuse=true' \
// RUN:         --verify-diagnostics %s

// A sequence the dynamic BD pool keeps rolled pushes @in %n times and @out
// once. Its loop runs a number of times the counts cannot follow, so it may
// leave input words behind for the next dispatch, and no await on @out in
// @rounds proves @in's pushes finished.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%t, S2MM, 0)
    aie.runtime_sequence @dyn(%a: memref<512xi32>, %c: memref<512xi32>, %n: index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %n step %c1 {
        %in = aiex.dma_configure_task(%t, MM2S, 0) {
          aie.dma_bd(%a : memref<512xi32> offset = 0 len = 64)
          aie.end
        } {alloc = @in, issue_token = true}
        // expected-note@+1 {{the transfer}}
        aiex.dma_start_task(%in)
        aiex.dma_await_task(%in)
      }
      %out = aiex.dma_configure_task(%t, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out)
      aiex.dma_await_task(%out)
    }
    aie.runtime_sequence @rounds(%a: memref<512xi32>, %c: memref<512xi32>) {
      %in0 = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in0)
      %out0 = aiex.dma_configure_task(%t, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out0)
      aiex.dma_await_task(%out0)
      // expected-note@+1 {{released here}}
      aiex.dma_free_task(%in0)
      %in1 = aiex.dma_configure_task(%t, MM2S, 0) {
        // expected-warning@+2 {{reuses buffer descriptor ID 0 on tile (0,0) after it was released by an aiex.dma_free_task that had no completion guarantee}}
        // expected-note@+1 {{not proven finished: through @out, a transfer on them moves a byte count the compiler cannot follow (a BD chain, a loop, or a length parameter without min_value >= 0)}}
        aie.dma_bd(%a : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in1)
      %out1 = aiex.dma_configure_task(%t, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out1)
      aiex.dma_await_task(%out1)
      aiex.dma_free_task(%in1)
    }
  }
}
