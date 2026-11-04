//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --pass-pipeline='builtin.module(aie.device(aie-substitute-shim-dma-allocations,aie-assign-runtime-sequence-bd-ids))' \
// RUN:         --verify-diagnostics --split-input-file %s

// A dispatch loads its own device first, so only a sequence that ends on
// another device's PDI leaves that array to a later dispatch. @main runs @a's
// balanced @rounds, materialized, and ends with @a loaded: @a's next dispatch
// starts clean.
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @x(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @in(%shim_noc_tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%shim_noc_tile_0_0, S2MM, 0)
    aie.runtime_sequence @main_seq(%arg0: memref<512xi32>, %arg1: memref<512xi32>) {
      aiex.npu.load_pdi {device_ref = @a}
      %0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<512xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg1 : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%0)
      %2 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<512xi32> offset = 64 len = 64)
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg1 : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%3)
      aiex.dma_await_task(%3)
      aiex.dma_free_task(%2)
    }
  }
  aie.device(npu2) @a {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%shim_noc_tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%shim_noc_tile_0_0, S2MM, 0)
    aie.runtime_sequence @rounds(%arg0: memref<512xi32>, %arg1: memref<512xi32>) {
      %0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<512xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg1 : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%0)
      %2 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<512xi32> offset = 64 len = 64)
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg1 : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%3)
      aiex.dma_await_task(%3)
      aiex.dma_free_task(%2)
    }
  }
}

// -----

// The same, but @main then pushes on @a's input channel through @x, so @a's
// next dispatch would find those words in the array while its proofs count on
// a clean start.
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @x(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @in(%shim_noc_tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%shim_noc_tile_0_0, S2MM, 0)
    aie.runtime_sequence @main_seq(%arg0: memref<512xi32>, %arg1: memref<512xi32>) {
      // expected-error@+1 {{leaves @a loaded with transfers through @in that @out does not prove consumed, or with other traffic on their channels}}
      aiex.npu.load_pdi {device_ref = @a}
      %0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<512xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg1 : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%0)
      %2 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<512xi32> offset = 64 len = 64)
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg1 : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%3)
      aiex.dma_await_task(%3)
      aiex.dma_free_task(%2)
      %4 = aiex.dma_configure_task_for @x {
        aie.dma_bd(%arg0 : memref<512xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%4)
    }
  }
  aie.device(npu2) @a {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%shim_noc_tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%shim_noc_tile_0_0, S2MM, 0)
    aie.runtime_sequence @rounds(%arg0: memref<512xi32>, %arg1: memref<512xi32>) {
      %0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<512xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg1 : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%0)
      %2 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<512xi32> offset = 64 len = 64)
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg1 : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%3)
      aiex.dma_await_task(%3)
      aiex.dma_free_task(%2)
    }
  }
}
