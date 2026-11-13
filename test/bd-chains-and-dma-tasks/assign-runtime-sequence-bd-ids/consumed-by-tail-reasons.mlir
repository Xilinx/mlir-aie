//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --pass-pipeline='builtin.module(aie.device(aie-substitute-shim-dma-allocations,aie-assign-runtime-sequence-bd-ids))' \
// RUN:         --verify-diagnostics --split-input-file %s

// A sequence that ends on another device's PDI leaves that array to the next
// dispatch, whose consumed_by proofs count on a clean start: the last stretch
// must balance under the loaded device's proofs, which @main's
// allocations carry as materialized copies naming @a. One outcome per reason.

// A raw write onto the input's task queue in the tail.
module {
  aie.device(npu2) @main {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1, device = @a>]}
    aie.shim_dma_allocation @out(%t, S2MM, 0)
    aie.runtime_sequence @main_seq(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>) {
      // expected-error@+1 {{has other traffic on their channels}}
      aiex.npu.load_pdi {device_ref = @a}
      %i0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%i0)
      %o0 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg1 : memref<4096xi32> offset = 0 len = 64)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%o0)
      aiex.dma_await_task(%o0)
      aiex.dma_free_task(%i0)
      %queue = arith.constant 119316 : i32
      %bd = arith.constant 15 : i32
      aiex.npu.write32(%queue, %bd) {column = 0 : i32, row = 0 : i32} : i32, i32
    }
  }
  aie.device(npu2) @a {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%t, S2MM, 0)
  }
}

// -----

// The proof counts output only in whole 512-byte objects of a join, and the
// tail's output stops 256 bytes into one.
module {
  aie.device(npu2) @main {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1, granule_bytes = 512, device = @a>]}
    aie.shim_dma_allocation @out(%t, S2MM, 0)
    aie.runtime_sequence @main_seq(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>) {
      // expected-error@+1 {{may end with the output partway through an object of the join the proof passes}}
      aiex.npu.load_pdi {device_ref = @a}
      %i0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%i0)
      %o0 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg1 : memref<4096xi32> offset = 0 len = 64)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%o0)
      aiex.dma_await_task(%o0)
      aiex.dma_free_task(%i0)
    }
  }
  aie.device(npu2) @a {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1, granule_bytes = 512>]}
    aie.shim_dma_allocation @out(%t, S2MM, 0)
  }
}

// -----

// The same proof with the tail ending on a whole object: balanced.
module {
  aie.device(npu2) @main {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1, granule_bytes = 512, device = @a>]}
    aie.shim_dma_allocation @out(%t, S2MM, 0)
    aie.runtime_sequence @main_seq(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>) {
      aiex.npu.load_pdi {device_ref = @a}
      %i0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 128)
        aie.end
      }
      aiex.dma_start_task(%i0)
      %o0 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg1 : memref<4096xi32> offset = 0 len = 128)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%o0)
      aiex.dma_await_task(%o0)
      aiex.dma_free_task(%i0)
    }
  }
  aie.device(npu2) @a {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1, granule_bytes = 512>]}
    aie.shim_dma_allocation @out(%t, S2MM, 0)
  }
}
