//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --pass-pipeline='builtin.module(aie.device(aie-substitute-shim-dma-allocations,aie-assign-runtime-sequence-bd-ids))' \
// RUN:         --verify-diagnostics --split-input-file %s

// A dispatch loads its own device first, so only a sequence that ends on
// another device's PDI leaves that array to a later dispatch, whose proofs
// count on a clean start. The last stretch is held to the loaded device's
// own proofs. Inputs using aiex.configure are given materialized, their
// copies' proofs naming the device they describe.

// @main runs @a's balanced round, then @b's, both on shim (0,0) channel 0,
// and ends with @b loaded. @b's proof holds over the last stretch; @a's copy
// describes @a's array, which a real load of @a resets.
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%shim_noc_tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1, device = @b>]}
    aie.shim_dma_allocation @out(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @in_0(%shim_noc_tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out_0, input_bytes = 1, output_bytes = 1, device = @a>]}
    aie.shim_dma_allocation @out_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.runtime_sequence @main_seq(%arg0: memref<512xi32>, %arg1: memref<512xi32>) {
      aiex.npu.load_pdi {device_ref = @a}
      %0 = aiex.dma_configure_task_for @in_0 {
        aie.dma_bd(%arg0 : memref<512xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @out_0 {
        aie.dma_bd(%arg1 : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%0)
      aiex.npu.load_pdi {device_ref = @b}
      %2 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<512xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg1 : memref<512xi32> offset = 0 len = 64)
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
    aie.runtime_sequence @seq(%arg0: memref<512xi32>, %arg1: memref<512xi32>) {
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
    }
  }
  aie.device(npu2) @b {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%shim_noc_tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%shim_noc_tile_0_0, S2MM, 0)
    aie.runtime_sequence @seq(%arg0: memref<512xi32>, %arg1: memref<512xi32>) {
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
    }
  }
}

// -----

// @main runs its own proven pair, then loads @a and runs @a's balanced round
// on the same circuit channels, through copies written without a device: on
// a circuit channel every push counts as the route's, so the tail balances.
module {
  aie.device(npu2) @main {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @mine_in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @mine_out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @mine_out(%t, S2MM, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%t, S2MM, 0)
    aie.runtime_sequence @main_seq(%arg0: memref<512xi32>, %arg1: memref<512xi32>) {
      %m0 = aiex.dma_configure_task_for @mine_in {
        aie.dma_bd(%arg0 : memref<512xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%m0)
      %m1 = aiex.dma_configure_task_for @mine_out {
        aie.dma_bd(%arg1 : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%m1)
      aiex.dma_await_task(%m1)
      aiex.dma_free_task(%m0)
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
    }
  }
  aie.device(npu2) @a {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%t, S2MM, 0)
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
    }
  }
}

// -----

// @main runs @a's balanced round, then pushes 64 more words on @a's input
// channel through @x: @a's next dispatch would find them in the array.
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @x(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @in(%shim_noc_tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1, device = @a>]}
    aie.shim_dma_allocation @out(%shim_noc_tile_0_0, S2MM, 0)
    aie.runtime_sequence @main_seq(%arg0: memref<512xi32>, %arg1: memref<512xi32>) {
      // expected-error@+1 {{moves more through that input than the output proves taken}}
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

// -----

// The tail moves @n units in and out, balanced for every @n, but @n has no
// lower bound, so the count cannot be proven.
module {
  aiex.scratchpad_parameter @n : i32 {kind = 0 : i32, max_value = 4 : i32, state_table_idx = 0 : ui8}
  aie.device(npu2) @main {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%t, S2MM, 0)
    aie.runtime_sequence @main_seq(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>) {
      // expected-error@+1 {{moves bytes on their channels whose count is not known, so cannot be proven balanced}}
      aiex.npu.load_pdi {device_ref = @a}
      %0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 64 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg1 : memref<4096xi32> offset = 0 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 64 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
    }
  }
  aie.device(npu2) @a {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%t, S2MM, 0)
  }
}

// -----

// Both of @a's proofs see the tail send 128 bytes in and take 64 out, so
// each pair is reported, in the order of their channels, whichever order
// @a declares its allocations in.
module {
  aie.device(npu2) @main {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in0(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out0, input_bytes = 1, output_bytes = 1, device = @a>]}
    aie.shim_dma_allocation @out0(%t, S2MM, 0)
    aie.shim_dma_allocation @in1(%t, MM2S, 1) {consumed_by = [#aie.consumed_by<output = @out1, input_bytes = 1, output_bytes = 1, device = @a>]}
    aie.shim_dma_allocation @out1(%t, S2MM, 1)
    aie.runtime_sequence @main_seq(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>) {
      // expected-error@+2 {{whose proof that @in0 is consumed by @out0 counts on a clean start, finds what this sequence's last stretch left: it moves more through that input than the output proves taken}}
      // expected-error@+1 {{whose proof that @in1 is consumed by @out1 counts on a clean start, finds what this sequence's last stretch left: it moves more through that input than the output proves taken}}
      aiex.npu.load_pdi {device_ref = @a}
      %i0 = aiex.dma_configure_task_for @in0 {
        aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 128)
        aie.end
      }
      aiex.dma_start_task(%i0)
      %i1 = aiex.dma_configure_task_for @in1 {
        aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 128)
        aie.end
      }
      aiex.dma_start_task(%i1)
      %o0 = aiex.dma_configure_task_for @out0 {
        aie.dma_bd(%arg1 : memref<4096xi32> offset = 0 len = 64)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%o0)
      %o1 = aiex.dma_configure_task_for @out1 {
        aie.dma_bd(%arg1 : memref<4096xi32> offset = 0 len = 64)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%o1)
      aiex.dma_await_task(%o0)
      aiex.dma_await_task(%o1)
      aiex.dma_free_task(%i0)
      aiex.dma_free_task(%i1)
    }
  }
  aie.device(npu2) @a {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in0(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out0, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out0(%t, S2MM, 0)
    aie.shim_dma_allocation @in1(%t, MM2S, 1) {consumed_by = [#aie.consumed_by<output = @out1, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out1(%t, S2MM, 1)
  }
}

// -----

module {
  aie.device(npu2) @main {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in0(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out0, input_bytes = 1, output_bytes = 1, device = @a>]}
    aie.shim_dma_allocation @out0(%t, S2MM, 0)
    aie.shim_dma_allocation @in1(%t, MM2S, 1) {consumed_by = [#aie.consumed_by<output = @out1, input_bytes = 1, output_bytes = 1, device = @a>]}
    aie.shim_dma_allocation @out1(%t, S2MM, 1)
    aie.runtime_sequence @main_seq(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>) {
      // expected-error@+2 {{whose proof that @in0 is consumed by @out0 counts on a clean start, finds what this sequence's last stretch left: it moves more through that input than the output proves taken}}
      // expected-error@+1 {{whose proof that @in1 is consumed by @out1 counts on a clean start, finds what this sequence's last stretch left: it moves more through that input than the output proves taken}}
      aiex.npu.load_pdi {device_ref = @a}
      %i0 = aiex.dma_configure_task_for @in0 {
        aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 128)
        aie.end
      }
      aiex.dma_start_task(%i0)
      %i1 = aiex.dma_configure_task_for @in1 {
        aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 128)
        aie.end
      }
      aiex.dma_start_task(%i1)
      %o0 = aiex.dma_configure_task_for @out0 {
        aie.dma_bd(%arg1 : memref<4096xi32> offset = 0 len = 64)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%o0)
      %o1 = aiex.dma_configure_task_for @out1 {
        aie.dma_bd(%arg1 : memref<4096xi32> offset = 0 len = 64)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%o1)
      aiex.dma_await_task(%o0)
      aiex.dma_await_task(%o1)
      aiex.dma_free_task(%i0)
      aiex.dma_free_task(%i1)
    }
  }
  aie.device(npu2) @a {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in1(%t, MM2S, 1) {consumed_by = [#aie.consumed_by<output = @out1, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out1(%t, S2MM, 1)
    aie.shim_dma_allocation @in0(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out0, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out0(%t, S2MM, 0)
  }
}
