//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids='warn-unsafe-bd-reuse=true warn-unreleased-transfers=false' \
// RUN:         --verify-diagnostics --split-input-file %s

// An input's shim allocation may list outputs whose progress proves its bytes
// consumed (#aie.consumed_by, recorded by aie-objectfifo-verify). When an
// await retires an output's push, every input push within the bytes that
// proves taken has finished, so freeing it and reusing its BD id is safe. The
// sequences below move as much through each input as their proofs cover, as a
// dispatch run without a load before it must for its counts to hold.

// A ProjectionGEMM-style round: the input is freed once only the output is
// awaited, and the next round reuses its id.
module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%tile_0_0, S2MM, 0)
    aie.runtime_sequence @rounds(%a: memref<512xi32>, %c: memref<512xi32>) {
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
    }
  }
}

// -----

// The same round with the input's len omitted: each push moves the whole
// 64-element buffer, as the output's explicit len does.
module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%tile_0_0, S2MM, 0)
    aie.runtime_sequence @rounds(%a: memref<64xi32>, %c: memref<64xi32>) {
      %in0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%a : memref<64xi32>)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in0)
      %out0 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<64xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out0)
      aiex.dma_await_task(%out0)
      aiex.dma_free_task(%in0)
      %in1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%a : memref<64xi32>)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in1)
      %out1 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<64xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out1)
      aiex.dma_await_task(%out1)
      aiex.dma_free_task(%in1)
    }
  }
}

// -----

// An MHA-style round: q is pushed before the output and k after it, k's
// length the parameter @n, as is part of the output's. Both are covered once
// the output is awaited, whatever @n is.
module {
  aiex.scratchpad_parameter @n : i32 {kind = 0 : i32, max_value = 4 : i32, min_value = 0 : i32, state_table_idx = 0 : ui8}
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @q(%tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @k(%tile_0_0, MM2S, 1) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%tile_0_0, S2MM, 0)
    aie.runtime_sequence @rounds(%a: memref<4096xi32>, %c: memref<4096xi32>) {
      %q0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%a : memref<4096xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @q}
      aiex.dma_start_task(%q0)
      %out0 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<4096xi32> offset = 0 len = 64) {length_state_table_idx = 0 : ui8, length_unit = 64 : i32}
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out0)
      %k0 = aiex.dma_configure_task(%tile_0_0, MM2S, 1) {
        aie.dma_bd(%a : memref<4096xi32> offset = 1024 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 64 : i32}
        aie.end
      } {alloc = @k}
      aiex.dma_start_task(%k0)
      aiex.dma_await_task(%out0)
      aiex.dma_free_task(%q0)
      aiex.dma_free_task(%k0)
      %q1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%a : memref<4096xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @q}
      aiex.dma_start_task(%q1)
      %out1 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<4096xi32> offset = 2048 len = 64) {length_state_table_idx = 0 : ui8, length_unit = 64 : i32}
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out1)
      %k1 = aiex.dma_configure_task(%tile_0_0, MM2S, 1) {
        aie.dma_bd(%a : memref<4096xi32> offset = 2048 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 64 : i32}
        aie.end
      } {alloc = @k}
      aiex.dma_start_task(%k1)
      aiex.dma_await_task(%out1)
      aiex.dma_free_task(%q1)
      aiex.dma_free_task(%k1)
    }
  }
}

// -----

// k's length is the parameter @m, which nothing ties to the output's @n: its
// bytes are not proven taken.
module {
  aiex.scratchpad_parameter @n : i32 {kind = 0 : i32, max_value = 4 : i32, min_value = 0 : i32, state_table_idx = 0 : ui8}
  aiex.scratchpad_parameter @m : i32 {kind = 0 : i32, max_value = 4 : i32, min_value = 0 : i32, state_table_idx = 1 : ui8}
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @k(%tile_0_0, MM2S, 1) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%tile_0_0, S2MM, 0)
    // expected-note@+1 {{the stretch ends here}}
    aie.runtime_sequence @other_parameter(%a: memref<4096xi32>, %c: memref<4096xi32>) {
      %k0 = aiex.dma_configure_task(%tile_0_0, MM2S, 1) {
        aie.dma_bd(%a : memref<4096xi32> offset = 0 len = 0) {length_state_table_idx = 1 : ui8, length_unit = 64 : i32}
        aie.end
      } {alloc = @k}
      %out0 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<4096xi32> offset = 0 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 64 : i32}
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out0)
      aiex.dma_start_task(%k0)
      aiex.dma_await_task(%out0)
      // expected-note@+1 {{released here}}
      aiex.dma_free_task(%k0)
      %k1 = aiex.dma_configure_task(%tile_0_0, MM2S, 1) {
        // expected-warning@+1 {{reuses buffer descriptor ID 0 on tile (0,0) after it was released by an aiex.dma_free_task that had no completion guarantee}}
        aie.dma_bd(%a : memref<4096xi32> offset = 1024 len = 0) {length_state_table_idx = 1 : ui8, length_unit = 64 : i32}
        aie.end
      } {alloc = @k}
      aiex.dma_start_task(%k1)
    }
  }
}

// -----

// A load of the PDI already loaded leaves the array as it was. @init loads,
// then pushes 128 words in and receives 64, leaving the rest in the array.
// @next, balanced on its own, may run after it without a real reload and
// receive those leftovers first: nothing is credited.
module {
  aie.device(npu2) @design {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%tile_0_0, S2MM, 0)
    // expected-note@+1 {{the stretch ends here}}
    aie.runtime_sequence @init(%a: memref<512xi32>, %c: memref<512xi32>) {
      aiex.npu.load_pdi {device_ref = @design}
      %in0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 0 len = 128)
        aie.end
      } {alloc = @in, issue_token = true}
      aiex.dma_start_task(%in0)
      %out0 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out0)
      aiex.dma_await_task(%in0)
      aiex.dma_await_task(%out0)
    }
    aie.runtime_sequence @next(%a: memref<512xi32>, %c: memref<512xi32>) {
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
    }
  }
}

// -----

// Balanced stretches before and after a load: the rounds are credited.
module {
  aie.device(npu2) @design {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%tile_0_0, S2MM, 0)
    aie.runtime_sequence @balanced_after_load(%a: memref<512xi32>, %c: memref<512xi32>) {
      aiex.npu.load_pdi {device_ref = @design}
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
    }
  }
}

// -----

// Slot 0 holds @n here and @o, which may be negative, in some other command.
// This command lists the parameters it reads, so the slot is @n, held at 0 or
// above, and k is covered once the output is awaited.
module {
  aiex.scratchpad_parameter @n : i32 {kind = 0 : i32, max_value = 4 : i32, min_value = 0 : i32, state_table_idx = 0 : ui8}
  aiex.scratchpad_parameter @o : i32 {kind = 0 : i32, max_value = 4 : i32, min_value = -4 : i32, state_table_idx = 0 : ui8}
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @k(%tile_0_0, MM2S, 1) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%tile_0_0, S2MM, 0)
    aie.runtime_sequence @listed(%a: memref<4096xi32>, %c: memref<4096xi32>) attributes {scratchpad_parameters = [@n]} {
      %k0 = aiex.dma_configure_task(%tile_0_0, MM2S, 1) {
        aie.dma_bd(%a : memref<4096xi32> offset = 0 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 64 : i32}
        aie.end
      } {alloc = @k}
      aiex.dma_start_task(%k0)
      %out0 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<4096xi32> offset = 0 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 64 : i32}
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out0)
      aiex.dma_await_task(%out0)
      aiex.dma_free_task(%k0)
      %k1 = aiex.dma_configure_task(%tile_0_0, MM2S, 1) {
        aie.dma_bd(%a : memref<4096xi32> offset = 1024 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 64 : i32}
        aie.end
      } {alloc = @k}
      aiex.dma_start_task(%k1)
      %out1 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<4096xi32> offset = 1024 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 64 : i32}
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out1)
      aiex.dma_await_task(%out1)
    }
  }
}

// -----

// The same command without its list: slot 0 may hold @o, which may be
// negative, so k's bytes are not proven covered.
module {
  aiex.scratchpad_parameter @n : i32 {kind = 0 : i32, max_value = 4 : i32, min_value = 0 : i32, state_table_idx = 0 : ui8}
  aiex.scratchpad_parameter @o : i32 {kind = 0 : i32, max_value = 4 : i32, min_value = -4 : i32, state_table_idx = 0 : ui8}
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @k(%tile_0_0, MM2S, 1) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%tile_0_0, S2MM, 0)
    aie.runtime_sequence @unlisted(%a: memref<4096xi32>, %c: memref<4096xi32>) {
      %k0 = aiex.dma_configure_task(%tile_0_0, MM2S, 1) {
        aie.dma_bd(%a : memref<4096xi32> offset = 0 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 64 : i32}
        aie.end
      } {alloc = @k}
      // expected-note@+1 {{the transfer}}
      aiex.dma_start_task(%k0)
      %out0 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<4096xi32> offset = 0 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 64 : i32}
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out0)
      aiex.dma_await_task(%out0)
      // expected-note@+1 {{released here}}
      aiex.dma_free_task(%k0)
      %k1 = aiex.dma_configure_task(%tile_0_0, MM2S, 1) {
        // expected-warning@+1 {{reuses buffer descriptor ID 0 on tile (0,0) after it was released by an aiex.dma_free_task that had no completion guarantee}}
        aie.dma_bd(%a : memref<4096xi32> offset = 1024 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 64 : i32}
        aie.end
      } {alloc = @k}
      aiex.dma_start_task(%k1)
      %out1 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<4096xi32> offset = 1024 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 64 : i32}
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out1)
      aiex.dma_await_task(%out1)
    }
  }
}

// -----

// A copy on the input's own channel counts as the input's own bytes (see
// consumed-by-copy.mlir), so it moves more through @in than @out proves
// taken by the time the sequence ends: nothing is credited.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%t, S2MM, 0)
    // expected-note@+1 {{the stretch ends here}}
    aie.runtime_sequence @rounds(%a: memref<512xi32>, %c: memref<512xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 15 : i64, metadata = @in} : memref<512xi32>
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
        // expected-warning@+1 {{reuses buffer descriptor ID 0 on tile (0,0) after it was released by an aiex.dma_free_task that had no completion guarantee}}
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
    }
  }
}

// -----

// A raw write to the input's task queue (shim MM2S 0's, 0x1D214, the word
// after its control register) pushes a task the counts cannot see, so no await
// on the output proves the input's pushes finished.
module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%tile_0_0, S2MM, 0)
    aie.runtime_sequence @rounds(%a: memref<512xi32>, %c: memref<512xi32>) {
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
      // expected-note@+1 {{the other transfer}}
      aiex.npu.write32(%queue, %bd) {column = 0 : i32, row = 0 : i32} : i32, i32
    }
  }
}

// -----

// @main's proof says its core takes @in before writing @out, but after
// load_pdi @b the array runs @b's design: no await on @out proves @in's
// pushes finished there.
module {
  aie.device(npu2) @b {
    %t = aie.tile(0, 0)
  }
  aie.device(npu2) @main {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%t, S2MM, 0)
    aie.runtime_sequence @seq(%a: memref<512xi32>, %c: memref<512xi32>) {
      aiex.npu.load_pdi {device_ref = @b}
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
        // expected-warning@+1 {{reuses buffer descriptor ID 0}}
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
      aiex.npu.load_pdi {device_ref = @main}
    }
  }
}
