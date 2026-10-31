//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids='warn-unsafe-bd-reuse=true' \
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

// The next round's input, pushed before this round's output is awaited, lies
// past the bytes that await proves taken: its id stays unsafe.
module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%tile_0_0, S2MM, 0)
    aie.runtime_sequence @prefetch(%a: memref<512xi32>, %c: memref<512xi32>) {
      %next = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @in}
      %in0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @in}
      %out0 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%in0)
      aiex.dma_start_task(%out0)
      aiex.dma_start_task(%next)
      aiex.dma_await_task(%out0)
      // expected-note@+1 {{released here}}
      aiex.dma_free_task(%next)
      %out1 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        // expected-warning@+1 {{reuses buffer descriptor ID 0 on tile (0,0) after it was released by an aiex.dma_free_task that had no completion guarantee}}
        aie.dma_bd(%c : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out1)
      aiex.dma_await_task(%out1)
      aiex.dma_free_task(%in0)
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

// An input whose rounds push more than the output covers is consumed only in
// part when the output finishes.
module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%tile_0_0, S2MM, 0)
    aie.runtime_sequence @partial(%a: memref<512xi32>, %c: memref<512xi32>) {
      %in0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 0 len = 128)
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
        aie.dma_bd(%a : memref<512xi32> offset = 128 len = 128)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in1)
    }
  }
}

// -----

// Without a load first, a sequence that leaves input bytes uncovered at its
// end could leave them for the next dispatch, whose counts would then be off:
// nothing is credited.
module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%tile_0_0, S2MM, 0)
    aie.runtime_sequence @leaves_input(%a: memref<512xi32>, %c: memref<512xi32>) {
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
    }
  }
}

// -----

// A load of the PDI already loaded leaves the array as it was, so a stretch
// after a load must leave no input bytes behind either: this one pushes in1
// past what its output covers, and nothing is credited.
module {
  aie.device(npu2) @design {
    %tile_0_0 = aie.tile(0, 0)
  }
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%tile_0_0, S2MM, 0)
    aie.runtime_sequence @after_load(%a: memref<512xi32>, %c: memref<512xi32>) {
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
      // expected-note@+1 {{released here}}
      aiex.dma_free_task(%in0)
      %in1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-warning@+1 {{reuses buffer descriptor ID 0 on tile (0,0) after it was released by an aiex.dma_free_task that had no completion guarantee}}
        aie.dma_bd(%a : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in1)
    }
  }
}

// -----

// @init loads, then pushes 128 words in and receives 64, leaving the rest in
// the array. @next, balanced on its own, may run after it without a real
// reload and receive those leftovers first: nothing is credited.
module {
  aie.device(npu2) @design {
    %tile_0_0 = aie.tile(0, 0)
  }
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%tile_0_0, S2MM, 0)
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
  }
  aie.device(npu2) {
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
