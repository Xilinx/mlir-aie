//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids='warn-unsafe-bd-reuse=true' \
// RUN:         --verify-diagnostics --split-input-file %s

// A proof with granule_bytes holds only for output bytes rounded down to a
// multiple of the granule: the input reaches the output through a later
// segment of a join, taken only once the segments ahead of it in each object
// are. Here an object of the join is 256 output bytes, and the input's bytes
// are half of the output's.

// Whole objects received: the input pushed before them is credited.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 2, granule_bytes = 256>]}
    aie.shim_dma_allocation @out(%t, S2MM, 0)
    aie.runtime_sequence @s(%a: memref<512xi32>, %c: memref<512xi32>) {
      %in0 = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 0 len = 32)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in0)
      %out0 = aiex.dma_configure_task(%t, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out0)
      aiex.dma_await_task(%out0)
      aiex.dma_free_task(%in0)
      %in1 = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 32 len = 32)
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

// -----

// Half an object received: the later segment may not have gone yet, so the
// input pushed before it is not credited, though its bytes are half the
// output's. The second half completes the object, so the stretch ends whole.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 2, granule_bytes = 256>]}
    aie.shim_dma_allocation @out(%t, S2MM, 0)
    aie.runtime_sequence @s(%a: memref<512xi32>, %c: memref<512xi32>) {
      %in0 = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 0 len = 16)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in0)
      %out0 = aiex.dma_configure_task(%t, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 0 len = 32)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out0)
      aiex.dma_await_task(%out0)
      // expected-note@+1 {{released here}}
      aiex.dma_free_task(%in0)
      %in1 = aiex.dma_configure_task(%t, MM2S, 0) {
        // expected-warning@+2 {{reuses buffer descriptor ID}}
        // expected-note@+1 {{not proven finished: through @out, the bytes its awaited transfers moved, less the part of an object of the join the proof passes (256 bytes) they may end in, times 1, -127, fall short of the bytes the input moved up to this transfer times 2, 128}}
        aie.dma_bd(%a : memref<512xi32> offset = 16 len = 16)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in1)
      %out1 = aiex.dma_configure_task(%t, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 32 len = 32)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out1)
      aiex.dma_await_task(%out1)
      aiex.dma_free_task(%in1)
    }
  }
}

// -----

// The output's length is a parameter in whole objects (64 words, 256 bytes,
// each): credited whatever @n is.
module {
  aiex.scratchpad_parameter @n : i32 {kind = 0 : i32, max_value = 4 : i32, min_value = 0 : i32, state_table_idx = 0 : ui8}
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 2, granule_bytes = 256>]}
    aie.shim_dma_allocation @out(%t, S2MM, 0)
    aie.runtime_sequence @s(%a: memref<4096xi32>, %c: memref<4096xi32>) {
      %in0 = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%a : memref<4096xi32> offset = 0 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 32 : i32}
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in0)
      %out0 = aiex.dma_configure_task(%t, S2MM, 0) {
        aie.dma_bd(%c : memref<4096xi32> offset = 0 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 64 : i32}
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out0)
      aiex.dma_await_task(%out0)
      aiex.dma_free_task(%in0)
      %in1 = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%a : memref<4096xi32> offset = 2048 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 32 : i32}
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in1)
      %out1 = aiex.dma_configure_task(%t, S2MM, 0) {
        aie.dma_bd(%c : memref<4096xi32> offset = 2048 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 64 : i32}
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out1)
      aiex.dma_await_task(%out1)
      aiex.dma_free_task(%in1)
    }
  }
}

// -----

// The output's length is a parameter in half objects (32 words): for an odd
// @n the first await may end partway through an object, so the input pushed
// before it is not credited. Two such pushes make whole objects, so the
// dispatch itself ends whole.
module {
  aiex.scratchpad_parameter @n : i32 {kind = 0 : i32, max_value = 4 : i32, min_value = 0 : i32, state_table_idx = 0 : ui8}
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 2, granule_bytes = 256>]}
    aie.shim_dma_allocation @out(%t, S2MM, 0)
    aie.runtime_sequence @s(%a: memref<4096xi32>, %c: memref<4096xi32>) {
      %in0 = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%a : memref<4096xi32> offset = 0 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 16 : i32}
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in0)
      %out0 = aiex.dma_configure_task(%t, S2MM, 0) {
        aie.dma_bd(%c : memref<4096xi32> offset = 0 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 32 : i32}
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out0)
      aiex.dma_await_task(%out0)
      // expected-note@+1 {{released here}}
      aiex.dma_free_task(%in0)
      %in1 = aiex.dma_configure_task(%t, MM2S, 0) {
        // expected-warning@+2 {{reuses buffer descriptor ID}}
        // expected-note@+1 {{not proven finished: through @out, the bytes its awaited transfers moved, less the part of an object of the join the proof passes (256 bytes) they may end in, times 1, -255 + 128*@n, fall short of the bytes the input moved up to this transfer times 2, 128*@n, for some parameter values}}
        aie.dma_bd(%a : memref<4096xi32> offset = 2048 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 16 : i32}
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in1)
      %out1 = aiex.dma_configure_task(%t, S2MM, 0) {
        aie.dma_bd(%c : memref<4096xi32> offset = 2048 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 32 : i32}
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out1)
      aiex.dma_await_task(%out1)
      aiex.dma_free_task(%in1)
    }
  }
}

// -----

// A stretch whose constant output ends partway through an object.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 2, granule_bytes = 256>]}
    aie.shim_dma_allocation @out(%t, S2MM, 0)
    // expected-note@+1 {{the stretch ends here}}
    aie.runtime_sequence @s(%a: memref<512xi32>, %c: memref<512xi32>) {
      %in0 = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 0 len = 32)
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
        // expected-warning@+2 {{reuses buffer descriptor ID}}
        // expected-note@+1 {{not proven finished: through @out, a stretch of @s may end with the output partway through an object of the join the proof passes (256 output bytes)}}
        aie.dma_bd(%a : memref<512xi32> offset = 32 len = 16)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in1)
      %out1 = aiex.dma_configure_task(%t, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 64 len = 32)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out1)
      aiex.dma_await_task(%out1)
      aiex.dma_free_task(%in1)
    }
  }
}
