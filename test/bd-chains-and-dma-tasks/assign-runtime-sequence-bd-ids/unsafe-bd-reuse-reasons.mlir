//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids='warn-unsafe-bd-reuse=true warn-unreleased-transfers=false' \
// RUN:         --verify-diagnostics --split-input-file %s

// The unsafe-reuse warning says why no #aie.consumed_by proof shows the freed
// transfer finished, so the fix is clear from the warning alone.

// No proof recorded for the input.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0)
    aie.runtime_sequence @s(%a: memref<512xi32>) {
      %in0 = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in0)
      // expected-note@+1 {{released here}}
      aiex.dma_free_task(%in0)
      %in1 = aiex.dma_configure_task(%t, MM2S, 0) {
        // expected-warning@+2 {{reuses buffer descriptor ID}}
        // expected-note@+1 {{not proven finished: aie-objectfifo-verify found no output whose progress proves @in consumed}}
        aie.dma_bd(%a : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in1)
    }
  }
}

// -----

// A task with no shim allocation.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.runtime_sequence @s(%a: memref<512xi32>) {
      %in0 = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%in0)
      // expected-note@+1 {{released here}}
      aiex.dma_free_task(%in0)
      %in1 = aiex.dma_configure_task(%t, MM2S, 0) {
        // expected-warning@+2 {{reuses buffer descriptor ID}}
        // expected-note@+1 {{not proven finished: it names no shim allocation, so no output's progress can prove it consumed}}
        aie.dma_bd(%a : memref<512xi32> offset = 64 len = 64)
        aie.end
      }
      aiex.dma_start_task(%in1)
    }
  }
}

// -----

// The output is pushed but not yet awaited when the id is reused.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%t, S2MM, 0)
    aie.runtime_sequence @s(%a: memref<512xi32>, %c: memref<512xi32>) {
      %in0 = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in0)
      %out0 = aiex.dma_configure_task(%t, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 0 len = 128)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out0)
      // expected-note@+1 {{released here}}
      aiex.dma_free_task(%in0)
      %in1 = aiex.dma_configure_task(%t, MM2S, 0) {
        // expected-warning@+2 {{reuses buffer descriptor ID}}
        // expected-note@+1 {{not proven finished: through @out, no await on it has retired a transfer yet}}
        aie.dma_bd(%a : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in1)
      aiex.dma_await_task(%out0)
    }
  }
}

// -----

// A prefetched input lies past the bytes the awaited output proves taken.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%t, S2MM, 0)
    aie.runtime_sequence @s(%a: memref<512xi32>, %c: memref<512xi32>) {
      %in0 = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @in}
      %next = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @in}
      %out0 = aiex.dma_configure_task(%t, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%in0)
      aiex.dma_start_task(%out0)
      aiex.dma_start_task(%next)
      aiex.dma_await_task(%out0)
      // expected-note@+1 {{released here}}
      aiex.dma_free_task(%next)
      %out1 = aiex.dma_configure_task(%t, S2MM, 0) {
        // expected-warning@+2 {{reuses buffer descriptor ID}}
        // expected-note@+1 {{not proven finished: through @out, the bytes its awaited transfers moved, 256, fall short of the bytes the input moved up to this transfer, 512}}
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

// The same with a parameter-sized prefetch, scaled by the proof's ratio: short
// whenever @n is above 0.
module {
  aiex.scratchpad_parameter @n : i32 {kind = 0 : i32, max_value = 4 : i32, min_value = 0 : i32, state_table_idx = 0 : ui8}
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 2, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%t, S2MM, 0)
    aie.runtime_sequence @s(%a: memref<4096xi32>, %c: memref<4096xi32>) {
      %in0 = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%a : memref<4096xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @in}
      %next = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%a : memref<4096xi32> offset = 64 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 64 : i32}
        aie.end
      } {alloc = @in}
      %out0 = aiex.dma_configure_task(%t, S2MM, 0) {
        aie.dma_bd(%c : memref<4096xi32> offset = 0 len = 32)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%in0)
      aiex.dma_start_task(%out0)
      aiex.dma_start_task(%next)
      aiex.dma_await_task(%out0)
      // expected-note@+1 {{released here}}
      aiex.dma_free_task(%next)
      %out1 = aiex.dma_configure_task(%t, S2MM, 0) {
        // expected-warning@+2 {{reuses buffer descriptor ID}}
        // expected-note@+1 {{not proven finished: through @out, the bytes its awaited transfers moved times 2, 256, fall short of the bytes the input moved up to this transfer times 1, 256 + 256*@n, for some parameter values}}
        aie.dma_bd(%c : memref<4096xi32> offset = 64 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 32 : i32}
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out1)
      aiex.dma_await_task(%out1)
      aiex.dma_free_task(%in0)
    }
  }
}

// -----

// Another allocation's transfer shares the input's channel, which the proof's
// counts do not describe.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @other(%t, MM2S, 0)
    aie.shim_dma_allocation @out(%t, S2MM, 0)
    aie.runtime_sequence @s(%a: memref<512xi32>, %c: memref<512xi32>) {
      %in0 = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in0)
      %x = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 128 len = 64)
        aie.end
      } {alloc = @other}
      // expected-note@+1 {{the other transfer}}
      aiex.dma_start_task(%x)
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
        // expected-note@+1 {{not proven finished: through @out, another transfer shares their channels, which the counts do not describe}}
        aie.dma_bd(%a : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in1)
      aiex.dma_free_task(%x)
    }
  }
}

// -----

// A length parameter that may be negative leaves the input's bytes unknown.
module {
  aiex.scratchpad_parameter @n : i32 {kind = 0 : i32, max_value = 4 : i32, state_table_idx = 0 : ui8}
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%t, S2MM, 0)
    aie.runtime_sequence @s(%a: memref<4096xi32>, %c: memref<4096xi32>) {
      %in0 = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%a : memref<4096xi32> offset = 0 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 64 : i32}
        aie.end
      } {alloc = @in}
      // expected-note@+1 {{the transfer}}
      aiex.dma_start_task(%in0)
      %out0 = aiex.dma_configure_task(%t, S2MM, 0) {
        aie.dma_bd(%c : memref<4096xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out0)
      aiex.dma_await_task(%out0)
      // expected-note@+1 {{released here}}
      aiex.dma_free_task(%in0)
      %in1 = aiex.dma_configure_task(%t, MM2S, 0) {
        // expected-warning@+2 {{reuses buffer descriptor ID}}
        // expected-note@+1 {{not proven finished: through @out, a transfer on them moves a byte count the compiler cannot follow (a BD chain, a loop, or a length parameter without min_value >= 0)}}
        aie.dma_bd(%a : memref<4096xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in1)
    }
  }
}

// -----

// The sequence ends having sent more through the input than the output
// proves taken, so a later dispatch's counts would not start clean.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%t, S2MM, 0)
    // expected-note@+1 {{the stretch ends here}}
    aie.runtime_sequence @s(%a: memref<512xi32>, %c: memref<512xi32>) {
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
        // expected-warning@+2 {{reuses buffer descriptor ID}}
        // expected-note@+1 {{not proven finished: through @out, a stretch of @s can end having moved more through the input than the output proves taken, so the next dispatch's counts do not start clean}}
        aie.dma_bd(%a : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in1)
    }
  }
}

// -----

// The input is pushed after the output's only await so far, which therefore
// proves nothing about it.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%t, S2MM, 0)
    aie.runtime_sequence @s(%a: memref<512xi32>, %c: memref<512xi32>) {
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
      aiex.dma_free_task(%in0)
      %in1 = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in1)
      // expected-note@+1 {{released here}}
      aiex.dma_free_task(%in1)
      %out1 = aiex.dma_configure_task(%t, S2MM, 0) {
        // expected-warning@+2 {{reuses buffer descriptor ID}}
        // expected-note@+1 {{not proven finished: through @out, no await on it has retired a transfer since this one was pushed}}
        aie.dma_bd(%c : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out1)
      aiex.dma_await_task(%out1)
    }
  }
}

// -----

// Two outputs each prove the input consumed one for one, but the input moves
// @a + @b blocks while each output moves only one of them: neither alone
// covers it, by how much depending on the parameters.
module {
  aiex.scratchpad_parameter @a : i32 {kind = 0 : i32, max_value = 4 : i32, min_value = 0 : i32, state_table_idx = 0 : ui8}
  aiex.scratchpad_parameter @b : i32 {kind = 0 : i32, max_value = 4 : i32, min_value = 0 : i32, state_table_idx = 1 : ui8}
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @o1, input_bytes = 1, output_bytes = 1>, #aie.consumed_by<output = @o2, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @o1(%t, S2MM, 0)
    aie.shim_dma_allocation @o2(%t, S2MM, 1)
    // expected-note@+1 {{the stretch ends here}}
    aie.runtime_sequence @s(%x: memref<4096xi32>, %y: memref<4096xi32>, %z: memref<4096xi32>) {
      %in0 = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%x : memref<4096xi32> offset = 0 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 16 : i32}
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in0)
      %in1 = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%x : memref<4096xi32> offset = 1024 len = 0) {length_state_table_idx = 1 : ui8, length_unit = 16 : i32}
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in1)
      %o1 = aiex.dma_configure_task(%t, S2MM, 0) {
        aie.dma_bd(%y : memref<4096xi32> offset = 0 len = 0) {length_state_table_idx = 0 : ui8, length_unit = 16 : i32}
        aie.end
      } {alloc = @o1, issue_token = true}
      aiex.dma_start_task(%o1)
      %o2 = aiex.dma_configure_task(%t, S2MM, 1) {
        aie.dma_bd(%z : memref<4096xi32> offset = 0 len = 0) {length_state_table_idx = 1 : ui8, length_unit = 16 : i32}
        aie.end
      } {alloc = @o2, issue_token = true}
      aiex.dma_start_task(%o2)
      aiex.dma_await_task(%o1)
      aiex.dma_await_task(%o2)
      // expected-note@+1 {{released here}}
      aiex.dma_free_task(%in0)
      aiex.dma_free_task(%in1)
      %in2 = aiex.dma_configure_task(%t, MM2S, 0) {
        // expected-warning@+4 {{reuses buffer descriptor ID}}
        // expected-note@+3 {{not proven finished: through @o1, a stretch of @s can end having moved more through the input than the output proves taken, so the next dispatch's counts do not start clean (input bytes, 64 + 64*@a + 64*@b, against output bytes, 64*@a)}}
        // expected-note@+2 {{not proven finished: through @o2, a stretch of @s can end having moved more through the input than the output proves taken, so the next dispatch's counts do not start clean (input bytes, 64 + 64*@a + 64*@b, against output bytes, 64*@b)}}
        // expected-note@+1 {{not proven finished: no one output's proof covers the input: it moves more than each proves alone, by how much depending on the runtime parameters @b, @a}}
        aie.dma_bd(%x : memref<4096xi32> offset = 2048 len = 16)
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in2)
    }
  }
}

// -----

// A copy that reprograms a freed task's BD is a reuse like a task's.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%t, MM2S, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 1)
    aie.runtime_sequence @s(%a: memref<512xi32>, %pkts: memref<?xi32>) {
      %in0 = aiex.dma_configure_task(%t, MM2S, 1) {
        aie.dma_bd(%a : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @in, issue_token = true}
      aiex.dma_start_task(%in0)
      // expected-note@+1 {{released here}}
      aiex.dma_free_task(%in0)
      // expected-warning@+2 {{reuses buffer descriptor ID 0 on tile (0,0) after it was released by an aiex.dma_free_task that had no completion guarantee}}
      // expected-note@+1 {{not proven finished: aie-objectfifo-verify found no output whose progress proves @in consumed}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      aiex.npu.sync(%c0, %c0, %c1, %c1, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
  }
}
