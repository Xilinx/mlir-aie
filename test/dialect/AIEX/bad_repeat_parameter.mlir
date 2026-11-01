//===- bad_repeat_parameter.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// repeat_parameter cases the verifier or the lowering rejects.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-lower-scratchpad-parameters --aie-split-long-repeats --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu --aie-dma-to-npu %s

aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence(%arg0 : memref<64xi32>) {
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
      aie.end
    } {repeat_count = 3 : i32}
    // expected-error @+1 {{takes a repeat_parameter or a repeat count, not both}}
    aiex.dma_start_task(%task) {repeat_count = 1 : i32, repeat_parameter = @n}
  }
}

// -----

// A repeat count is written shifted into the queue word, so its parameter
// cannot also be a length.
// expected-error @+1 {{parameter 'n' is an aiex.dma_start_task repeat_parameter (repeat) and also read by a core or used as a DMA offset or length}}
aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence(%arg0 : memref<64xi32>, %arg1 : memref<256xi32>) {
    %out = aiex.dma_configure_task(%t, S2MM, 0) {
      aie.dma_bd(%arg1 : memref<256xi32> offset = 0 len = 0) {length_parameter = @n, length_unit = 64 : i32}
      aie.end
    }
    %in = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
      aie.end
    } {repeat_count = 3 : i32}
    aiex.dma_start_task(%out)
    aiex.dma_start_task(%in) {repeat_parameter = @n}
  }
}

// -----

aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence(%arg0 : memref<64xi32>, %r : i32) {
    %task = aiex.dma_configure_task(%t, MM2S, 0) repeat %r : i32 {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
      aie.end
    }
    // expected-error @+1 {{has a repeat_parameter, which its task's repeat count bounds, but that is a runtime value}}
    aiex.dma_start_task(%task) {repeat_parameter = @n}
  }
}

// -----

// The update clears the queue word's low two bits, so a given head bd_id
// must be a multiple of 4.
aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence(%arg0 : memref<64xi32>) {
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64) {bd_id = 1 : i32}
      aie.end
    } {repeat_count = 3 : i32}
    // expected-error @+1 {{bd_id must be a multiple of 4, got 1}}
    aiex.dma_start_task(%task) {repeat_parameter = @n}
  }
}

// -----

aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 2)
  %buf = aie.buffer(%t) {address = 1024 : i32} : memref<64xi32>
  aie.runtime_sequence(%arg0 : memref<64xi32>) {
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64)
      aie.end
    } {repeat_count = 3 : i32}
    // expected-error @+1 {{repeat_parameter is only supported on shim NOC and mem tiles, got tile (0, 2)}}
    aiex.dma_start_task(%task) {repeat_parameter = @n}
  }
}

// -----

// The task's repeat count, the parameter's maximum, must fit one push.
aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence(%arg0 : memref<64xi32>) {
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
      aie.end
    } {repeat_count = 300 : i32}
    // expected-error @+1 {{takes its repeat count from a scratchpad parameter, which reaches one queue push, but its task's repeat count, the parameter's maximum, is 300, past the target's 255}}
    aiex.dma_start_task(%task) {repeat_parameter = @n}
  }
}

// -----

// A start of a task picked at run time cannot know which task bounds it.
aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence(%arg0 : memref<64xi32>, %c : i1) {
    %a = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
      aie.end
    } {repeat_count = 3 : i32}
    %b = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
      aie.end
    } {repeat_count = 7 : i32}
    %task = arith.select %c, %a, %b : index
    // expected-error @+1 {{has a repeat_parameter, so it must start the task where it is configured: the task's repeat count bounds the parameter}}
    aiex.dma_start_task(%task) {repeat_parameter = @n}
  }
}
