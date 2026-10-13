// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -verify-diagnostics

// Verify that read_scratchpad_parameter must be inside aie.core.

aiex.scratchpad_parameter @foo : i32
aie.device(npu2) {
  aie.runtime_sequence() {
    // expected-error @+1 {{'aiex.read_scratchpad_parameter' op must be inside an aie.core}}
    %x = aiex.read_scratchpad_parameter @foo : i32
    aie.end
  }
}

// -----

// Verify that read_scratchpad_parameter rejects unknown parameter references.

aie.device(npu2) {
  %t = aie.tile(0, 2)
  aie.core(%t) {
    // expected-error @+1 {{'aiex.read_scratchpad_parameter' op references unknown parameter 'nonexistent'}}
    %x = aiex.read_scratchpad_parameter @nonexistent : i32
    aie.end
  }
}

// -----

// Verify that read_scratchpad_parameter rejects f32 result type.

aiex.scratchpad_parameter @foo : f32
aie.device(npu2) {
  %t = aie.tile(0, 2)
  aie.core(%t) {
    // expected-error @+1 {{'aiex.read_scratchpad_parameter' op f32 parameters are not supported}}
    %x = aiex.read_scratchpad_parameter @foo : f32
    aie.end
  }
}

// -----

// Verify that DMA BD offset parameters reject sub-byte element types.

aiex.scratchpad_parameter @offset : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence(%arg0 : memref<64xi4>) {
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      // expected-error @+1 {{'aie.dma_bd' op offset_parameter requires a whole-byte element type}}
      aie.dma_bd(%arg0 : memref<64xi4> offset = 0 len = 8) {offset_parameter = @offset}
      aie.end
    }
  }
}

// -----

// Verify that NPU DMA offset parameters reject sub-byte element types.

aiex.scratchpad_parameter @offset : i32
aie.device(npu2) {
  aie.runtime_sequence(%arg0 : memref<64xi1>) {
    // expected-error @+1 {{'aiex.npu.dma_memcpy_nd' op offset_parameter requires a whole-byte element type}}
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 32][0, 0, 0, 1]) {id = 0 : i64, metadata = @dma, offset_parameter = @offset} : memref<64xi1>
  }
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @dma(%tile, MM2S, 0)
}
