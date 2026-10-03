// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -verify-diagnostics \
// RUN:   --aie-lower-scratchpad-parameters --aie-dma-tasks-to-npu

// A parameter is an offset or a length, not both.

module {
  // expected-error @+1 {{parameter 'p' is used as more than one of an aiex.read_scratchpad_parameter source (core), a DMA offset_parameter (addr) and a DMA length_parameter (len); a parameter must have a single kind}}
  aiex.scratchpad_parameter @p : i32
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.runtime_sequence(%arg0 : memref<64xi32>) {
      %task = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 16) {offset_parameter = @p, length_parameter = @p}
        aie.end
      }
    }
  }
}

// -----

// A length parameter is an i32.

module {
  // expected-note @+1 {{Parameter declared here.}}
  aiex.scratchpad_parameter @p : i16
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.runtime_sequence(%arg0 : memref<64xi32>) {
      %task = aiex.dma_configure_task(%t, MM2S, 0) {
        // expected-error @+1 {{'aie.dma_bd' op length_parameter 'p' must have type i32, got 'i16'.}}
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 16) {length_parameter = @p}
        aie.end
      }
    }
  }
}

// -----

// Only a shim BD has its length alone in its first word.

module {
  aiex.scratchpad_parameter @p : i32
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %buf = aie.buffer(%t) : memref<64xi32>
    aie.runtime_sequence(%arg0 : memref<64xi32>) {
      %task = aiex.dma_configure_task(%t, MM2S, 0) {
        // expected-error @+1 {{'aie.dma_bd' op length_parameter is only supported on shim tile BDs.}}
        aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 16) {bd_id = 0 : i32, length_parameter = @p}
        aie.end
      }
      aiex.dma_start_task(%task)
    }
  }
}
