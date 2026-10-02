//===- bad_length_parameter_lowering.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// length_parameter cases that the verifier cannot see and the lowering
// rejects: the BD's tile and bd_id are only known once the task is lowered,
// and a parameter's kind is only known from all of its uses.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-lower-scratchpad-parameters --aie-dma-tasks-to-npu %s

aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 1)
  %buf = aie.buffer(%t) {address = 0 : i32} : memref<4096xi32>
  aie.runtime_sequence(%arg0 : memref<4096xi32>) {
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      // expected-error @+1 {{'aie.dma_bd' op length_parameter is only supported on shim NOC tiles, got tile (0, 1)}}
      aie.dma_bd(%buf : memref<4096xi32> offset = 0 len = 64) {bd_id = 0 : i32, length_parameter = @n, length_unit = 4 : i32}
      aie.end
    }
    aiex.dma_start_task(%task)
  }
}

// -----

aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence(%arg0 : memref<4096xi32>) {
    %bd = aiex.dma_bd_pool_pop(0, 0) partition [0, 16) : i32
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      // expected-error @+1 {{'aie.dma_bd' op length_parameter requires a constant bd_id}}
      aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 64) bd_id_val %bd : i32 {length_parameter = @n, length_unit = 4 : i32}
      aie.end
    }
    aiex.dma_start_task(%task)
  }
}

// -----

// A length takes either kind, but a core read and an offset still need
// different kinds.

// expected-error @+1 {{parameter 'n' is used both as an aiex.read_scratchpad_parameter source (core) and as a DMA offset_parameter (addr); a parameter must have a single kind}}
aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  %t02 = aie.tile(0, 2)
  aie.core(%t02) {
    %v = aiex.read_scratchpad_parameter @n : i32
    aie.end
  }
  aie.runtime_sequence(%arg0 : memref<4096xi32>) {
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 64) {bd_id = 0 : i32, length_parameter = @n, length_unit = 4 : i32, offset_parameter = @n}
      aie.end
    }
    aiex.dma_start_task(%task)
  }
}

// -----

// Even n = 0 moves the static 64 elements from 16, past the end of the buffer.

aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence(%arg0 : memref<64xi32>) {
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      // expected-error @+1 {{'aie.dma_bd' op no value of parameter 'n' keeps every transfer using it within its buffer}}
      aie.dma_bd(%arg0 : memref<64xi32> offset = 16 len = 64) {bd_id = 0 : i32, length_parameter = @n, length_unit = 4 : i32}
      aie.end
    }
    aiex.dma_start_task(%task)
  }
}
