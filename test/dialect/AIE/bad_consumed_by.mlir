//===- bad_consumed_by.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --split-input-file %s

// A zero share would credit every input push by any await on the output.
aie.device(npu2) {
  %t = aie.tile(0, 0)
  // expected-error@+1 {{consumed_by needs positive byte counts, got 1 and 0}}
  aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 0>]}
  aie.shim_dma_allocation @out(%t, S2MM, 0)
}

// -----

aie.device(npu2) {
  %t = aie.tile(0, 0)
  // expected-error@+1 {{consumed_by needs positive byte counts, got -3 and 1}}
  aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = -3, output_bytes = 1>]}
  aie.shim_dma_allocation @out(%t, S2MM, 0)
}

// -----

aie.device(npu2) {
  %t = aie.tile(0, 0)
  // expected-error@+1 {{consumed_by output @in names no shim output (S2MM aie.shim_dma_allocation)}}
  aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @in, input_bytes = 1, output_bytes = 1>]}
}

// -----

aie.device(npu2) {
  %t = aie.tile(0, 0)
  %c = aie.tile(0, 2)
  %buf = aie.buffer(%c) {sym_name = "notalloc"} : memref<16xi32>
  // expected-error@+1 {{consumed_by output @notalloc names no shim output}}
  aie.shim_dma_allocation @in(%t, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @notalloc, input_bytes = 1, output_bytes = 1>]}
}

// -----

aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.shim_dma_allocation @in(%t, MM2S, 0)
  // expected-error@+1 {{consumed_by belongs on a shim input (MM2S)}}
  aie.shim_dma_allocation @out(%t, S2MM, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
}
