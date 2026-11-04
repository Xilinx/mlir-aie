//===- offset_parameter_local_tile.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics \
// RUN:   --pass-pipeline='builtin.module(aie-lower-scratchpad-parameters,aie.device(aie-assign-runtime-sequence-bd-ids,aie-dma-tasks-to-npu))' %s \
// RUN:   | FileCheck %s

// A mem tile BD's address field counts 32-bit words from bit 0, so a runtime
// offset of n i32 elements adds n to it.
// CHECK-LABEL: aie.device(npu2) @mem
// CHECK: aiex.npu.update_from_scratchpad<mul> {address = [[ADDR:[0-9]+]] : ui32, func_arg = 1 : ui32
aiex.scratchpad_parameter @off : i32
aie.device(npu2) @mem {
  %t = aie.tile(0, 1)
  %buf = aie.buffer(%t) {address = 1024 : i32, sym_name = "b"} : memref<128xi32>
  aie.runtime_sequence(%a : memref<16xi32>) {
    %ta = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%buf : memref<128xi32> offset = 0 len = 32) {offset_parameter = @off, bd_id = 0 : i32}
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%ta)
    aiex.dma_await_task(%ta)
  }
}

// -----

// A core tile BD's word address sits at bit 14, so n i64 elements add
// 2n words there.
// CHECK-LABEL: aie.device(npu2) @core
// CHECK: aiex.npu.update_from_scratchpad<mul> {address = {{[0-9]+}} : ui32, func_arg = 32768 : ui32
aiex.scratchpad_parameter @off : i32
aie.device(npu2) @core {
  %t = aie.tile(0, 2)
  %buf = aie.buffer(%t) {address = 1024 : i32, sym_name = "b"} : memref<64xi64>
  aie.runtime_sequence(%a : memref<16xi32>) {
    %ta = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%buf : memref<64xi64> offset = 0 len = 16) {offset_parameter = @off, bd_id = 0 : i32}
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%ta)
    aiex.dma_await_task(%ta)
  }
}

// -----

// Half a word of offset cannot be added to a word address.
aiex.scratchpad_parameter @off : i32
aie.device(npu2) @narrow {
  %t = aie.tile(0, 1)
  %buf = aie.buffer(%t) {address = 1024 : i32, sym_name = "b"} : memref<256xi16>
  aie.runtime_sequence(%a : memref<16xi32>) {
    %ta = aiex.dma_configure_task(%t, MM2S, 0) {
      // expected-error@+1 {{a runtime offset into 2-byte elements on a mem or core tile buffer descriptor, whose address counts 32-bit words}}
      aie.dma_bd(%buf : memref<256xi16> offset = 0 len = 64) {offset_parameter = @off, bd_id = 0 : i32}
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%ta)
    aiex.dma_await_task(%ta)
  }
}
