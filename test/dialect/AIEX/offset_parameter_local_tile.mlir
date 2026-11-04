//===- offset_parameter_local_tile.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics \
// RUN:   --pass-pipeline='builtin.module(aie-lower-scratchpad-parameters,aie.device(aie-assign-runtime-sequence-bd-ids,aie-dma-tasks-to-npu))' %s \
// RUN:   | FileCheck %s

// npu.update_from_scratchpad adds StateTable * func_arg into a BD register
// and clears that register's bits [1:0].

// A mem tile BD's address counts words from bit 0 of word 1, so an i32
// offset adds 1 a element there, the buffer starts on 16 bytes, and the
// offset moves in 4 elements.
// CHECK-LABEL: aiex.scratchpad_parameter @off
// CHECK-SAME: offset_multiple = 4 : i32
// CHECK-LABEL: aie.device(npu2) @mem
// CHECK: aiex.npu.update_from_scratchpad<mul> {address = 1703940 : ui32, func_arg = 1 : ui32
// CHECK-NOT: aiex.npu.maskwrite32
// CHECK: aiex.npu.push_queue
aiex.scratchpad_parameter @off : i32
aie.device(npu2) @mem {
  %t = aie.tile(0, 1)
  %buf = aie.buffer(%t) {address = 1024 : i32, sym_name = "b"} : memref<128xi32>
  aie.runtime_sequence(%a : memref<16xi32>) {
    %ta = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%buf : memref<128xi32> offset = 0 len = 13) {offset_parameter = @off, bd_id = 0 : i32}
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%ta)
    aiex.dma_await_task(%ta)
  }
}

// -----

// Off a 16-byte boundary, the cleared bits would move the buffer.
aiex.scratchpad_parameter @off : i32
aie.device(npu2) @mem_unaligned {
  %t = aie.tile(0, 1)
  %buf = aie.buffer(%t) {address = 1028 : i32, sym_name = "b"} : memref<128xi32>
  aie.runtime_sequence(%a : memref<16xi32>) {
    %ta = aiex.dma_configure_task(%t, MM2S, 0) {
      // expected-error@+1 {{a runtime offset needs the buffer at a multiple of 16 bytes}}
      aie.dma_bd(%buf : memref<128xi32> offset = 0 len = 13) {offset_parameter = @off, bd_id = 0 : i32}
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%ta)
    aiex.dma_await_task(%ta)
  }
}

// -----

// A quarter word cannot be added to a word address.
aiex.scratchpad_parameter @off : i32
aie.device(npu2) @mem_i16 {
  %t = aie.tile(0, 1)
  %buf = aie.buffer(%t) {address = 1024 : i32, sym_name = "b"} : memref<256xi16>
  aie.runtime_sequence(%a : memref<16xi32>) {
    %ta = aiex.dma_configure_task(%t, MM2S, 0) {
      // expected-error@+1 {{a runtime offset into 2-byte elements cannot be added to this tile's buffer descriptor address, which counts 4-byte units from bit 0}}
      aie.dma_bd(%buf : memref<256xi16> offset = 0 len = 64) {offset_parameter = @off, bd_id = 0 : i32}
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%ta)
    aiex.dma_await_task(%ta)
  }
}

// -----

// A core tile BD's word address sits at bit 14 of word 0, above its length,
// so an i32 offset adds 1 << 14 an element and moves in whole elements; the
// update clears the length's low two bits (13 = 0b1101), which a maskwrite
// puts back.
// CHECK-LABEL: aie.device(npu2) @core_i32
// CHECK: aiex.npu.update_from_scratchpad<mul> {address = [[W0:[0-9]+]] : ui32, func_arg = 16384 : ui32
// CHECK-NEXT: %[[A:.+]] = arith.constant [[W0]] : i32
// CHECK-NEXT: %[[V:.+]] = arith.constant 1 : i32
// CHECK-NEXT: %[[M:.+]] = arith.constant 3 : i32
// CHECK-NEXT: aiex.npu.maskwrite32(%[[A]], %[[V]], %[[M]])
aiex.scratchpad_parameter @off : i32
aie.device(npu2) @core_i32 {
  %t = aie.tile(0, 2)
  %buf = aie.buffer(%t) {address = 1024 : i32, sym_name = "b"} : memref<128xi32>
  aie.runtime_sequence(%a : memref<16xi32>) {
    %ta = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%buf : memref<128xi32> offset = 0 len = 13) {offset_parameter = @off, bd_id = 0 : i32}
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%ta)
    aiex.dma_await_task(%ta)
  }
}

// -----

// i16 moves half a word an element: 1 << 13, in steps of 2 elements.
// CHECK-LABEL: aiex.scratchpad_parameter @off
// CHECK-SAME: offset_multiple = 2 : i32
// CHECK-LABEL: aie.device(npu2) @core_i16
// CHECK: aiex.npu.update_from_scratchpad<mul> {address = {{[0-9]+}} : ui32, func_arg = 8192 : ui32
// CHECK: aiex.npu.maskwrite32
aiex.scratchpad_parameter @off : i32
aie.device(npu2) @core_i16 {
  %t = aie.tile(0, 2)
  %buf = aie.buffer(%t) {address = 1024 : i32, sym_name = "b"} : memref<256xi16>
  aie.runtime_sequence(%a : memref<16xi32>) {
    %ta = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi16> offset = 0 len = 26) {offset_parameter = @off, bd_id = 0 : i32}
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%ta)
    aiex.dma_await_task(%ta)
  }
}

// -----

// i64 moves two words an element.
// CHECK-LABEL: aie.device(npu2) @core_i64
// CHECK: aiex.npu.update_from_scratchpad<mul> {address = {{[0-9]+}} : ui32, func_arg = 32768 : ui32
aiex.scratchpad_parameter @off : i32
aie.device(npu2) @core_i64 {
  %t = aie.tile(0, 2)
  %buf = aie.buffer(%t) {address = 1024 : i32, sym_name = "b"} : memref<64xi64>
  aie.runtime_sequence(%a : memref<16xi32>) {
    %ta = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%buf : memref<64xi64> offset = 0 len = 8) {offset_parameter = @off, bd_id = 0 : i32}
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%ta)
    aiex.dma_await_task(%ta)
  }
}
