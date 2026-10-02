//===- length_parameter_lowering.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A length_parameter lowers to an update_from_scratchpad<mul> on word 0 of the
// BD, after the BD write has set the static length. Word 0 counts 32-bit
// words, so func_arg is the unit in words for an addr parameter (the state
// table holds n) and in 16-byte blocks for a core parameter (it holds n << 2).

// RUN: aie-opt --split-input-file --aie-lower-scratchpad-parameters --aie-assign-buffer-addresses --aie-dma-tasks-to-npu --aie-dma-to-npu %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-lower-scratchpad-parameters %s | FileCheck %s --check-prefix=MARK

// The scratchpad pass marks a core-kind length for the device-level lowering.

// A contiguous task: 64 bf16 static (32 words), plus n units of 64 bf16
// (128 B, func_arg 32). A parameter used only as a length is addr kind. BD 3
// of tile (0, 0) is at 0x1D060 = 118880.

// CHECK-LABEL: module
// CHECK: aiex.scratchpad_parameter @n : i32 {kind = 1 : i32, state_table_idx = 0 : ui8}
// CHECK: dense<[32,
// CHECK: aie.runtime_sequence @contiguous
// CHECK: aiex.npu.create_scratchpad {size = 4 : ui32}
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 118880 : ui32}
// CHECK: aiex.npu.address_patch
// CHECK: aiex.npu.update_from_scratchpad<mul> {address = 118880 : ui32, func_arg = 32 : ui32, state_table_idx = 0 : ui8}
// CHECK: aiex.npu.write32
// MARK-LABEL: aie.runtime_sequence @contiguous
// MARK: aie.dma_bd({{.*}}) {bd_id = 3 : i32, length_state_table_idx = 0 : ui8, length_unit = 64 : i32}
aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence @contiguous(%arg0 : memref<4096xbf16>) {
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<4096xbf16> offset = 0 len = 64) {bd_id = 3 : i32, length_parameter = @n, length_unit = 64 : i32}
      aie.end
    }
    aiex.dma_start_task(%task)
  }
}

// -----

// A strided task with both a runtime offset and a runtime length. Each unit is
// one more step of the third dimension (32 i32, func_arg 32). The offset is
// scaled by the element size into word 1; the length goes into word 0. BD 5
// of tile (1, 0) is at 0x201D0A0 = 33673376.

// CHECK-LABEL: module
// CHECK: aiex.scratchpad_parameter @off : i32 {kind = 1 : i32, state_table_idx = 0 : ui8}
// CHECK: aiex.scratchpad_parameter @rows : i32 {kind = 1 : i32, state_table_idx = 1 : ui8}
// CHECK: dense<[64,
// CHECK: aie.runtime_sequence @strided
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 33673376 : ui32}
// CHECK: aiex.npu.update_from_scratchpad<mul> {address = 33673380 : ui32, func_arg = 4 : ui32, state_table_idx = 0 : ui8}
// CHECK: aiex.npu.update_from_scratchpad<mul> {address = 33673376 : ui32, func_arg = 32 : ui32, state_table_idx = 1 : ui8}
aiex.scratchpad_parameter @off : i32
aiex.scratchpad_parameter @rows : i32
aie.device(npu2) {
  %t = aie.tile(1, 0)
  aie.runtime_sequence @strided(%arg0 : memref<4096xi32>) {
    %task = aiex.dma_configure_task(%t, S2MM, 1) {
      aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 64 sizes = [2, 4, 8] strides = [64, 16, 1]) {bd_id = 5 : i32, offset_parameter = @off, length_parameter = @rows, length_unit = 32 : i32}
      aie.end
    }
    aiex.dma_start_task(%task)
  }
}

// -----

// The same strided pattern through npu.dma_memcpy_nd. BD 1 of tile (0, 0) is
// at 0x1D020 = 118816.

// CHECK-LABEL: module
// CHECK: aiex.scratchpad_parameter @n : i32 {kind = 1 : i32, state_table_idx = 0 : ui8}
// CHECK: dense<[64,
// CHECK: aie.runtime_sequence @memcpy
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 118816 : ui32}
// CHECK: aiex.npu.address_patch
// CHECK: aiex.npu.update_from_scratchpad<mul> {address = 118816 : ui32, func_arg = 32 : ui32, state_table_idx = 0 : ui8}
// CHECK: aiex.npu.write32
aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  aie.runtime_sequence @memcpy(%arg0 : memref<4096xi32>) {
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 2, 4, 8][0, 64, 16, 1]) {id = 1 : i64, metadata = @dma, length_parameter = @n, length_unit = 32 : i64} : memref<4096xi32>
  }
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @dma(%tile, MM2S, 0)
}

// -----

// A parameter a core also reads is core kind, so the length's func_arg counts
// 16-byte blocks: 64 bf16 is 128 B, func_arg 8.

// CHECK-LABEL: module
// CHECK: aiex.scratchpad_parameter @n : i32 {kind = 0 : i32, state_table_idx = 0 : ui8}
// CHECK: aie.runtime_sequence @core_kind
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 118880 : ui32}
// CHECK: aiex.npu.update_from_scratchpad<mul> {address = 118880 : ui32, func_arg = 8 : ui32, state_table_idx = 0 : ui8}
// MARK-LABEL: aie.runtime_sequence @core_kind
// MARK: aie.dma_bd({{.*}}) {bd_id = 3 : i32, length_core_encoded, length_state_table_idx = 0 : ui8, length_unit = 64 : i32}
aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  %t02 = aie.tile(0, 2)
  aie.core(%t02) {
    %v = aiex.read_scratchpad_parameter @n : i32
    aie.end
  }
  aie.runtime_sequence @core_kind(%arg0 : memref<4096xbf16>) {
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<4096xbf16> offset = 0 len = 64) {bd_id = 3 : i32, length_parameter = @n, length_unit = 64 : i32}
      aie.end
    }
    aiex.dma_start_task(%task)
  }
}

// -----

// One parameter as both the offset and the length of a transfer: n elements
// from the start, n more elements moved. Offset func_arg 4 (i32) into word 1,
// length func_arg 16 (16 i32 is 64 B, 16 words) into word 0.

// CHECK-LABEL: module
// CHECK: aiex.scratchpad_parameter @n : i32 {kind = 1 : i32, state_table_idx = 0 : ui8}
// CHECK: aie.runtime_sequence @offset_and_length
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 118880 : ui32}
// CHECK: aiex.npu.update_from_scratchpad<mul> {address = 118884 : ui32, func_arg = 4 : ui32, state_table_idx = 0 : ui8}
// CHECK: aiex.npu.update_from_scratchpad<mul> {address = 118880 : ui32, func_arg = 16 : ui32, state_table_idx = 0 : ui8}
aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence @offset_and_length(%arg0 : memref<4096xi32>) {
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 16) {bd_id = 3 : i32, offset_parameter = @n, length_parameter = @n, length_unit = 16 : i32}
      aie.end
    }
    aiex.dma_start_task(%task)
  }
}

