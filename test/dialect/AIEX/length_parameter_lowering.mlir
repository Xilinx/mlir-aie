//===- length_parameter_lowering.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A length_parameter lowers to an update_from_scratchpad<mul> on word 0 of the
// BD, after the BD write has set the static length. Word 0 counts 32-bit
// words and the firmware multiplies the core-encoded value (n << 2) by
// func_arg, so func_arg is the unit in 16-byte blocks.

// RUN: aie-opt --split-input-file --aie-lower-scratchpad-parameters --aie-dma-tasks-to-npu --aie-dma-to-npu %s | FileCheck %s

// A contiguous task: 64 bf16 static (32 words), plus n units of 64 bf16
// (128 B, func_arg 8). BD 3 of tile (0, 0) is at 0x1D060 = 118880.

// CHECK-LABEL: module
// CHECK: aiex.scratchpad_parameter @n : i32 {kind = 0 : i32, state_table_idx = 0 : ui8}
// CHECK: dense<[32,
// CHECK: aie.runtime_sequence @contiguous
// CHECK: aiex.npu.create_scratchpad {size = 4 : ui32}
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 118880 : ui32}
// CHECK: aiex.npu.address_patch
// CHECK: aiex.npu.update_from_scratchpad<mul> {address = 118880 : ui32, func_arg = 8 : ui32, state_table_idx = 0 : ui8}
// CHECK: aiex.npu.write32
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
// one more step of the third dimension (32 i32, func_arg 8). The offset is an
// addr parameter scaled by the element size into word 1; the length is a core
// parameter into word 0. BD 5 of tile (1, 0) is at 0x201D0A0 = 33673376.

// CHECK-LABEL: module
// CHECK: aiex.scratchpad_parameter @off : i32 {kind = 1 : i32, state_table_idx = 0 : ui8}
// CHECK: aiex.scratchpad_parameter @rows : i32 {kind = 0 : i32, state_table_idx = 1 : ui8}
// CHECK: dense<[64,
// CHECK: aie.runtime_sequence @strided
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 33673376 : ui32}
// CHECK: aiex.npu.update_from_scratchpad<mul> {address = 33673380 : ui32, func_arg = 4 : ui32, state_table_idx = 0 : ui8}
// CHECK: aiex.npu.update_from_scratchpad<mul> {address = 33673376 : ui32, func_arg = 8 : ui32, state_table_idx = 1 : ui8}
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
// CHECK: aiex.scratchpad_parameter @n : i32 {kind = 0 : i32, state_table_idx = 0 : ui8}
// CHECK: dense<[64,
// CHECK: aie.runtime_sequence @memcpy
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 118816 : ui32}
// CHECK: aiex.npu.address_patch
// CHECK: aiex.npu.update_from_scratchpad<mul> {address = 118816 : ui32, func_arg = 8 : ui32, state_table_idx = 0 : ui8}
// CHECK: aiex.npu.write32
aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  aie.runtime_sequence @memcpy(%arg0 : memref<4096xi32>) {
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 2, 4, 8][0, 64, 16, 1]) {id = 1 : i64, metadata = @dma, length_parameter = @n, length_unit = 32 : i64} : memref<4096xi32>
  }
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @dma(%tile, MM2S, 0)
}
