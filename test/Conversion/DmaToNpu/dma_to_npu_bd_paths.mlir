//===- dma_to_npu_bd_paths.mlir - Static vs dynamic BD lowering --*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Characterization of the two BD-lowering paths in aie-dma-to-npu's
// DmaToNpuPattern. Pins the semantically meaningful output so that refactoring
// the pattern (e.g. splitting the static/dynamic paths into helpers) cannot
// silently change behavior.
//
//   - Static path: all sizes/strides/offset are constants -> a single
//     blockwrite of the BD words, an address_patch for the buffer pointer, and
//     a queue push (lowered to write32 with the issue-token bit).
//   - Dynamic path: an SSA size forces per-word write32 overrides tagged with
//     bd_group, plus the address_patch.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file -aie-dma-to-npu %s | FileCheck %s

// Static BD: fully constant transfer.
// CHECK-LABEL: @static_bd
// The BD words are emitted as a single blockwrite (not per-word write32s).
// CHECK: %[[G:.+]] = memref.get_global
// CHECK: aiex.npu.blockwrite(%[[G]]) {address = 118816 : ui32}
// CHECK-NOT: bd_group
// The buffer pointer is patched from runtime arg 0.
// CHECK: %[[AP:.+]] = arith.constant 0 : i32
// CHECK: aiex.npu.address_patch(%[[AP]] : i32) {addr = 118820 : ui32, arg_idx = 0 : i32}
// The queue push is a write32 with the issue-token bit set (0x80000000).
// CHECK: arith.constant -2147483647 : i32
// CHECK: aiex.npu.write32
module {
  aie.device(npu1) {
    aie.runtime_sequence @static_bd(%arg0: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 16, 16][0, 0, 64, 1]) {issue_token = true, metadata = @toMem, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @toMem}
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @toMem (%tile_0_0, S2MM, 0)
  }
}

// -----
// Dynamic BD: an SSA outer size forces per-word write32 overrides instead of a
// single static blockwrite.
// CHECK-LABEL: @dynamic_bd
// Dynamic BD words are written via write32 ops tagged with the BD's bd_group.
// CHECK: aiex.npu.write32({{.*}}) {bd_group = 118816 : ui32}
// The buffer pointer is still patched from runtime arg 0.
// CHECK: aiex.npu.address_patch({{.*}}) {addr = 118820 : ui32, arg_idx = 0 : i32}
module {
  aie.device(npu2) {
    aie.runtime_sequence @dynamic_bd(%arg0: memref<16384xbf16>, %arg1: i32) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c32 = arith.constant 32 : i64
      %dim0 = arith.extui %arg1 : i32 to i64
      aiex.npu.dma_memcpy_nd (%arg0[%c0, %c0, %c0, %c0][%dim0, %c1, %c32, %c32][%c0, %c32, %c32, %c1]) {metadata = @toMem, id = 1 : i64} : memref<16384xbf16>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @toMem (%tile_0_0, S2MM, 0)
  }
}
