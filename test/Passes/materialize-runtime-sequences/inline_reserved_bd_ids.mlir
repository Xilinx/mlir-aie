//===- inline_reserved_bd_ids.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-materialize-runtime-sequences %s | FileCheck %s

// The caller's shim tile takes the callee's tile in place of a copy although
// only the callee's pins BD ids, and keeps the ids of both.

// CHECK-LABEL: aie.device(npu2) {
// CHECK: aie.tile(0, 0) {aiex.reserved_bd_ids = array<i32: 0, 1, 5>}
// CHECK-NOT: aie.tile(0, 0)
// CHECK: aiex.npu.load_pdi {device_ref = @a}
module {
  aie.device(npu2) @main {
    %tile00 = aie.tile(0, 0) {aiex.reserved_bd_ids = array<i32: 5>}
    aie.runtime_sequence @main_seq(%arg0: memref<16xi32>) {
      aiex.configure @a {
        aiex.run @seq(%arg0) : (memref<16xi32>)
      }
    }
  }

  aie.device(npu2) @a {
    %tile00 = aie.tile(0, 0) {aiex.reserved_bd_ids = array<i32: 0, 1>}
    aie.shim_dma_allocation @pin(%tile00, MM2S, 0)
    aie.runtime_sequence @seq(%a: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd (%a[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {id = 0 : i64, metadata = @pin} : memref<16xi32>
    }
  }
}
