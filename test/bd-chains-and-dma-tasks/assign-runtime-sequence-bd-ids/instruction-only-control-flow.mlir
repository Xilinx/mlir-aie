// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s
// RUN: aiecc --get=npu_lowered.mlir --output-dir=%t.d --tmpdir=%t.d/work %s
// RUN: FileCheck %s < %t.d/npu_lowered.mlir

// No DMA tasks remain to allocate, but instruction emission still depends on
// runtime scalars. Neither path should reject or unroll this control flow.
// CHECK: aie.runtime_sequence
// CHECK-NOT: aiex.dma_bd_pool
// CHECK: scf.for
// CHECK: scf.if
// CHECK: aiex.npu.address_patch
// CHECK-NOT: aiex.dma_bd_pool
aie.device(npu1_1col) {
  aie.runtime_sequence @seq(%a: memref<8xi32>, %offset: i32, %n: index, %pred: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      scf.if %pred {
        aiex.npu.address_patch(%offset : i32) {addr = 119300 : ui32, arg_idx = 0 : i32}
      }
    }
  }
}
