//===- apply_offset_57_fields.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The assembly carries num_entries and offset as unsigned decimal, including
// the largest 16-bit value.

// RUN: aie-translate --aie-cert-to-asm %s | FileCheck %s

// CHECK: APPLY_OFFSET_57        @bd, 1, 65535
// CHECK: APPLY_OFFSET_57        @bd, 65535, 0
// CHECK: APPLY_OFFSET_57        @bd, 3, 2

module {
  aie.device(npu2) {
    memref.global "private" constant @bd : memref<9xi32> = dense<0>
    aiex.cert.job(0) {
      aiex.cert.apply_offset_57(@bd, 1, 0xffff)
      aiex.cert.apply_offset_57(@bd, 0xffff, 0)
      aiex.cert.apply_offset_57(@bd, 3, 2)
    }
  }
}
