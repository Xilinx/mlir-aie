//===- cert_apply_offset_57.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// num_entries and offset are unsigned 16-bit fields, so the largest value
// prints as 65535 and parses back to itself.

// RUN: aie-opt %s | FileCheck %s
// RUN: aie-opt %s | aie-opt | FileCheck %s

module {
  aie.device(npu2) {
    memref.global "private" constant @bd : memref<9xi32> = dense<0>
    aiex.cert.job(0) {
      // CHECK: aiex.cert.apply_offset_57(@bd, 1, 65535)
      aiex.cert.apply_offset_57(@bd, 1, 0xffff)
      // CHECK: aiex.cert.apply_offset_57(@bd, 65535, 0)
      aiex.cert.apply_offset_57(@bd, 0xffff, 0)
      // CHECK: aiex.cert.apply_offset_57(@bd, 1, 2)
      aiex.cert.apply_offset_57(@bd, 1, 2)
    }
  }
}
