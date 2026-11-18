//===- cert_apply_offset_57_invalid.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// num_entries and offset are unsigned 16-bit fields. Negative spellings are
// rejected, and a num_entries of 0 patches nothing.

// RUN: aie-opt %s --split-input-file --verify-diagnostics

module {
  aie.device(npu2) {
    memref.global "private" constant @bd : memref<9xi32> = dense<0>
    aiex.cert.job(0) {
      // expected-error@+1 {{negative integer literal not valid for unsigned integer type}}
      aiex.cert.apply_offset_57(@bd, -1, 2)
    }
  }
}

// -----

module {
  aie.device(npu2) {
    memref.global "private" constant @bd : memref<9xi32> = dense<0>
    aiex.cert.job(0) {
      // expected-error@+1 {{negative integer literal not valid for unsigned integer type}}
      aiex.cert.apply_offset_57(@bd, 1, -1)
    }
  }
}

// -----

module {
  aie.device(npu2) {
    memref.global "private" constant @bd : memref<9xi32> = dense<0>
    aiex.cert.job(0) {
      // expected-error@+1 {{'aiex.cert.apply_offset_57' op num_entries must be at least 1}}
      aiex.cert.apply_offset_57(@bd, 0, 2)
    }
  }
}
