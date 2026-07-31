//===- cert_page_placement.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Round-trip the optional `placement` attribute on cert.page.

// RUN: aie-opt %s | aie-opt | FileCheck %s

module {
  aie.device(npu2) {
    // A placed page carries its resolved group/uC id.
    // CHECK: aiex.cert.page {
    // CHECK:   aiex.cert.job(1)
    // CHECK: } {placement = 2 : i32}
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.write32(0x2000, 43)
      }
    } {placement = 2 : i32}

    // An unplaced page has no placement attribute (defaults to group 0).
    // CHECK: aiex.cert.page {
    // CHECK:   aiex.cert.job(0)
    // CHECK-NOT: placement
    aiex.cert.page {
      aiex.cert.job(0) {
        aiex.cert.write32(0x1000, 42)
      }
    }
  }
}
