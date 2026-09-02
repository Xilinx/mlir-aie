//===- npu_to_cert_pages.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-npu-to-cert %s | FileCheck %s
// RUN: aie-opt --aie-npu-to-cert %s | aie-opt --cert-legalize-pages | FileCheck %s --check-prefix=PAGES

// Test that runtime sequences are lowered to a bare cert.job at device level by
// aie-npu-to-cert (no enclosing page yet), and that cert-legalize-pages then
// wraps the contiguous job(s) into a cert.page.

module {
  aie.device(npu2) @main {
    // After aie-npu-to-cert: a bare job, no enclosing page.
    // CHECK-NOT: aiex.cert.page
    // CHECK: aiex.cert.job({{[0-9]+}})
    // CHECK: aiex.cert.write32(4096, 1)

    // After cert-legalize-pages: the job is wrapped in a page.
    // PAGES: aiex.cert.page
    // PAGES-NEXT: aiex.cert.job({{[0-9]+}})
    // PAGES: aiex.cert.write32(4096, 1)
    aie.runtime_sequence @main_seq(%arg0: memref<1024xi32>) {
      %wa0 = arith.constant 4096 : i32
      %wv0 = arith.constant 1 : i32
      aiex.npu.write32(%wa0, %wv0) {column = 0 : i32, row = 0 : i32} : i32, i32
    }
  }
}
