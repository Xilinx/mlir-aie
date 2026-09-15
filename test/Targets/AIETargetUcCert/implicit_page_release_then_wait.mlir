//===- implicit_page_release_then_wait.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A non-blocking "release" job placed textually before a runtime-sequence job
// containing wait_tcts ends up in the SAME implicit page, release first.
// Because they share a page (one .eop), the WAIT_TCTS has no forward-page
// dependency on the release — the release runs to completion (non-blocking)
// before WAIT_TCTS blocks. This is the safe form of the release-then-wait
// pattern a multi-uC design needs.

// RUN: aie-opt --aie-npu-to-cert --cert-legalize-pages %s | FileCheck %s
// RUN: aie-opt --aie-npu-to-cert --cert-legalize-pages %s | aie-translate --aie-cert-to-asm | FileCheck %s --check-prefix=ASM

// Both jobs are in one page: no new cert.page opens between them.
// CHECK: aiex.cert.page {
// CHECK: aiex.cert.job(2)
// CHECK: aiex.cert.write32(8052816, 1)
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.job(3)
// CHECK: aiex.cert.wait_tcts(0, 0, 1)

// In the emitted asm, the release WRITE_32 precedes WAIT_TCTS with no .eop
// (page break) between them; a single .eop follows both.
// ASM: WRITE_32{{.*}}0x007ae050
// ASM-NOT: .eop
// ASM: WAIT_TCTS
// ASM: .eop

module {
  aie.device(npu2) @main {
    // Non-blocking release (writes a lock register), textually first.
    aie.runtime_sequence @release() {
      %ra = arith.constant 0x7AE050 : i32
      %rv = arith.constant 1 : i32
      aiex.npu.write32(%ra, %rv) : i32, i32
    }
    // Runtime sequence that waits for task-completion tokens.
    aie.runtime_sequence @seq() {
      %col = arith.constant 0 : i32
      %row = arith.constant 0 : i32
      %dir = arith.constant 0 : i32
      %chan = arith.constant 0 : i32
      %colnum = arith.constant 1 : i32
      %rownum = arith.constant 1 : i32
      aiex.npu.sync(%col, %row, %dir, %chan, %colnum, %rownum) : i32, i32, i32, i32, i32, i32
    }
  }
}
