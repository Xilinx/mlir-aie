//===- cert_page_placement_lowering.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// cert-legalize-pages lowers cert.page {placement = N} to an enclosing
// cert.attach_to_group(N), dropping the redundant placement attr.

// RUN: aie-opt --cert-legalize-pages %s | FileCheck %s
// RUN: aie-opt --cert-legalize-pages %s | aie-translate --aie-cert-to-asm | FileCheck %s --check-prefix=ASM

// The placed page moves under attach_to_group(2); an unplaced page stays at
// device level (implicit group 0).
// CHECK: aiex.cert.page {
// CHECK:   aiex.cert.job(0)
// CHECK: aiex.cert.attach_to_group(2)
// CHECK:   aiex.cert.page
// CHECK:     aiex.cert.job(1)
// CHECK-NOT: placement

// ASM: .attach_to_group 0
// ASM: START_JOB 0
// ASM: .attach_to_group 2
// ASM: START_JOB 1

module {
  aie.device(npu2) {
    aiex.cert.page {
      aiex.cert.job(0) {
        aiex.cert.write32(0x1000, 42)
      }
    }
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.write32(0x2000, 43)
      }
    } {placement = 2 : i32}
  }
}
