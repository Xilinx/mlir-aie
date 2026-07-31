//===- attach_group0_inline.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// attach_to_group(0) names uC 0, which is exactly the default (unspecified)
// group. It must be a placement no-op, not a page boundary: its content joins
// the device-level group-0 stream and merges (implicit page) with an adjacent
// device-level job. Previously it was isolated on its own page.

// RUN: aie-opt -cert-legalize-pages %s | FileCheck %s --implicit-check-not=attach_to_group

// The two jobs end up in ONE implicit page (no attach_to_group survives, no page
// boundary between them), in textual order.
// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(0)
// CHECK: aiex.cert.write32(4096, 1)
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.job(1)
// CHECK: aiex.cert.write32(8192, 2)

module {
  aie.device(npu2) {
    aiex.cert.attach_to_group(0) {
      aiex.cert.job(0) {
        aiex.cert.write32(0x1000, 1)
      }
    }
    aiex.cert.job(1) {
      aiex.cert.write32(0x2000, 2)
    }
  }
}
