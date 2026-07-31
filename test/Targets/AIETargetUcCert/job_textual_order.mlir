//===- job_textual_order.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-cert-to-asm %s | FileCheck %s

// Jobs must be emitted in textual (IR) order, not sorted by job_id. Here the
// job with the *larger* id appears first in the IR, so it must be emitted
// first. (Under the old job_id-sorting emitter this would fail: START_JOB 2
// would be emitted before START_JOB 5.)

// CHECK: START_JOB 5
// CHECK:   WRITE_32               0x00001000, 0x0000002a
// CHECK: START_JOB 2
// CHECK:   WRITE_32               0x00002000, 0x0000002b

aie.device(npu2) {
  aiex.cert.page {
    aiex.cert.job(5) {
      aiex.cert.write32(0x1000, 42)
    }
    aiex.cert.job(2) {
      aiex.cert.write32(0x2000, 43)
    }
  }
}
