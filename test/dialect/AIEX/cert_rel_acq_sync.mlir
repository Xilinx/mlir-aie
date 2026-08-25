//===- cert_rel_acq_sync.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Round-trip the release/acquire address pair unchanged.

// RUN: aie-opt %s | FileCheck %s
// RUN: aie-opt %s | aie-opt | FileCheck %s

// CHECK-LABEL: aie.device(npu2)
// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(1)
// CHECK: aiex.cert.rel_acq_sync(90116, 90632)

module {
  aie.device(npu2) {
    aiex.cert.page {
      aiex.cert.job(1) {
        // rel(+1), then acq_eq(2), on lock 8 of a shim NOC module.
        aiex.cert.rel_acq_sync(0x00016004, 0x00016208)
      }
    }
  }
}
