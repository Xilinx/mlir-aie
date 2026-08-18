//===- rel_acq_sync.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate -aie-cert-to-asm %s | FileCheck %s

// CHECK: START_JOB 1
// CHECK:   REL_ACQ_SYNC           0x00016004, 0x00016208
// CHECK:   WRITE_32               0x007ae050, 0x00000002
// CHECK: END_JOB

module {
  aie.device(npu2) {
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.rel_acq_sync(0x00016004, 0x00016208)
        aiex.cert.write32(0x7AE050, 0x2)
      }
    }
  }
}
