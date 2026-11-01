//===- rel_acq_sync.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Addresses are synthetic; this checks operand order and formatting.
// RUN: aie-translate -aie-cert-to-asm %s | FileCheck %s

// CHECK: START_JOB 1
// CHECK:   REL_ACQ_SYNC           0x00001000, 0x00001004
// CHECK:   WRITE_32               0x00002000, 0x00000002
// CHECK: END_JOB

module {
  aie.device(npu2) {
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.rel_acq_sync(0x00001000, 0x00001004)
        aiex.cert.write32(0x2000, 0x2)
      }
    }
  }
}
