//===- uc_stream_group0_unified.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// device-level (group 0) content and an explicit attach_to_group(0) are
// emitted as a single uC-0 stream, in IR order, under one ".attach_to_group 0"
// header with one EOF (no second group header).

// RUN: aie-translate --aie-cert-to-asm %s | FileCheck %s

// CHECK: .attach_to_group 0
// CHECK: START_JOB 0
// CHECK: START_JOB 1
// CHECK-NOT: .attach_to_group
// CHECK: EOF

module {
  aie.device(npu2) {
    aiex.cert.page {
      aiex.cert.job(0) {
        aiex.cert.write32(0x1000, 1)
      }
    }
    aiex.cert.attach_to_group(0) {
      aiex.cert.page {
        aiex.cert.job(1) {
          aiex.cert.write32(0x1000, 2)
        }
      }
    }
  }
}
