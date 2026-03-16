//===- cert_groups.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-cert-to-asm %s | FileCheck %s

// CHECK: .attach_to_group 0
// CHECK: START_JOB 0
// CHECK: .attach_to_group 1
// CHECK: START_JOB 1
// CHECK: EOF

aie.device(npu2) {

  aiex.cert.attach_to_group(0) {
    aiex.cert.job(0) {
      aiex.cert.write32(0x1000, 42)
    }
  }
  aiex.cert.attach_to_group(1) {
    aiex.cert.job(1) {
      aiex.cert.write32(0x2000, 43)
    }
  }
}
