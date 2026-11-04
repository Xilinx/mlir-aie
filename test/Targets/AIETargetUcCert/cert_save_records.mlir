// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Emit the debug record ops.
// RUN: aie-translate --aie-cert-to-asm %s | FileCheck %s

// CHECK: START_JOB 1
// CHECK-NEXT: SAVE_TIMESTAMPS        7
// CHECK-NEXT: SAVE_REGISTER          0x00001000, 9
// CHECK-NEXT: SAVE_TIMESTAMPS        4294967295
// CHECK-NEXT: SAVE_REGISTER          0xffffffff, 4294967295
// CHECK-NEXT: END_JOB

module {
  aie.device(xcve3858) {
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.save_timestamps(7)
        aiex.cert.save_register(0x1000, 9)
        aiex.cert.save_timestamps(0xFFFFFFFF)
        aiex.cert.save_register(0xFFFFFFFF, 0xFFFFFFFF)
      }
    }
  }
}
