// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Emit the register reads and register-addressed writes. The
// WRITE_32_D flags byte sets bit 0 when the address is a register index and
// bit 1 when the value is.
// RUN: aie-translate --aie-cert-to-asm %s | FileCheck %s

// CHECK: START_JOB 1
// CHECK-NEXT: READ_32                $r1, 0x00001000
// CHECK-NEXT: READ_32_D              $r1, $r2
// CHECK-NEXT: WRITE_32_D             0, 0x00001000, 0x00000005
// CHECK-NEXT: WRITE_32_D             1, 0x00000001, 0x00000005
// CHECK-NEXT: WRITE_32_D             2, 0x00001000, 0x00000002
// CHECK-NEXT: WRITE_32_D             3, 0x00000001, 0x00000002
// CHECK-NEXT: END_JOB

module {
  aie.device(xcve3858) {
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.read32(1, 0x1000)
        aiex.cert.read32_d(1, 2)
        aiex.cert.write32_d(0x1000, 5)
        aiex.cert.write32_d(1, 5) {address_is_reg}
        aiex.cert.write32_d(0x1000, 2) {value_is_reg}
        aiex.cert.write32_d(1, 2) {address_is_reg, value_is_reg}
      }
    }
  }
}
