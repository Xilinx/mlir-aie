//===- cert_register_ops.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Round-trip the register reads and the register-addressed write.
// An unflagged write32_d field keeps a full 32-bit immediate.

// RUN: aie-opt %s | FileCheck %s
// RUN: aie-opt %s | aie-opt | FileCheck %s

// CHECK-LABEL: aie.device(xcve3858)
// CHECK: aiex.cert.read32(1, 4096)
// CHECK-NEXT: aiex.cert.read32_d(1, 2)
// CHECK-NEXT: aiex.cert.write32_d(4096, 5)
// CHECK-NEXT: aiex.cert.write32_d(1, 5) {address_is_reg}
// CHECK-NEXT: aiex.cert.write32_d(4096, 2) {value_is_reg}
// CHECK-NEXT: aiex.cert.write32_d(1, 2) {address_is_reg, value_is_reg}
// CHECK-NEXT: aiex.cert.write32_d(4294967295, 4294967295)
aie.device(xcve3858) {
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.read32(1, 0x1000)
      aiex.cert.read32_d(1, 2)
      aiex.cert.write32_d(0x1000, 5)
      aiex.cert.write32_d(1, 5) {address_is_reg}
      aiex.cert.write32_d(0x1000, 2) {value_is_reg}
      aiex.cert.write32_d(1, 2) {address_is_reg, value_is_reg}
      aiex.cert.write32_d(0xFFFFFFFF, 0xFFFFFFFF)
    }
  }
}
