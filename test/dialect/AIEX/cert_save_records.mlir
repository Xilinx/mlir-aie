//===- cert_save_records.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Round-trip the debug record ops, including full 32-bit identifiers.

// RUN: aie-opt %s | FileCheck %s
// RUN: aie-opt %s | aie-opt | FileCheck %s

// CHECK-LABEL: aie.device(xcve3858)
// CHECK: aiex.cert.save_timestamps(7)
// CHECK-NEXT: aiex.cert.save_register(4096, 9)
// CHECK-NEXT: aiex.cert.save_timestamps(4294967295)
// CHECK-NEXT: aiex.cert.save_register(4294967295, 4294967295)
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
