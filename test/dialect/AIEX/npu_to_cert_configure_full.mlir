//===- npu_to_cert_configure_full.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-npu-to-cert %s | FileCheck %s

// Test complete configure/run flow with multiple operations

module {
  // The @main_seq runtime sequence is lowered to a bare cert.job (no enclosing
  // page yet); the callee @config_device is absorbed into a cert.section.
  // CHECK: aie.device(npu2) {
  // CHECK: aiex.cert.job({{[0-9]+}})
  // CHECK: aiex.cert.write32(4096, 1)
  // CHECK: aiex.cert.load_pdi(1, @config_device)
  // CHECK: aiex.cert.write32(6750208, 42)
  // CHECK: aiex.cert.write32(8192, 99)
  // CHECK: aiex.cert.section @config_device
  // CHECK: aiex.cert.page
  // CHECK: aiex.cert.job({{[0-9]+}})
  // CHECK-NOT: aie.device(npu2) @config_device
  aie.device(npu2) @main {
    aie.runtime_sequence @main_seq(%arg0: memref<1024xi32>) {
      %wa0 = arith.constant 4096 : i32
      %wv0 = arith.constant 1 : i32
      aiex.npu.write32(%wa0, %wv0) {column = 0 : i32, row = 0 : i32} : i32, i32
      aiex.configure @config_device {
        aiex.run @setup_sequence(%arg0) : (memref<1024xi32>)
      }
      %wa1 = arith.constant 8192 : i32
      %wv1 = arith.constant 99 : i32
      aiex.npu.write32(%wa1, %wv1) {column = 0 : i32, row = 0 : i32} : i32, i32
    }
  }

  aie.device(npu2) @config_device {
    aie.runtime_sequence @setup_sequence(%buf: memref<1024xi32>) {
      %wa2 = arith.constant 6750208 : i32
      %wv2 = arith.constant 42 : i32
      aiex.npu.write32(%wa2, %wv2) : i32, i32
    }
  }
}
