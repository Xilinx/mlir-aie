//===- npu_to_cert_configure.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-npu-to-cert %s | FileCheck %s

// Test that aiex.configure operations are converted to cert.section + cert.load_pdi

module {
  // The @configure runtime sequence is lowered to a bare cert.job (no enclosing
  // page yet) carrying the {cert.configure} unit attribute.
  // CHECK: aie.device(npu2) {
  // CHECK: aiex.cert.job({{[0-9]+}})
  // CHECK: aiex.cert.load_pdi(1, @config_device)
  // CHECK: aiex.cert.write32(6750208, 1)
  // CHECK: aiex.cert.write32(6750212, 2)
  // CHECK: {cert.configure}
  // CHECK: aiex.cert.section @config_device
  // CHECK: aiex.cert.page
  // CHECK: aiex.cert.job({{[0-9]+}})
  // CHECK-NOT: aie.device(npu2) @config_device
  aie.device(npu2) @main {
    aie.runtime_sequence @configure(%arg0: memref<16xi32>) {
      aiex.configure @config_device {
        aiex.run @setup(%arg0) : (memref<16xi32>)
      }
    }
  }

  aie.device(npu2) @config_device {
    aie.runtime_sequence @setup(%buf: memref<16xi32>) {
      %wa0 = arith.constant 6750208 : i32
      %wv0 = arith.constant 1 : i32
      aiex.npu.write32(%wa0, %wv0) : i32, i32
      %wa1 = arith.constant 6750212 : i32
      %wv1 = arith.constant 2 : i32
      aiex.npu.write32(%wa1, %wv1) : i32, i32
    }
  }
}
