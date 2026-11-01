//===- inline_and_load_pdi.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-materialize-runtime-sequences --convert-aie-to-transaction --aie-npu-to-cert %s | FileCheck %s

// Test that configure/run are converted to CERT with load_pdi and proper sections

module {
  // CHECK: aie.device(npu2) {
  // CHECK-NOT: aie.device(npu2) @config_a
  aie.device(npu2) @main {
    %tile00 = aie.tile(0, 0)

    // The main sequence is lowered to a bare cert.job (no enclosing page yet).
    // CHECK: aiex.cert.job({{[0-9]+}}) {
    // CHECK: aiex.cert.load_pdi(1, @config_a)
    // CHECK: aiex.cert.write32(33554532, 42)
    // CHECK: aiex.cert.write32(200, 99)
    aie.runtime_sequence @main_seq(%arg0: memref<16xi32>) {
      // CHECK: aiex.cert.section @config_a {
      // CHECK: aiex.cert.page {
      // CHECK: aiex.cert.job({{[0-9]+}}) {
      // CHECK-NEXT: }
      aiex.configure @config_a {
        aiex.run @seq_a(%arg0) : (memref<16xi32>)
      }
      %wa0 = arith.constant 200 : i32
      %wv0 = arith.constant 99 : i32
      aiex.npu.write32(%wa0, %wv0) {column = 0 : i32, row = 0 : i32} : i32, i32
    }
  }

  aie.device(npu2) @config_a {
    %tile10 = aie.tile(1, 0)

    aie.runtime_sequence @seq_a(%arg0: memref<16xi32>) {
      %wa1 = arith.constant 100 : i32
      %wv1 = arith.constant 42 : i32
      aiex.npu.write32(%wa1, %wv1) {column = 1 : i32, row = 0 : i32} : i32, i32
    }
  }
}
