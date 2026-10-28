//===- cert_configure_first.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// the implicit @configure sequence must become the FIRST page on uC0,
// even when it appears textually after a user runtime sequence.

// RUN: aie-opt --aie-npu-to-cert --cert-legalize-pages %s | FileCheck %s

// Config (job 1, writes 0x1000=4096) is emitted before the user job (job 2,
// writes 0x2000=8192), despite @user_seq preceding @configure in the input.
// CHECK: aiex.cert.job(1)
// CHECK: aiex.cert.write32(4096, 1)
// CHECK: aiex.cert.job(2)
// CHECK: aiex.cert.write32(8192, 99)

module {
  aie.device(npu2) @main {
    aie.runtime_sequence @user_seq() {
      %ua = arith.constant 0x2000 : i32
      %uv = arith.constant 99 : i32
      aiex.npu.write32(%ua, %uv) : i32, i32
    }
    aie.runtime_sequence @configure() {
      %ca = arith.constant 0x1000 : i32
      %cv = arith.constant 1 : i32
      aiex.npu.write32(%ca, %cv) : i32, i32
    }
  }
}
