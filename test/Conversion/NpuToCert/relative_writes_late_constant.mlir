// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A tile-relative address that is constant only once folded is resolved by the
// write lowering, which still warns when its upper bits disagree with the
// tile coordinates.

// RUN: aie-opt --aie-npu-to-cert %s 2>&1 | FileCheck %s

// CHECK: warning: address upper bits (0x12300000) don't match row=2 col=1 computed bits (0x2200000)
// CHECK: warning: address upper bits (0x12300000) don't match row=2 col=1 computed bits (0x2200000)
// CHECK-NOT: warning:
// CHECK: aiex.cert.write32(35651840, 9)
// CHECK: aiex.cert.maskwrite32(35651840, 255, 9)
// CHECK-NOT: aiex.npu.
module {
  aie.device(xcve3858) @main {
    aie.runtime_sequence @seq() {
      %hi = arith.constant 0x12300000 : i32
      %lo = arith.constant 0x100 : i32
      %addr = arith.addi %hi, %lo : i32
      %v = arith.constant 9 : i32
      %mask = arith.constant 0xff : i32
      aiex.npu.write32(%addr, %v) {column = 1 : i32, row = 2 : i32} : i32, i32
      aiex.npu.maskwrite32(%addr, %v, %mask) {column = 1 : i32, row = 2 : i32} : i32, i32, i32
    }
  }
}
