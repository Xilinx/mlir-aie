//===- npu_maskpoll.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// aiex.npu.maskpoll encodes as a 7-word MASKPOLL transaction.
//
// Same shape as MASKWRITE, deliberately: XAie_MaskPoll32Hdr and
// XAie_MaskWrite32Hdr have identical layouts. Opcode 4, a 64-bit register
// offset, then value, mask and the operation size.

// RUN: aie-translate --aie-npu-to-binary --aie-output-binary=false %s | FileCheck %s

// CHECK: 00000004
// CHECK-NEXT: 00000000
// CHECK-NEXT: 0021D000
// CHECK-NEXT: 00000000
// CHECK-NEXT: 00000001
// CHECK-NEXT: FFFFFFFF
// CHECK-NEXT: 0000001C

module {
  aie.device(npu2) {
    aie.runtime_sequence @seq() {
      %addr = arith.constant 0x21D000 : i32
      %val  = arith.constant 1 : i32
      %mask = arith.constant 0xFFFFFFFF : i32
      aiex.npu.maskpoll(%addr, %val, %mask) : i32, i32, i32
      aie.end
    }
  }
}
