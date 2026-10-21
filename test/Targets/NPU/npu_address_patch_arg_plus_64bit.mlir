//===- npu_address_patch_arg_plus_64bit.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// arg_plus is a 64-bit field on the wire: aie-rt's patch_op_t declares
// `u64 argplus`, and the 12-word DDR_PATCH op carries it in words 10 AND 11,
// low half first, exactly as regaddr and argidx occupy 6-7 and 8-9. These
// checks pin both halves.
//
// Each check anchors on the register address (word 6, 0x12340 = 74560) and
// counts forward: w7 pad, w8 arg_idx, w9 pad, then the two arg_plus halves.
// arg_idx is 0 -- inside the firmware-translated set -- so no DDR aperture
// offset is folded in and the checked words are the offset itself.

// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false %s | FileCheck %s

module {
  aie.device(npu2) {
    aie.runtime_sequence(%a0: memref<8xi32>) {
      // Largest offset that still fits the low word: high word stays zero.
      // CHECK:      00012340
      // CHECK-NEXT: 00000000
      // CHECK-NEXT: 00000000
      // CHECK-NEXT: 00000000
      // CHECK-NEXT: FFFFFFFF
      // CHECK-NEXT: 00000000
      %max32 = arith.constant 4294967295 : i64
      aiex.npu.address_patch(%max32 : i64) {addr = 74560 : ui32, arg_idx = 0 : i32}

      // 2^32 exactly: the low word is zero and the whole offset sits high.
      // CHECK:      00012340
      // CHECK-NEXT: 00000000
      // CHECK-NEXT: 00000000
      // CHECK-NEXT: 00000000
      // CHECK-NEXT: 00000000
      // CHECK-NEXT: 00000001
      %four_gib = arith.constant 4294967296 : i64
      aiex.npu.address_patch(%four_gib : i64) {addr = 74560 : ui32, arg_idx = 0 : i32}

      // 8 GiB + 0x100: both halves non-zero, so neither is being dropped.
      // CHECK:      00012340
      // CHECK-NEXT: 00000000
      // CHECK-NEXT: 00000000
      // CHECK-NEXT: 00000000
      // CHECK-NEXT: 00000100
      // CHECK-NEXT: 00000002
      %eight_gib = arith.constant 8589934848 : i64
      aiex.npu.address_patch(%eight_gib : i64) {addr = 74560 : ui32, arg_idx = 0 : i32}

      // An i32 arg_plus is accepted and fills the low word, high word zero.
      // CHECK:      00012340
      // CHECK-NEXT: 00000000
      // CHECK-NEXT: 00000000
      // CHECK-NEXT: 00000000
      // CHECK-NEXT: 00000100
      // CHECK-NEXT: 00000000
      %small = arith.constant 256 : i32
      aiex.npu.address_patch(%small : i32) {addr = 74560 : ui32, arg_idx = 0 : i32}
    }
  }
}
