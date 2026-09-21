//===- roundtrip_address_patch_arg_plus_64bit.mlir -------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// arg_plus occupies words 10-11 of the DDR_PATCH op, so re-import has to read
// both to round-trip an offset past 4 GiB. An offset that fits stays i32.

// RUN: aie-translate -aie-npu-to-binary %s -o %t.cfg
// RUN: %python txn2mlir.py -f %t.cfg | FileCheck %s

// CHECK: %[[SMALL:.*]] = arith.constant 256 : i32
// CHECK: aiex.npu.address_patch(%[[SMALL]] : i32) {addr = 74560 : ui32, arg_idx = 0 : i32}
// CHECK: %[[BIG:.*]] = arith.constant 8589934848 : i64
// CHECK: aiex.npu.address_patch(%[[BIG]] : i64) {addr = 74560 : ui32, arg_idx = 0 : i32}
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence() {
      %small = arith.constant 256 : i32
      aiex.npu.address_patch(%small : i32) {addr = 74560 : ui32, arg_idx = 0 : i32}
      %big = arith.constant 8589934848 : i64
      aiex.npu.address_patch(%big : i64) {addr = 74560 : ui32, arg_idx = 0 : i32}
    }
  }
}
