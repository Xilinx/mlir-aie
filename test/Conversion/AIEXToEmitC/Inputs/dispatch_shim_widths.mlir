//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Input for dispatch_shim.mlir's SIGNED prefix: one runtime-sequence parameter
// per shape the ABI string has to spell -- unsigned, i1, a narrow signed int,
// and index.

module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @widths(%arg0: memref<8xi32>, %u: ui32, %b: i1, %s8: i8, %idx: index) {
      %c = arith.constant 0 : i32
      aiex.npu.address_patch(%c : i32) {addr = 119300 : ui32, arg_idx = 0 : i32}
    }
  }
}
