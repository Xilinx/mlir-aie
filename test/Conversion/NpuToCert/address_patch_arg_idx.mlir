// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// cert.apply_offset_57 names the argument by its first 32-bit register,
// 2 * arg_idx, in a 16-bit field where 0xFFFF means the control code's own
// address. So arg_idx 32767 is the largest it can express, and a larger one is
// an error rather than a silently wrapped or redirected offset.

// RUN: aie-opt --aie-npu-to-cert --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK: aiex.cert.apply_offset_57(@t, 1, 32767)
module {
  aie.device(xcve3858) {
    memref.global "private" constant @t : memref<2xi32> = dense<[1, 0]>
    aie.runtime_sequence @seq(%a: memref<64xi32>) {
      %0 = memref.get_global @t : memref<2xi32>
      aiex.npu.blockwrite(%0) {address = 36864 : ui32} : memref<2xi32>
      %c0 = arith.constant 0 : i32
      aiex.npu.address_patch(%c0 : i32) {addr = 36868 : ui32, arg_idx = 32767 : i32}
    }
  }
}

// -----

// 2 * 32768 does not fit in 16 bits.
module {
  aie.device(xcve3858) {
    memref.global "private" constant @t : memref<2xi32> = dense<[1, 0]>
    aie.runtime_sequence @seq(%a: memref<64xi32>) {
      %0 = memref.get_global @t : memref<2xi32>
      aiex.npu.blockwrite(%0) {address = 36864 : ui32} : memref<2xi32>
      %c0 = arith.constant 0 : i32
      // expected-error@+1 {{arg_idx 32768 does not fit apply_offset_57's 16-bit argument offset; the largest is 32767}}
      aiex.npu.address_patch(%c0 : i32) {addr = 36868 : ui32, arg_idx = 32768 : i32}
    }
  }
}

// -----

// 65535 truncates to the control-code marker 0xFFFF.
module {
  aie.device(xcve3858) {
    memref.global "private" constant @t : memref<2xi32> = dense<[1, 0]>
    aie.runtime_sequence @seq(%a: memref<64xi32>) {
      %0 = memref.get_global @t : memref<2xi32>
      aiex.npu.blockwrite(%0) {address = 36864 : ui32} : memref<2xi32>
      %c0 = arith.constant 0 : i32
      // expected-error@+1 {{arg_idx 65535 does not fit apply_offset_57's 16-bit argument offset; the largest is 32767}}
      aiex.npu.address_patch(%c0 : i32) {addr = 36868 : ui32, arg_idx = 65535 : i32}
    }
  }
}
