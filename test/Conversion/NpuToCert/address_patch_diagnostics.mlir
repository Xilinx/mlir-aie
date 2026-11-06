// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// An address_patch that cert.apply_offset_57 cannot express is an error, not
// a silently dropped offset.

// RUN: aie-opt --aie-npu-to-cert --split-input-file --verify-diagnostics %s

module {
  aie.device(xcve3858) {
    memref.global "private" constant @t : memref<2xi32> = dense<[1, 0]>
    aie.runtime_sequence @seq(%a: memref<64xi32>, %off: i32) {
      %0 = memref.get_global @t : memref<2xi32>
      aiex.npu.blockwrite(%0) {address = 36864 : ui32} : memref<2xi32>
      // expected-error@+1 {{cert lowering requires a constant arg_plus: apply_offset_57 cannot add a runtime offset}}
      aiex.npu.address_patch(%off : i32) {addr = 36868 : ui32, arg_idx = 0 : i32}
    }
  }
}

// -----

module {
  aie.device(xcve3858) {
    memref.global "private" constant @t : memref<2xi32> = dense<[1, 0]>
    aie.runtime_sequence @seq(%a: memref<64xi32>) {
      %0 = memref.get_global @t : memref<2xi32>
      aiex.npu.blockwrite(%0) {address = 36864 : ui32} : memref<2xi32>
      %big = arith.constant 8589934592 : i64
      // expected-error@+1 {{arg_plus 8589934592 does not fit the BD's 32-bit address word}}
      aiex.npu.address_patch(%big : i64) {addr = 36868 : ui32, arg_idx = 0 : i32}
    }
  }
}

// -----

module {
  aie.device(xcve3858) {
    memref.global "private" constant @t : memref<2xi32> = dense<[1, 0]>
    aie.runtime_sequence @seq(%a: memref<64xi32>) {
      %0 = memref.get_global @t : memref<2xi32>
      aiex.npu.blockwrite(%0) {address = 36864 : ui32} : memref<2xi32>
      %c0 = arith.constant 0 : i32
      // A buffer operand that was never resolved to an argument index.
      // expected-error@+1 {{cert lowering requires an address_patch with arg_idx}}
      aiex.npu.address_patch(%c0 : i32) buffer %a : memref<64xi32> {addr = 36868 : ui32}
    }
  }
}
