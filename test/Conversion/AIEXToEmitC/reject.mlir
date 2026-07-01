//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-aiex-to-emitc --verify-diagnostics

// Control flow inside a runtime sequence is out of scope for the straight-line
// C++ TXN target; it must be rejected with a diagnostic, never silently dropped.
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @seq_scf_for(%arg0: memref<8xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      // expected-error @+1 {{control flow in runtime sequences is not yet supported by the C++ TXN target}}
      scf.for %i = %c0 to %c4 step %c1 {
        %addr = arith.constant 100 : i32
        %val = arith.constant 7 : i32
        aiex.npu.write32(%addr, %val) : i32, i32
      }
    }
  }
}

// -----

module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @seq_scf_if(%arg0: memref<8xi32>, %cond: i1) {
      // expected-error @+1 {{control flow in runtime sequences is not yet supported by the C++ TXN target}}
      scf.if %cond {
        %addr = arith.constant 100 : i32
        %val = arith.constant 7 : i32
        aiex.npu.write32(%addr, %val) : i32, i32
      }
    }
  }
}

// -----

// A runtime (non-constant) register ADDRESS cannot be resolved statically: the
// address selects which hardware register and must fold in buffer/col/row at
// compile time. (Runtime data values are fine; only the address is rejected.)
// This matches the static binary emitter, which also requires a constant address.
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @seq_runtime_addr(%arg0: memref<8xi32>, %addr: i32, %val: i32) {
      // expected-error @+1 {{cannot convert a symbolic/unresolved write32 address to the C++ TXN target}}
      aiex.npu.write32(%addr, %val) : i32, i32
    }
  }
}
