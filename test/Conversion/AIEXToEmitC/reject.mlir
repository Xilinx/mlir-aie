//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-aiex-to-emitc --verify-diagnostics

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
