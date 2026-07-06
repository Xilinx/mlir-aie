//===- npu_runtime_operand_error.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// The static TXN binary path can only encode npu ops whose operands are
// compile-time constants. A runtime-valued operand (e.g. a runtime-sequence
// argument) must fail translation with a clear diagnostic rather than emit a
// silently incomplete binary.

// RUN: not aie-translate --aie-npu-to-binary -aie-output-binary=false %s 2>&1 | FileCheck %s

// CHECK: 'aiex.npu.write32' op Cannot translate write32 with non-constant address or value to a static TXN binary
module {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: i32) {
      %addr = arith.constant 100 : i32
      aiex.npu.write32(%addr, %arg0) : i32, i32
    }
  }
}
