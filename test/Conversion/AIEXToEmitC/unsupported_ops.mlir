//===- unsupported_ops.mlir - Unsupported op error test ----------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Negative test: verifies that the ConvertAIEXToEmitC pass emits an error
// when encountering unsupported ops (npu.push_queue) inside a runtime_sequence.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --convert-aiex-to-emitc %s 2>&1 | FileCheck %s

// CHECK: not supported in dynamic TXN C++ generation

module {
  aie.device(npu2) {
    aie.runtime_sequence() {
      aiex.npu.push_queue(0, 0, S2MM : 0) {issue_token = false, repeat_count = 0 : i32, bd_id = 0 : i32}
    }
  }
}
