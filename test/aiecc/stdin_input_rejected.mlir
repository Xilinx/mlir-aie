//===- stdin_input_rejected.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Every aiecc stage reopens its input by name, so stdin cannot be the input.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: cd %t.d && not %aiecc --no-progress - < %s 2>&1 | FileCheck %s

// CHECK: aiecc: reads its input from a file; '-' (stdin) is not supported

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
  }
}
