//===- only_nsts.mlir ------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// Check that passing the --no-compile flag with --aie-generate-npu-insts generates _only_ the NPU instructions and skips other expensive compilation steps.

// RUN: %PYTHON aiecc.py -v --no-compile --aie-generate-npu-insts --npu-insts-name=my_insts.bin %s | FileCheck %s
// RUN: ls | FileCheck %s --check-prefix=LS
// CHECK-NOT: xchesscc_wrapper
// LS: my_insts.bin

module {
  aie.device(npu2) {
    %12 = aie.tile(1, 2)
    aie.runtime_sequence @seq(%buf : memref<1xi32>) {
        aiex.npu.sync { channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32 }
    }
  }
}