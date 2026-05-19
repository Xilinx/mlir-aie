//===- only_nsts.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that passing the --no-compile flag with --aie-generate-npu-insts generates _only_ the NPU instructions and skips other expensive compilation steps.

// RUN: %PYTHON aiecc.py -v --no-compile --aie-generate-npu-insts --npu-insts-name=my_insts.bin %s 2>&1 | FileCheck %s
// RUN: ls | FileCheck %s --check-prefix=LS
// CHECK-NOT: xchesscc_wrapper
// LS: my_insts.bin

module {
  aie.device(npu2) {
    %12 = aie.tile(1, 2)
    aie.runtime_sequence @seq(%buf : memref<1xi32>) {
        %cst_npu_0 = arith.constant 0 : i32
        %cst_npu_1 = arith.constant 0 : i32
        %cst_npu_2 = arith.constant 1 : i32
        %cst_npu_3 = arith.constant 0 : i32
        %cst_npu_4 = arith.constant 1 : i32
        %cst_npu_5 = arith.constant 1 : i32
        aiex.npu.sync(%cst_npu_0, %cst_npu_1, %cst_npu_2, %cst_npu_3, %cst_npu_4, %cst_npu_5) : i32, i32, i32, i32, i32, i32
    }
  }
}