//===- simple.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py -v --aie-generate-pdi --pdi-name=MlirAie.pdi --npu-insts-name=insts.txt %s | FileCheck %s
// RUN: ls | grep MlirAie.pdi | FileCheck %s --check-prefix=CHECK-FILE

// Note that llc determines the architecture from the llvm IR.
// CHECK: bootgen
// CHECK: copy{{.*}} to MlirAie.pdi
// CHECK-NOT: xclbinutil
// CHECK-FILE: MlirAie.pdi

module {
  aie.device(npu1_4col) {
  %12 = aie.tile(1, 2)
  %buf = aie.buffer(%12) : memref<256xi32>
  %4 = aie.core(%12)  {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 0 : index
    memref.store %0, %buf[%1] : memref<256xi32>
    aie.end
  }
  }
}
