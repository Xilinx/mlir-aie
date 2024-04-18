//===- simple.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: chess
// REQUIRES: peano

// RUN: %PYTHON aiecc.py --xchesscc --no-link -nv --aie-generate-cdo --aie-generate-npu --no-compile-host --xclbin-name=aie.xclbin --npu-insts-name=insts.txt %s | FileCheck %s --check-prefix=XCHESSCC
// RUN: %PYTHON aiecc.py --no-xchesscc --no-link -nv --aie-generate-cdo --aie-generate-npu --no-compile-host --xclbin-name=aie.xclbin --npu-insts-name=insts.txt %s | FileCheck %s --check-prefix=PEANO

// Note that llc determines the architecture from the llvm IR.
// XCHESSCC-NOT: {{^[^ ]*llc}}
// XCHESSCC: xchesscc_wrapper aie2
// XCHESSCC: bootgen
// XCHESSCC: xclbinutil
// XCHESSCC-NOT: {{^[^ ]*llc}}
// PEANO-NOT: xchesscc_wrapper
// PEANO: llc
// PEANO-SAME: --march=aie2
// PEANO: bootgen
// PEANO: xclbinutil

module {
  aie.device(npu) {
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
