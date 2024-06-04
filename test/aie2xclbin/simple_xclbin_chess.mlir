//===- simple.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie2xclbin -v --use-chess --host-target=aarch64-linux-gnu %s --xclbin-name=test.xclbin | FileCheck %s
// REQUIRES: valid_xchess_license, xrt

// Note that llc determines the architecture from the llvm IR.
// CHECK-NOT: llc
// CHECK: xchesscc_wrapper
// CHECK: bootgen
// CHECK: xclbinutil
// CHECK-NOT: llc

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
