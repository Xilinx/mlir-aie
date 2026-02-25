//===- example0.mlir -------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-standard-lowering %s | FileCheck %s

module {
  aie.device(npu1) {
    %t23 = aie.tile(2, 3)
    %t32 = aie.tile(3, 2)
    %t34 = aie.tile(3, 4)

    // CHECK: memref.global "public" @buf44 : memref<3x2xi32> = dense<{{\[}}[0, 1], [2, 3], [4, 5]]>
    // CHECK: memref.global "public" @buf42 : memref<2x3xi32> = dense<{{\[}}[0, 1, 2], [3, 4, 5]]>
    // CHECK: memref.global "public" @buf33 : memref<2x2xi32> = dense<{{\[}}[0, 1], [2, 3]]>
    %buf33 = aie.buffer(%t23) { sym_name = "buf33" } : memref<2x2xi32> = dense<[[0, 1], [2, 3]]>
    %buf42 = aie.buffer(%t32) { sym_name = "buf42" } : memref<2x3xi32> = dense<[[0, 1, 2], [3, 4, 5]]>
    %buf44 = aie.buffer(%t34) { sym_name = "buf44" } : memref<3x2xi32> = dense<[[0, 1], [2, 3], [4, 5]]>
  }
}
