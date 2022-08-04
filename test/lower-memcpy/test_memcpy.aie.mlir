//===- test_memcpy.aie.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
// Date: August 4th 2022
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-memcpy %s | FileCheck %s

module @lower_memcpy {
    %tile12 = AIE.tile(1, 2)
    %tile33 = AIE.tile(3, 3)

    %buff12 = AIE.buffer(%tile12) : memref<256xi32>
    %buff33 = AIE.buffer(%tile33) : memref<256xi32>

    AIE.token(0) {sym_name = "token12"}

    %mem12 = AIE.mem(%tile12) {
        ^end:
            AIE.end
    }

    %mem33 = AIE.mem(%tile33) {
        ^end:
            AIE.end
    }

    AIE.memcpy @token12(1, 0) (%tile12 : <%buff12, 0, 256>, %tile33 : <%buff33, 0, 256>) : (memref<256xi32>, memref<256xi32>)
}