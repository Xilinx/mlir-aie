//===- badtiledma5.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s

module {
    %t21 = aie.tile(2, 2)
    %buf21_0 = aie.buffer(%t21) { sym_name = "buf21_0" } : memref<7168xi32>
    %l21_0 = aie.lock(%t21, 0)
    %m21 = aie.mem(%t21) {
      %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
        ^bd0:
        // CHECK: Burst length is only supported in Shim NOC tiles that are connected to the memory-mapped NOC.
        aie.dma_bd(%buf21_0 : memref<7168xi32>, 0, 7168){ burst_length = 256 : i32 }
        aie.next_bd ^end
      ^end:
        aie.end
    }
}
