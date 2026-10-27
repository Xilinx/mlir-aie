//===- badtiledma5.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s

module {
    %t21 = aie.tile(2, 2)
    aie.buffer @buf21_0(%t21) : memref<7168xi32>
    aie.lock @l21_0(%t21, 0)
    %m21 = aie.mem(%t21) {
      %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
        ^bd0:
        // CHECK: Burst length is only supported in Shim NOC tiles that are connected to the memory-mapped NOC.
        %buf21_0 = aie.get_buffer @buf21_0 : memref<7168xi32>
        aie.dma_bd(%buf21_0 : memref<7168xi32> offset = 0 len = 7168){ burst_length = 256 : i32 }
        aie.next_bd ^end
      ^end:
        aie.end
    }
}
