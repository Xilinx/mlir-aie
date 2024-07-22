//===- nd-dma-bad-pad.mlir -------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics %s

// CHECK-LABEL: module {
// CHECK:       }

module {
  aie.device(xcve2802) {
    %t1 = aie.tile(1, 1)
    %buf = aie.buffer(%t1) : memref<256xi8>
    %mem = aie.memtile_dma(%t1) {
      aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        // expected-error@+1 {{'aie.dma_bd' op Inner-most padding-before count must result in padding in 32-bit words.}}
        aie.dma_bd(%buf : memref<256xi8>, 0, 8, [<size = 4, stride = 1>], [<const_pad_before = 2, const_pad_after = 2>], pad_value = 0)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
