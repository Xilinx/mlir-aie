//===- nd-dma-pad-exceeds-len.mlir -----------------------------*- MLIR -*-===//
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
    %buf = aie.buffer(%t1) : memref<256xi32>
    %mem = aie.memtile_dma(%t1) {
      aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        // expected-error@+1 {{'aie.dma_bd' op Data exceeds len after padding.}}
        aie.dma_bd(%buf : memref<256xi32>, dims = [<size = 2, stride = 128>], pad_dims = [<const_pad_before = 2, const_pad_after = 1>]) { len = 4 : i32, pad_value = 0 : i32 }
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
