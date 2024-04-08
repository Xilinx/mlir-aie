// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023, Advanced Micro Devices, Inc.

module {
  func.func @dut(%arg0: memref<1024xbf16>, %arg1: memref<1024xbf16>) {
    affine.for %arg3 = 0 to 1024 {
      %0 = affine.load %arg0[%arg3] : memref<1024xbf16>
      %1 = math.exp %0 : bf16
      affine.store %1, %arg1[%arg3] : memref<1024xbf16>
    }
    return
  }
}