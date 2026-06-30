//===- buffers_xclbin_few.mlir ----------------------------------*- MLIR -*-===//
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %python aiecc.py -n --no-compile --no-link --aie-generate-xclbin %s
// RUN: FileCheck %s --input-file=buffers_xclbin_few.mlir.prj/main_kernels.json

// A runtime_sequence with 2 host buffer arguments should emit exactly bo0 and
// bo1 — no more. Verify the exact count is derived from the sequence, not a
// hardcoded minimum.

// CHECK: "name": "bo0"
// CHECK: "offset": "0x14"
// CHECK: "name": "bo1"
// CHECK: "offset": "0x1C"
// CHECK-NOT: "name": "bo2"

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in0 (%tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @out0 (%tile_0_0, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 0 : i64, metadata = @in0} : memref<1024xi32>
      aiex.npu.dma_memcpy_nd (%arg1[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 1 : i64, metadata = @out0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
  }
}
