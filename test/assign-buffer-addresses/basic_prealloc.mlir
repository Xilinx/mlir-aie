//===- basic_prealloc.mlir -------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-buffer-addresses='alloc-scheme=basic-sequential' %s | FileCheck %s

// CHECK: aie.buffer(%{{.*}}) {address = 4096 : i32, sym_name = "b"} : memref<1024xi32>
// CHECK: aie.buffer(%{{.*}}) {address = 12288 : i32, sym_name = "c"} : memref<1024xi32>
// CHECK: aie.buffer(%{{.*}}) {address = 20000 : i32, sym_name = "d"} : memref<1024xi32>

module @test {
  aie.device(npu1) {
    %tile22 = aie.tile(2, 2)

    %buf0 = aie.buffer(%tile22) : memref<200xi32>
    %buf1 = aie.buffer(%tile22) : memref<100xi32>
    %buf2 = aie.buffer(%tile22) { sym_name = "b", address = 4096 : i32 } : memref<1024xi32>
    %buf3 = aie.buffer(%tile22) { sym_name = "c", address = 12288 : i32 } : memref<1024xi32>
    %buf4 = aie.buffer(%tile22) { sym_name = "d", address = 20000 : i32 } : memref<1024xi32>
    %buf5 = aie.buffer(%tile22) : memref<800xi32>
  }
}
