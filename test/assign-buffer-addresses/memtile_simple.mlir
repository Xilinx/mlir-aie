//===- memtile_simple.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-buffer-addresses %s |& FileCheck %s
// CHECK:   {{.*}} AIE.buffer({{.*}}) {address = 0 : i32, sym_name = "a"} : memref<65536xi32>

module @test {
 AIE.device(xcve2302) {
  %0 = AIE.tile(3, 1)
  %b1 = AIE.buffer(%0) { sym_name = "a" } : memref<65536xi32>
  AIE.memTileDMA(%0) {
    AIE.end
  }
 }
}
