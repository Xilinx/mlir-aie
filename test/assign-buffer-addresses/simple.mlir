//===- simple.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-buffer-addresses %s | FileCheck %s
// CHECK:   {{.*}} AIE.buffer({{.*}}) {address = 3104 : i32, sym_name = "a"} : memref<16xi8>
// CHECK:   {{.*}} AIE.buffer({{.*}}) {address = 1024 : i32, sym_name = "b"} : memref<512xi32>
// CHECK:   {{.*}} AIE.buffer({{.*}}) {address = 3072 : i32, sym_name = "c"} : memref<16xi16>
// CHECK:   {{.*}} AIE.buffer({{.*}}) {address = 1024 : i32, sym_name = "_anonymous0"} : memref<500xi32>

module @test {
 AIE.device(xcvc1902) {
  %0 = AIE.tile(3, 3)
  %b1 = AIE.buffer(%0) { sym_name = "a" } : memref<16xi8>
  %1 = AIE.buffer(%0) { sym_name = "b" } : memref<512xi32>
  %b2 = AIE.buffer(%0) { sym_name = "c" } : memref<16xi16>
  %3 = AIE.tile(4, 4)
  %4 = AIE.buffer(%3) : memref<500xi32>
 }
}
