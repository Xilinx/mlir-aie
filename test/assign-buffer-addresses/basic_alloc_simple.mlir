//===- simple.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=basic-sequential" %s | FileCheck %s
// CHECK:   {{.*}} aie.buffer({{.*}}) {address = 3104 : i32, sym_name = "a"} : memref<16xi8>
// CHECK:   {{.*}} aie.buffer({{.*}}) {address = 1024 : i32, sym_name = "b"} : memref<512xi32>
// CHECK:   {{.*}} aie.buffer({{.*}}) {address = 3072 : i32, sym_name = "c"} : memref<16xi16>
// CHECK:   {{.*}} aie.buffer({{.*}}) {address = 1024 : i32, sym_name = "_anonymous0"} : memref<500xi32>

module @test {
 aie.device(xcvc1902) {
  %0 = aie.tile(3, 3)
  %b1 = aie.buffer(%0) { sym_name = "a" } : memref<16xi8>
  %1 = aie.buffer(%0) { sym_name = "b" } : memref<512xi32>
  %b2 = aie.buffer(%0) { sym_name = "c" } : memref<16xi16>
  %3 = aie.tile(4, 4)
  %4 = aie.buffer(%3) : memref<500xi32>
  aie.core(%0) {
    aie.end
  }
  aie.core(%3) {
    aie.end
  }
 }
}
