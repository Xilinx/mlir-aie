//===- per_tile_alloc.mlir ----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | FileCheck %s

module @test {
  aie.device(xcvc1902) {
    // CHECK: {{.*}} aie.tile(0, 1) {allocation_scheme = "bank-aware"}
    %t1 = aie.tile(0, 1) { allocation_scheme="bank-aware" }
    // CHECK: {{.*}} = aie.buffer({{.*}}) {address = {{.*}} : i32, mem_bank = {{.*}} : i32, sym_name = {{.*}}}
    %b1 = aie.buffer(%t1) : memref<16xi8>
    // CHECK: {{.*}} = aie.buffer({{.*}}) {address = {{.*}} : i32, mem_bank = {{.*}} : i32, sym_name = {{.*}}}
    %b2 = aie.buffer(%t1) : memref<512xi32>
    // CHECK: {{.*}} = aie.buffer({{.*}}) {address = {{.*}} : i32, mem_bank = {{.*}} : i32, sym_name = {{.*}}}
    %b3 = aie.buffer(%t1) : memref<16xi16>

    // The tile allocation scheme overrides the alloc-scheme flag.
    // CHECK: aie.tile(4, 4) {allocation_scheme = "basic-sequential"}
    %t2 = aie.tile(4, 4) { allocation_scheme="basic-sequential" }
    // CHECK: {address = {{.*}} : i32, sym_name = {{.*}}}
    %b4 = aie.buffer(%t2) : memref<500xi32>

    // The default allocation scheme is given by the falg in this case.
    // CHECK-NOT: allocation_scheme =
    %t3 = aie.tile(4,5)
    // CHECK: {{.*}} = aie.buffer({{.*}}) {address = {{.*}} : i32, mem_bank = {{.*}} : i32, sym_name = {{.*}}}
    %b5 = aie.buffer(%t3) : memref<500xi32>

    aie.core(%t1) {
      aie.end
    }

    aie.core(%t2) {
      aie.end
    } {stackSize = 2048}
  }
}
