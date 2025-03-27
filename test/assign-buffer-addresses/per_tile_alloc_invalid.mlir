//===- per_tile_alloc.mlir ----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-assign-buffer-addresses %s 2>&1 | FileCheck %s
// CHECK: error: 'aie.tile' op Shim tiles cannot have an allocation scheme

module @test {
  aie.device(xcvc1902) {
    %t1 = aie.tile(0, 3) { allocation_scheme="bank-aware" }
    %b1 = aie.buffer(%t1) { sym_name = "a" } : memref<16xi8>
    %b2 = aie.buffer(%t1) { sym_name = "b" } : memref<512xi32>
    %b3 = aie.buffer(%t1) { sym_name = "c" } : memref<16xi16>

    %t2 = aie.tile(4, 4) { allocation_scheme="basic-sequential" }
    %b4 = aie.buffer(%t2) : memref<500xi32>

    %t3 = aie.tile(0, 0) { allocation_scheme="basic-sequential" }

    aie.core(%t1) {
      aie.end
    }

    aie.core(%t2) {
      aie.end
    } {stackSize = 2048}
  }
}
