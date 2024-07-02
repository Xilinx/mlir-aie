//===- test_mmap1.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-mmap %s | FileCheck %s

// CHECK-LABEL: Tile(2, 4)
// CHECK: _symbol a 0x38000 16
// CHECK-LABEL: Tile(3, 3)
// CHECK: _symbol a 0x30000 16
// CHECK-LABEL: Tile(3, 4)
// CHECK: _symbol a 0x28000 16
// CHECK-LABEL: Tile(3, 5)
// CHECK: _symbol a 0x20000 16
// CHECK-LABEL: Tile(4, 4)
// CHECK-NOT: _symbol a

module @test_mmap1 {
 aie.device(xcvc1902) {
  %t34 = aie.tile(3, 4)
  %t24 = aie.tile(2, 4) // Different column
  %t44 = aie.tile(4, 4) // Different column
  %t33 = aie.tile(3, 3) // Different row
  %t35 = aie.tile(3, 5) // Different row

  %buf34_0 = aie.buffer(%t34) { sym_name = "a", address = 0x0 : i32 } : memref<4xi32>
 }
}

