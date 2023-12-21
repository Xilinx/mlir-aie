//===- test_mmap3.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-mmap %s | FileCheck %s

// CHECK-LABEL: Tile(3, 4)
// CHECK-NEXT: Memory map: name base_address num_bytes
// CHECK-NEXT: _symbol south 0x20000 16
// CHECK-NEXT: _symbol same 0x28000 16
// CHECK-NEXT: _symbol north 0x30000 16
// CHECK-NEXT: _symbol east 0x38000 16

module @test_mmap1 {
 aie.device(xcvc1902) {
  %tsame = aie.tile(3, 4)
  %twest = aie.tile(2, 4) // Different column
  %teast = aie.tile(4, 4) // Different column
  %tsouth = aie.tile(3, 3) // Different row
  %tnorth = aie.tile(3, 5) // Different row

  %bufsame = aie.buffer(%tsame) { sym_name = "same", address = 0x0 } : memref<4xi32>
  %bufeast = aie.buffer(%teast) { sym_name = "east", address = 0x0 } : memref<4xi32>
  %bufwest = aie.buffer(%twest) { sym_name = "west", address = 0x0 } : memref<4xi32>
  %bufsouth = aie.buffer(%tsouth) { sym_name = "south", address = 0x0 } : memref<4xi32>
  %bufnorth = aie.buffer(%tnorth) { sym_name = "north", address = 0x0 } : memref<4xi32>
 }
}
